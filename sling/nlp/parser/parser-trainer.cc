// Copyright 2017 Google Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "sling/nlp/parser/parser-trainer.h"

#include <math.h>

#include "sling/myelin/gradient.h"
#include "sling/nlp/document/document.h"
#include "sling/nlp/document/lexicon.h"
#include "sling/util/unicode.h"

namespace sling {
namespace nlp {

using namespace task;
using namespace myelin;

ParserTrainer::~ParserTrainer() {
  for (auto *d : delegates_) delete d;
  delete training_corpus_;
  delete evaluation_corpus_;
}

void ParserTrainer::Run(task::Task *task) {
  // Get training parameters.
  task->Fetch("lstm_dim", &lstm_dim_);
  task->Fetch("max_source", &max_source_);
  task->Fetch("max_target", &max_target_);
  task->Fetch("mark_depth", &mark_depth_);
  task->Fetch("frame_limit", &frame_limit_);
  task->Fetch("attention_depth", &attention_depth_);
  task->Fetch("history_size", &history_size_);
  task->Fetch("out_roles_size", &out_roles_size_);
  task->Fetch("in_roles_size", &in_roles_size_);
  task->Fetch("labeled_roles_size", &labeled_roles_size_);
  task->Fetch("unlabeled_roles_size", &unlabeled_roles_size_);
  task->Fetch("roles_dim", &roles_dim_);
  task->Fetch("activations_dim", &activations_dim_);
  task->Fetch("link_dim_lstm", &link_dim_lstm_);
  task->Fetch("link_dim_ff", &link_dim_ff_);
  task->Fetch("mark_dim", &mark_dim_);
  task->Fetch("seed", &seed_);
  task->Fetch("batch_size", &batch_size_);
  task->Fetch("learning_rate", &learning_rate_);
  task->Fetch("min_learning_rate", &min_learning_rate_);

  // Statistics.
  num_tokens_ = task->GetCounter("tokens");
  num_documents_ = task->GetCounter("documents");
  num_transitions_ = task->GetCounter("transitions");

  // Open training and evaluation corpora.
  training_corpus_ =
    new DocumentCorpus(&commons_, task->GetInputFiles("training_corpus"));
  evaluation_corpus_ =
    new DocumentCorpus(&commons_, task->GetInputFiles("evaluation_corpus"));

  // Set up encoder lexicon.
  string normalization = task->Get("normalization", "d");
  spec_.lexicon.normalization = ParseNormalization(normalization);
  spec_.lexicon.threshold = task->Get("lexicon_threshold", 0);
  spec_.lexicon.max_prefix = task->Get("max_prefix", 0);
  spec_.lexicon.max_suffix = task->Get("max_suffix", 3);
  spec_.feature_padding = 16;

  // Set up word embeddings.
  spec_.word_dim = task->Get("word_dim", 32);
  auto *word_embeddings_input = task->GetInput("word_embeddings");
  if (word_embeddings_input != nullptr) {
    spec_.word_embeddings = word_embeddings_input->resource()->name();
  }
  spec_.train_word_embeddings = task->Get("train_word_embeddings", true);

  // Set up lexical back-off features.
  spec_.prefix_dim = task->Get("prefix_dim", 0);
  spec_.suffix_dim = task->Get("suffix_dim", 16);
  spec_.hyphen_dim = task->Get("hypen_dim", 8);
  spec_.caps_dim = task->Get("caps_dim", 8);;
  spec_.punct_dim = task->Get("punct_dim", 8);;
  spec_.quote_dim = task->Get("quote_dim", 8);;
  spec_.digit_dim = task->Get("digit_dim", 8);;

  // Custom parser model initialization. This should set up the word and role
  // vocabularies as well as the delegate cascade.
  Setup(task);

  // Build parser model flow graph.
  BuildFlow(&flow_, true);
  optimizer_ = GetOptimizer(task);
  optimizer_->Build(&flow_);

  // Compile model.
  compiler_.Compile(&flow_, &net_);

  // Get decoder cells and tensors.
  decoder_ = net_.GetCell("ff_trunk");
  activations_ = decoder_->GetParameter("ff_trunk/steps");
  gdecoder_ = decoder_->Gradient();
  primal_ = decoder_->Primal();
  dactivations_ = activations_->Gradient();
  dactivation_ = gdecoder_->GetParameter("ff_trunk/hidden")->Gradient();
  dlr_ = gdecoder_->GetParameter("gradients/ff_trunk/d_lr_lstm");
  drl_ = gdecoder_->GetParameter("gradients/ff_trunk/d_rl_lstm");

  // Initialize model.
  feature_model_.Init(net_.GetCell("ff_trunk"),
                      flow_.DataBlock("spec"),
                      &roles_, frame_limit_);
  net_.InitLearnableWeights(seed_, 0.0, 0.01);
  encoder_.Initialize(net_);
  optimizer_->Initialize(net_);
  for (auto *d : delegates_) d->Initialize(net_);
  commons_.Freeze();

  // Train model.
  Train(task, &net_);

  // Clean up.
  delete optimizer_;
}

void ParserTrainer::Worker(int index, Network *model) {
  // Create instances.
  LexicalEncoderLearner encoder(encoder_);
  Instance gdecoder(gdecoder_);
  std::vector<DelegateLearnerInstance *> delegates;
  for (auto *d : delegates_) delegates.push_back(d->CreateInstance());

  // Collect gradients.
  std::vector<Instance *> gradients;
  encoder.CollectGradients(&gradients);
  gradients.push_back(&gdecoder);
  for (auto *d : delegates) d->CollectGradients(&gradients);

  // Training loop.
  std::vector<ParserAction> transitions;
  std::vector<Instance *> decoders;
  myelin::Channel activations(activations_);
  myelin::Channel dactivations(dactivations_);
  for (;;) {
    // Prepare next batch.
    for (auto *g : gradients) g->Clear();
    float epoch_loss = 0.0;
    int epoch_count = 0;

    for (int b = 0; b < batch_size_; b++) {
      // Get next training document.
      Store store(&commons_);
      Document *document = GetNextTrainingDocument(&store);
      CHECK(document != nullptr);
      num_documents_->Increment();
      num_tokens_->Increment(document->length());

      // Generate transitions for document.
      GenerateTransitions(*document, &transitions);
      num_transitions_->Increment(transitions.size());
      document->ClearAnnotations();

      // Compute the number of decoder steps.
      int steps = 0;
      for (const ParserAction &action : transitions) {
        if (action.type != ParserAction::CASCADE) steps++;
      }

      // Set up parser state.
      ParserState state(document, 0, document->length());
      ParserFeatureExtractor features(&feature_model_, &state);

      // Set up channels and instances for decoder.
      activations.resize(steps);
      dactivations.resize(steps);
      while (decoders.size() < steps) {
        decoders.push_back(new Instance(decoder_));
      }

      // Run document through encoder to produce contextual token encodings.
      auto bilstm = encoder.Compute(*document, 0, document->length());

      // Run decoder and delegates on all steps in the transition sequence.
      int t = 0;
      for (int s = 0; s < steps; ++s) {
        // Run next step of decoder.
        Instance *decoder = decoders[s];
        activations.zero(s);
        dactivations.zero(s);

        // Attach instance to recurrent layers.
        decoder->Clear();
        features.Attach(bilstm, &activations, decoder);

        // Extract features.
        features.Extract(decoder);

        // Compute decoder activations.
        decoder->Compute();

        // Run the cascade.
        float *fwd = reinterpret_cast<float *>(activations.at(s));
        float *bkw = reinterpret_cast<float *>(dactivations.at(s));
        int d = 0;
        for (;;) {
          ParserAction &action = transitions[t];
          float loss = delegates[d]->Compute(fwd, bkw, action);
          epoch_loss += loss;
          epoch_count++;
          if (action.type != ParserAction::CASCADE) break;
          CHECK_GT(action.delegate, d);
          d = action.delegate;
          t++;
        }

        // Apply action to parser state.
        state.Apply(transitions[t++]);
      }

      // Propagate gradients back through decoder.
      auto grad = encoder.PrepareGradientChannels(document->length());
      for (int s = steps - 1; s >= 0; --s) {
        gdecoder.Set(primal_, decoders[s]);
        gdecoder.Set(dactivations_, &dactivations);
        gdecoder.Set(dactivation_, &dactivations, s);
        gdecoder.Set(dlr_, grad.lr);
        gdecoder.Set(drl_, grad.rl);
        gdecoder.Compute();
      }

      // Propagate gradients back through encoder.
      encoder.Backpropagate();

      delete document;
    }

    // Update parameters.
    update_mu_.Lock();
    optimizer_->Apply(gradients);
    loss_sum_ += epoch_loss;
    loss_count_ += epoch_count;
    update_mu_.Unlock();

    // Check if we are done.
    if (EpochCompleted()) break;
  }

  // Clean up.
  for (auto *d : decoders) delete d;
  for (auto *d : delegates) delete d;
}

void ParserTrainer::Parse(Document *document) const {
  // Create delegates.
  std::vector<DelegateLearnerInstance *> delegates;
  for (auto *d : delegates_) delegates.push_back(d->CreateInstance());

  // Parse each sentence of the document.
  for (SentenceIterator s(document); s.more(); s.next()) {
    // Run the lexical encoder for sentence.
    LexicalEncoderInstance encoder(encoder_);
    auto bilstm = encoder.Compute(*document, s.begin(), s.end());

    // Initialize decoder.
    ParserState state(document, s.begin(), s.end());
    ParserFeatureExtractor features(&feature_model_, &state);
    myelin::Instance decoder(decoder_);
    myelin::Channel activations(feature_model_.hidden());

    // Run decoder to predict transitions.
    for (;;) {
      // Allocate space for next step.
      activations.push();

      // Attach instance to recurrent layers.
      decoder.Clear();
      features.Attach(bilstm, &activations, &decoder);

      // Extract features.
      features.Extract(&decoder);

      // Compute decoder activations.
      decoder.Compute();

      // Run the cascade.
      ParserAction action(ParserAction::CASCADE, 0);
      int step = state.step();
      float *activation = reinterpret_cast<float *>(activations.at(step));
      int d = 0;
      for (;;) {
        delegates[d]->Predict(activation, &action);
        if (action.type != ParserAction::CASCADE) break;
        CHECK_GT(action.delegate, d);
        d = action.delegate;
      }

      // Shift or stop if predicted action is invalid.
      if (!state.CanApply(action)) {
        if (state.current() < state.end()) {
          action.type = ParserAction::SHIFT;
        } else {
          action.type = ParserAction::STOP;
        }
      }

      // Apply action to parser state.
      state.Apply(action);

      // Check if we are done.
      if (action.type == ParserAction::STOP) break;
    }
  }

  for (auto *d : delegates) delete d;
}

bool ParserTrainer::Evaluate(int64 epoch, Network *model) {
  // Skip evaluation if there are no data.
  if (loss_count_ == 0) return true;

  // Compute average loss of epochs since last eval.
  float loss = loss_sum_ / loss_count_;
  float p = exp(-loss) * 100.0;
  loss_sum_ = 0.0;
  loss_count_ = 0;

  // Decay learning rate if loss increases.
  if (prev_loss_ != 0.0 &&
      prev_loss_ < loss &&
      learning_rate_ > min_learning_rate_) {
    learning_rate_ = optimizer_->DecayLearningRate();
  }
  prev_loss_ = loss;

  LOG(INFO) << "epoch=" << epoch
            << " lr=" << learning_rate_
            << " loss=" << loss
            << " p=" << p;

  // Evaluate current model on held-out evaluation corpus.
  ParserEvaulationCorpus corpus(this);
  FrameEvaluation::Output eval;
  FrameEvaluation::Evaluate(&corpus, &eval);
  LOG(INFO) << "SPAN:  " << eval.mention.Summary();
  LOG(INFO) << "FRAME: " << eval.frame.Summary();
  LOG(INFO) << "TYPE:  " << eval.type.Summary();
  LOG(INFO) << "ROLE:  " << eval.role.Summary();
  LOG(INFO) << "LABEL: " << eval.label.Summary();
  LOG(INFO) << "SLOT:  " << eval.slot.Summary();
  LOG(INFO) << "TOTAL: " << eval.combined.Summary();

  return true;
}

void ParserTrainer::Checkpoint(int64 epoch, Network *model) {
}

void ParserTrainer::BuildFlow(Flow *flow, bool learn) {
  // Build document input encoder.
  BiLSTM::Outputs lstm;
  if (learn) {
    Vocabulary::HashMapIterator vocab(words_);
    lstm = encoder_.Build(flow, spec_, &vocab, lstm_dim_, true);
  } else {
    lstm = encoder_.Build(flow, spec_, nullptr, lstm_dim_, false);
  }

  // Build parser decoder.
  FlowBuilder f(flow, "ff_trunk");
  std::vector<Flow::Variable *> features;
  Flow::Blob *spec = flow->AddBlob("spec", "");

  // Add inputs for recurrent channels.
  auto *lr = f.Placeholder("link/lr_lstm", DT_FLOAT, {1, lstm_dim_}, true);
  auto *rl = f.Placeholder("link/rl_lstm", DT_FLOAT, {1, lstm_dim_}, true);
  auto *steps = f.Placeholder("steps", DT_FLOAT, {1, activations_dim_}, true);

  // Role features.
  if (in_roles_size_ > 0) {
    features.push_back(f.Feature("in-roles", roles_.size() * frame_limit_,
                                 in_roles_size_, roles_dim_));
  }
  if (out_roles_size_ > 0) {
    features.push_back(f.Feature("out-roles", roles_.size() * frame_limit_,
                                 out_roles_size_, roles_dim_));
  }
  if (labeled_roles_size_ > 0) {
    features.push_back(f.Feature("labeled-roles",
                                 roles_.size() * frame_limit_ * frame_limit_,
                                 labeled_roles_size_, roles_dim_));
  }
  if (unlabeled_roles_size_ > 0) {
    features.push_back(f.Feature("unlabeled-roles",
                                 frame_limit_ * frame_limit_,
                                 unlabeled_roles_size_, roles_dim_));
  }

  // Link features.
  features.push_back(LinkedFeature(&f, "frame-creation-steps",
                                   steps, frame_limit_, link_dim_ff_));
  features.push_back(LinkedFeature(&f, "frame-focus-steps",
                                   steps, frame_limit_, link_dim_ff_));
  features.push_back(LinkedFeature(&f, "history",
                                   steps, history_size_, link_dim_ff_));
  features.push_back(LinkedFeature(&f, "frame-end-lr",
                                   lr, frame_limit_, link_dim_lstm_));
  features.push_back(LinkedFeature(&f, "frame-end-rl",
                                   rl, frame_limit_, link_dim_lstm_));
  features.push_back(LinkedFeature(&f, "lr", lr, 1, link_dim_lstm_));
  features.push_back(LinkedFeature(&f, "rl", rl, 1, link_dim_lstm_));

  // Mark features.
  features.push_back(f.Feature("mark-distance",
                               mark_distance_bins_.size() + 1,
                               mark_depth_, mark_dim_));
  features.push_back(LinkedFeature(&f, "mark-lr",
                                   lr, mark_depth_, link_dim_lstm_));
  features.push_back(LinkedFeature(&f, "mark-rl",
                                   rl, mark_depth_, link_dim_lstm_));
  features.push_back(LinkedFeature(&f, "mark-step",
                                   steps, mark_depth_, link_dim_ff_));
  string bins;
  for (int d : mark_distance_bins_) {
    if (!bins.empty()) bins.push_back(' ');
    bins.append(std::to_string(d));
  }
  spec->SetAttr("mark_distance_bins", bins);

  // Concatenate mapped feature inputs.
  auto *fv = f.Concat(features);
  int fvsize = fv->dim(1);

  // Feed-forward layer.
  auto *W = f.Random(f.Parameter("W0", DT_FLOAT, {fvsize, activations_dim_}));
  auto *b = f.Random(f.Parameter("b0", DT_FLOAT, {1, activations_dim_}));
  auto *activations = f.Name(f.Relu(f.Add(f.MatMul(fv, W), b)), "hidden");
  activations->set_in()->set_out()->set_ref();

  // Build function decoder gradient.
  Flow::Variable *dactivations = nullptr;
  if (learn) {
    Gradient(flow, f.func());
    dactivations = flow->GradientVar(activations);
  }

  // Build flows for delegates.
  for (DelegateLearner *delegate : delegates_) {
    delegate->Build(flow, activations, dactivations, learn);
  }

  // Link recurrences.
  flow->Connect({lstm.lr, lr});
  flow->Connect({lstm.rl, rl});
  flow->Connect({steps, activations});
  if (learn) {
    auto *dsteps = flow->GradientVar(steps);
    flow->Connect({dsteps, dactivations});
  }
}

Flow::Variable *ParserTrainer::LinkedFeature(FlowBuilder *f,
                                             const string &name,
                                             Flow::Variable *embeddings,
                                             int size, int dim) {
  int link_dim = embeddings->dim(1);
  auto *features = f->Placeholder(name, DT_INT32, {1, size});
  auto *oov = f->Parameter(name + "_oov", DT_FLOAT, {1, link_dim});
  auto *gather = f->Gather(embeddings, features, oov);
  auto *transform = f->Parameter(name + "_transform", DT_FLOAT,
                                 {link_dim, dim});
  return f->Reshape(f->MatMul(gather, transform), {1, size * dim});
}

Document *ParserTrainer::GetNextTrainingDocument(Store *store) {
  MutexLock lock(&input_mu_);
  Document *document = training_corpus_->Next(store);
  if (document == nullptr) {
    // Loop around if the end of the training corpus has been reached.
    training_corpus_->Rewind();
    document = training_corpus_->Next(store);
  }
  return document;
}

ParserTrainer::ParserEvaulationCorpus::ParserEvaulationCorpus(
    ParserTrainer *trainer) : trainer_(trainer) {
  trainer_->evaluation_corpus_->Rewind();
}

bool ParserTrainer::ParserEvaulationCorpus::Next(Store **store,
                                                 Document **golden,
                                                 Document **predicted) {
  // Create a store for both golden and parsed document.
  Store *local = new Store(&trainer_->commons_);

  // Read next document from corpus.
  Document *document = trainer_->evaluation_corpus_->Next(local);
  if (document == nullptr) {
    delete local;
    return false;
  }

  // Clone document without annotations.
  Document *parsed = new Document(*document, false);

  // Parse the document using the current model.
  trainer_->Parse(parsed);
  parsed->Update();

  // Return golden and predicted documents.
  *store = local;
  *golden = document;
  *predicted = parsed;

  return true;
}

}  // namespace nlp
}  // namespace sling

