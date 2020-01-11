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

#include "sling/frame/serialization.h"
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
  task->Fetch("rnn_dim", &rnn_dim_);
  task->Fetch("rnn_layers", &rnn_layers_);
  task->Fetch("rnn_type", &rnn_type_);
  task->Fetch("rnn_bidir", &rnn_bidir_);
  task->Fetch("rnn_highways", &rnn_highways_);

  task->Fetch("mark_depth", &mark_depth_);
  task->Fetch("mark_dim", &mark_dim_);
  task->Fetch("frame_limit", &frame_limit_);
  task->Fetch("history_size", &history_size_);
  task->Fetch("out_roles_size", &out_roles_size_);
  task->Fetch("in_roles_size", &in_roles_size_);
  task->Fetch("labeled_roles_size", &labeled_roles_size_);
  task->Fetch("unlabeled_roles_size", &unlabeled_roles_size_);
  task->Fetch("roles_dim", &roles_dim_);
  task->Fetch("activations_dim", &activations_dim_);
  task->Fetch("link_dim_token", &link_dim_token_);
  task->Fetch("link_dim_step", &link_dim_step_);

  task->Fetch("seed", &seed_);
  task->Fetch("batch_size", &batch_size_);
  task->Fetch("learning_rate", &learning_rate_);
  task->Fetch("min_learning_rate", &min_learning_rate_);
  task->Fetch("learning_rate_cliff", &learning_rate_cliff_);
  task->Fetch("dropout", &dropout_);
  task->Fetch("ff_l2reg", &ff_l2reg_);

  // Statistics.
  num_tokens_ = task->GetCounter("tokens");
  num_documents_ = task->GetCounter("documents");
  num_transitions_ = task->GetCounter("transitions");

  // Open training and evaluation corpora.
  training_corpus_ =
    new DocumentCorpus(&commons_, task->GetInputFiles("training_corpus"));
  evaluation_corpus_ =
    new DocumentCorpus(&commons_, task->GetInputFiles("evaluation_corpus"));

  // Output file for model.
  auto *model_file = task->GetOutput("model");
  if (model_file != nullptr) {
    model_filename_ = model_file->resource()->name();
  }

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

  // Set up RNNs.
  RNN::Spec rnn_spec;
  rnn_spec.type = static_cast<RNN::Type>(rnn_type_);
  rnn_spec.dim = rnn_dim_;
  rnn_spec.highways = rnn_highways_;
  rnn_spec.dropout = dropout_;
  encoder_.AddLayers(rnn_layers_, rnn_spec, rnn_bidir_);

  // Custom parser model initialization. This should set up the word and role
  // vocabularies as well as the delegate cascade.
  Setup(task);

  // Build parser model flow graph.
  Build(&flow_, true);
  optimizer_ = GetOptimizer(task);
  optimizer_->Build(&flow_);

  // Compile model.
  compiler_.Compile(&flow_, &model_);

  // Get decoder cells and tensors.
  decoder_ = model_.GetCell("decoder");
  encodings_ = decoder_->GetParameter("decoder/tokens");
  activations_ = decoder_->GetParameter("decoder/steps");
  activation_ = decoder_->GetParameter("decoder/activation");

  gdecoder_ = decoder_->Gradient();
  primal_ = decoder_->Primal();
  dencodings_ = encodings_->Gradient();
  dactivations_ = activations_->Gradient();
  dactivation_ = activation_->Gradient();

  // Initialize model.
  feature_model_.Init(decoder_, &roles_, frame_limit_);
  model_.InitModelParameters(seed_);
  encoder_.Initialize(model_);
  optimizer_->Initialize(model_);
  for (auto *d : delegates_) d->Initialize(model_);
  commons_.Freeze();

  // Optionally load initial model parameters for restart.
  if (task->Get("restart", false) && !model_filename_.empty()) {
    LOG(INFO) << "Load model parameters from " << model_filename_;
    Flow initial;
    CHECK(initial.Load(model_filename_));
    model_.LoadParameters(initial);
  }

  // Train model.
  Train(task, &model_);

  // Save final model.
  if (!model_filename_.empty()) {
    LOG(INFO) << "Writing parser model to " << model_filename_;
    Save(model_filename_);
  }

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
  myelin::Channel dencodings(dencodings_);
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
      myelin::Channel *encodings =
          encoder.Compute(*document, 0, document->length());

      // Run decoder and delegates on all steps in the transition sequence.
      int t = 0;
      for (int s = 0; s < steps; ++s) {
        // Run next step of decoder.
        Instance *decoder = decoders[s];
        activations.zero(s);
        dactivations.zero(s);

        // Attach instance to recurrent layers.
        decoder->Clear();
        features.Attach(encodings, &activations, decoder);

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
      dencodings.reset(document->length());
      for (int s = steps - 1; s >= 0; --s) {
        gdecoder.Set(primal_, decoders[s]);
        gdecoder.Set(dencodings_, &dencodings);
        gdecoder.Set(dactivations_, &dactivations);
        gdecoder.Set(dactivation_, &dactivations, s);
        gdecoder.Compute();
      }

      // Propagate gradients back through encoder.
      encoder.Backpropagate(&dencodings);

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
    myelin::Channel *encodings = encoder.Compute(*document, s.begin(), s.end());

    // Initialize decoder.
    ParserState state(document, s.begin(), s.end());
    ParserFeatureExtractor features(&feature_model_, &state);
    myelin::Instance decoder(decoder_);
    myelin::Channel activations(feature_model_.activation());

    // Run decoder to predict transitions.
    while (!state.done()) {
      // Allocate space for next step.
      activations.push();

      // Attach instance to recurrent layers.
      decoder.Clear();
      features.Attach(encodings, &activations, &decoder);

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

      // Fall back to SHIFT if predicted action is not valid.
      if (!state.CanApply(action)) {
        action.type = ParserAction::SHIFT;
      }

      // Apply action to parser state.
      state.Apply(action);
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
  bool decay = prev_loss_ != 0.0 && prev_loss_ < loss;
  if (learning_rate_cliff_ != 0 && epoch >= learning_rate_cliff_) decay = true;
  if (learning_rate_ <= min_learning_rate_) decay = false;
  if (decay) learning_rate_ = optimizer_->DecayLearningRate();
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
  LOG(INFO) << "PAIR:  " << eval.pair.Summary();
  LOG(INFO) << "EDGE:  " << eval.edge.Summary();
  LOG(INFO) << "ROLE:  " << eval.role.Summary();
  LOG(INFO) << "TYPE:  " << eval.type.Summary();
  LOG(INFO) << "LABEL: " << eval.label.Summary();
  LOG(INFO) << "SLOT:  " << eval.slot.Summary();
  LOG(INFO) << "TOTAL: " << eval.combined.Summary();

  return true;
}

void ParserTrainer::Checkpoint(int64 epoch, Network *model) {
  if (!model_filename_.empty()) {
    LOG(INFO) << "Checkpointing model to " << model_filename_;
    Save(model_filename_);
  }
}

void ParserTrainer::Build(Flow *flow, bool learn) {
  // Build document input encoder.
  RNN::Variables rnn;
  if (learn) {
    Vocabulary::HashMapIterator vocab(words_);
    rnn = encoder_.Build(flow, spec_, &vocab, true);
  } else {
    rnn = encoder_.Build(flow, spec_, nullptr, false);
  }
  int token_dim = rnn.output->elements();

  // Build parser decoder.
  FlowBuilder f(flow, "decoder");
  std::vector<Flow::Variable *> features;

  // Add inputs for recurrent channels.
  auto *tokens = f.Placeholder("tokens", DT_FLOAT, {1, token_dim}, true);
  auto *steps = f.Placeholder("steps", DT_FLOAT, {1, activations_dim_}, true);

  // Role features.
  if (in_roles_size_ > 0) {
    features.push_back(f.Feature("in_roles", roles_.size() * frame_limit_,
                                 in_roles_size_, roles_dim_));
  }
  if (out_roles_size_ > 0) {
    features.push_back(f.Feature("out_roles", roles_.size() * frame_limit_,
                                 out_roles_size_, roles_dim_));
  }
  if (labeled_roles_size_ > 0) {
    features.push_back(f.Feature("labeled_roles",
                                 roles_.size() * frame_limit_ * frame_limit_,
                                 labeled_roles_size_, roles_dim_));
  }
  if (unlabeled_roles_size_ > 0) {
    features.push_back(f.Feature("unlabeled_roles",
                                 frame_limit_ * frame_limit_,
                                 unlabeled_roles_size_, roles_dim_));
  }

  // Link features.
  features.push_back(LinkedFeature(&f, "token", tokens, 1, link_dim_token_));
  features.push_back(LinkedFeature(&f, "attention_tokens",
                                   tokens, frame_limit_, link_dim_token_));
  features.push_back(LinkedFeature(&f, "attention_steps",
                                   steps, frame_limit_, link_dim_step_));
  features.push_back(LinkedFeature(&f, "history",
                                   steps, history_size_, link_dim_step_));

  // Mark features.
  features.push_back(LinkedFeature(&f, "mark_tokens",
                                   tokens, mark_depth_, link_dim_token_));
  features.push_back(LinkedFeature(&f, "mark_steps",
                                   steps, mark_depth_, link_dim_step_));

  // Pad feature vector.
  const static int alignment = 16;
  int n = 0;
  for (auto *f : features) n += f->elements();
  if (n % alignment != 0) {
    int padding = alignment - n % alignment;
    auto *zeroes = f.Const(nullptr, DT_FLOAT, {1, padding});
    features.push_back(zeroes);
  }

  // Concatenate mapped feature inputs.
  auto *fv = f.Concat(features);
  int fvsize = fv->dim(1);

  // Feed-forward layer.
  auto *W = f.Parameter("W0", DT_FLOAT, {fvsize, activations_dim_});
  auto *b = f.Parameter("b0", DT_FLOAT, {1, activations_dim_});
  f.RandomNormal(W);
  if (ff_l2reg_ != 0.0) W->SetAttr("l2reg", ff_l2reg_);
  auto *activation = f.Name(f.Relu(f.Add(f.MatMul(fv, W), b)), "activation");
  activation->set_in()->set_out()->set_ref();

  // Build function decoder gradient.
  Flow::Variable *dactivation = nullptr;
  if (learn) {
    Gradient(flow, f.func());
    dactivation = flow->GradientVar(activation);
  }

  // Build flows for delegates.
  for (DelegateLearner *delegate : delegates_) {
    delegate->Build(flow, activation, dactivation, learn);
  }

  // Link recurrences.
  flow->Connect({tokens, rnn.output});
  flow->Connect({steps, activation});
  if (learn) {
    auto *dsteps = flow->GradientVar(steps);
    flow->Connect({dsteps, dactivation});
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

void ParserTrainer::Save(const string &filename) {
  // Build model.
  Flow flow;
  Build(&flow, false);

  // Copy weights from trained model.
  model_.SaveParameters(&flow);

  // Save lexicon.
  encoder_.SaveLexicon(&flow);

  // Make parser specification frame.
  Store store(&commons_);
  Builder spec(&store);

  // Save encoder spec.
  Builder encoder_spec(&store);
  encoder_spec.Add("type", "lexrnn");
  encoder_spec.Add("rnn", static_cast<int>(rnn_type_));
  encoder_spec.Add("dim", rnn_dim_);
  encoder_spec.Add("layers", rnn_layers_);
  encoder_spec.Add("bidir", rnn_bidir_);
  encoder_spec.Add("highways", rnn_highways_);
  spec.Set("encoder", encoder_spec.Create());

  // Save decoder spec.
  Builder decoder_spec(&store);
  decoder_spec.Add("type", "transition");
  decoder_spec.Set("frame_limit", frame_limit_);
  decoder_spec.Set("sentence_reset", sentence_reset_);

  Handles role_list(&store);
  roles_.GetList(&role_list);
  decoder_spec.Set("roles", Array(&store, role_list));

  Array delegates(&store, delegates_.size());
  for (int i = 0; i < delegates_.size(); ++i) {
    Builder delegate_spec(&store);
    delegates_[i]->Save(&flow, &delegate_spec);
    delegates.set(i, delegate_spec.Create().handle());
  }
  decoder_spec.Set("delegates", delegates);

  spec.Set("decoder", decoder_spec.Create());

  // Save parser spec in flow.
  StringEncoder encoder(&store);
  encoder.Encode(spec.Create());

  Flow::Blob *blob = flow.AddBlob("parser", "frame");
  blob->data = flow.AllocateMemory(encoder.buffer());
  blob->size = encoder.buffer().size();

  // Save model to file.
  DCHECK(flow.IsConsistent());
  flow.Save(filename);
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

