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

#include <math.h>

#include "sling/nlp/parser/parser.h"

#include "sling/frame/serialization.h"
#include "sling/myelin/cuda/cuda-runtime.h"
#include "sling/myelin/kernel/cuda.h"
#include "sling/myelin/kernel/dragnn.h"
#include "sling/myelin/kernel/tensorflow.h"
#include "sling/nlp/document/document.h"
#include "sling/nlp/document/features.h"
#include "sling/nlp/document/lexicon.h"

namespace sling {
namespace nlp {

static myelin::CUDARuntime cudart;

void Parser::EnableGPU() {
  if (myelin::CUDA::Supported()) {
    // Initialize CUDA runtime for Myelin.
    if (!cudart.connected()) {
      cudart.Connect();
    }

    // Always use fast fallback when running on GPU.
    use_gpu_ = true;
    fast_fallback_ = true;
  }
}

void Parser::Load(Store *store, const string &model) {
  // Register kernels for implementing parser ops.
  RegisterTensorflowLibrary(&library_);
  RegisterDragnnLibrary(&library_);
  if (use_gpu_) RegisterCUDALibrary(&library_);

  // Load and analyze parser flow file.
  myelin::Flow flow;
  CHECK(flow.Load(model));

  // Add argmax for fast fallback.
  if (fast_fallback_) {
    auto *ff = flow.Func("ff");
    auto *output = flow.Var("ff/output");
    auto *prediction = flow.AddVariable("ff/prediction", myelin::DT_INT32, {1});
    flow.AddOperation(ff, "ff/ArgMax", "ArgMax", {output}, {prediction});
  }

  // Analyze parser flow file.
  flow.Analyze(library_);

  // Compile parser flow.
  if (use_gpu_) network_.set_runtime(&cudart);
  CHECK(network_.Compile(flow, library_));

  // Initialize cells.
  InitLSTM("lr_lstm", &lr_, false);
  InitLSTM("rl_lstm", &rl_, true);
  InitFF("ff", &ff_);

  // Initialize profiling.
  if (ff_.cell->profile()) profile_ = new Profile(this);

  // Load lexicon.
  myelin::Flow::Blob *vocabulary = flow.DataBlock("lexicon");
  CHECK(vocabulary != nullptr);
  lexicon_.InitWords(vocabulary->data, vocabulary->size);
  bool normalize = vocabulary->attrs.Get("normalize_digits", false);
  int oov = vocabulary->attrs.Get("oov", -1);
  lexicon_.set_normalize_digits(normalize);
  lexicon_.set_oov(oov);

  // Load affix tables.
  myelin::Flow::Blob *prefix_table = flow.DataBlock("prefixes");
  if (prefix_table != nullptr) {
    lexicon_.InitPrefixes(prefix_table->data, prefix_table->size);
  }
  myelin::Flow::Blob *suffix_table = flow.DataBlock("suffixes");
  if (suffix_table != nullptr) {
    lexicon_.InitSuffixes(suffix_table->data, suffix_table->size);
  }

  // Load commons and action stores.
  myelin::Flow::Blob *commons = flow.DataBlock("commons");
  if (commons != nullptr) {
    StringDecoder decoder(store, commons->data, commons->size);
    decoder.DecodeAll();
  }
  myelin::Flow::Blob *actions = flow.DataBlock("actions");
  if (actions != nullptr) {
    StringDecoder decoder(store, actions->data, actions->size);
    decoder.DecodeAll();
  }

  // Initialize action table.
  store_ = store;
  actions_.Init(store);
  num_actions_ = actions_.NumActions();
  CHECK_GT(num_actions_, 0);
  roles_.Init(actions_);
}

void Parser::InitLSTM(const string &name, LSTM *lstm, bool reverse) {
  // Get cell.
  lstm->cell = GetCell(name);
  lstm->reverse = reverse;
  lstm->profile = lstm->cell->profile();

  // Get connectors.
  lstm->control = GetConnector(name + "/control");
  lstm->hidden = GetConnector(name + "/hidden");

  // Get feature inputs.
  lstm->word_feature = GetParam(name + "/words", true);
  lstm->prefix_feature = GetParam(name + "/prefix", true);
  lstm->suffix_feature = GetParam(name + "/suffix", true);
  lstm->hyphen_feature = GetParam(name + "/hyphen", true);
  lstm->caps_feature = GetParam(name + "/capitalization", true);
  lstm->punct_feature = GetParam(name + "/punctuation", true);
  lstm->quote_feature = GetParam(name + "/quote", true);
  lstm->digit_feature = GetParam(name + "/digit", true);

  // Get feature sizes.
  if (lstm->prefix_feature != nullptr) {
    lstm->prefix_size = lstm->prefix_feature->elements();
  }
  if (lstm->suffix_feature != nullptr) {
    lstm->suffix_size = lstm->suffix_feature->elements();
  }

  // Get links.
  lstm->c_in = GetParam(name + "/c_in");
  lstm->c_out = GetParam(name + "/c_out");
  lstm->h_in = GetParam(name + "/h_in");
  lstm->h_out = GetParam(name + "/h_out");
}

void Parser::InitFF(const string &name, FF *ff) {
  // Get cell.
  ff->cell = GetCell(name);
  ff->profile = ff->cell->profile();

  // Get connector for recurrence.
  ff->step = GetConnector(name + "/step");

  // Get feature inputs.
  ff->lr_focus_feature = GetParam(name + "/lr", true);
  ff->rl_focus_feature = GetParam(name + "/rl", true);
  ff->lr_attention_feature = GetParam(name + "/frame-end-lr", true);
  ff->rl_attention_feature = GetParam(name + "/frame-end-rl", true);
  ff->frame_create_feature = GetParam(name + "/frame-creation-steps", true);
  ff->frame_focus_feature = GetParam(name + "/frame-focus-steps", true);
  ff->history_feature = GetParam(name + "/history", true);
  ff->out_roles_feature = GetParam(name + "/out-roles", true);
  ff->in_roles_feature = GetParam(name + "/in-roles", true);
  ff->unlabeled_roles_feature = GetParam(name + "/unlabeled-roles", true);
  ff->labeled_roles_feature = GetParam(name + "/labeled-roles", true);

  // Get feature sizes.
  std::vector<myelin::Tensor *> attention_features {
    ff->lr_attention_feature,
    ff->rl_attention_feature,
    ff->frame_create_feature,
    ff->frame_focus_feature,
  };
  for (auto *f : attention_features) {
    if (!f) continue;
    if (f->elements() > ff->attention_depth) {
      ff->attention_depth = f->elements();
    }
  }
  for (auto *f : attention_features) {
    if (!f) continue;
    CHECK_EQ(ff->attention_depth, f->elements());
  }
  if (ff->history_feature != nullptr) {
    ff->history_size = ff->history_feature->elements();
  }
  if (ff->out_roles_feature != nullptr) {
    ff->out_roles_size = ff->out_roles_feature->elements();
  }
  if (ff->in_roles_feature != nullptr) {
    ff->in_roles_size = ff->in_roles_feature->elements();
  }
  if (ff->unlabeled_roles_feature != nullptr) {
    ff->unlabeled_roles_size = ff->unlabeled_roles_feature->elements();
  }
  if (ff->labeled_roles_feature != nullptr) {
    ff->labeled_roles_size = ff->labeled_roles_feature->elements();
  }

  // Get links.
  ff->lr_lstm = GetParam(name + "/link/lr_lstm");
  ff->rl_lstm = GetParam(name + "/link/rl_lstm");
  ff->steps = GetParam(name + "/steps");
  ff->hidden = GetParam(name + "/hidden");
  ff->output = GetParam(name + "/output");
  ff->prediction = GetParam(name + "/prediction", true);
}

void Parser::Parse(Document *document) const {
  // Extract lexical features from document.
  DocumentFeatures features(&lexicon_);
  features.Extract(*document);

  // Parse each sentence of the document.
  for (SentenceIterator s(document); s.more(); s.next()) {
    // Initialize parser model instance data.
    ParserInstance data(this, document, s.begin(), s.end());
    ParserState &state = data.state_;

    // Compute left-to-right LSTM.
    for (int i = 0; i < s.length(); ++i) {
      // Attach hidden and control layers.
      data.lr_.Clear();
      int in = i > 0 ? i - 1 : s.length();
      int out = i;
      data.AttachLR(in, out);

      // Extract features.
      data.ExtractFeaturesLSTM(s.begin() + out, features, lr_, &data.lr_);

      // Compute LSTM cell.
      if (profile_) data.lr_.set_profile(&profile_->lr);
      data.lr_.Compute();
    }

    // Compute right-to-left LSTM.
    for (int i = 0; i < s.length(); ++i) {
      // Attach hidden and control layers.
      data.rl_.Clear();
      int in = s.length() - i;
      int out = in - 1;
      data.AttachRL(in, out);

      // Extract features.
      data.ExtractFeaturesLSTM(s.begin() + out, features, rl_, &data.rl_);

      // Compute LSTM cell.
      if (profile_) data.rl_.set_profile(&profile_->rl);
      data.rl_.Compute();
    }

    // Run FF to predict transitions.
    bool done = false;
    int steps_since_shift = 0;
    int step = 0;
    while (!done) {
      // Allocate space for next step.
      data.ff_step_.push();

      // Attach instance to recurrent layers.
      data.ff_.Clear();
      data.AttachFF(step);

      // Extract features.
      data.ExtractFeaturesFF(step);

      // Predict next action.
      if (profile_) data.ff_.set_profile(&profile_->ff);
      data.ff_.Compute();
      int prediction = 0;
      if (fast_fallback_) {
        // Get highest scoring action.
        prediction = *data.ff_.Get<int>(ff_.prediction);
        const ParserAction &action = actions_.Action(prediction);
        if (!state.CanApply(action) || actions_.Beyond(prediction)) {
          // Fall back to SHIFT or STOP action.
          if (state.current() == state.end()) {
            prediction = actions_.StopIndex();
          } else {
            prediction = actions_.ShiftIndex();
          }
        }
      } else {
        // Get highest scoring allowed action.
        float *output = data.ff_.Get<float>(ff_.output);
        float max_score = -INFINITY;
        for (int a = 0; a < num_actions_; ++a) {
          if (output[a] > max_score) {
            const ParserAction &action = actions_.Action(a);
            if (state.CanApply(action) && !actions_.Beyond(a)) {
              prediction = a;
              max_score = output[a];
            }
          }
        }
      }

      // Apply action to parser state.
      const ParserAction &action = actions_.Action(prediction);
      state.Apply(action);

      // Update state.
      switch (action.type) {
        case ParserAction::SHIFT:
          steps_since_shift = 0;
          break;

        case ParserAction::STOP:
          done = true;
          break;

        case ParserAction::EVOKE:
        case ParserAction::REFER:
        case ParserAction::CONNECT:
        case ParserAction::ASSIGN:
        case ParserAction::EMBED:
        case ParserAction::ELABORATE:
          steps_since_shift++;
          if (state.AttentionSize() > 0) {
            int focus = state.Attention(0);
            if (data.create_step_.size() < focus + 1) {
              data.create_step_.resize(focus + 1);
              data.create_step_[focus] = step;
            }
            if (data.focus_step_.size() < focus + 1) {
              data.focus_step_.resize(focus + 1);
            }
            data.focus_step_[focus] = step;
          }
      }

      // Next step.
      step += 1;
    }

    // Add frames for sentence to the document.
    state.AddParseToDocument(document);
  }
}

myelin::Cell *Parser::GetCell(const string &name) {
  myelin::Cell *cell = network_.GetCell(name);
  if (cell == nullptr) {
    LOG(FATAL) << "Unknown parser cell: " << name;
  }
  return cell;
}

myelin::Connector *Parser::GetConnector(const string &name) {
  myelin::Connector *cnx = network_.GetConnector(name);
  if (cnx == nullptr) {
    LOG(FATAL) << "Unknown parser connector: " << name;
  }
  return cnx;
}

myelin::Tensor *Parser::GetParam(const string &name, bool optional) {
  myelin::Tensor *param = network_.GetParameter(name);
  if (param == nullptr && !optional) {
    LOG(FATAL) << "Unknown parser parameter: " << name;
  }
  return param;
}

ParserInstance::ParserInstance(const Parser *parser, Document *document,
                               int begin, int end)
    : parser_(parser),
      state_(document->store(), begin, end),
      lr_(parser->lr_.cell),
      rl_(parser->rl_.cell),
      ff_(parser->ff_.cell),
      lr_c_(parser->lr_.control),
      lr_h_(parser->lr_.hidden),
      rl_c_(parser->rl_.control),
      rl_h_(parser->rl_.hidden),
      ff_step_(parser->ff_.step) {
  // Add one extra element to LSTM activations for boundary element.
  int length = end - begin;
  lr_c_.resize(length + 1);
  lr_h_.resize(length + 1);
  rl_c_.resize(length + 1);
  rl_h_.resize(length + 1);

  // Reserve two transitions per token.
  ff_step_.reserve(length * 2);
}

void ParserInstance::AttachLR(int input, int output) {
  lr_.Set(parser_->lr_.c_in, &lr_c_, input);
  lr_.Set(parser_->lr_.c_out, &lr_c_, output);
  lr_.Set(parser_->lr_.h_in, &lr_h_, input);
  lr_.Set(parser_->lr_.h_out, &lr_h_, output);
}

void ParserInstance::AttachRL(int input, int output) {
  rl_.Set(parser_->rl_.c_in, &rl_c_, input);
  rl_.Set(parser_->rl_.c_out, &rl_c_, output);
  rl_.Set(parser_->rl_.h_in, &rl_h_, input);
  rl_.Set(parser_->rl_.h_out, &rl_h_, output);
}

void ParserInstance::AttachFF(int output) {
  ff_.Set(parser_->ff_.lr_lstm, &lr_h_);
  ff_.Set(parser_->ff_.rl_lstm, &rl_h_);
  ff_.Set(parser_->ff_.steps, &ff_step_);
  ff_.Set(parser_->ff_.hidden, &ff_step_, output);
}

void ParserInstance::ExtractFeaturesLSTM(int token,
                                         const DocumentFeatures &features,
                                         const Parser::LSTM &lstm,
                                         myelin::Instance *data) {
  // Extract word feature.
  if (lstm.word_feature) {
    *data->Get<int>(lstm.word_feature) = features.word(token);
  }

  // Extract prefix feature.
  if (lstm.prefix_feature) {
    Affix *affix = features.prefix(token);
    int *a = data->Get<int>(lstm.prefix_feature);
    for (int n = 0; n < lstm.prefix_size; ++n) {
      if (affix != nullptr) {
        *a++ = affix->id();
        affix = affix->shorter();
      } else {
        *a++ = -2;
      }
    }
  }

  // Extract suffix feature.
  if (lstm.suffix_feature) {
    Affix *affix = features.suffix(token);
    int *a = data->Get<int>(lstm.suffix_feature);
    for (int n = 0; n < lstm.suffix_size; ++n) {
      if (affix != nullptr) {
        *a++ = affix->id();
        affix = affix->shorter();
      } else {
        *a++ = -2;
      }
    }
  }

  // Extract hyphen feature.
  if (lstm.hyphen_feature) {
    *data->Get<int>(lstm.hyphen_feature) = features.hyphen(token);
  }

  // Extract capitalization feature.
  if (lstm.caps_feature) {
    *data->Get<int>(lstm.caps_feature) = features.capitalization(token);
  }

  // Extract punctuation feature.
  if (lstm.punct_feature) {
    *data->Get<int>(lstm.punct_feature) = features.punctuation(token);
  }

  // Extract quote feature.
  if (lstm.quote_feature) {
    *data->Get<int>(lstm.quote_feature) = features.quote(token);
  }

  // Extract digit feature.
  if (lstm.digit_feature) {
    *data->Get<int>(lstm.digit_feature) = features.digit(token);
  }
}

void ParserInstance::ExtractFeaturesFF(int step) {
  // Extract LSTM focus features.
  const Parser::FF &ff = parser_->ff_;
  int current = state_.current() - state_.begin();
  if (state_.current() == state_.end()) current = -1;
  int *lr_focus = GetFF(ff.lr_focus_feature);
  int *rl_focus = GetFF(ff.rl_focus_feature);
  if (lr_focus != nullptr) *lr_focus = current;
  if (rl_focus != nullptr) *rl_focus = current;

  // Extract frame attention, create, and focus features.
  if (ff.attention_depth > 0) {
    int *lr = GetFF(ff.lr_attention_feature);
    int *rl = GetFF(ff.rl_attention_feature);
    int *create = GetFF(ff.frame_create_feature);
    int *focus = GetFF(ff.frame_focus_feature);
    for (int d = 0; d < ff.attention_depth; ++d) {
      int att = -1;
      int created = -1;
      int focused = -1;
      if (d < state_.AttentionSize()) {
        // Get frame from attention buffer.
        int frame = state_.Attention(d);

        // Get end token for phrase that evoked frame.
        att = state_.FrameEvokeEnd(frame);
        if (att != -1) att -= state_.begin() + 1;

        // Get the step numbers that created and focused the frame.
        created = create_step_[frame];
        focused = focus_step_[frame];
      }
      if (lr != nullptr) lr[d] = att;
      if (rl != nullptr) rl[d] = att;
      if (create != nullptr) create[d] = created;
      if (focus != nullptr) focus[d] = focused;
    }
  }

  // Extract history feature.
  int *history = GetFF(ff.history_feature);
  if (history != nullptr) {
    int h = 0;
    int s = step - 1;
    while (h < ff.history_size && s >= 0) history[h++] = s--;
    while (h < ff.history_size) history[h++] = -1;
  }

  // Extract role features.
  if (parser_->frame_limit_ > 0 && parser_->roles_.size() > 0) {
    // Construct role graph for center of attention.
    RoleGraph graph;
    graph.Compute(state_, parser_->frame_limit_, parser_->roles_);

    // Extract out roles.
    int *out = GetFF(ff.out_roles_feature);
    if (out != nullptr) {
      int *end = out + ff.out_roles_size;
      graph.out([&out, end](int f) {
        if (out < end) *out++ = f;
      });
      while (out < end) *out++ = -2;
    }

    // Extract in roles.
    int *in = GetFF(ff.in_roles_feature);
    if (in != nullptr) {
      int *end = in + ff.in_roles_size;
      graph.in([&in, end](int f) {
        if (in < end) *in++ = f;
      });
      while (in < end) *in++ = -2;
    }

    // Extract unlabeled roles.
    int *unlabeled = GetFF(ff.unlabeled_roles_feature);
    if (unlabeled != nullptr) {
      int *end = unlabeled + ff.unlabeled_roles_size;
      graph.unlabeled([&unlabeled, end](int f) {
        if (unlabeled < end) *unlabeled++ = f;
      });
      while (unlabeled < end) *unlabeled++ = -2;
    }

    // Extract labeled roles.
    int *labeled = GetFF(ff.labeled_roles_feature);
    if (labeled != nullptr) {
      int *end = labeled + ff.labeled_roles_size;
      graph.labeled([&labeled, end](int f) {
        if (labeled < end) *labeled++ = f;
      });
      while (labeled < end) *labeled++ = -2;
    }
  }
}

}  // namespace nlp
}  // namespace sling

