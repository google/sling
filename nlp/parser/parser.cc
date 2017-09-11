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

#include "nlp/parser/parser.h"

#include "frame/serialization.h"
#include "myelin/kernel/dragnn.h"
#include "myelin/kernel/tensorflow.h"
#include "stream/memory.h"

namespace sling {
namespace nlp {

void Parser::Load(Store *store, const string &model) {
  // Register kernels for implementing parser ops.
  RegisterTensorflowLibrary(&library_);
  RegisterDragnnLibrary(&library_);

  // Load and analyze parser flow file.
  myelin::Flow flow;
  CHECK(flow.Load(model));
  flow.Analyze(library_);

  // Compile parser flow.
  CHECK(network_.Compile(flow, library_));

  // Initialize cells.
  InitLSTM("lr_lstm", &lr_, false);
  InitLSTM("rl_lstm", &rl_, true);
  InitFF("ff", &ff_);

  // Get attention depth.
  attention_depth_ = ff_.feature_lr_attention->elements();
  CHECK_EQ(attention_depth_, ff_.feature_rl_attention->elements());
  CHECK_EQ(attention_depth_, ff_.feature_frame_create->elements());
  CHECK_EQ(attention_depth_, ff_.feature_frame_focus->elements());

  // Get history size.
  history_size_ = ff_.feature_history->elements();

  // Get maximum number of role features.
  max_roles_ = ff_.feature_roles->elements();

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

  // Get the set of roles that connect two frames.
  for (int i = 0; i < num_actions_; ++i) {
    const auto &action = actions_.Action(i);
    if (action.type == ParserAction::CONNECT ||
        action.type == ParserAction::EMBED ||
        action.type == ParserAction::ELABORATE) {
      if (roles_.find(action.role) == roles_.end()) {
        int index = roles_.size();
        roles_[action.role] = index;
      }
    }
  }

  // Compute the offsets for the four types of role features. These are laid
  // out in this order: all (i, r) features, all (r, j) features, all (i, j)
  // features, all (i, r, j) features.
  int combinations = frame_limit_ * roles_.size();
  outlink_offset_ = 0;
  inlink_offset_ = outlink_offset_ + combinations;
  unlabeled_link_offset_ = inlink_offset_ + combinations;
  labeled_link_offset_ = unlabeled_link_offset_ + frame_limit_ * frame_limit_;
}

void Parser::InitLSTM(const string &name, LSTM *lstm, bool reverse) {
  // Get cell.
  lstm->cell = GetCell(name);
  lstm->reverse = reverse;

  // Get connectors.
  lstm->control = GetConnector(name + "/control");
  lstm->hidden = GetConnector(name + "/hidden");

  // Get feature inputs.
  lstm->feature_words = GetParam(name + "/words", true);
  lstm->feature_prefix = GetParam(name + "/prefix", true);
  lstm->feature_suffix = GetParam(name + "/suffix", true);
  lstm->feature_shape = GetParam(name + "/shape", true);

  // Get links.
  lstm->c_in = GetParam(name + "/c_in");
  lstm->c_out = GetParam(name + "/c_out");
  lstm->h_in = GetParam(name + "/h_in");
  lstm->h_out = GetParam(name + "/h_out");
}

void Parser::InitFF(const string &name, FF *ff) {
  // Get cell.
  ff->cell = GetCell(name);

  // Get connector for recurrence.
  ff->step = GetConnector(name + "/step");

  // Get feature inputs.
  ff->feature_lr_focus = GetParam(name + "/lr", true);
  ff->feature_rl_focus = GetParam(name + "/rl", true);
  ff->feature_lr_attention = GetParam(name + "/frame-end-lr", true);
  ff->feature_rl_attention = GetParam(name + "/frame-end-rl", true);
  ff->feature_frame_create = GetParam(name + "/frame-creation-steps", true);
  ff->feature_frame_focus = GetParam(name + "/frame-focus-steps", true);
  ff->feature_history = GetParam(name + "/history", true);
  ff->feature_roles = GetParam(name + "/roles", true);

  // Get links.
  ff->lr_lstm = GetParam(name + "/link/lr_lstm");
  ff->rl_lstm = GetParam(name + "/link/rl_lstm");
  ff->steps = GetParam(name + "/steps");
  ff->hidden = GetParam(name + "/hidden");
  ff->output = GetParam(name + "/output");
}

void Parser::Parse(Document *document) const {
  // Parse each sentence of the document.
  for (SentenceIterator s(document); s.more(); s.next()) {
    // Initialize parser model instance data.
    ParserInstance data(this, document, s.begin(), s.end());
    ParserState &state = data.state_;

    // Look up words in vocabulary.
    for (int i = s.begin(); i < s.end(); ++i) {
      int word = lexicon_.LookupWord(document->token(i).text());
      data.words_[i - s.begin()] = word;
    }

    // Compute left-to-right LSTM.
    for (int i = 0; i < s.length(); ++i) {
      // Attach hidden and control layers.
      data.lr_.Clear();
      int in = i > 0 ? i - 1 : s.length();
      int out = i;
      data.AttachLR(in, out);

      // Extract features.
      data.ExtractFeaturesLR(out);

      // Compute LSTM cell.
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
      data.ExtractFeaturesRL(out);

      // Compute LSTM cell.
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
      data.ff_.Compute();
      float *output = data.ff_.Get<float>(ff_.output);
      int prediction = 0;
      float max_score = -INFINITY;
      for (int a = 0; a < num_actions_; ++a) {
        if (output[a] > max_score) {
          const ParserAction &action = actions_.Action(a);
          if (state.CanApply(action)) {
            prediction = a;
            max_score = output[a];
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
  // Allocate space for word ids.
  int length = end - begin;
  words_.resize(length);

  // Add one extra element to LSTM activations for boundary element.
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

void ParserInstance::ExtractFeaturesLR(int current) {
  int word = words_[current];
  *lr_.Get<int>(parser_->lr_.feature_words) = word;
}

void ParserInstance::ExtractFeaturesRL(int current) {
  int word = words_[current];
  *rl_.Get<int>(parser_->rl_.feature_words) = word;
}

void ParserInstance::ExtractFeaturesFF(int step) {
  // Compute LSTM focus features.
  int current = state_.current() - state_.begin();
  if (current == state_.end()) current = -1;
  *ff_.Get<int>(parser_->ff_.feature_lr_focus) = current;
  *ff_.Get<int>(parser_->ff_.feature_rl_focus) = current;

  // Compute frame attention, create, and focus features.
  int *lr = ff_.Get<int>(parser_->ff_.feature_lr_attention);
  int *rl = ff_.Get<int>(parser_->ff_.feature_rl_attention);
  int *create = ff_.Get<int>(parser_->ff_.feature_frame_create);
  int *focus = ff_.Get<int>(parser_->ff_.feature_frame_focus);
  for (int d = 0; d < parser_->attention_depth_; ++d) {
    int att = -2;
    int created = -2;
    int focused = -2;
    if (d < state_.AttentionSize()) {
      // Get frame from attention buffer.
      int frame = state_.Attention(d);

      // Get end token for phrase that evoked frame.
      att = state_.FrameEvokeEnd(frame);
      if (att != -1) att -= state_.begin() + 1;

      // Get the step numbers that created and focused the frame.
      if (frame < create_step_.size()) {
        created = create_step_[frame];
      }
      if (frame < focus_step_.size()) {
        focused = focus_step_[frame];
      }
    }
    lr[d] = att;
    rl[d] = att;
    create[d] = created;
    focus[d] = focused;
  }

  // Compute history feature.
  int *history = ff_.Get<int>(parser_->ff_.feature_history);
  int h = 0;
  int s = step - 1;
  while (h < parser_->history_size_ && s >= 0) history[h++] = s--;
  while (h < parser_->history_size_) history[h++] = -2;

  // Construct a mapping from absolute frame index -> attention index.
  std::unordered_map<int, int> frame_to_attention;
  for (int i = 0; i < parser_->frame_limit_; ++i) {
    if (i < state_.AttentionSize()) {
      frame_to_attention[state_.Attention(i)] = i;
    } else {
      break;
    }
  }

  // Compute role features.
  int *r = ff_.Get<int>(parser_->ff_.feature_roles);
  int *rend = r + parser_->max_roles_;
  for (const auto &kv : frame_to_attention) {
    // Attention index of the source frame.
    int source = kv.second;
    int outlink_base = parser_->outlink_offset_ +
                       source * parser_->roles_.size();

    // Go over each slot of the source frame.
    Handle handle = state_.frame(kv.first);
    const FrameDatum *frame = state_.store()->GetFrame(handle);
    for (const Slot *slot = frame->begin(); slot < frame->end(); ++slot) {
      const auto &it = parser_->roles_.find(slot->name);
      if (it == parser_->roles_.end()) continue;
      int role = it->second;

      if (r < rend) {
        // (source, role).
        *r++ = outlink_base + role;
      }
      if (slot->value.IsIndex()) {
        const auto &it2 = frame_to_attention.find(slot->value.AsIndex());
        if (it2 != frame_to_attention.end()) {
          // Attention index of the target frame.
          int target = it2->second;
          if (r < rend) {
            // (role, target)
            *r++ = parser_->inlink_offset_ +
                   target * parser_->roles_.size() +
                   role;
          }
          if (r < rend) {
            // (source, target)
            *r++ = parser_->unlabeled_link_offset_ +
                   source * parser_->frame_limit_ +
                   target;
          }
          if (r < rend) {
            // (source, role, target)
            *r++ = parser_->labeled_link_offset_ +
                   source * parser_->frame_limit_ * parser_->roles_.size() +
                   target * parser_->roles_.size() +
                   role;
          }
        }
      }
    }
  }
  while (r < rend) *r++ = -2;
}

}  // namespace nlp
}  // namespace sling

