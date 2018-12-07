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

#include <functional>
#include <iostream>
#include <math.h>

#include "sling/nlp/parser/parser.h"

#include "sling/frame/serialization.h"
#include "sling/myelin/kernel/dragnn.h"
#include "sling/nlp/document/document.h"
#include "sling/nlp/document/features.h"
#include "sling/nlp/document/lexicon.h"
#include "sling/string/strcat.h"

using namespace std::placeholders;

namespace sling {
namespace nlp {

void Parser::Load(Store *store, const string &model) {
  // Load and analyze parser flow file.
  myelin::Flow flow;
  CHECK(flow.Load(model));

  // FIXME(ringgaard): Patch feature cell output.
  flow.Var("features/feature_vector")->set_in();

  // Register DRAGNN kernel to support legacy parser models.
  RegisterDragnnLibrary(compiler_.library());

  // Compile parser flow.
  compiler_.Compile(&flow, &network_);

  // Initialize lexical encoder.
  encoder_.Initialize(network_);
  encoder_.LoadLexicon(&flow);

  // Initialize feed-forward trunk.
  myelin::Flow::Blob *spec = flow.DataBlock("spec");
  InitFF("ff_trunk", spec, &ff_);

  // Load commons and action stores.
  myelin::Flow::Blob *commons = flow.DataBlock("commons");
  CHECK(commons != nullptr);
  StringDecoder decoder(store, commons->data, commons->size);
  decoder.DecodeAll();

  // Read the cascade specification and implementation from the flow.
  Frame cascade_spec(store, "/cascade");
  CHECK(cascade_spec.valid());
  cascade_.Initialize(network_, cascade_spec);

  // Initialize action table.
  store_ = store;
  actions_.Init(store);
  cascade_.set_actions(&actions_);
  frame_limit_ = actions_.frame_limit();
  roles_.Init(actions_);
}

void Parser::InitFF(const string &name, myelin::Flow::Blob *spec, FF *ff) {
  // Get cell.
  ff->cell = GetCell(name);

  // Get feature inputs.
  ff->lr_focus_feature = GetParam(name + "/lr", true);
  ff->rl_focus_feature = GetParam(name + "/rl", true);
  ff->lr_attention_feature = GetParam(name + "/frame-end-lr", true);
  ff->rl_attention_feature = GetParam(name + "/frame-end-rl", true);
  ff->frame_create_feature = GetParam(name + "/frame-creation-steps", true);
  ff->frame_focus_feature = GetParam(name + "/frame-focus-steps", true);
  ff->history_feature = GetParam(name + "/history", true);
  ff->mark_lr_feature = GetParam(name + "/mark-lr", true);
  ff->mark_rl_feature = GetParam(name + "/mark-rl", true);
  ff->mark_step_feature = GetParam(name + "/mark-step", true);
  ff->mark_distance_feature = GetParam(name + "/mark-distance", true);
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
  if (ff->mark_lr_feature != nullptr) {
    ff->mark_depth = ff->mark_lr_feature->elements();
  }
  if (ff->mark_rl_feature != nullptr) {
    if (ff->mark_depth == 0) {
      ff->mark_depth = ff->mark_rl_feature->elements();
    } else {
      CHECK_EQ(ff->mark_depth, ff->mark_rl_feature->elements());
    }
  }
  if (ff->mark_distance_feature != nullptr) {
    CHECK(spec != nullptr);
    string bins_str = spec->GetAttr("mark_distance_bins");
    std::vector<int> bins;
    int start = 0;
    while (true) {
      ssize_t index = bins_str.find(' ', start);
      if (index != string::npos) {
        string s = bins_str.substr(start, index - start);
        bins.push_back(std::stoi(s));
        start = index + 1;
      } else {
        bins.push_back(std::stoi(bins_str.substr(start)));
        break;
      }
    }
    int distance = 0;
    for (int i = 0; i < bins.size(); ++i) {
      while (distance <= bins[i]) {
        ff->mark_distance_bins.push_back(i);
        distance++;
      }
    }
    ff->mark_distance_bins.push_back(bins.size());
  }

  // Get links.
  ff->lr_lstm = GetParam(name + "/link/lr_lstm");
  ff->rl_lstm = GetParam(name + "/link/rl_lstm");
  ff->steps = GetParam(name + "/steps");

  ff->hidden = GetParam(name + "/hidden");
}

void Parser::Parse(Document *document) const {
  // Parse each sentence of the document.
  for (SentenceIterator s(document); s.more(); s.next()) {
    // Initialize parser model instance data.
    ParserInstance data(this, document, s.begin(), s.end());
    LexicalEncoderInstance &encoder = data.encoder_;

    // Run the lexical encoder.
    auto bilstm = encoder.Compute(*document, s.begin(), s.end());

    // Run FF to predict transitions.
    ParserState &state = data.state_;
    bool done = false;
    int steps_since_shift = 0;
    int step = 0;
    while (!done) {
      // Allocate space for next step.
      data.ff_step_.push();

      // Attach instance to recurrent layers.
      data.ff_.Clear();
      data.AttachFF(step, bilstm);

      // Extract features.
      data.ExtractFeaturesFF(step);

      // Compute FF hidden layer.
      data.ff_.Compute();

      // Apply the cascade.
      ParserAction action;
      data.cascade_.Compute(&data.ff_step_, step, &state, &action, data.trace_);
      state.Apply(action);

      // Update state.
      switch (action.type) {
        case ParserAction::SHIFT:
          steps_since_shift = 0;
          break;

        case ParserAction::STOP:
          done = true;
          break;

        case ParserAction::CASCADE:
          LOG(FATAL) << "CASCADE action should not reach ParserState.";
          break;

        case ParserAction::MARK:
          data.marks_.emplace_back(state.current(), step);
          break;

        case ParserAction::EVOKE:
          if (action.length == 0) data.marks_.pop_back();
          FALLTHROUGH_INTENDED;

        case ParserAction::REFER:
        case ParserAction::CONNECT:
        case ParserAction::ASSIGN:
        case ParserAction::EMBED:
        case ParserAction::ELABORATE:
          steps_since_shift++;
          if (state.AttentionSize() > 0) {
            Handle focus = state.Attention(0);
            if (data.create_step_.find(focus) == data.create_step_.end()) {
              data.create_step_[focus] = step;
            }
            data.focus_step_[focus] = step;
          }
      }

      // Next step.
      step += 1;
    }
    if (data.trace_ != nullptr) data.trace_->Write(document);
  }
}

myelin::Cell *Parser::GetCell(const string &name) {
  myelin::Cell *cell = network_.LookupCell(name);
  if (cell == nullptr) {
    LOG(FATAL) << "Unknown parser cell: " << name;
  }
  return cell;
}

myelin::Tensor *Parser::GetParam(const string &name, bool optional) {
  myelin::Tensor *param = network_.LookupParameter(name);
  if (param == nullptr && !optional) {
    LOG(FATAL) << "Unknown parser parameter: " << name;
  }
  return param;
}

ParserInstance::ParserInstance(const Parser *parser, Document *document,
                               int begin, int end)
    : parser_(parser),
      encoder_(parser->encoder()),
      state_(document, begin, end),
      ff_(parser->ff_.cell),
      ff_step_(parser->ff_.hidden),
      cascade_(&parser->cascade_),
      trace_(nullptr) {
  // Reserve two transitions per token.
  int length = end - begin;
  ff_step_.reserve(length * 2);
  if (parser->trace()) {
    trace_ = new Trace();
    trace_->begin = begin;
    trace_->end = end;
    encoder_.set_trace(std::bind(&Trace::AddLSTM, trace_, _1, _2, _3));
  }
}

void ParserInstance::AttachFF(int output, const myelin::BiChannel &bilstm) {
  ff_.Set(parser_->ff_.lr_lstm, bilstm.lr);
  ff_.Set(parser_->ff_.rl_lstm, bilstm.rl);
  ff_.Set(parser_->ff_.steps, &ff_step_);
  ff_.Set(parser_->ff_.hidden, &ff_step_, output);
}

void ParserInstance::TraceFFFeatures() {
  trace_->steps.emplace_back();
  auto &step = trace_->steps.back();
  step.current = state_.current();

  const Parser::FF &ff = parser_->ff_;
  step.Add(GetFF(ff.lr_focus_feature), 1, "lr");
  step.Add(GetFF(ff.rl_focus_feature), 1, "rl");
  step.Add(GetFF(ff.mark_lr_feature), ff.mark_depth, "mark-lr");
  step.Add(GetFF(ff.mark_rl_feature), ff.mark_depth, "mark-rl");
  step.Add(GetFF(ff.mark_step_feature), ff.mark_depth, "mark-step");

  int depth = ff.attention_depth;
  step.Add(GetFF(ff.lr_attention_feature), depth, "frame-end-lr");
  step.Add(GetFF(ff.rl_attention_feature), depth, "frame-end-rl");
  step.Add(GetFF(ff.frame_create_feature), depth, "frame-creation-steps");
  step.Add(GetFF(ff.frame_focus_feature), depth, "frame-focus-steps");
  step.Add(GetFF(ff.history_feature), ff.history_size, "history");
  step.Add(GetFF(ff.mark_distance_feature), 1, "mark-distance");
  step.Add(GetFF(ff.out_roles_feature), ff.out_roles_size, "out-roles");
  step.Add(GetFF(ff.in_roles_feature), ff.in_roles_size, "in-roles");
  step.Add(GetFF(ff.unlabeled_roles_feature),
    ff.unlabeled_roles_size, "unlabeled-roles");
  step.Add(GetFF(ff.labeled_roles_feature),
    ff.labeled_roles_size, "labeled-roles");
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

  // Extract features from the mark stack.
  int *lr_mark = GetFF(ff.mark_lr_feature);
  int *rl_mark = GetFF(ff.mark_rl_feature);
  int *mark_step = GetFF(ff.mark_step_feature);
  for (int d = 0; d < ff.mark_depth; ++d) {
    const auto *m =
      (d < marks_.size()) ? &marks_[marks_.size() - 1 - d] : nullptr;
    int token = (m != nullptr) ? (m->token - state_.begin()) : -1;
    if (lr_mark != nullptr) lr_mark[d] = token;
    if (rl_mark != nullptr) rl_mark[d] = token;
    if (mark_step != nullptr) mark_step[d] = (m != nullptr) ? m->step : -1;
  }

  int *mark_distance = GetFF(ff.mark_distance_feature);
  if (mark_distance != nullptr) {
    *mark_distance = -2;
    if (marks_.size() > 0) {
      int distance = state_.current() - marks_[marks_.size() - 1].token;
      *mark_distance = ff.mark_distance_bins.back();
      if (distance < ff.mark_distance_bins.size()) {
        *mark_distance = ff.mark_distance_bins[distance];
      }
    }
  }

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
        Handle frame = state_.Attention(d);

        // Get end token for phrase that evoked frame.
        att = state_.FrameEvokeEnd(d);
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

  if (trace_ != nullptr) TraceFFFeatures();
}

}  // namespace nlp
}  // namespace sling

