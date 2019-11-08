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

#include "sling/nlp/parser/parser-features.h"

#include <string>
#include <vector>

#include "sling/base/logging.h"

namespace sling {
namespace nlp {

myelin::Tensor *ParserFeatureModel::GetParam(const string &name,
                                             bool optional) {
  string full_name = cell_->name() + "/" + name;
  myelin::Tensor *param = cell_->LookupParameter(full_name);
  if (param == nullptr && !optional) {
    LOG(FATAL) << "Unknown parser parameter: " << full_name;
  }
  return param;
}

void ParserFeatureModel::Init(myelin::Cell *cell,
                              myelin::Flow::Blob *spec,
                              const RoleSet *roles,
                              int frame_limit) {
  // Store cell that contains the feature inputs.
  cell_ = cell;
  roles_ = roles;
  frame_limit_ = frame_limit;

  // Get feature inputs.
  lr_focus_feature_ = GetParam("lr", true);
  rl_focus_feature_ = GetParam("rl", true);
  lr_attention_feature_ = GetParam("frame-end-lr", true);
  rl_attention_feature_ = GetParam("frame-end-rl", true);
  frame_create_feature_ = GetParam("frame-creation-steps", true);
  frame_focus_feature_ = GetParam("frame-focus-steps", true);
  history_feature_ = GetParam("history", true);
  mark_lr_feature_ = GetParam("mark-lr", true);
  mark_rl_feature_ = GetParam("mark-rl", true);
  mark_step_feature_ = GetParam("mark-step", true);
  mark_distance_feature_ = GetParam("mark-distance", true);
  out_roles_feature_ = GetParam("out-roles", true);
  in_roles_feature_ = GetParam("in-roles", true);
  unlabeled_roles_feature_ = GetParam("unlabeled-roles", true);
  labeled_roles_feature_ = GetParam("labeled-roles", true);

  // Get feature sizes.
  std::vector<myelin::Tensor *> attention_features {
    lr_attention_feature_,
    rl_attention_feature_,
    frame_create_feature_,
    frame_focus_feature_,
  };
  for (auto *f : attention_features) {
    if (!f) continue;
    if (f->elements() > attention_depth_) {
      attention_depth_ = f->elements();
    }
  }
  for (auto *f : attention_features) {
    if (!f) continue;
    CHECK_EQ(attention_depth_, f->elements());
  }
  if (history_feature_ != nullptr) {
    history_size_ = history_feature_->elements();
  }
  if (out_roles_feature_ != nullptr) {
    out_roles_size_ = out_roles_feature_->elements();
  }
  if (in_roles_feature_ != nullptr) {
    in_roles_size_ = in_roles_feature_->elements();
  }
  if (unlabeled_roles_feature_ != nullptr) {
    unlabeled_roles_size_ = unlabeled_roles_feature_->elements();
  }
  if (labeled_roles_feature_ != nullptr) {
    labeled_roles_size_ = labeled_roles_feature_->elements();
  }
  if (mark_lr_feature_ != nullptr) {
    mark_depth_ = mark_lr_feature_->elements();
  }
  if (mark_rl_feature_ != nullptr) {
    if (mark_depth_ == 0) {
      mark_depth_ = mark_rl_feature_->elements();
    } else {
      CHECK_EQ(mark_depth_, mark_rl_feature_->elements());
    }
  }
  if (mark_distance_feature_ != nullptr) {
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
        mark_distance_bins_.push_back(i);
        distance++;
      }
    }
    mark_distance_bins_.push_back(bins.size());
  }

  // Get links.
  lr_lstm_ = GetParam("link/lr_lstm");
  rl_lstm_ = GetParam("link/rl_lstm");
  steps_ = GetParam("steps");
  hidden_ = GetParam("hidden");
};

void ParserFeatureExtractor::Attach(const myelin::BiChannel &bilstm,
                                    myelin::Channel *activations,
                                    myelin::Instance *instance) {
  const ParserFeatureModel *fm = features_;
  instance->Set(fm->lr_lstm_, bilstm.lr);
  instance->Set(fm->rl_lstm_, bilstm.rl);
  instance->Set(fm->steps_, activations);
  instance->Set(fm->hidden_, activations, state_->step());
}

void ParserFeatureExtractor::Extract(myelin::Instance *instance) {
  const ParserFeatureModel *fm = features_;
  Data data(instance);

  // Extract LSTM focus features.
  int current = state_->current() - state_->begin();
  if (state_->current() == state_->end()) current = -1;
  int *lr_focus = data.Get(fm->lr_focus_feature_);
  int *rl_focus = data.Get(fm->rl_focus_feature_);
  if (lr_focus != nullptr) *lr_focus = current;
  if (rl_focus != nullptr) *rl_focus = current;

  // Extract features from the mark stack.
  auto &marks = state_->marks();
  int *lr_mark = data.Get(fm->mark_lr_feature_);
  int *rl_mark = data.Get(fm->mark_rl_feature_);
  int *mark_step = data.Get(fm->mark_step_feature_);
  for (int d = 0; d < fm->mark_depth_; ++d) {
    if (d < marks.size()) {
      const auto &m = marks[marks.size() - 1 - d];
      int token = m.token - state_->begin();
      if (lr_mark != nullptr) lr_mark[d] = token;
      if (rl_mark != nullptr) rl_mark[d] = token;
      if (mark_step != nullptr) mark_step[d] = m.step;
    } else {
      if (lr_mark != nullptr) lr_mark[d] = -1;
      if (rl_mark != nullptr) rl_mark[d] = -1;
      if (mark_step != nullptr) mark_step[d] = -1;
    }
  }

  int *mark_distance = data.Get(fm->mark_distance_feature_);
  if (mark_distance != nullptr) {
    *mark_distance = fm->mark_distance_bins_.back();
    if (!marks.empty()) {
      int distance = state_->current() - marks[marks.size() - 1].token;
      if (distance < fm->mark_distance_bins_.size()) {
        *mark_distance = fm->mark_distance_bins_[distance];
      }
    }
  }

  // Extract frame attention, create, and focus features.
  if (fm->attention_depth_ > 0) {
    int *lr = data.Get(fm->lr_attention_feature_);
    int *rl = data.Get(fm->rl_attention_feature_);
    int *create = data.Get(fm->frame_create_feature_);
    int *focus = data.Get(fm->frame_focus_feature_);
    for (int d = 0; d < fm->attention_depth_; ++d) {
      int token = -1;
      int created = -1;
      int focused = -1;
      if (d < state_->AttentionSize()) {
        // Get frame from attention buffer.
        const auto &attention = state_->Attention(d);

        // Get end token for phrase that evoked frame.
        if (attention.span != nullptr) {
          token = attention.span->end();
          if (token != -1) token -= state_->begin() + 1;
        }

        // Get the step numbers that created and focused the frame.
        created = attention.created;
        focused = attention.focused;
      }
      if (lr != nullptr) lr[d] = token;
      if (rl != nullptr) rl[d] = token;
      if (create != nullptr) create[d] = created;
      if (focus != nullptr) focus[d] = focused;
    }
  }

  // Extract history feature.
  int *history = data.Get(fm->history_feature_);
  if (history != nullptr) {
    int h = 0;
    int s = state_->step() - 1;
    while (h < fm->history_size_ && s >= 0) history[h++] = s--;
    while (h < fm->history_size_) history[h++] = -1;
  }

  // Extract role features.
  if (features_->frame_limit_ > 0 && fm->roles_->size() > 0) {
    // Construct role graph for center of attention.
    RoleGraph graph;
    graph.Compute(*state_, fm->frame_limit_, *fm->roles_);

    // Extract out roles.
    int *out = data.Get(fm->out_roles_feature_);
    if (out != nullptr) {
      int *end = out + fm->out_roles_size_;
      graph.out([&out, end](int f) {
        if (out < end) *out++ = f;
      });
      while (out < end) *out++ = -2;
    }

    // Extract in roles.
    int *in = data.Get(fm->in_roles_feature_);
    if (in != nullptr) {
      int *end = in + fm->in_roles_size_;
      graph.in([&in, end](int f) {
        if (in < end) *in++ = f;
      });
      while (in < end) *in++ = -2;
    }

    // Extract unlabeled roles.
    int *unlabeled = data.Get(fm->unlabeled_roles_feature_);
    if (unlabeled != nullptr) {
      int *end = unlabeled + fm->unlabeled_roles_size_;
      graph.unlabeled([&unlabeled, end](int f) {
        if (unlabeled < end) *unlabeled++ = f;
      });
      while (unlabeled < end) *unlabeled++ = -2;
    }

    // Extract labeled roles.
    int *labeled = data.Get(fm->labeled_roles_feature_);
    if (labeled != nullptr) {
      int *end = labeled + fm->labeled_roles_size_;
      graph.labeled([&labeled, end](int f) {
        if (labeled < end) *labeled++ = f;
      });
      while (labeled < end) *labeled++ = -2;
    }
  }
};

void ParserFeatureExtractor::TraceFeatures(myelin::Instance *instance,
                                           Trace *trace) const {
  trace->steps.emplace_back();
  auto &step = trace->steps.back();
  step.current = state_->current();

  Data data(instance);
  const ParserFeatureModel *fm = features_;
  step.Add(data.Get(fm->lr_focus_feature_), 1, "lr");
  step.Add(data.Get(fm->rl_focus_feature_), 1, "rl");
  step.Add(data.Get(fm->mark_lr_feature_), fm->mark_depth_, "mark-lr");
  step.Add(data.Get(fm->mark_rl_feature_), fm->mark_depth_, "mark-rl");
  step.Add(data.Get(fm->mark_step_feature_), fm->mark_depth_, "mark-step");

  int depth = fm->attention_depth_;
  step.Add(data.Get(fm->lr_attention_feature_), depth, "frame-end-lr");
  step.Add(data.Get(fm->rl_attention_feature_), depth, "frame-end-rl");
  step.Add(data.Get(fm->frame_create_feature_), depth, "frame-creation-steps");
  step.Add(data.Get(fm->frame_focus_feature_), depth, "frame-focus-steps");
  step.Add(data.Get(fm->history_feature_), fm->history_size_, "history");
  step.Add(data.Get(fm->mark_distance_feature_), 1, "mark-distance");
  step.Add(data.Get(fm->out_roles_feature_), fm->out_roles_size_, "out-roles");
  step.Add(data.Get(fm->in_roles_feature_), fm->in_roles_size_, "in-roles");
  step.Add(data.Get(fm->unlabeled_roles_feature_),
    fm->unlabeled_roles_size_, "unlabeled-roles");
  step.Add(data.Get(fm->labeled_roles_feature_),
    fm->labeled_roles_size_, "labeled-roles");
}

}  // namespace nlp
}  // namespace sling

