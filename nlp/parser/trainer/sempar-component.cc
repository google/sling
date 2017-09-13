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

#include "nlp/parser/trainer/sempar-component.h"

#include <iostream>
#include <memory>

#include "base/logging.h"
#include "dragnn/core/component_registry.h"
#include "dragnn/core/input_batch_cache.h"
#include "dragnn/core/interfaces/component.h"
#include "dragnn/core/interfaces/transition_state.h"
#include "file/file.h"
#include "nlp/parser/trainer/document-batch.h"
#include "string/strcat.h"

namespace sling {
namespace nlp {

using syntaxnet::dragnn::ComponentSpec;
using syntaxnet::dragnn::InputBatchCache;
using syntaxnet::dragnn::LinkFeatures;
using syntaxnet::dragnn::TransitionState;

namespace {

// Splits the given string on every occurrence of the given delimiter char.
std::vector<string> Split(const string &text, char delim) {
  std::vector<string> result;
  int token_start = 0;
  if (!text.empty()) {
    for (size_t i = 0; i < text.size() + 1; i++) {
      if (i == text.size() || text[i] == delim) {
        result.push_back(string(text.data() + token_start, i - token_start));
        token_start = i + 1;
      }
    }
  }
  return result;
}

}  // namespace

SemparComponent::~SemparComponent() {
  for (SemparState *state : batch_) delete state;
}

void SemparComponent::InitializeComponent(const ComponentSpec &spec) {
  // Save off the passed spec for future reference.
  spec_ = spec;

  // Set up the underlying transition system.
  const string &system = spec.transition_system().registered_name();
  if (system == "shift-only") {
    system_type_ = SHIFT_ONLY;
  } else if (system == "sempar") {
    system_type_ = SEMPAR;
  } else {
    LOG(FATAL) << "Unknown/unsupported transition system: " << system;
  }

  resources_.LoadGlobalStore(SemparFeature::GetResource(spec_, "commons"));
  if (system_type_ == SEMPAR) {
    resources_.LoadActionTable(
        SemparFeature::GetResource(spec_, "action-table"));
  } else {
    for (const auto &param : spec_.transition_system().parameters()) {
      const string &val = param.second;
      if (param.first == "left_to_right") {
        CHECK(val == "false" || val == "true") << val;
        if (val == "false") left_to_right_ = false;
      }
    }
  }

  // Set up the fixed feature extractor.
  for (const auto &fixed_channel : spec_.fixed_feature()) {
    feature_extractor_.AddChannel(fixed_channel);
  }
  feature_extractor_.Init(spec_, &resources_);
  feature_extractor_.RequestWorkspaces(&workspace_registry_);

  // Set up link feature extractors.
  for (const auto &linked_channel : spec_.linked_feature()) {
    link_feature_extractor_.AddChannel(linked_channel);
  }
  link_feature_extractor_.Init(spec_, &resources_);
  link_feature_extractor_.RequestWorkspaces(&workspace_registry_);

  // Initialize gold transition generator.
  gold_transition_generator_.Init(resources_.global);
}

void SemparComponent::InitializeData(InputBatchCache *input_data) {
  // Save off the input data object.
  input_data_ = input_data;

  DocumentBatch *input = input_data->GetAs<DocumentBatch>();
  input->Decode(resources_.global);

  // Get rid of the previous batch.
  for (SemparState *old : batch_) delete old;

  // Create states for the new batch.
  batch_.clear();
  for (int batch_index = 0; batch_index < input->size(); ++batch_index) {
    batch_.push_back(CreateState(input->item(batch_index)));
  }
}

bool SemparComponent::IsReady() const { return input_data_ != nullptr; }

string SemparComponent::Name() const {
  return "SemparComponent";
}

int SemparComponent::BatchSize() const { return batch_.size(); }

int SemparComponent::StepsTaken(int batch_index) const {
  return batch_.at(batch_index)->NumSteps();
}

std::function<int(int, int)> SemparComponent::GetStepLookupFunction(
    const string &method) {
  if (method == "reverse-token") {
    CHECK_EQ(system_type_, SHIFT_ONLY)
        << "'reverse-token' only supported for shift-only systems.";
    return [this](int batch_index, int value) {
      SemparState *state = batch_.at(batch_index);
      int size = state->document()->num_tokens();
      int result = size - value - 1;
      if (result >= 0 && result < size) {
        return result;
      } else {
        return -1;
      }
    };
  } else {
    LOG(FATAL) << "Unsupported step lookup function: " << method;
    return 0;
  }
}

void SemparComponent::AdvanceFromPrediction(const float scores[],
                                            int transition_matrix_length) {
  VLOG(2) << "Advancing from prediction.";
  int offset = 0;
  int num_actions = 1;
  if (!shift_only()) num_actions = resources_.table.NumActions();
  CHECK_EQ(transition_matrix_length, batch_.size() * num_actions);
  for (int i = 0; i < batch_.size(); ++i) {
    CHECK_LE(offset + num_actions, transition_matrix_length) << offset;
    SemparState *state = batch_.at(i);
    if (!state->IsFinal()) {
      int best = -1;
      for (int action = 0; action < num_actions; ++action) {
        if (!state->Allowed(action)) continue;
        if (best == -1 || scores[offset + best] < scores[offset + action]) {
          best = action;
        }
      }
      CHECK_NE(best, -1) << state->DebugString();
      Advance(state, best);
      state->SetScore(state->GetScore() + scores[offset + best]);
    }
    offset += num_actions;
  }
}

void SemparComponent::AdvanceFromOracle() {
  for (SemparState *state : batch_) {
    if (state->IsFinal()) continue;
    Advance(state, GetOracleLabel(state));
    state->SetScore(0.0f);
  }
}

bool SemparComponent::IsTerminal() const {
  for (SemparState *state : batch_) {
    if (!state->IsFinal()) return false;
  }
  return true;
}

std::vector<const TransitionState *> SemparComponent::GetStates() {
  std::vector<const TransitionState *> states;
  for (auto *s : batch_) states.push_back(s);
  return states;
}

int SemparComponent::GetFixedFeatures(
    std::function<int32 *(int)> allocate_indices,
    std::function<int64 *(int)> allocate_ids,
    int channel_id) const {
  SemparFeature::Args args;

  std::vector<int> segment_indices;
  int next_segment_index = 0;
  int channel_size = spec_.fixed_feature(channel_id).size();
  for (SemparState *state : batch_) {
    args.state = state;
    int old_size = args.output.size();
    feature_extractor_.Extract(&args, channel_id);

    for (int i = old_size; i < args.output.size(); ++i) {
      segment_indices.emplace_back(
          next_segment_index + args.output[i].feature_index);
    }

    next_segment_index += channel_size;
  }
  int feature_count = args.output.size();
  CHECK_EQ(feature_count, segment_indices.size());

  int32 *indices_tensor = allocate_indices(feature_count);
  int64 *ids_tensor = allocate_ids(feature_count);
  for (int i = 0; i < feature_count; ++i) {
    ids_tensor[i] = args.output[i].id;
    indices_tensor[i] = segment_indices[i];
  }

  return feature_count;
}

std::vector<LinkFeatures> SemparComponent::GetRawLinkFeatures(
    int channel_id) const {
  std::vector<string> feature_names;
  int channel_size = link_feature_extractor_.ChannelSize(channel_id);
  std::vector<LinkFeatures> features;
  features.resize(batch_.size() * channel_size);

  SemparFeature::Args args;
  for (int batch_idx = 0; batch_idx < batch_.size(); ++batch_idx) {
    args.Clear();
    args.state = batch_.at(batch_idx);
    link_feature_extractor_.Extract(&args, channel_id);

    // Add the raw feature values to the LinkFeatures proto.
    int base = batch_idx * channel_size;
    for (int i = 0; i < args.output.size(); ++i) {
      auto &f = features[base + args.output[i].feature_index];
      f.set_feature_value(args.output[i].id);
      f.set_batch_idx(batch_idx);
    }
  }

  return features;
}

std::vector<int> SemparComponent::GetOracleLabels() const {
  std::vector<int> oracle_labels;
  for (SemparState *state : batch_) {
    oracle_labels.push_back(GetOracleLabel(state));
  }

  return oracle_labels;
}

void SemparComponent::FinalizeData() {
  for (SemparState *state : batch_) {
    if (!state->shift_only()) {
      state->parser_state()->AddParseToDocument(state->document());
      state->document()->Update();
    }
  }
}

void SemparComponent::ResetComponent() {
  for (SemparState *state : batch_) delete state;
  batch_.clear();
  input_data_ = nullptr;
}

SemparState *SemparComponent::CreateState(SemparInstance *instance) {
  instance->workspaces->Reset(workspace_registry_);
  SemparState *state =
      new SemparState(instance, resources_, system_type_, left_to_right_);
  state->set_gold_transition_generator(&gold_transition_generator_);
  feature_extractor_.Preprocess(state);
  link_feature_extractor_.Preprocess(state);

  return state;
}

bool SemparComponent::IsAllowed(SemparState *state, int action) const {
  return state->Allowed(action);
}

bool SemparComponent::IsFinal(SemparState *state) const {
  return state->IsFinal();
}

int SemparComponent::GetOracleLabel(SemparState *state) const {
  return state->NextGoldAction();
}

void SemparComponent::Advance(SemparState *state, int action) {
  state->PerformAction(action);
}

REGISTER_DRAGNN_COMPONENT(SemparComponent);

}  // namespace nlp
}  // namespace sling
