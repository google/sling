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

#include "sling/nlp/parser/trainer/sempar-component.h"

#include <iostream>
#include <memory>

#include "dragnn/core/component_registry.h"
#include "dragnn/core/input_batch_cache.h"
#include "dragnn/core/interfaces/component.h"
#include "dragnn/core/interfaces/transition_state.h"
#include "sling/base/logging.h"
#include "sling/file/file.h"
#include "sling/nlp/parser/trainer/document-batch.h"
#include "sling/string/strcat.h"

namespace sling {
namespace nlp {

using syntaxnet::dragnn::ComponentSpec;
using syntaxnet::dragnn::InputBatchCache;
using syntaxnet::dragnn::TransitionState;

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

  // Load shared resources.
  resources_.Load(spec_);
  const auto &lexicon = resources_.lexicon;
  const string &name = spec.name();
  LOG(INFO) << name << ": loaded " << lexicon.size() << " words";
  LOG(INFO) << name << ": loaded " << lexicon.prefixes().size() << " prefixes";
  LOG(INFO) << name << ": loaded " << lexicon.suffixes().size() << " suffixes";
  if (lexicon.size() > 0) {
    LOG(INFO) << "Lexicon OOV: " << lexicon.oov();
    LOG(INFO) << "Lexicon normalize digits: " << lexicon.normalize_digits();
  }

  // Determine if processing will be done left to right or not.
  left_to_right_ = true;
  for (const auto &param : spec_.transition_system().parameters()) {
    const string &val = param.second;
    if (param.first == "left_to_right") {
      CHECK(val == "false" || val == "true") << val;
      if (val == "false") left_to_right_ = false;
    }
  }

  // Set up the feature extractors.
  fixed_feature_extractor_.Init(spec_, &resources_);
  link_feature_extractor_.Init(spec_, &resources_);

  // Initialize gold transition generator.
  gold_transition_generator_.Init(resources_.global);
}

void SemparComponent::InitializeData(
    InputBatchCache *input_data, bool clear_existing_annotations) {
  // Save off the input data object.
  input_data_ = input_data;

  DocumentBatch *input = input_data->GetAs<DocumentBatch>();
  input->Decode(resources_.global, clear_existing_annotations);

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

void SemparComponent::GetFixedFeatures(int channel_id, int64 *output) const {
  int columns = fixed_feature_extractor_.MaxNumIds(channel_id);
  int size = batch_.size() * columns;
  for (int i = 0; i < size; ++i) output[i] = -1;

  for (SemparState *state : batch_) {
    fixed_feature_extractor_.Extract(channel_id, state, output);
    output += columns;
  }
}

void SemparComponent::GetRawLinkFeatures(
    int channel_id, int *steps, int *batch) const {
  int channel_size = link_feature_extractor_.ChannelSize(channel_id);
  for (int batch_idx = 0; batch_idx < batch_.size(); ++batch_idx) {
    SemparState *state = batch_[batch_idx];
    int base = batch_idx * channel_size;
    link_feature_extractor_.Extract(channel_id, state, steps + base);

    for (int i = base; i < base + channel_size; ++i) {
      batch[i] = batch_idx;
    }
  }
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
  SemparState *state =
      new SemparState(instance, resources_, system_type_, left_to_right_);
  state->set_gold_transition_generator(&gold_transition_generator_);
  fixed_feature_extractor_.Preprocess(state);

  return state;
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
