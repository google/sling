// Copyright 2017 Google Inc. All Rights Reserved.
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
#include "file/file.h"
#include "dragnn/components/util/bulk_feature_extractor.h"
#include "dragnn/core/component_registry.h"
#include "dragnn/core/input_batch_cache.h"
#include "dragnn/core/interfaces/component.h"
#include "dragnn/core/interfaces/transition_state.h"
#include "nlp/parser/trainer/document-batch.h"
#include "string/strcat.h"
#include "syntaxnet/sparse.pb.h"
#include "syntaxnet/task_context.h"
#include "syntaxnet/task_spec.pb.h"
#include "syntaxnet/utils.h"

namespace sling {
namespace nlp {

using sling::StrCat;
using syntaxnet::utils::Join;
using syntaxnet::utils::Split;

using syntaxnet::SparseFeatures;
using syntaxnet::dragnn::ComponentSpec;
using syntaxnet::dragnn::ComponentStepTrace;
using syntaxnet::dragnn::ComponentTrace;
using syntaxnet::dragnn::InputBatchCache;
using syntaxnet::dragnn::LinkFeatures;
using syntaxnet::dragnn::TransitionState;

namespace {

// Returns a new step in a trace based on a ComponentSpec.
ComponentStepTrace GetNewStepTrace(const ComponentSpec &spec,
                                   const TransitionState &state) {
  ComponentStepTrace step;
  for (auto &linked_spec : spec.linked_feature()) {
    auto &channel_trace = *step.add_linked_feature_trace();
    channel_trace.set_name(linked_spec.name());
    channel_trace.set_source_component(linked_spec.source_component());
    channel_trace.set_source_translator(linked_spec.source_translator());
    channel_trace.set_source_layer(linked_spec.source_layer());
  }
  for (auto &fixed_spec : spec.fixed_feature()) {
    step.add_fixed_feature_trace()->set_name(fixed_spec.name());
  }
  step.set_html_representation(state.HTMLRepresentation());
  return step;
}

// Returns the last step in the trace.
ComponentStepTrace *GetLastStepInTrace(ComponentTrace *trace) {
  CHECK_GT(trace->step_trace_size(), 0) << "Trace has no steps added yet";
  return trace->mutable_step_trace(trace->step_trace_size() - 1);
}

}  // namespace

SemparComponent::SemparComponent()
    : max_beam_size_(1),
      input_data_(nullptr),
      do_tracing_(false) {}

SemparComponent::~SemparComponent() {
  for (SemparState *state : batch_) delete state;
}

void SemparComponent::InitializeComponent(const ComponentSpec &spec) {
  File::Init();

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

void SemparComponent::InitializeData(
    const std::vector<std::vector<const TransitionState *>> &parent_states,
    int max_beam_size,
    InputBatchCache *input_data) {
  CHECK_EQ(max_beam_size, 1) << "BeamSize not supported:" << max_beam_size;
  max_beam_size_ = max_beam_size;

  // Save off the input data object.
  input_data_ = input_data;

  DocumentBatch *input = input_data->GetAs<DocumentBatch>();
  input->Decode(resources_.global);

  // Expect that the sentence data is the same size as the input states batch.
  if (!parent_states.empty()) {
    CHECK_EQ(parent_states.size(), input->size());
  }

  // Get rid of the previous batch.
  for (SemparState *old : batch_) delete old;

  // Fill the beams with the relevant data for that batch.
  batch_.clear();
  for (int batch_index = 0; batch_index < input->size(); ++batch_index) {
    SemparState *state = CreateState(input->item(batch_index));
    if (!parent_states.empty()) {
      CHECK_GE(parent_states.at(batch_index).size(), 1);
      state->Init(*parent_states.at(batch_index).at(0));
    }
    batch_.push_back(state);
  }
}

bool SemparComponent::IsReady() const { return input_data_ != nullptr; }

string SemparComponent::Name() const {
  return "SemparComponent";
}

int SemparComponent::BatchSize() const { return batch_.size(); }

int SemparComponent::BeamSize() const { return max_beam_size_; }

int SemparComponent::StepsTaken(int batch_index) const {
  return batch_.at(batch_index)->NumSteps();
}

int SemparComponent::GetBeamIndexAtStep(int step,
                                        int current_index,
                                        int batch) const {
  CHECK_EQ(current_index, 0) << "Only BeamSize=1 supported: " << current_index;
  return 0;  // since beam size is 1
}

int SemparComponent::GetSourceBeamIndex(int current_index, int batch) const {
  CHECK_EQ(current_index, 0) << "Only BeamSize=1 supported: " << current_index;
  return 0;  // since beam size is 1
}

std::function<int(int, int, int)> SemparComponent::GetStepLookupFunction(
    const string &method) {
  if (method == "reverse-token") {
    CHECK_EQ(system_type_, SHIFT_ONLY)
        << "'reverse-token' only supported for shift-only systems.";
    return [this](int batch_index, int beam_index, int value) {
      CHECK_EQ(beam_index, 0);
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
      CHECK_NE(best, -1) << state->HTMLRepresentation();
      Advance(state, best);
      state->SetScore(state->GetScore() + scores[offset + best]);
      state->SetBeamIndex(0);
    }
    offset += num_actions;
  }
}

void SemparComponent::AdvanceFromOracle() {
  for (SemparState *state : batch_) {
    if (state->IsFinal()) continue;
    Advance(state, GetOracleLabel(state));
    state->SetScore(0.0f);
    state->SetBeamIndex(0);
  }
}

bool SemparComponent::IsTerminal() const {
  for (SemparState *state : batch_) {
    if (!state->IsFinal()) return false;
  }
  return true;
}

std::vector<std::vector<const TransitionState *>> SemparComponent::GetBeam() {
  std::vector<std::vector<const TransitionState *>> state_beam;
  for (SemparState *state : batch_) {
    state_beam.push_back({state});
  }
  return state_beam;
}

int SemparComponent::GetFixedFeatures(
    std::function<int32 *(int)> allocate_indices,
    std::function<int64 *(int)> allocate_ids,
    std::function<float *(int)> allocate_weights, int channel_id) const {
  SemparFeature::Args args;
  args.debug = do_tracing_;

  std::vector<int> segment_indices;
  int next_segment_index = -1;
  for (SemparState *state : batch_) {
    args.state = state;
    int old_size = args.output.size();
    feature_extractor_.Extract(&args, channel_id);

    for (int i = old_size; i < args.output.size(); ++i) {
      if ((i == old_size) ||
          (args.output[i].feature_index < args.output[i - 1].feature_index)) {
        next_segment_index++;
      }
      segment_indices.emplace_back(next_segment_index);
    }

    if (do_tracing_) {
      auto *trace = GetLastStepInTrace(state->mutable_trace());
      auto *fixed_trace = trace->mutable_fixed_feature_trace(channel_id);

      for (int i = 0; i < feature_extractor_.ChannelSize(channel_id); ++i) {
        fixed_trace->add_value_trace();
      }

      for (const SemparFeature::Value &value : args.output) {
        fixed_trace->mutable_value_trace(value.feature_index)->add_value_name(
            value.debug);
      }
    }
  }
  int feature_count = args.output.size();
  CHECK_EQ(feature_count, segment_indices.size());

  int32 *indices_tensor = allocate_indices(feature_count);
  int64 *ids_tensor = allocate_ids(feature_count);
  float *weights_tensor = allocate_weights(feature_count);

  // TODO: If beam size increases to >1, then pad the feature vector with
  // empty entries wherever current_beam_size  < max_beam_size.
  // Look at SyntaxnetComponent::GetFixedFeatures() for details.
  for (int i = 0; i < feature_count; ++i) {
    weights_tensor[i] = 1.0;  // we only support weight = 1.0
    ids_tensor[i] = args.output[i].id;
    indices_tensor[i] = segment_indices[i];
  }

  return feature_count;
}

int SemparComponent::BulkGetFixedFeatures(
    const syntaxnet::dragnn::BulkFeatureExtractor &extractor) {
  LOG(FATAL) << "Not implemented";
  return -1;
}

std::vector<LinkFeatures> SemparComponent::GetRawLinkFeatures(
    int channel_id) const {
  std::vector<string> feature_names;
  if (do_tracing_) {
    feature_names = Split(spec_.linked_feature(channel_id).fml(), ' ');
  }

  int channel_size = link_feature_extractor_.ChannelSize(channel_id);
  std::vector<LinkFeatures> features;
  features.resize(batch_.size() * channel_size);

  SemparFeature::Args args;
  args.debug = do_tracing_;
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
      f.set_beam_idx(0);
      if (do_tracing_) {
        f.set_feature_name(feature_names.at(args.output[i].feature_index));
      }
    }
  }

  return features;
}

std::vector<std::vector<int>> SemparComponent::GetOracleLabels() const {
  std::vector<std::vector<int>> oracle_labels;
  for (SemparState *state : batch_) {
    oracle_labels.emplace_back();  // beamsize = 1
    oracle_labels.back().push_back(GetOracleLabel(state));
  }
  return oracle_labels;
}

void SemparComponent::FinalizeData() {
  // This chooses the top-scoring beam item to annotate the underlying document.
  for (SemparState *state : batch_) {
    if (!state->shift_only()) {
      state->parser_state()->AddParseToDocument(state->instance()->document);
    }
  }
}

void SemparComponent::ResetComponent() {
  for (SemparState *state : batch_) delete state;
  batch_.clear();
  input_data_ = nullptr;
  max_beam_size_ = 0;
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
  if (do_tracing_) {
    auto *trace = state->mutable_trace();
    auto *last_step = GetLastStepInTrace(trace);

    // Add action to the prior step.
    if (!shift_only()) {
      const ParserAction &a = resources_.table.Action(action);
      last_step->set_caption(a.ToString(resources_.global));
    } else {
      last_step->set_caption("SHIFT");
    }
    last_step->set_step_finished(true);
  }
  state->PerformAction(action);

  if (do_tracing_) {
    // Add info for the next step.
    *state->mutable_trace()->add_step_trace() = GetNewStepTrace(spec_, *state);
  }
}

void SemparComponent::InitializeTracing() {
  do_tracing_ = true;
  CHECK(IsReady()) << "Cannot initialize trace before InitializeData().";

  // Initialize each element of the beam with a new trace.
  for (SemparState *state : batch_) {
    ComponentTrace *trace = new ComponentTrace();
    trace->set_name(spec_.name());
    *trace->add_step_trace() = GetNewStepTrace(spec_, *state);
    state->set_trace(trace);
  }
}

void SemparComponent::DisableTracing() {
  do_tracing_ = false;
}

void SemparComponent::AddTranslatedLinkFeaturesToTrace(
    const std::vector<LinkFeatures> &features, int channel_id) {
  CHECK(do_tracing_) << "Tracing is not enabled.";
  int channel_size = link_feature_extractor_.ChannelSize(channel_id);
  int linear_idx = 0;

  for (SemparState *state : batch_) {
    for (int feature_idx = 0; feature_idx < channel_size; ++feature_idx) {
      auto *trace = GetLastStepInTrace(state->mutable_trace());
      auto *link_trace = trace->mutable_linked_feature_trace(channel_id);
      if (features[linear_idx].feature_value() >= 0 &&
          features[linear_idx].step_idx() >= 0) {
        *link_trace->add_value_trace() = features[linear_idx];
      }
      ++linear_idx;
    }
  }
}

std::vector<std::vector<ComponentTrace>> SemparComponent::GetTraceProtos()
    const {
  std::vector<std::vector<ComponentTrace>> traces;
  for (SemparState *state : batch_) {
    std::vector<ComponentTrace> beam_trace;
    beam_trace.push_back(*state->mutable_trace());
    traces.push_back(beam_trace);
  }
  return traces;
};

REGISTER_DRAGNN_COMPONENT(SemparComponent);

}  // namespace nlp
}  // namespace sling
