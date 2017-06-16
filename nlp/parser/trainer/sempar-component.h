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
// =============================================================================

#ifndef NLP_PARSER_TRAINER_SEMPAR_COMPONENT_H_
#define NLP_PARSER_TRAINER_SEMPAR_COMPONENT_H_

#include <string>
#include <vector>

#include "dragnn/core/input_batch_cache.h"
#include "dragnn/core/interfaces/component.h"
#include "dragnn/core/interfaces/transition_state.h"
#include "dragnn/protos/data.pb.h"
#include "dragnn/protos/spec.pb.h"
#include "dragnn/protos/trace.pb.h"
#include "nlp/parser/trainer/feature.h"
#include "nlp/parser/trainer/sempar-instance.h"
#include "nlp/parser/trainer/shared-resources.h"
#include "nlp/parser/trainer/transition-state.h"
#include "nlp/parser/trainer/transition-system-type.h"
#include "syntaxnet/registry.h"

namespace sling {
namespace nlp {

class SemparComponent : public syntaxnet::dragnn::Component {
 public:
  SemparComponent();
  ~SemparComponent();

  // Initializes this component from the spec.
  void InitializeComponent(
      const syntaxnet::dragnn::ComponentSpec &spec) override;

  // Provides the previous beam to the component.
  void InitializeData(
      const std::vector<std::vector<const syntaxnet::dragnn::TransitionState *>>
      &states,
      int max_beam_size,
      syntaxnet::dragnn::InputBatchCache *input_data) override;

  // Returns true if the component has had InitializeData called on it since
  // the last time it was reset.
  bool IsReady() const override;

  // Returns the string name of this component.
  string Name() const override;

  // Returns the number of steps taken by the given batch item.
  int StepsTaken(int batch_index) const override;

  // Returns the current batch size of the component's underlying data.
  int BatchSize() const override;

  // Returns the maximum beam size of this component.
  int BeamSize() const override;

  // Return the beam index of the item which is currently at index
  // 'index', when the beam was at step 'step', for batch element 'batch'.
  int GetBeamIndexAtStep(int step, int current_index, int batch) const override;

  // Return the source index of the item which is currently at index 'index'
  // for batch element 'batch'. This index is into the final beam of the
  // Component that this Component was initialized from.
  int GetSourceBeamIndex(int current_index, int batch) const override;

  // Request a translation function based on the given method string.
  // The translation function will be called with arguments (batch, beam, value)
  // and should return the step index corresponding to the given value, for the
  // data in the given beam and batch.
  std::function<int(int, int, int)> GetStepLookupFunction(
      const string &method) override;

  // Advances this component from the given transition matrix.
  void AdvanceFromPrediction(const float transition_matrix[],
                             int transition_matrix_length) override;

  // Advances this component from the state oracles.
  void AdvanceFromOracle() override;

  // Returns true if all states within this component are terminal.
  bool IsTerminal() const override;

  // Returns the current batch of beams for this component.
  std::vector<std::vector<const syntaxnet::dragnn::TransitionState *>> GetBeam()
      override;

  // Extracts and populates the FixedFeatures vector for the specified channel.
  int GetFixedFeatures(std::function<int32 *(int)> allocate_indices,
                       std::function<int64 *(int)> allocate_ids,
                       std::function<float *(int)> allocate_weights,
                       int channel_id) const override;

  // Extracts and populates all FixedFeatures for all channels, advancing this
  // component via the oracle until it is terminal.
  int BulkGetFixedFeatures(
      const syntaxnet::dragnn::BulkFeatureExtractor &extractor) override;

  // Extracts and returns the vector of LinkFeatures for the specified
  // channel. Note: these are NOT translated.
  std::vector<syntaxnet::dragnn::LinkFeatures> GetRawLinkFeatures(
      int channel_id) const override;

  // Returns a vector of oracle labels for each element in the beam and
  // batch.
  std::vector<std::vector<int>> GetOracleLabels() const override;

  // Annotates the underlying instance with this component's calculation.
  void FinalizeData() override;

  // Resets this component.
  void ResetComponent() override;

  // Initializes the component for tracing execution. This will typically have
  // the side effect of slowing down all subsequent Component calculations
  // and storing a trace in memory that can be returned by GetTraceProtos().
  void InitializeTracing() override;

  // Disables tracing, freeing any additional memory and avoiding triggering
  // additional computation in the future.
  void DisableTracing() override;

  std::vector<std::vector<syntaxnet::dragnn::ComponentTrace>> GetTraceProtos()
      const override;

  // Adds tracing information for specified channel's link features.
  void AddTranslatedLinkFeaturesToTrace(
      const std::vector<syntaxnet::dragnn::LinkFeatures> &features,
      int channel_id) override;

  // Returns the underlying component spec.
  syntaxnet::dragnn::ComponentSpec *spec() { return &spec_; }

  TransitionSystemType system_type() const { return system_type_; }
  bool left_to_right() const { return left_to_right_; }
  bool shift_only() const { return system_type_ == SHIFT_ONLY; }

 private:
  // Permission function for this component.
  bool IsAllowed(SemparState *state, int action) const;

  // Returns true if this state is final
  bool IsFinal(SemparState *state) const;

  // Oracle function for this component.
  int GetOracleLabel(SemparState *state) const;

  // State advance function for this component.
  void Advance(SemparState *state, int action);

  // Creates a new state for the given instance.
  SemparState *CreateState(SemparInstance *instance);

  // Transition system type.
  TransitionSystemType system_type_;

  // If the tokens are traversed left to right (only for SHIFT_ONLY).
  bool left_to_right_ = true;

  // Shared resources.
  SharedResources resources_;

  // Gold sequence generator (only used during training).
  GoldTransitionGenerator gold_transition_generator_;

  // Extractor for fixed features
  SemparFeatureExtractor feature_extractor_;

  // Extractor for linked features.
  SemparFeatureExtractor link_feature_extractor_;

  // Internal workspace registry for use in feature extraction.
  syntaxnet::WorkspaceRegistry workspace_registry_;

  // The ComponentSpec used to initialize this component.
  syntaxnet::dragnn::ComponentSpec spec_;

  // Current batch of states (corresponds to beamsize = 1). Owned.
  std::vector<SemparState *> batch_;

  // Current max beam size.
  int max_beam_size_;

  // Underlying input data. Not owned.
  syntaxnet::dragnn::InputBatchCache *input_data_;

  // Whether or not to trace for each batch and beam element.
  bool do_tracing_;
};

}  // namespace nlp
}  // namespace sling

#endif  // NLP_PARSER_TRAINER_SEMPAR_COMPONENT_H_
