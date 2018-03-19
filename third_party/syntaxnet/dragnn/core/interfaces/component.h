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

#ifndef SYNTAXNET_DRAGNN_CORE_INTERFACES_COMPONENT_H_
#define SYNTAXNET_DRAGNN_CORE_INTERFACES_COMPONENT_H_

#include <functional>
#include <vector>

#include "dragnn/core/input_batch_cache.h"
#include "dragnn/core/interfaces/transition_state.h"
#include "dragnn/protos/spec.pb.h"
#include "sling/base/registry.h"

namespace syntaxnet {
namespace dragnn {

class Component : public sling::Component<Component> {
 public:
  virtual ~Component() {}

  // Initializes this component from the spec.
  virtual void InitializeComponent(const ComponentSpec &spec) = 0;

  // Initializes the component with data for the next batch.
  virtual void InitializeData(
      InputBatchCache *input_data, bool clear_existing_annotations) = 0;

  // Returns true if the component has had InitializeData called on it since
  // the last time it was reset.
  virtual bool IsReady() const = 0;

  // Returns the string name of this component.
  virtual string Name() const = 0;

  // Returns the current batch size of the component's underlying data.
  virtual int BatchSize() const = 0;

  // Returns the number of steps taken by this component so far.
  virtual int StepsTaken(int batch_index) const = 0;

  // Request a translation function based on the given method string.
  // The translation function will be called with arguments (batch, value)
  // and should return the step index corresponding to the given value, for the
  // data in the given batch.
  virtual std::function<int(int, int)> GetStepLookupFunction(
      const string &method) = 0;

  // Advances this component from the given transition matrix.
  virtual void AdvanceFromPrediction(const float transition_matrix[],
                                     int transition_matrix_length) = 0;

  // Advances this component from the state oracles.
  virtual void AdvanceFromOracle() = 0;

  // Returns true if all states within this component are terminal.
  virtual bool IsTerminal() const = 0;

  // Returns the current batch of states for this component.
  virtual std::vector<const TransitionState *> GetStates() = 0;

  // Extracts and populates into 'output' onwards the fixed features for the
  // specified channel.
  virtual void GetFixedFeatures(int channel_id, int64 *output) const = 0;

  // Returns the linked features for the specified channel.
  virtual void GetRawLinkFeatures(int channel_id, int *steps, int *batch)
      const = 0;

  // Returns a vector of oracle labels for each element in the batch.
  virtual std::vector<int> GetOracleLabels() const = 0;

  // Annotate the underlying data object with the results of this Component's
  // calculation.
  virtual void FinalizeData() = 0;

  // Reset this component.
  virtual void ResetComponent() = 0;
};

}  // namespace dragnn
}  // namespace syntaxnet

#endif  // SYNTAXNET_DRAGNN_CORE_INTERFACES_COMPONENT_H_
