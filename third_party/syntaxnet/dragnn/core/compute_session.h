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

#ifndef SYNTAXNET_DRAGNN_CORE_COMPUTE_SESSION_H_
#define SYNTAXNET_DRAGNN_CORE_COMPUTE_SESSION_H_

#include <string>

#include "dragnn/core/index_translator.h"
#include "dragnn/core/interfaces/component.h"
#include "dragnn/protos/spec.pb.h"

namespace syntaxnet {
namespace dragnn {

// ComputeSession object that drives a DRAGNN session.
class ComputeSession {
 public:
  // Creates a ComputeSessionwith the provided component builder function.
  ComputeSession(
      int id,
      std::function<std::unique_ptr<Component>(const string &component_name,
                                               const string &backend_type)>
          component_builder);

  // Initialize this ComputeSession to compute the graph defined in the given
  // MasterSpec with the hyperparameters passed in the GridPoint. This should
  // only be called once, when the ComputeSession is created.
  void Init(const MasterSpec &master_spec, const GridPoint &hyperparams);

  // Initialize a component with data.
  // Note that attempting to initialize a component that depends on
  // another component that has not yet finished will cause a CHECK failure.
  void InitializeComponentData(const string &component_name);

  // Return the batch size for the given component.
  int BatchSize(const string &component_name) const;

  // Returns the spec used to create this ComputeSession.
  const ComponentSpec &Spec(const string &component_name) const;

  // Advance the given component using the component's oracle.
  void AdvanceFromOracle(const string &component_name);

  // Advance the given component using the given score matrix.
  void AdvanceFromPrediction(const string &component_name,
                             const float score_matrix[],
                             int score_matrix_length);

  // Get the fixed features for the given component and channel. This passes
  // through to the relevant Component's GetFixedFeatures() call.
  void GetInputFeatures(
      const string &component_name,
      int channel_id,
      int64 *output) const;

  // Get the linked features for the given component and channel.i
  void GetTranslatedLinkFeatures(
      const string &component_name, int channel_id, int output_size,
      int *steps, int *batch);

  // Get the oracle labels for the given component.
  std::vector<int> EmitOracleLabels(const string &component_name);

  // Returns true if the given component is terminal.
  bool IsTerminal(const string &component_name);

  // Force the given component to write out its predictions to the backing data.
  void FinalizeData(const string &component_name);

  // Return the finalized predictions from this compute session.
  std::vector<string> GetSerializedPredictions();

  // Provides the ComputeSession with a batch of data to compute.
  void SetInputData(const std::vector<string> &data);

  // Resets all components owned by this ComputeSession.
  void ResetSession();

  // Returns a unique identifier for this ComputeSession.
  int Id() const;

  // Returns a string describing the given component.
  string GetDescription(const string &component_name) const;

  // Get all the translators for the given component. Should only be used to
  // validate correct construction of translators in tests.
  const std::vector<const IndexTranslator *> Translators(
      const string &component_name) const;
 private:
  // Get a given component. Fails if the component is not found.
  Component *GetComponent(const string &component_name) const;

  // Get a given component. CHECK-fail if the component's IsReady method
  // returns false.
  Component *GetReadiedComponent(const string &component_name) const;

  // Get the index translators for the given component.
  const std::vector<IndexTranslator *> &GetTranslators(
      const string &component_name) const;

  // Create an index translator.
  std::unique_ptr<IndexTranslator> CreateTranslator(
      const LinkedFeatureChannel &channel, Component *start_component);

  // Perform initialization on the given Component.
  void InitComponent(Component *component);

  // Holds all of the components owned by this ComputeSession, associated with
  // their names in the MasterSpec.
  std::map<string, std::unique_ptr<Component>> components_;

  // Holds a vector of translators for each component, indexed by the name
  // of the component they belong to.
  std::map<string, std::vector<IndexTranslator *>> translators_;

  // Holds ownership of all the IndexTranslators for this compute session.
  std::vector<std::unique_ptr<IndexTranslator>> owned_translators_;

  // The predecessor component for every component.
  // If a component is not in this map, it has no predecessor component and
  // will be initialized without any data from other components.
  std::map<Component *, Component *> predecessors_;

  // Holds the current input data for this ComputeSession.
  std::unique_ptr<InputBatchCache> input_data_;

  // Function that, given a string, will return a Component.
  std::function<std::unique_ptr<Component>(const string &component_name,
                                           const string &backend_type)>
      component_builder_;

  // The master spec for this compute session.
  MasterSpec spec_;

  // The hyperparameters for this compute session.
  GridPoint grid_point_;

  // Unique identifier, assigned at construction.
  int id_;

};

}  // namespace dragnn
}  // namespace syntaxnet

#endif  // SYNTAXNET_DRAGNN_CORE_COMPUTE_SESSION_H_
