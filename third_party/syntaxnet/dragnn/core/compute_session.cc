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

#include "dragnn/core/compute_session.h"

#include <algorithm>
#include <utility>

#include "base/registry.h"
#include "dragnn/protos/spec.pb.h"
#include "tensorflow/core/platform/logging.h"

namespace syntaxnet {
namespace dragnn {

ComputeSession::ComputeSession(
    int id,
    std::function<std::unique_ptr<Component>(const string &component_name,
                                             const string &backend_type)>
        component_builder)
    : component_builder_(std::move(component_builder)), id_(id) {}

void ComputeSession::Init(const MasterSpec &master_spec,
                              const GridPoint &hyperparams) {
  spec_ = master_spec;
  grid_point_ = hyperparams;

  VLOG(2) << "Creating components.";
  bool is_input = true;
  Component *predecessor;
  for (const ComponentSpec &spec : master_spec.component()) {
    // Construct the component using the specified backend.
    VLOG(2) << "Creating component '" << spec.name()
            << "' with backend: " << spec.backend().registered_name();
    auto component =
        component_builder_(spec.name(), spec.backend().registered_name());

    // Initializes the component.
    component->InitializeComponent(spec);

    // Adds a predecessor to non-input components.
    if (!is_input) {
      predecessors_.insert(
          std::pair<Component *, Component *>(component.get(), predecessor));
    }

    // The current component will be the predecessor component next time around.
    predecessor = component.get();

    // All components after the first are non-input components.
    is_input = false;

    // Move into components list.
    components_.insert(std::pair<string, std::unique_ptr<Component>>(
        spec.name(), std::move(component)));
  }
  VLOG(2) << "Done creating components.";

  VLOG(2) << "Adding translators.";
  for (const ComponentSpec &spec : master_spec.component()) {
    // First, get the component object for this spec.
    VLOG(2) << "Examining component: " << spec.name();
    auto map_result = components_.find(spec.name());
    CHECK(map_result != components_.end()) << "Unable to find component.";
    Component *start_component = map_result->second.get();

    if (spec.linked_feature_size() > 0) {
      VLOG(2) << "Adding " << spec.linked_feature_size() << " translators for "
              << spec.name();

      // Attach all the translators described in the spec.
      std::vector<IndexTranslator *> translator_set;
      for (const LinkedFeatureChannel &channel : spec.linked_feature()) {
        // For every translator, save off a non-unique ptr in the component name
        // to translator map, then push the unique ptr onto the management
        // vector.
        auto translator = CreateTranslator(channel, start_component);
        translator_set.push_back(translator.get());
        owned_translators_.push_back(std::move(translator));
      }

      // Once all translators have been created, associate this group of
      // translators with a component.
      translators_.insert(std::pair<string, std::vector<IndexTranslator *>>(
          spec.name(), std::move(translator_set)));
    } else {
      VLOG(2) << "No translators found for " << spec.name();
    }
  }
  VLOG(2) << "Done adding translators.";
  VLOG(2) << "Initialization complete.";
}

void ComputeSession::InitializeComponentData(const string &component_name) {
  CHECK(input_data_ != nullptr) << "Attempted to access a component without "
                                   "providing input data for this session.";
  Component *component = GetComponent(component_name);

  // Try and find the source component. If one exists, check that it is terminal
  // and get its data; if not, pass in an empty vector for source data.
  auto source_result = predecessors_.find(component);
  if (source_result != predecessors_.end()) {
    auto source = source_result->second;
    CHECK(source->IsTerminal()) << "Source is not terminal for component '"
                                << component_name << "'. Exiting.";
  }
  component->InitializeData(input_data_.get());
}

int ComputeSession::BatchSize(const string &component_name) const {
  return GetReadiedComponent(component_name)->BatchSize();
}

const ComponentSpec &ComputeSession::Spec(
    const string &component_name) const {
  for (const auto &component : spec_.component()) {
    if (component.name() == component_name) {
      return component;
    }
  }
  LOG(FATAL) << "Missing component '" << component_name << "'. Exiting.";
}

void ComputeSession::AdvanceFromOracle(const string &component_name) {
  GetReadiedComponent(component_name)->AdvanceFromOracle();
}

void ComputeSession::AdvanceFromPrediction(const string &component_name,
                                           const float score_matrix[],
                                           int score_matrix_length) {
  GetReadiedComponent(component_name)
      ->AdvanceFromPrediction(score_matrix, score_matrix_length);
}

void ComputeSession::GetInputFeatures(
    const string &component_name, int channel_id, int64 *output) const {
  GetReadiedComponent(component_name)->GetFixedFeatures(channel_id, output);
}

void ComputeSession::GetTranslatedLinkFeatures(
    const string &component_name, int channel_id, int size,
    int *steps, int *batch) {
  auto *component = GetReadiedComponent(component_name);
  component->GetRawLinkFeatures(channel_id, steps, batch);

  IndexTranslator *translator = GetTranslators(component_name).at(channel_id);
  for (int i = 0; i < size; ++i) {
    if (steps[i] >= 0) {
      VLOG(2) << "Raw feature[" << i << "] step: " << steps[i];
      IndexTranslator::Index index = translator->Translate(batch[i], steps[i]);
      steps[i] = index.step_index;
      batch[i] = index.batch_index;
      VLOG(2) << "Translated feature[" << i << "] step: " << steps[i];
    } else {
      // Clip missing steps to -1 and their batch index to 0.
      steps[i] = -1;
      batch[i] = 0;
      VLOG(2) << "Raw feature[" << i << "]: PADDING (empty proto)";
    }
  }
}

std::vector<int> ComputeSession::EmitOracleLabels(
    const string &component_name) {
  return GetReadiedComponent(component_name)->GetOracleLabels();
}

bool ComputeSession::IsTerminal(const string &component_name) {
  return GetReadiedComponent(component_name)->IsTerminal();
}

void ComputeSession::FinalizeData(const string &component_name) {
  VLOG(2) << "Finalizing data for " << component_name;
  GetReadiedComponent(component_name)->FinalizeData();
}

std::vector<string> ComputeSession::GetSerializedPredictions() {
  VLOG(2) << "Geting serialized predictions.";
  return input_data_->SerializedData();
}

void ComputeSession::SetInputData(const std::vector<string> &data) {
  input_data_.reset(new InputBatchCache(data));
}

void ComputeSession::ResetSession() {
  // Reset all component states.
  for (auto &component_pair : components_) {
    component_pair.second->ResetComponent();
  }

  // Reset the input data pointer.
  input_data_.reset();
}

int ComputeSession::Id() const { return id_; }

string ComputeSession::GetDescription(const string &component_name) const {
  return GetComponent(component_name)->Name();
}

const std::vector<const IndexTranslator *> ComputeSession::Translators(
    const string &component_name) const {
  auto translators = GetTranslators(component_name);
  std::vector<const IndexTranslator *> const_translators;
  for (const auto &translator : translators) {
    const_translators.push_back(translator);
  }
  return const_translators;
}

Component *ComputeSession::GetReadiedComponent(
    const string &component_name) const {
  auto component = GetComponent(component_name);
  CHECK(component->IsReady())
      << "Attempted to access component " << component_name
      << " without first initializing it.";
  return component;
}

Component *ComputeSession::GetComponent(
    const string &component_name) const {
  auto result = components_.find(component_name);
  if (result == components_.end()) {
    LOG(ERROR) << "Could not find component \"" << component_name
               << "\" in the component set. Current components are: ";
    for (const auto &component_pair : components_) {
      LOG(ERROR) << component_pair.first;
    }
    LOG(FATAL) << "Missing component. Exiting.";
  }

  auto component = result->second.get();
  return component;
}

const std::vector<IndexTranslator *> &ComputeSession::GetTranslators(
    const string &component_name) const {
  auto result = translators_.find(component_name);
  if (result == translators_.end()) {
    LOG(ERROR) << "Could not find component " << component_name
               << " in the translator set. Current components are: ";
    for (const auto &component_pair : translators_) {
      LOG(ERROR) << component_pair.first;
    }
    LOG(FATAL) << "Missing component. Exiting.";
  }
  return result->second;
}

std::unique_ptr<IndexTranslator> ComputeSession::CreateTranslator(
    const LinkedFeatureChannel &channel, Component *start_component) {
  const int num_components = spec_.component_size();
  VLOG(2) << "Channel spec: " << channel.ShortDebugString();

  // Find the linked feature's source component, if it exists.
  auto source_map_result = components_.find(channel.source_component());
  CHECK(source_map_result != components_.end())
      << "Unable to find source component " << channel.source_component();
  const Component *end_component = source_map_result->second.get();

  // Our goal here is to iterate up the source map from the
  // start_component to the end_component.
  Component *current_component = start_component;
  std::vector<Component *> path;
  path.push_back(current_component);
  while (current_component != end_component) {
    // Try to find the next link upwards in the source chain.
    auto source_result = predecessors_.find(current_component);

    // If this component doesn't have a source to find, that's an error.
    CHECK(source_result != predecessors_.end())
        << "No link to source " << channel.source_component();

    // If we jump more times than there are components in the graph, that
    // is an error state.
    CHECK_LT(path.size(), num_components) << "Too many jumps. Is there a "
                                             "loop in the MasterSpec "
                                             "component definition?";

    // Add the source to the vector and repeat.
    path.push_back(source_result->second);
    current_component = source_result->second;
  }

  // At this point, we have the source chain for the traslator and can
  // build it.
  std::unique_ptr<IndexTranslator> translator(
      new IndexTranslator(path, channel.source_translator()));
  return translator;
}

}  // namespace dragnn
}  // namespace syntaxnet
