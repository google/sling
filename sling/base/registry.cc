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

#include "sling/base/registry.h"

namespace sling {

// Global list of all component registries.
RegistryMetadata *RegistryMetadata::global_registry_list = nullptr;

void RegistryMetadata::GetComponents(
    std::vector<const ComponentMetadata *> *components) const {
  components->clear();
  ComponentMetadata *meta = *components_;
  while (meta != nullptr) {
    components->push_back(meta);
    meta = meta->link();
  }
}

const ComponentMetadata *RegistryMetadata::GetComponent(
    const string &name) const {
  ComponentMetadata *meta = *components_;
  while (meta != nullptr) {
    if (name == meta->name()) return meta;
    meta = meta->link();
  }
  return nullptr;
}

void RegistryMetadata::Register(RegistryMetadata *registry) {
  registry->set_link(global_registry_list);
  global_registry_list = registry;
}

void RegistryMetadata::GetRegistries(
    std::vector<const RegistryMetadata *> *registries) {
  registries->clear();
  RegistryMetadata *meta = global_registry_list;
  while (meta != nullptr) {
    registries->push_back(meta);
    meta = meta->next();
  }
}

const RegistryMetadata *RegistryMetadata::GetRegistry(const string &name) {
  RegistryMetadata *meta = global_registry_list;
  while (meta != nullptr) {
    if (name == meta->name()) return meta;
    meta = meta->next();
  }
  return nullptr;
}

}  // namespace sling

