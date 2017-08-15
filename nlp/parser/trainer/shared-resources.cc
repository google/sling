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

#include "nlp/parser/trainer/shared-resources.h"

#include "base/macros.h"
#include "frame/serialization.h"

namespace sling {
namespace nlp {

void SharedResources::LoadActionTable(const string &file) {
  CHECK(global != nullptr);
  Store temp(global);
  sling::LoadStore(file, &temp);
  table.Init(&temp);
  table.set_action_checks(false);
}

void SharedResources::LoadGlobalStore(const string &file) {
  delete global;
  global = new Store();
  sling::LoadStore(file, global);
  global->Freeze();
}

}  // namespace nlp
}  // namespace sling
