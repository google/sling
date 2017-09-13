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

#ifndef NLP_PARSER_TRAINER_SHARED_RESOURCES_H_
#define NLP_PARSER_TRAINER_SHARED_RESOURCES_H_

#include "frame/store.h"
#include "nlp/parser/action-table.h"

namespace sling {
namespace nlp {

// Container for resources that are typically shared (e.g. across features).
struct SharedResources {
  ActionTable table;
  Store *global = nullptr;  // owned

  ~SharedResources() { delete global; }

  // Loads global store from 'file'.
  void LoadGlobalStore(const string &file);

  // Loads action table from 'file'.
  void LoadActionTable(const string &file);
};

}  // namespace nlp
}  // namespace sling

#endif // NLP_PARSER_TRAINER_SHARED_RESOURCES_H_
