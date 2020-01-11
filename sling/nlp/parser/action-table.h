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

#ifndef SLING_NLP_PARSER_ACTION_TABLE_H_
#define SLING_NLP_PARSER_ACTION_TABLE_H_

#include <unordered_map>
#include <vector>

#include "sling/base/types.h"
#include "sling/frame/object.h"
#include "sling/frame/store.h"
#include "sling/nlp/parser/parser-action.h"

namespace sling {
namespace nlp {

// The action table is a set of parser actions indexed by id.
class ActionTable {
 public:
  // Add action to the table.
  void Add(const ParserAction &action);

  // Return the index of action.
  int Index(const ParserAction &action) const {
    const auto &it = mapping_.find(action);
    return it == mapping_.end() ? -1 : it->second;
  }

  // Return the number of parser actions.
  int size() const { return actions_.size(); }

  // Return the ith parser action.
  const ParserAction &Action(int index) const { return actions_[index]; }

  // Return list of actions.
  const std::vector<ParserAction> &list() const { return actions_; }

  // Read action table from frame.
  void Read(const Frame &frame);

  // Write action table to frame.
  void Write(Builder *frame) const;

 private:
  // List of actions.
  std::vector<ParserAction> actions_;

  // Mapping from parser action to index.
  std::unordered_map<ParserAction, int, ParserActionHash> mapping_;
};

}  // namespace nlp
}  // namespace sling

#endif  // SLING_NLP_PARSER_ACTION_TABLE_H_
