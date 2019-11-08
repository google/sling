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

#ifndef SLING_NLP_PARSER_TRACE_H_
#define SLING_NLP_PARSER_TRACE_H_

#include <unordered_map>
#include <utility>
#include <vector>

#include "sling/frame/object.h"
#include "sling/frame/store.h"
#include "sling/nlp/document/document.h"
#include "sling/nlp/parser/parser-action.h"

namespace sling {
namespace nlp {

// Tracing information for a semantic parse.
struct Trace {
  // Single step of the decoder.
  struct Step {
    // Index of the current token,
    int current;

    // Decoder feature -> List of feature value(s).
    std::unordered_map<string, std::vector<int>> ff_features;

    // List of (predicted action, final action) in this step's cascade.
    std::vector<std::pair<ParserAction, ParserAction>> actions;

    // Adds feature values for feature 'name'. If 'ptr' is not nullptr,
    // 'num' values are taken starting at 'ptr'.
    void Add(int *ptr, int num, const string &name);
  };

  Trace(int begin, int end) : begin(begin), end(end) {}

  // Beginning and ending tokens of the parser state.
  int begin;
  int end;

  // Token -> (encoder feature name -> feature values).
  std::vector<std::unordered_map<string, std::vector<int>>> lstm_features;

  // List of steps.
  std::vector<Step> steps;

  // Adds a predicted (=final) action to the latest step.
  void Action(const ParserAction &action);

  // Sets the last final action to 'fallback' for the latest step.
  void Fallback(const ParserAction &fallback);

  // Adds encoder feature values for 'token'.
  void AddLSTM(int token, const string &name, int val);

  // Writes tracing information as a frame to 'document'.
  void Write(Document *document) const;
};

}  // namespace nlp
}  // namespace sling

#endif  // SLING_NLP_PARSER_TRACE_H_
