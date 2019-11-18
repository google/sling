// Copyright 2019 Google Inc.
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

#ifndef SLING_NLP_PARSER_TRANSITION_GENERATOR_H_
#define SLING_NLP_PARSER_TRANSITION_GENERATOR_H_

#include <functional>

#include "sling/nlp/document/document.h"
#include "sling/nlp/parser/parser-action.h"

namespace sling {
namespace nlp {

// Generates transition sequences for [begin, end) token range in 'document',
// calling 'callback' for every transition.
void Generate(const Document &document,
              int begin, int end,
              std::function<void(const ParserAction &)> callback);

// Generates transition sequences for all tokens in 'document', calling
// 'callback' for every transition.
void Generate(const Document &document,
              std::function<void(const ParserAction &)> callback);

}  // namespace nlp
}  // namespace sling

#endif  // SLING_NLP_PARSER_TRANSITION_GENERATOR_H_
