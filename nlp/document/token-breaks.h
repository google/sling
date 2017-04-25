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

#ifndef NLP_DOCUMENT_TOKEN_BREAKS_H_
#define NLP_DOCUMENT_TOKEN_BREAKS_H_

namespace sling {
namespace nlp {

// Token break types.
enum BreakType {
  NO_BREAK         = 0,
  SPACE_BREAK      = 1,
  LINE_BREAK       = 2,
  SENTENCE_BREAK   = 3,
  PARAGRAPH_BREAK  = 4,
  SECTION_BREAK    = 5,
  CHAPTER_BREAK    = 6,
};

}  // namespace nlp
}  // namespace sling

#endif  // NLP_DOCUMENT_TOKEN_BREAKS_H_

