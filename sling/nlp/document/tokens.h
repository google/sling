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

#ifndef SLING_NLP_DOCUMENT_TOKENS_H_
#define SLING_NLP_DOCUMENT_TOKENS_H_

#include <string>
#include <vector>

#include "sling/base/types.h"
#include "sling/frame/object.h"
#include "sling/frame/store.h"

namespace sling {
namespace nlp {

class Tokens {
 public:
  // Initialize tokens from document.
  Tokens(const Frame &document);

  // Initialize tokens from token array.
  Tokens(const Array &tokens);

  // Locate token at byte position.
  int Locate(int position) const;

  // Return phrase for token span.
  string Phrase(int begin, int end) const;

 private:
  // Token array.
  sling::Array tokens_;

  // Start positions for tokens.
  std::vector<int> positions_;

  // Symbol names.
  Names names_;
  Name n_document_tokens_{names_, "/s/document/tokens"};
  Name n_token_start_{names_, "/s/token/start"};
  Name n_token_text_{names_, "/s/token/text"};
  Name n_token_break_{names_, "/s/token/break"};
};

}  // namespace nlp
}  // namespace sling

#endif  // SLING_NLP_DOCUMENT_TOKENS_H_

