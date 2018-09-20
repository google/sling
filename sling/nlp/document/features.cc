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

#include "sling/nlp/document/features.h"

#include "sling/base/types.h"
#include "sling/nlp/document/document.h"
#include "sling/util/unicode.h"

namespace sling {
namespace nlp {

void DocumentFeatures::Extract(const Document &document, int begin, int end) {
  if (end == -1) end = document.num_tokens();
  int length = end - begin;
  features_.resize(length);
  bool in_quote = false;
  for (int i = 0; i < length; ++i) {
    const string &word = document.token(begin + i).word();
    TokenFeatures &f = features_[i];

    // Look up token word in lexicon and get word features.
    f.word = lexicon_->Lookup(word, &f.prefix, &f.suffix, &f.shape);

    // Re-compute context-sensitive features.
    if (i == 0 || document.token(i).brk() >= SENTENCE_BREAK) {
      if (f.shape.capitalization == WordShape::CAPITALIZED) {
        f.shape.capitalization = WordShape::INITIAL;
      }
    }
    if (f.shape.quote == WordShape::UNKNOWN_QUOTE) {
      f.shape.quote = in_quote ? WordShape::CLOSE_QUOTE : WordShape::OPEN_QUOTE;
      in_quote = !in_quote;
    }
  }
}

}  // namespace nlp
}  // namespace sling

