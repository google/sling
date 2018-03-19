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

#include "sling/nlp/document/tokens.h"

#include <algorithm>
#include <string>
#include <vector>

#include "sling/base/logging.h"
#include "sling/base/types.h"
#include "sling/frame/object.h"
#include "sling/frame/store.h"
#include "sling/nlp/document/text-tokenizer.h"
#include "sling/nlp/document/token-breaks.h"

namespace sling {
namespace nlp {

Tokens::Tokens(const Frame &document) {
  // Bind names.
  Store *store = document.store();
  names_.Bind(store);

  // Get tokens.
  tokens_ = document.Get(n_document_tokens_).AsArray();
  if (!tokens_.valid()) return;

  // Build token position index.
  int num_tokens = tokens_.length();
  positions_.resize(num_tokens);
  int last_position = 0;
  for (int i = 0; i < num_tokens; ++i) {
    Frame token(store, tokens_.get(i));
    positions_[i] = token.GetInt(n_token_start_);
    CHECK_GE(positions_[i], last_position);
    last_position = positions_[i];
  }
}

Tokens::Tokens(const Array &tokens) : tokens_(tokens) {
  // Bind names.
  Store *store = tokens.store();
  names_.Bind(store);

  // Build token position index.
  int num_tokens = tokens.length();
  positions_.resize(num_tokens);
  for (int i = 0; i < num_tokens; ++i) {
    Frame token(store, tokens_.get(i));
    positions_[i] = token.GetInt(n_token_start_);
  }
}

int Tokens::Locate(int position) const {
  auto it = std::lower_bound(positions_.begin(), positions_.end(), position);
  return std::distance(positions_.begin(), it);
}

string Tokens::Phrase(int begin, int end) const {
  string phrase;
  CHECK(tokens_.valid());
  CHECK_GE(begin, 0);
  CHECK_LE(begin, end);
  CHECK_LE(end, tokens_.length());
  for (int t = begin; t < end; ++t) {
    Frame token(tokens_.store(), tokens_.get(t));
    Text text = token.GetText(n_token_text_);
    int brk = token.GetInt(n_token_break_, SPACE_BREAK);
    if (t > begin && brk > NO_BREAK) phrase.push_back(' ');
    phrase.append(text.data(), text.size());
  }
  return phrase;
}

}  // namespace nlp
}  // namespace sling

