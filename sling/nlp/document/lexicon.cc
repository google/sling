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

#include "sling/nlp/document/lexicon.h"

#include "sling/base/types.h"
#include "sling/nlp/document/affix.h"
#include "sling/stream/memory.h"
#include "sling/util/vocabulary.h"

namespace sling {
namespace nlp {

void Lexicon::InitWords(const char *data, size_t size) {
  // Initialize mapping from words to ids.
  const static char kTerminator = '\n';
  vocabulary_.Init(data, size, kTerminator);

  // Initialize mapping from ids to words.
  words_.resize(vocabulary_.size());
  const char *current = data;
  const char *end = data + size;
  int index = 0;
  while (current < end) {
    // Find next word.
    const char *next = current;
    while (next < end && *next != kTerminator) next++;
    if (next == end) break;

    // Initialize item for word.
    words_[index].word.assign(current, next - current);

    current = next + 1;
    index++;
  }
}

void Lexicon::InitPrefixes(const char *data, size_t size) {
  // Read prefixes.
  ArrayInputStream stream(data, size);
  prefixes_.Read(&stream);

  // Pre-compute the longest prefix for all words in lexicon.
  for (Entry &entry : words_) {
    entry.prefix = prefixes_.GetLongestAffix(entry.word);
  }
}

void Lexicon::InitSuffixes(const char *data, size_t size) {
  // Read suffixes.
  ArrayInputStream stream(data, size);
  suffixes_.Read(&stream);

  // Pre-compute the longest suffix for all words in lexicon.
  for (Entry &entry : words_) {
    entry.suffix = suffixes_.GetLongestAffix(entry.word);
  }
}

int Lexicon::LookupWord(const string &word) const {
  // Lookup word in vocabulary.
  int id = vocabulary_.Lookup(word);

  if (id == -1 && normalize_digits_) {
    // Check if word has digits.
    bool has_digits = false;
    for (char c : word) {
      if (c >= '0' && c <= '9') has_digits = true;
    }

    if (has_digits) {
      // Normalize digits and lookup the normalized word.
      string normalized = word;
      for (char &c : normalized) {
        if (c >= '0' && c <= '9') c = '9';
      }
      id = vocabulary_.Lookup(normalized);
    }
  }

  return id != -1 ? id : oov_;
}

}  // namespace nlp
}  // namespace sling

