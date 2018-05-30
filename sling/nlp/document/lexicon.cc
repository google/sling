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

void Lexicon::InitWords(Vocabulary::Iterator *words) {
  // Initialize mapping from words to ids.
  vocabulary_.Init(words);

  // Initialize mapping from ids to words.
  words_.resize(words->Size());
  int index = 0;
  words->Reset();
  Text word;
  while (words->Next(&word, nullptr)) {
    words_[index].word.assign(word.data(), word.size());
    index++;
  }
}

void Lexicon::WriteVocabulary(string *buffer, char terminator) const {
  for (const Entry &e : words_) {
    buffer->append(e.word);
    buffer->push_back(terminator);
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

void Lexicon::WritePrefixes(string *buffer) const {
  StringOutputStream stream(buffer);
  prefixes_.Write(&stream);
}

void Lexicon::WriteSuffixes(string *buffer) const {
  StringOutputStream stream(buffer);
  suffixes_.Write(&stream);
}

void Lexicon::BuildPrefixes(int max_prefix) {
  prefixes_.Reset(max_prefix);
  if (max_prefix > 0) {
    for (Entry &entry : words_) {
      entry.prefix = prefixes_.AddAffixesForWord(entry.word);
    }
  }
}

void Lexicon::BuildSuffixes(int max_suffix) {
  suffixes_.Reset(max_suffix);
  if (max_suffix > 0) {
    for (Entry &entry : words_) {
      entry.suffix = suffixes_.AddAffixesForWord(entry.word);
    }
  }
}

int Lexicon::LookupWord(const string &word, bool *changed) const {
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
      if (changed != nullptr) *changed = true;
    }
  }

  return id != -1 ? id : oov_;
}

}  // namespace nlp
}  // namespace sling

