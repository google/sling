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

#ifndef NLP_DOCUMENT_LEXICON_H_
#define NLP_DOCUMENT_LEXICON_H_

#include <string>
#include <vector>

#include "base/types.h"
#include "nlp/document/affix.h"
#include "util/vocabulary.h"

namespace sling {
namespace nlp {

// Lexicon for extracting lexical features from documents.
class Lexicon {
 public:
  // Initialize lexicon with newline-terminated word list.
  void InitWords(const char *data, size_t size);

  // Initialize affix tables.
  void InitPrefixes(const char *data, size_t size);
  void InitSuffixes(const char *data, size_t size);

  // Look up word in vocabulary. Return OOV if word is not found.
  int LookupWord(const string &word) const;

  // Return number of words in vocabulary.
  size_t size() const { return words_.size(); }

  // Return word in vocabulary.
  const string &word(int index) const { return words_[index].word; }

  // Get longest prefix for known word.
  Affix *prefix(int index) const { return words_[index].prefix; }

  // Get longest suffix for known word.
  Affix *suffix(int index) const { return words_[index].suffix; }

  // Get affix tables.
  const AffixTable &prefixes() const { return prefixes_; }
  const AffixTable &suffixes() const { return suffixes_; }

  // Out-of-vocabulary id.
  int oov() const { return oov_; }
  void set_oov(int oov) { oov_ = oov; }

  // Digit normalization of words.
  bool normalize_digits() const { return normalize_digits_; }
  void set_normalize_digits(bool normalize) { normalize_digits_ = normalize; }

 private:
  // Lexicon entry.
  struct Entry {
    string word;                // lexical word form
    Affix *prefix = nullptr;    // longest known prefix for word
    Affix *suffix = nullptr;    // longest known suffix for word
  };

  // Mapping from words to ids.
  Vocabulary vocabulary_;
  bool normalize_digits_ = false;
  int oov_ = -1;

  // Lexicon entries.
  std::vector<Entry> words_;

  // Word prefixes.
  AffixTable prefixes_{AffixTable::PREFIX, 0};

  // Word suffixes.
  AffixTable suffixes_{AffixTable::SUFFIX, 0};
};

}  // namespace nlp
}  // namespace sling

#endif  // NLP_DOCUMENT_LEXICON_H_

