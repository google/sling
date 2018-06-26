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

#ifndef SLING_NLP_DOCUMENT_LEXICON_H_
#define SLING_NLP_DOCUMENT_LEXICON_H_

#include <string>
#include <vector>

#include "sling/base/types.h"
#include "sling/nlp/document/affix.h"
#include "sling/util/unicode.h"
#include "sling/util/vocabulary.h"

namespace sling {
namespace nlp {

// Word shape features for lexical backoff.
struct WordShape {
  // Hyphen feature.
  enum Hyphen {
    NO_HYPHEN = 0,
    HAS_HYPHEN = 1,
    HYPHEN_CARDINALITY = 2,
  };

  // Capitalization feature.
  enum Capitalization {
    LOWERCASE = 0,
    UPPERCASE = 1,
    CAPITALIZED = 2,
    INITIAL = 3,
    NON_ALPHABETIC = 4,
    CAPITALIZATION_CARDINALITY = 5,
  };

  // Punctuation feature.
  enum Punctuation {
    NO_PUNCTUATION = 0,
    SOME_PUNCTUATION = 1,
    ALL_PUNCTUATION = 2,
    PUNCTUATION_CARDINALITY = 3,
  };

  // Quote feature.
  enum Quote {
    NO_QUOTE = 0,
    OPEN_QUOTE = 1,
    CLOSE_QUOTE = 2,
    UNKNOWN_QUOTE = 3,
    QUOTE_CARDINALITY = 4,
  };

  // Digit feature.
  enum Digit {
    NO_DIGIT = 0,
    SOME_DIGIT = 1,
    ALL_DIGIT = 2,
    DIGIT_CARDINALITY = 3,
  };

  // Extract shape features from word.
  void Extract(const string &word);

  Hyphen hyphen = NO_HYPHEN;                  // hyphenation
  Capitalization capitalization = LOWERCASE;  // capitalization
  Punctuation punctuation = NO_PUNCTUATION;   // punctuation
  Quote quote = NO_QUOTE;                     // quotes
  Digit digit = NO_DIGIT;                     // digits
};

// Lexicon for extracting lexical features from documents.
class Lexicon {
 public:
  // Initialize lexicon from word vocabulary. The vocabulary words must already
  // be normalized.
  void InitWords(Vocabulary::Iterator *words);

  // Initialize affix tables from serialized format.
  void InitPrefixes(const char *data, size_t size);
  void InitSuffixes(const char *data, size_t size);

  // Write vocabulary to buffer.
  void WriteVocabulary(string *buffer, char terminator = 0) const;

  // Write affix tables into serialized format.
  void WritePrefixes(string *buffer) const;
  void WriteSuffixes(string *buffer) const;

  // Build affix tables from current dictionary.
  void BuildPrefixes(int max_prefix);
  void BuildSuffixes(int max_suffix);

  // Pre-compute word shape features for all words in the lexicon. The shape
  // features are computed on the normalized forms of the words.
  void PrecomputeShapes();

  // Look up word in vocabulary. If (normalized) word is found in the lexicon
  // the pre-computed affix and shape information from the lexicon is returned.
  // Otherwise, OOV is returned, and the affix and shape information is computed
  // on-the-fly.
  int Lookup(const string &word,
             Affix **prefix, Affix **suffix,
             WordShape *shape) const;

  // Look up word in lexicon. No normalization is performed.
  int Lookup(const string &word) const;

  // Return number of words in vocabulary.
  size_t size() const { return words_.size(); }

  // Get affix tables.
  const AffixTable &prefixes() const { return prefixes_; }
  const AffixTable &suffixes() const { return suffixes_; }

  // Out-of-vocabulary id.
  int oov() const { return oov_; }
  void set_oov(int oov) { oov_ = oov; }

  // Token normalization flags.
  Normalization normalization() const { return normalization_; }
  void set_normalization(Normalization normalization) {
    normalization_ = normalization;
  }

 private:
  // Lexicon entry.
  struct Entry {
    string word;                // normalized word form
    Affix *prefix = nullptr;    // longest known prefix for word
    Affix *suffix = nullptr;    // longest known suffix for word
    WordShape shape;            // word shape features for normalized word
  };

  // Mapping from words to ids.
  Vocabulary vocabulary_;
  Normalization normalization_ = NORMALIZE_NONE;
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

#endif  // SLING_NLP_DOCUMENT_LEXICON_H_

