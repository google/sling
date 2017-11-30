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

#ifndef SLING_NLP_DOCUMENT_FEATURES_H_
#define SLING_NLP_DOCUMENT_FEATURES_H_

#include <vector>

#include "sling/base/types.h"
#include "sling/nlp/document/document.h"
#include "sling/nlp/document/lexicon.h"

namespace sling {
namespace nlp {

// Extract lexical features from the tokens in a document.
class DocumentFeatures {
 public:
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
    DIGIT__CARDINALITY = 3,
  };

  // Initialize lexical feature extractor.
  DocumentFeatures(const Lexicon *lexicon) : lexicon_(lexicon) {}

  // Extract features from document.
  void Extract(const Document &document);

  // Get features for token.
  int word(int index) const {
    return features_[index].word;
  }
  Affix *prefix(int index) const {
    return features_[index].prefix;
  }
  Affix *suffix(int index) const {
    return features_[index].suffix;
  }
  Hyphen hyphen(int index) const {
    return features_[index].hyphen;
  }
  Capitalization capitalization(int index) const {
    return features_[index].capitalization;
  }
  Punctuation punctuation(int index) const {
    return features_[index].punctuation;
  }
  Quote quote(int index) const {
    return features_[index].quote;
  }
  Digit digit(int index) const {
    return features_[index].digit;
  }

 private:
  // Lexical features for token.
  struct TokenFeatures {
    int word;                                   // word id
    Affix *prefix = nullptr;                    // longest prefix
    Affix *suffix = nullptr;                    // longest suffix
    Hyphen hyphen = NO_HYPHEN;                  // hyphenation
    Capitalization capitalization = LOWERCASE;  // capitalization
    Punctuation punctuation = NO_PUNCTUATION;   // punctuation
    Quote quote = NO_QUOTE;                     // quotes
    Digit digit = NO_DIGIT;                     // digits
  };

  // Lexicon for looking up feature values.
  const Lexicon *lexicon_;

  // Features for tokens.
  std::vector<TokenFeatures> features_;
};

}  // namespace nlp
}  // namespace sling

#endif  // SLING_NLP_DOCUMENT_FEATURES_H_

