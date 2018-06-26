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
  // Initialize lexical feature extractor.
  DocumentFeatures(const Lexicon *lexicon) : lexicon_(lexicon) {}

  // Extract features from document.
  void Extract(const Document &document, int begin = 0, int end = -1);

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
  WordShape::Hyphen hyphen(int index) const {
    return features_[index].shape.hyphen;
  }
  WordShape::Capitalization capitalization(int index) const {
    return features_[index].shape.capitalization;
  }
  WordShape::Punctuation punctuation(int index) const {
    return features_[index].shape.punctuation;
  }
  WordShape::Quote quote(int index) const {
    return features_[index].shape.quote;
  }
  WordShape::Digit digit(int index) const {
    return features_[index].shape.digit;
  }

 private:
  // Lexical features for token.
  struct TokenFeatures {
    int word;                 // word id
    Affix *prefix = nullptr;  // longest prefix
    Affix *suffix = nullptr;  // longest suffix
    WordShape shape;          // word shape features
  };

  // Lexicon for looking up feature values.
  const Lexicon *lexicon_;

  // Features for tokens.
  std::vector<TokenFeatures> features_;
};

}  // namespace nlp
}  // namespace sling

#endif  // SLING_NLP_DOCUMENT_FEATURES_H_

