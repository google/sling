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

#ifndef SLING_NLP_DOCUMENT_PHRASE_TOKENIZER_H_
#define SLING_NLP_DOCUMENT_PHRASE_TOKENIZER_H_

#include "sling/base/types.h"
#include "sling/nlp/document/text-tokenizer.h"
#include "sling/nlp/document/fingerprinter.h"
#include "sling/string/text.h"
#include "sling/util/unicode.h"

namespace sling {
namespace nlp {

class PhraseTokenizer {
 public:
  PhraseTokenizer();

  // Tokenize phrase into tokens.
  void Tokenize(Text text, std::vector<string> *tokens) const;

  // Tokenize phrase and return token fingerprints for each token.
  uint64 TokenFingerprints(Text text, std::vector<uint64> *tokens) const;

  // Compute fingerprint for phrase.
  uint64 Fingerprint(Text text) const;

  // Compute fingerprint and case form for phrase.
  void FingerprintAndForm(Text text, uint64 *fingerprint, CaseForm *form) const;

  // Set/get phrase normalization flags.
  Normalization normalization() const { return normalization_; }
  void set_normalization(Normalization normalization) {
    normalization_ = normalization;
  }

 private:
  // Phrase text normalization.
  Normalization normalization_ = NORMALIZE_DEFAULT;

  // Text tokenizer.
  Tokenizer tokenizer_;
};

}  // namespace nlp
}  // namespace sling

#endif  // SLING_NLP_DOCUMENT_PHRASE_TOKENIZER_H_

