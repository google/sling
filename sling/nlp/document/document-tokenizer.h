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

#ifndef SLING_NLP_DOCUMENT_DOCUMENT_TOKENIZER_H_
#define SLING_NLP_DOCUMENT_DOCUMENT_TOKENIZER_H_

#include "sling/nlp/document/document.h"
#include "sling/nlp/document/text-tokenizer.h"
#include "sling/string/text.h"

namespace sling {
namespace nlp {

class DocumentTokenizer {
 public:
  DocumentTokenizer();

  // Add tokenized text to document
  void Tokenize(Document *document, Text text) const;

  // Tokenize text in document.
  void Tokenize(Document *document) const;

 private:
  // Text tokenizer.
  Tokenizer tokenizer_;
};

}  // namespace nlp
}  // namespace sling

#endif  // SLING_NLP_DOCUMENT_DOCUMENT_TOKENIZER_H_

