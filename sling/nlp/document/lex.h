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

// LEX is a light-weight frame annotation format for text.

#ifndef SLING_NLP_DOCUMENT_LEX_H_
#define SLING_NLP_DOCUMENT_LEX_H_

#include "sling/base/types.h"
#include "sling/nlp/document/document.h"
#include "sling/nlp/document/document-tokenizer.h"

namespace sling {
namespace nlp {

class DocumentLexer {
 public:
  // Initialize document lexer.
  DocumentLexer(const DocumentTokenizer *tokenizer) : tokenizer_(tokenizer) {}

  // parse text in LEX format and add text and annotations to document.
  bool Lex(Document *document, Text lex) const;

 private:
  // Markable span in LEX-encoded text.
  struct Markable {
    Markable(int pos) : begin(pos) {}
    // Range of bytes in plain text covering the span.
    int begin;
    int end = -1;

    // Annotation object number.
    int object = -1;
  };

  // Document tokenizer.
  const DocumentTokenizer *tokenizer_;
};

// Convert document to LEX format.
string ToLex(const Document &document);

}  // namespace nlp
}  // namespace sling

#endif  // SLING_NLP_DOCUMENT_LEX_H_

