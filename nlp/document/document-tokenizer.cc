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

#include "nlp/document/document-tokenizer.h"

#include "base/types.h"
#include "nlp/document/document.h"
#include "nlp/document/text-tokenizer.h"
#include "string/text.h"

namespace sling {
namespace nlp {

DocumentTokenizer::DocumentTokenizer() {
  // Initialize tokenizer.
  tokenizer_.InitLDC();
}

void DocumentTokenizer::Tokenize(Document *document, Text text) const {
  document->SetText(text);
  Tokenize(document);
}

void DocumentTokenizer::Tokenize(Document *document) const {
  string text = document->GetText();
  tokenizer_.Tokenize(text,
    [document](const Tokenizer::Token &t) {
      document->AddToken(t.begin, t.end, t.text, t.brk);
    }
  );
}

}  // namespace nlp
}  // namespace sling

