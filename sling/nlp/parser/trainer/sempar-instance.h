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

#ifndef SLING_NLP_PARSER_TRAINER_SEMPAR_SENTENCE_H_
#define SLING_NLP_PARSER_TRAINER_SEMPAR_SENTENCE_H_

#include "sling/frame/store.h"
#include "sling/nlp/document/document.h"

namespace sling {
namespace nlp {

// Container for holding a document and its local store.
struct SemparInstance {
  ~SemparInstance() {
    delete document;
    delete store;
  }

  // Serialization of the document. Cleared once the document is decoded.
  string encoded;

  // The document itself and its local store.
  Document *document = nullptr;
  Store *store = nullptr;
};

}  // namespace nlp
}  // namespace sling

#endif  // SLING_NLP_PARSER_TRAINER_SEMPAR_SENTENCE_H_
