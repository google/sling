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

#ifndef SLING_NLP_DOCUMENT_DOCUMENT_SOURCE_H_
#define SLING_NLP_DOCUMENT_DOCUMENT_SOURCE_H_

#include <string>

#include "sling/frame/store.h"
#include "sling/nlp/document/document.h"

namespace sling {
namespace nlp {

// Interface for iterating over a corpus of documents.
// At each iteration, it can return either a decoded Document
// or the encoded document string.
class DocumentSource {
 public:
  virtual ~DocumentSource() {}

  // Outputs the next docid and encoded document in 'name' and 'serialized'
  // respectively. Returns false if at the end of the corpus, else true.
  virtual bool NextSerialized(string *name, string *serialized) = 0;

  // Returns the next document, which is decoded using 'store'.
  // Returns nullptr if at the end of the corpus.
  // 'store' should outlive the returned document, which is owned by the caller.
  virtual Document *Next(Store *store);

  // Same as above, except also returns the name of the document.
  virtual Document *Next(Store *store, string *name);

  // Rewinds to the start of the corpus.
  virtual void Rewind() = 0;

  // Returns an iterator implementation depending on 'file_pattern'.
  static DocumentSource *Create(const string &file_pattern);
};

}  // namespace nlp
}  // namespace sling

#endif  // SLING_NLP_DOCUMENT_DOCUMENT_SOURCE_H_
