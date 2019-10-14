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

#ifndef SLING_NLP_DOCUMENT_DOCUMENT_CORPUS_H_
#define SLING_NLP_DOCUMENT_DOCUMENT_CORPUS_H_

#include <string>
#include <vector>

#include "sling/file/recordio.h"
#include "sling/frame/store.h"
#include "sling/nlp/document/document.h"

namespace sling {
namespace nlp {

// A document corpus is a set of record files with SLING-encoded documents.
class DocumentCorpus {
 public:
  // Initialize document corpus.
  DocumentCorpus(Store *commons, const string &filepattern);
  DocumentCorpus(Store *commons, const std::vector<string> &filenames);
  ~DocumentCorpus();

  // Read next document into store and return it or null of there are no
  // more document. The returned document is owned by the caller.
  Document *Next(Store *store);

  // Rewind to the start of the corpus.
  void Rewind();

 private:
  // Record files with documents.
  RecordDatabase corpus_;

  // Document schema.
  DocumentNames *docnames_;
};

}  // namespace nlp
}  // namespace sling

#endif  // SLING_NLP_DOCUMENT_DOCUMENT_CORPUS_H_
