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

#include "sling/nlp/document/document-corpus.h"

#include "sling/frame/serialization.h"

namespace sling {
namespace nlp {

DocumentCorpus::DocumentCorpus(Store *commons, const string &filepattern)
    : corpus_(filepattern, RecordFileOptions()) {
  docnames_ = commons->frozen() ? nullptr : new DocumentNames(commons);
}

DocumentCorpus::DocumentCorpus(Store *commons,
                               const std::vector<string> &filenames)
    : corpus_(filenames, RecordFileOptions()) {
  docnames_ = commons->frozen() ? nullptr : new DocumentNames(commons);
}

DocumentCorpus::~DocumentCorpus() {
  if (docnames_ != nullptr) docnames_->Release();
}

Document *DocumentCorpus::Next(Store *store) {
  // Return null if there are no more document.
  if (corpus_.Done()) return nullptr;

  // Read next record.
  Record record;
  CHECK(corpus_.Next(&record));

  // Decode document frame.
  ArrayInputStream stream(record.value.data(), record.value.size());
  InputParser parser(store, &stream);
  Frame frame = parser.Read().AsFrame();
  CHECK(frame.valid());

  // Return new document.
  return new Document(frame, docnames_);
}

void DocumentCorpus::Rewind() {
  CHECK(corpus_.Rewind());
}

}  // namespace nlp
}  // namespace sling

