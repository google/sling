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

#ifndef SLING_NLP_NER_IDF_H_
#define SLING_NLP_NER_IDF_H_

#include "sling/base/port.h"
#include "sling/file/repository.h"
#include "sling/util/unicode.h"

namespace sling {
namespace nlp {

// Word vocabulary table for inverse document frequency (IDF).
class IDFTable {
 public:
  // Load IDF repository from file.
  void Load(const string &filename);

  // Look up word fingerprint and return IDF for word.
  float GetIDF(uint64 fingerprint) const;

  // Get text normalization flags for IDF table.
  Normalization normalization() const { return normalization_; }

  // IDF repository header information.
  static const int VERSION = 1;
  struct Header {
    int version;
    float num_docs;
    char normalization[16];
  };

 private:
  // Word entry.
  struct Word {
    uint64 fingerprint;
    float idf;
  } ABSL_ATTRIBUTE_PACKED;

  // Word index in repository.
  class WordIndex : public RepositoryMap<Word> {
   public:
    // Initialize word index.
    void Initialize(const Repository &repository) { Init(repository, "IDF"); }

    // Return first element in bucket.
    const Word *GetBucket(int bucket) const { return GetObject(bucket); }
  };

  // Find word in word index.
  const Word *Find(uint64 fp) const;

  // Repository with name table.
  Repository repository_;

  // IDF header information.
  const Header *header_ = nullptr;

  // Word index.
  WordIndex index_;

  // IDF for out-of-vocabulary words.
  float oov_idf_ = 0.0;

  // Text normalization for fingerprints.
  Normalization normalization_ = NORMALIZE_DEFAULT;
};

}  // namespace nlp
}  // namespace sling

#endif  // SLING_NLP_NER_IDF_H_
