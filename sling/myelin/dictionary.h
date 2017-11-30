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

#ifndef SLING_MYELIN_DICTIONARY_H_
#define SLING_MYELIN_DICTIONARY_H_

#include "sling/base/types.h"
#include "sling/myelin/flow.h"

namespace sling {
namespace myelin {

// Read-only dictionary for looking up lexicon ids for words. This uses a
// compact memory layout with a bucket array and an item array. This is more
// compact and faster that a traditional hash table like std::unordered_map.
// Only the 64-bit hash of the word is stored so there could in principle be
// collisions, although these would be rare.
// The dictionary is initialized from a Flow blob which contains a list of
// strings. Each string is terminated by a nul character, or any other character
// specified in the "delimiter" attribute of the blob.
class Dictionary {
 public:
  ~Dictionary();

  // Initialize dictionary from lexicon blob.
  void Init(Flow::Blob *lexicon);

  // Lookup word in dictionary. Returns OOV (default zero) if word is not found.
  int64 Lookup(const char *data, size_t size) const;
  int64 Lookup(const string &word) const {
    return Lookup(word.data(), word.size());
  }

 private:
  // Item in dictionary.
  struct Item {
    uint64 hash;
    int64 value;
  };

  typedef Item *Bucket;

  // Hash buckets for dictionary. There is one additional sentinel element in
  // the bucket array.
  Bucket *buckets_ = nullptr;

  // Dictionary items sorted in bucket order.
  Item *items_ = nullptr;

  // Number of buckets.
  int num_buckets_ = 0;

  // Number of elements in dictionary.
  int size_ = 0;

  // Out-of-vocabulary id returned if word is not in dictionary.
  int64 oov_ = -1;
};

}  // namespace myelin
}  // namespace sling

#endif  // SLING_MYELIN_DICTIONARY_H_

