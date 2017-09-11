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

#ifndef UTIL_VOCABULARY_H_
#define UTIL_VOCABULARY_H_

#include <string>

#include "base/types.h"

namespace sling {

// Read-only dictionary mapping words to ids. This uses a compact memory
// layout with a bucket array and an item array. This is more compact and faster
// that a traditional hash table like std::unordered_map. Only the 64-bit hash
// of the word is stored so there could in principle be collisions, although
// these would be rare.
class Vocabulary {
 public:
  ~Vocabulary();

  // Initialize dictionary from list of words. The words in the data buffer
  // are terminated by the terminator character.
  void Init(const char *data, size_t size, char terminator = 0);

  // Lookup word in dictionary. Returns - 1 if word is not found.
  int64 Lookup(const char *data, size_t size) const;
  int64 Lookup(const string &word) const {
    return Lookup(word.data(), word.size());
  }

  // Return the vocabulary size.
  int size() const { return size_; }

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
};

}  // namespace sling

#endif  // UTIL_VOCABULARY_H_

