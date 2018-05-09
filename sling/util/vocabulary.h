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

#ifndef SLING_UTIL_VOCABULARY_H_
#define SLING_UTIL_VOCABULARY_H_

#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "sling/base/types.h"
#include "sling/string/text.h"

namespace sling {

// Read-only dictionary mapping words to ids. This uses a compact memory
// layout with a bucket array and an item array. This is more compact and faster
// that a traditional hash table like std::unordered_map. Only the 64-bit hash
// of the word is stored so there could in principle be collisions, although
// these would be rare.
class Vocabulary {
 public:
  // Iterator for word vocabulary.
  class Iterator {
   public:
    virtual ~Iterator() = default;

    // Returns the size of the word vocabulary.
    virtual int Size() = 0;

    // Reset iterator to beginning.
    virtual void Reset() = 0;

    // Get next word in vocabulary, returning false if there are no more words.
    virtual bool Next(Text *word, int *count) = 0;
  };

  // Vocabulary iterator for data buffer with words. Each word is terminated
  // by the terminator character.
  class BufferIterator : public Iterator {
   public:
    BufferIterator(const char *data, size_t size, char terminator = 0)
      : data_(data), end_(data + size), current_(data),
        terminator_(terminator) {}

    // Iterator interface.
    int Size() override;
    void Reset() override;
    bool Next(Text *word, int *count) override;

   private:
    const char *data_;       // begining of data buffer
    const char *end_;        // end of data buffer
    const char *current_;    // current position of iterator
    char terminator_;        // word terminator character
  };

  // Vocabulary iterator for word vector.
  class VectorIterator : public Iterator {
   public:
    VectorIterator(const std::vector<string> &words) : words_(words) {}

    // Iterator interface.
    int Size() override;
    void Reset() override;
    bool Next(Text *word, int *count) override;

   private:
    const std::vector<string> &words_;
    int current_ = 0;
  };

  // Vocabulary iterator for word map.
  template<class T> class MapIterator : public Iterator {
   public:
    MapIterator(const T &words) : words_(words), current_(words_.begin()) {}

    // Iterator interface.
    int Size() override { return words_.size(); }
    void Reset() override { current_ = words_.begin(); }
    bool Next(Text *word, int *count) override {
      if (current_ == words_.end()) return false;
      const string &str = current_->first;
      word->set(str.data(), str.size());
      if (count != nullptr) *count = current_->second;
      ++current_;
      return true;
    }

   private:
    const T &words_;
    typename T::const_iterator current_;
  };

  typedef MapIterator<std::unordered_map<string, int>> HashMapIterator;
  typedef MapIterator<std::vector<std::pair<string, int>>> VectorMapIterator;

  ~Vocabulary();

  // Initialize dictionary from list of words.
  void Init(Iterator *words);

  // Look up word in dictionary. Returns - 1 if word is not found.
  int64 Lookup(Text word) const;

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

#endif  // SLING_UTIL_VOCABULARY_H_

