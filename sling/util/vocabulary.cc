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

#include <stddef.h>
#include <algorithm>

#include "sling/util/fingerprint.h"
#include "sling/util/vocabulary.h"

namespace sling {

int Vocabulary::BufferIterator::Size() {
  int n = 0;
  for (const char *p = data_; p < end_; p++) {
    if (*p == terminator_) n++;
  }
  return n;
}

void Vocabulary::BufferIterator::Reset() {
  current_ = data_;
}

bool Vocabulary::BufferIterator::Next(Text *word, int *count) {
  const char *next = current_;
  while (next < end_ && *next != terminator_) next++;
  if (next == end_) return false;
  word->set(current_, next - current_);
  if (count != nullptr) *count = 1;
  current_ = next + 1;
  return true;
}

int Vocabulary::VectorIterator::Size() {
  return words_.size();
}

void Vocabulary::VectorIterator::Reset() {
  current_ = 0;
}

bool Vocabulary::VectorIterator::Next(Text *word, int *count) {
  if (current_ == words_.size()) return false;
  const string &str = words_[current_++];
  word->set(str.data(), str.size());
  if (count != nullptr) *count = 1;
  return true;
}

Vocabulary::~Vocabulary() {
  delete [] buckets_;
  delete [] items_;
}

void Vocabulary::Init(Iterator *words) {
  // Get number of items in the lexicon.
  size_ = words->Size();
  num_buckets_ = size_;

  // Allocate items and buckets. We allocate one extra bucket to mark the end of
  // the items. This ensures that all items in a bucket b are in the range from
  // bucket[b] to bucket[b + 1], even for the last bucket.
  items_ = new Item[size_];
  buckets_ = new Bucket[num_buckets_ + 1];

  // Build item for each word in the lexicon.
  words->Reset();
  Text word;
  int64 index = 0;
  while (words->Next(&word, nullptr)) {
    // Initialize item for word.
    items_[index].hash = Fingerprint(word.data(), word.size());
    items_[index].value = index;
    index++;
  }

  // Sort the items in bucket order.
  int modulo = num_buckets_;
  std::sort(items_, items_ + size_,
    [modulo](const Item &a, const Item &b) {
      return (a.hash % modulo) < (b.hash % modulo);
    }
  );

  // Build bucket array.
  int bucket = -1;
  for (int i = 0; i < size_; ++i) {
    int b = items_[i].hash % modulo;
    while (bucket < b) buckets_[++bucket] = &items_[i];
  }
  while (bucket < num_buckets_) buckets_[++bucket] = &items_[size_];
};

int64 Vocabulary::Lookup(Text word) const {
  uint64 hash = Fingerprint(word.data(), word.size());
  int b = hash % num_buckets_;
  Item *item = buckets_[b];
  Item *end = buckets_[b + 1];
  while (item < end) {
    if (hash == item->hash) return item->value;
    item++;
  }
  return -1;
}

}  // namespace sling

