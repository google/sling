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

Vocabulary::~Vocabulary() {
  delete [] buckets_;
  delete [] items_;
}

void Vocabulary::Init(const char *data, size_t size, char terminator) {
  // Count the number of items in the lexicon.
  size_ = 0;
  for (int i = 0; i < size; ++i) {
    if (data[i] == terminator) size_++;
  }
  num_buckets_ = size_;

  // Allocate items and buckets. We allocate one extra bucket to mark the end of
  // the items. This ensures that all items in a bucket b are in the range from
  // bucket[b] to bucket[b + 1], even for the last bucket.
  items_ = new Item[size_];
  buckets_ = new Bucket[num_buckets_ + 1];

  // Build item for each word in the lexicon.
  const char *current = data;
  const char *end = data + size;
  int64 index = 0;
  while (current < end) {
    // Find next word.
    const char *next = current;
    while (next < end && *next != terminator) next++;
    if (next == end) break;

    // Initialize item for word.
    items_[index].hash = Fingerprint(current, next - current);
    items_[index].value = index;

    current = next + 1;
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

int64 Vocabulary::Lookup(const char *word, size_t size) const {
  uint64 hash = Fingerprint(word, size);
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

