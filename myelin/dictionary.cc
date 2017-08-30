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

#include "myelin/dictionary.h"

namespace sling {
namespace myelin {

static const uint64_t mul1 = 0xC6A4A7935BD1E995u;
static const uint64_t mul2 = 0x0228876A7198B743u;
static const uint64_t seed = 0xA5B85C5E198ED849u;

// Mix two fingerprints.
static uint64 Mix(uint64 fp1, uint64 fp2) {
  const uint64 a = fp1 * mul1 + fp2 * mul2;
  return a + (~a >> 47);
}

// Compute hash of string.
static uint64 Hash(const char *bytes, size_t len) {
  uint64 fp = seed;
  const char *end = bytes + len;
  while (bytes + sizeof(uint64) <= end) {
    fp = Mix(fp, *(reinterpret_cast<const uint64 *>(bytes)));
    bytes += sizeof(uint64);
  }
  uint64 residual = 0;
  while (bytes < end) {
    residual <<= 8;
    residual |= *bytes;
    bytes++;
  }

  return Mix(fp, residual);
}

Dictionary::~Dictionary() {
  delete [] buckets_;
  delete [] items_;
}

void Dictionary::Init(Flow::Blob *lexicon) {
  // Count the number of items in the lexicon.
  oov_ = lexicon->attrs.Get("oov", -1);
  char delimiter = lexicon->attrs.Get("delimiter", 0);
  size_ = 0;
  for (int i = 0; i < lexicon->size; ++i) {
    if (lexicon->data[i] == delimiter) size_++;
  }
  num_buckets_ = size_;

  // Allocate items and buckets. We allocate one extra bucket to mark the end of
  // the items. This ensures that all items in a bucket b are in the range from
  // bucket[b] to bucket[b + 1], even for the last bucket.
  items_ = new Item[size_];
  buckets_ = new Bucket[num_buckets_ + 1];

  // Build item for each word in the lexicon.
  const char *data = lexicon->data;
  const char *end = data + lexicon->size;
  int64 index = 0;
  while (data < end) {
    // Find next word.
    const char *next = data;
    while (next < end && *next != delimiter) next++;
    if (next == end) break;

    // Initialize item for word.
    items_[index].hash = Hash(data, next - data);
    items_[index].value = index;

    data = next + 1;
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

int64 Dictionary::Lookup(const char *word, size_t size) const {
  uint64 hash = Hash(word, size);
  int b = hash % num_buckets_;
  Item *item = buckets_[b];
  Item *end = buckets_[b + 1];
  while (item < end) {
    if (hash == item->hash) return item->value;
    item++;
  }
  return oov_;
}

}  // namespace myelin
}  // namespace sling

