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

#ifndef SLING_UTIL_BLOOM_H_
#define SLING_UTIL_BLOOM_H_

#include <vector>

#include "sling/base/types.h"

namespace sling {

// Bloom filter for implementing a probabilistic set.
// To achieve a false-positive rate of p for a set of n elements, use the
// following parameters:
//   size = -n ln p / (ln 2)^2
//   hashes = size/n ln 2
// See https://en.wikipedia.org/wiki/Bloom_filter

class BloomFilter {
 public:
  // Hash mixer using a simple multiplicative congruential prng.
  struct Mixer {
    Mixer(uint64 seed, size_t size) : value(seed), size(size) {}

    // Get next hash value.
    uint64 operator ()() {
      value = value * 2862933555777941757 + 3037000493;
      return value % size;
    }

    uint64 value;
    size_t size;
  };

  // Initialize Bloom filter.
  BloomFilter(size_t size, int hashes) : bits_(size), hashes_(hashes) {}

  // Insert element in set.
  void insert(uint64 fp) {
    Mixer mixer(fp, bits_.size());
    for (int n = 0; n < bits_.size(); ++n) bits_[mixer()] = true;
  }

  // Add element to set and check if element was possibly already in the set.
  bool add(uint64 fp) {
    Mixer mixer(fp, bits_.size());
    bool member = true;
    for (int n = 0; n < hashes_; ++n) {
      uint64 h = mixer();
      member &= bits_[h];
      bits_[h] = true;
    }
    return member;
  }

  // Check if element is possibly in the set.
  bool contains(uint64 fp) {
    Mixer mixer(fp, bits_.size());
    for (int n = 0; n < hashes_; ++n) {
      if (!bits_[mixer()]) return false;
    }
    return true;
  }

 private:
  // Bit vector for Bloom filter.
  std::vector<bool> bits_;

  // Number of hash functions in filter.
  int hashes_;
};

}  // namespace sling

#endif  // SLING_UTIL_BLOOM_H_

