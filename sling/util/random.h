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

#ifndef SLING_UTIL_RANDOM_H_
#define SLING_UTIL_RANDOM_H_

#include <random>

namespace sling {

// Random number generator.
class Random {
 public:
  // Initialize random number generator.
  Random() : dist_(0.0, 1.0) {}

  // Set seed for random number generator.
  void seed(int seed) { prng_.seed(seed); }

  // Return random number between 0.0 (inclusive) and 1.0 (exclusive).
  float UniformProb() {
    return dist_(prng_);
  }

  // Return uniformly distributed random number r=p*scale+bias, 0<=p<1.
  float UniformFloat(float scale, float bias) {
    return dist_(prng_) * scale + bias;
  }

  // Return uniformly distributed random number between 0 and n (exclusive).
  int UniformInt(int n) {
    return prng_() % n;
  }

 private:
  // Mersenne Twister pseudo-random generator of 64-bit numbers.
  std::mt19937_64 prng_;

  // Uniform distribution.
  std::uniform_real_distribution<float> dist_;
};

}  // namespace sling

#endif  // SLING_UTIL_RANDOM_H_

