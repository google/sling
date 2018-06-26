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

#ifndef SLING_NLP_DOCUMENT_FINGERPRINTER_H_
#define SLING_NLP_DOCUMENT_FINGERPRINTER_H_

#include <string>
#include <vector>

#include "sling/base/types.h"
#include "sling/string/text.h"
#include "sling/util/fingerprint.h"
#include "sling/util/unicode.h"

namespace sling {
namespace nlp {

class Fingerprinter {
 public:
  // Compute simple fingerprint hash with no normalization. This will never
  // return zero.
  static uint64 Hash(const char *s, size_t len) {
    return sling::Fingerprint(s, len);
  }

  static uint64 Hash(Text t) {
    return sling::Fingerprint(t.data(), t.size());
  }

  static uint64 Hash(const string &s) {
    return sling::Fingerprint(s.data(), s.size());
  }

  static uint64 Hash(int n) {
    return Hash(reinterpret_cast<const char *>(&n), sizeof(n));
  }

  // Simple fingerprint hash with seed.
  static uint64 Hash(const char *s, size_t len, uint64 seed) {
    return Mix(Hash(s, len), seed);
  }

  static uint64 Hash(Text t, uint64 seed) {
    return Mix(Hash(t), seed);
  }

  static uint64 Hash(const string &s, uint64 seed) {
    return Mix(Hash(s.data(), s.size()), seed);
  }

  static uint64 Hash(int n, uint64 seed) {
    return Mix(Hash(n), seed);
  }

  // Combine two fingerprints into one. If seed is one this will just return
  // fp, otherwise the seed will be mixed will the new fingerprint. This will
  // never return zero unless both seed and fp are zero.
  static uint64 Mix(uint64 fp, uint64 seed) {
    return seed != 1 ? sling::FingerprintCat(fp, seed) : fp;
  }

  // Return the fingerprint for a normalized version of the given string.
  // Never returns zero. Returns one if the string should be ignored.
  static uint64 Fingerprint(Text word,
                            Normalization normalization = NORMALIZE_DEFAULT);

  // Return the fingerprint for a normalized version of the given string,
  // using a given seed. Never returns zero. Returns the seed if the string
  // should be ignored.
  static uint64 Fingerprint(Text word, uint64 seed,
                            Normalization normalization = NORMALIZE_DEFAULT);

  // Return the fingerprint for the given vector of strings by combining
  // the fingerprints of each string's normalized version.
  static uint64 Fingerprint(const std::vector<Text> &words,
                            Normalization normalization = NORMALIZE_DEFAULT);
};

}  // namespace nlp
}  // namespace sling

#endif  // SLING_NLP_DOCUMENT_FINGERPRINTER_H_

