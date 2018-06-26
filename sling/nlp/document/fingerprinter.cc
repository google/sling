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

#include "sling/nlp/document/fingerprinter.h"

#include <string>

#include "sling/util/unicode.h"

namespace sling {
namespace nlp {

uint64 Fingerprinter::Fingerprint(Text word, Normalization normalization) {
  // Normalize string.
  string normalized;
  UTF8::Normalize(word.data(), word.size(), normalization, &normalized);

  // Ignore degenerate words.
  if (normalized.empty()) return 1;

  // Return fingerprint for normalized word.
  return Hash(normalized);
}

uint64 Fingerprinter::Fingerprint(Text word, uint64 seed,
                                  Normalization normalization) {
  uint64 fp = Fingerprint(word, normalization);
  return fp == 1 ? seed : Mix(fp, seed);
}

uint64 Fingerprinter::Fingerprint(const std::vector<Text> &words,
                                  Normalization normalization) {
  uint64 fp = 1;
  for (const Text &word : words) {
    uint64 word_fp = Fingerprint(word, normalization);
    if (word_fp == 1) continue;
    fp = Mix(word_fp, fp);
  }
  return fp;
}

}  // namespace nlp
}  // namespace sling

