// Copyright 2010-2014 Google
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

// This code comes from:
// https://code.google.com/p/or-tools/source/browse/trunk/src/base/fingerprint2011.h
// and was adapted to the needs of this project.

#include "sling/util/fingerprint.h"

#include "sling/base/types.h"

namespace sling {

uint64 FingerprintCat(uint64 fp1, uint64 fp2) {
  // Two big prime numbers.
  const uint64 mul1 = 0xC6A4A7935BD1E995u;
  const uint64 mul2 = 0x228876A7198B743u;

  const uint64 a = fp1 * mul1 + fp2 * mul2;

  // Note: The following line also makes sure we never return 0 or 1, because we
  // will only add something to 'a' if there are any MSBs (the remaining bits
  // after the shift) being 0, in which case wrapping around would not happen.
  return a + (~a >> 47);
}

// This should be better (collision-wise) than the default hash<string>,
// without being much slower. It never returns 0 or 1.
uint64 Fingerprint(const char *bytes, size_t len) {
  // Some big prime number.
  uint64 fp = 0xA5B85C5E198ED849u;
  const char *end = bytes + len;
  while (bytes + sizeof(uint64) <= end) {
    fp = FingerprintCat(fp, *(reinterpret_cast<const uint64 *>(bytes)));
    bytes += sizeof(uint64);
  }
  uint64 residual = 0;
  while (bytes < end) {
    residual = residual << 8 | *reinterpret_cast<const uint8 *>(bytes);
    bytes++;
  }

  return FingerprintCat(fp, residual);
}

uint32 Fingerprint32(const char *bytes, size_t len) {
  uint64 fp = Fingerprint(bytes, len);
  return fp ^(fp >> 32);
}

}  // namespace sling

