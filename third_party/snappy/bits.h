// Copyright 2011 Google Inc. All Rights Reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
//     * Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//     * Redistributions in binary form must reproduce the above
// copyright notice, this list of conditions and the following disclaimer
// in the documentation and/or other materials provided with the
// distribution.
//     * Neither the name of Google Inc. nor the names of its
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//

#ifndef SNAPPY_BITS_H_
#define SNAPPY_BITS_H_

#include "sling/base/macros.h"
#include "sling/base/types.h"

namespace snappy {

#define HAVE_BUILTIN_CTZ

// Some bit-manipulation functions.
class Bits {
 public:
  // Return floor(log2(n)) for positive integer n.  Returns -1 iff n == 0.
  static int Log2Floor(uint32 n);

  // Return the first set least / most significant bit, 0-indexed.  Returns an
  // undefined value if n == 0.  FindLSBSetNonZero() is similar to ffs() except
  // that it's 0-indexed.
  static int FindLSBSetNonZero(uint32 n);
  static int FindLSBSetNonZero64(uint64 n);

 private:
  DISALLOW_COPY_AND_ASSIGN(Bits);
};

#ifdef HAVE_BUILTIN_CTZ

inline int Bits::Log2Floor(uint32 n) {
  return n == 0 ? -1 : 31 ^ __builtin_clz(n);
}

inline int Bits::FindLSBSetNonZero(uint32 n) {
  return __builtin_ctz(n);
}

inline int Bits::FindLSBSetNonZero64(uint64 n) {
  return __builtin_ctzll(n);
}

#else  // portable versions

inline int Bits::Log2Floor(uint32 n) {
  if (n == 0) return -1;
  int log = 0;
  uint32 value = n;
  for (int i = 4; i >= 0; --i) {
    int shift = (1 << i);
    uint32 x = value >> shift;
    if (x != 0) {
      value = x;
      log += shift;
    }
  }
  assert(value == 1);
  return log;
}

inline int Bits::FindLSBSetNonZero(uint32 n) {
  int rc = 31;
  for (int i = 4, shift = 1 << 4; i >= 0; --i) {
    const uint32 x = n << shift;
    if (x != 0) {
      n = x;
      rc -= shift;
    }
    shift >>= 1;
  }
  return rc;
}

// FindLSBSetNonZero64() is defined in terms of FindLSBSetNonZero().
inline int Bits::FindLSBSetNonZero64(uint64 n) {
  const uint32 bottombits = static_cast<uint32>(n);
  if (bottombits == 0) {
    // Bottom bits are zero, so scan in top bits
    return 32 + FindLSBSetNonZero(static_cast<uint32>(n >> 32));
  } else {
    return FindLSBSetNonZero(bottombits);
  }
}

#endif  // end portable versions

}  // namespace snappy

#endif  // SNAPPY_BITS_H_

