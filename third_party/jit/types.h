// Copyright 2014, the V8 project authors. All rights reserved.
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//     * Redistributions in binary form must reproduce the above
//       copyright notice, this list of conditions and the following
//       disclaimer in the documentation and/or other materials provided
//       with the distribution.
//     * Neither the name of Google Inc. nor the names of its
//       contributors may be used to endorse or promote products derived
//       from this software without specific prior written permission.
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

#ifndef JIT_TYPES_H_
#define JIT_TYPES_H_

#include <stdint.h>

#include "sling/base/logging.h"

namespace sling {
namespace jit {

// Data type sizes.
const int kIntSize = sizeof(int);
const int kInt32Size = sizeof(int32_t);
const int kInt64Size = sizeof(int64_t);
const int kPointerSize = sizeof(void *);

const int kBitsPerByte = 8;
const int kBitsPerInt = kIntSize * kBitsPerByte;

// Check number width.
inline bool is_intn(int64_t x, unsigned n) {
  DCHECK((0 < n) && (n < 64));
  int64_t limit = static_cast<int64_t>(1) << (n - 1);
  return (-limit <= x) && (x < limit);
}

inline bool is_uintn(int64_t x, unsigned n) {
  DCHECK((0 < n) && (n < (sizeof(x) * kBitsPerByte)));
  return !(x >> n);
}

inline bool is_int1(int64_t x) { return is_intn(x, 1); }
inline bool is_int2(int64_t x) { return is_intn(x, 2); }
inline bool is_int3(int64_t x) { return is_intn(x, 3); }
inline bool is_int4(int64_t x) { return is_intn(x, 4); }
inline bool is_int5(int64_t x) { return is_intn(x, 5); }
inline bool is_int6(int64_t x) { return is_intn(x, 6); }
inline bool is_int7(int64_t x) { return is_intn(x, 7); }
inline bool is_int8(int64_t x) { return is_intn(x, 8); }
inline bool is_int16(int64_t x) { return is_intn(x, 16); }
inline bool is_int32(int64_t x) { return is_intn(x, 32); }

template <class T> inline bool is_uint1(T x) { return is_uintn(x, 1); }
template <class T> inline bool is_uint2(T x) { return is_uintn(x, 2); }
template <class T> inline bool is_uint3(T x) { return is_uintn(x, 3); }
template <class T> inline bool is_uint4(T x) { return is_uintn(x, 4); }
template <class T> inline bool is_uint5(T x) { return is_uintn(x, 5); }
template <class T> inline bool is_uint6(T x) { return is_uintn(x, 6); }
template <class T> inline bool is_uint7(T x) { return is_uintn(x, 7); }
template <class T> inline bool is_uint8(T x) { return is_uintn(x, 8); }
template <class T> inline bool is_uint16(T x) { return is_uintn(x, 16); }
template <class T> inline bool is_uint32(T x) { return is_uintn(x, 32); }

}  // namespace jit
}  // namespace sling

#endif  // JIT_TYPES_H_
