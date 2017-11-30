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

#include "sling/util/varint.h"

#include <string>

#include "sling/base/types.h"

namespace sling {

#ifndef COMPILER_MSVC
const int Varint::kMax32;
const int Varint::kMax64;
#endif

char *Varint::Encode32(char *sptr, uint32 v) {
  return Encode32Inline(sptr, v);
}

char *Varint::Encode64(char *sptr, uint64 v) {
  if (v < (1u << 28)) {
    return Varint::Encode32(sptr, v);
  } else {
    // Operate on characters as unsigneds.
    unsigned char *ptr = reinterpret_cast<unsigned char *>(sptr);
    static const int B = 128;
    uint32 v32 = v;
    *(ptr++) = v32 | B;
    *(ptr++) = (v32 >> 7) | B;
    *(ptr++) = (v32 >> 14) | B;
    *(ptr++) = (v32 >> 21) | B;
    if (v < (1ull << 35)) {
      *(ptr++) = (v >> 28);
      return reinterpret_cast<char *>(ptr);
    } else {
      *(ptr++) = (v >> 28) | B;
      return Varint::Encode32(reinterpret_cast<char *>(ptr), v >> 35);
    }
  }
}

const char *Varint::Parse32Fallback(const char *ptr, uint32 *output) {
  return Parse32FallbackInline(ptr, output);
}

const char *Varint::Parse64Fallback(const char *p, uint64 *output) {
  const unsigned char *ptr = reinterpret_cast<const unsigned char *>(p);
  // Fast path: need to accumulate data in upto three result fragments
  //    res1    bits 0..27
  //    res2    bits 28..55
  //    res3    bits 56..63

  uint32 byte, res1, res2 = 0, res3 = 0;
  byte = *(ptr++); res1 = byte & 127;
  byte = *(ptr++); res1 |= (byte & 127) <<  7; if (byte < 128) goto done1;
  byte = *(ptr++); res1 |= (byte & 127) << 14; if (byte < 128) goto done1;
  byte = *(ptr++); res1 |= (byte & 127) << 21; if (byte < 128) goto done1;

  byte = *(ptr++); res2 = byte & 127;          if (byte < 128) goto done2;
  byte = *(ptr++); res2 |= (byte & 127) <<  7; if (byte < 128) goto done2;
  byte = *(ptr++); res2 |= (byte & 127) << 14; if (byte < 128) goto done2;
  byte = *(ptr++); res2 |= (byte & 127) << 21; if (byte < 128) goto done2;

  byte = *(ptr++); res3 = byte & 127;          if (byte < 128) goto done3;
  byte = *(ptr++); res3 |= (byte & 127) <<  7; if (byte < 128) goto done3;

  return nullptr;       // value is too long to be a varint64

 done1:
  DCHECK(res2 == 0);
  DCHECK(res3 == 0);
  *output = res1;
  return reinterpret_cast<const char *>(ptr);

 done2:
  DCHECK(res3 == 0);
  *output = res1 | (uint64(res2) << 28);
  return reinterpret_cast<const char *>(ptr);

 done3:
  *output = res1 | (uint64(res2) << 28) | (uint64(res3) << 56);
  return reinterpret_cast<const char *>(ptr);
}

const char *Varint::Parse32BackwardSlow(const char *ptr, const char *base,
                                        uint32 *output) {
  // Since this method is rarely called, for simplicity, we just skip backward
  // and then parse forward.
  const char *prev = Skip32BackwardSlow(ptr, base);
  if (prev == nullptr) return nullptr;  // no value before 'ptr'
  Parse32(prev, output);
  return prev;
}

const char *Varint::Parse64BackwardSlow(const char *ptr, const char *base,
                                        uint64 *output) {
  // Since this method is rarely called, for simplicity, we just skip backward
  // and then parse forward.
  const char *prev = Skip64BackwardSlow(ptr, base);
  if (prev == nullptr) return nullptr;  // no value before 'ptr'
  Parse64(prev, output);
  return prev;
}

const char *Varint::Parse64WithLimit(const char *p,
                                     const char *l,
                                     uint64 *output) {
  if (p + kMax64 <= l) {
    return Parse64(p, output);
  } else {
    const unsigned char *ptr = reinterpret_cast<const unsigned char *>(p);
    const unsigned char *limit = reinterpret_cast<const unsigned char *>(l);
    uint64 b, result;
    if (ptr >= limit) return nullptr;
    b = *(ptr++); result = b & 127;          if (b < 128) goto done;
    if (ptr >= limit) return nullptr;
    b = *(ptr++); result |= (b & 127) <<  7; if (b < 128) goto done;
    if (ptr >= limit) return nullptr;
    b = *(ptr++); result |= (b & 127) << 14; if (b < 128) goto done;
    if (ptr >= limit) return nullptr;
    b = *(ptr++); result |= (b & 127) << 21; if (b < 128) goto done;
    if (ptr >= limit) return nullptr;
    b = *(ptr++); result |= (b & 127) << 28; if (b < 128) goto done;
    if (ptr >= limit) return nullptr;
    b = *(ptr++); result |= (b & 127) << 35; if (b < 128) goto done;
    if (ptr >= limit) return nullptr;
    b = *(ptr++); result |= (b & 127) << 42; if (b < 128) goto done;
    if (ptr >= limit) return nullptr;
    b = *(ptr++); result |= (b & 127) << 49; if (b < 128) goto done;
    if (ptr >= limit) return nullptr;
    b = *(ptr++); result |= (b & 127) << 56; if (b < 128) goto done;
    if (ptr >= limit) return nullptr;
    b = *(ptr++); result |= (b & 127) << 63; if (b < 2) goto done;
    return nullptr;       // value is too long to be a varint64
   done:
    *output = result;
    return reinterpret_cast<const char *>(ptr);
  }
}

const char *Varint::Skip32BackwardSlow(const char *p, const char *b) {
  const unsigned char *ptr = reinterpret_cast<const unsigned char *>(p);
  const unsigned char *base = reinterpret_cast<const unsigned char *>(b);
  DCHECK(ptr >= base);

  // If the initial pointer is at the base or if the previous byte is not
  // the last byte of a varint, we return null since there is nothing to skip.
  if (ptr == base) return nullptr;
  if (*(--ptr) > 127) return nullptr;

  for (int i = 0; i < 5; i++) {
    if (ptr == base) return reinterpret_cast<const char *>(ptr);
    if (*(--ptr) < 128) return reinterpret_cast<const char *>(ptr + 1);
  }

  return nullptr;  // value is too long to be a varint32
}

const char *Varint::Skip64BackwardSlow(const char *p, const char *b) {
  const unsigned char *ptr = reinterpret_cast<const unsigned char *>(p);
  const unsigned char *base = reinterpret_cast<const unsigned char *>(b);
  DCHECK(ptr >= base);

  // If the initial pointer is at the base or if the previous byte is not
  // the last byte of a varint, we return null since there is nothing to skip.
  if (ptr == base) return nullptr;
  if (*(--ptr) > 127) return nullptr;

  for (int i = 0; i < 10; i++) {
    if (ptr == base) return reinterpret_cast<const char *>(ptr);
    if (*(--ptr) < 128) return reinterpret_cast<const char *>(ptr + 1);
  }

  return nullptr;  // value is too long to be a varint64
}

void Varint::Append32Slow(string *s, uint32 value) {
  char buf[Varint::kMax32];
  const char *p = Varint::Encode32(buf, value);
  s->append(buf, p - buf);
}

void Varint::Append64Slow(string *s, uint64 value) {
  char buf[Varint::kMax64];
  const char *p = Varint::Encode64(buf, value);
  s->append(buf, p - buf);
}

void Varint::EncodeTwo32Values(string *s, uint32 a, uint32 b) {
  uint64 v = 0;
  int shift = 0;
  while ((a > 0) || (b > 0)) {
    uint8 one_byte = (a & 0xf) | ((b & 0xf) << 4);
    v |= ((static_cast<uint64>(one_byte)) << shift);
    shift += 8;
    a >>= 4;
    b >>= 4;
  }
  Append64(s, v);
}

const char *Varint::DecodeTwo32ValuesSlow(const char *ptr,
                                          uint32 *a, uint32 *b) {
  uint64 v = 0;
  const char *result = Varint::Parse64(ptr, &v);
  *a = 0;
  *b = 0;
  int shift = 0;
  while (v > 0) {
    *a |= ((v & 0xf) << shift);
    *b |= (((v >> 4) & 0xf) << shift);
    v >>= 8;
    shift += 4;
  }
  return result;
}

inline int FastLength32(uint32 v) {
  if (v < (1u << 14)) {
    if (v < (1u << 7)) {
      return 1;
    } else {
      return 2;
    }
  } else {
    if (v < (1u << 28)) {
      if (v < (1u << 21)) {
        return 3;
      } else {
        return 4;
      }
    } else {
      return 5;
    }
  }
}

const char Varint::length32_bytes_required[33] = {
  1,  // entry for "-1", which happens when the value is 0
  1,1,1,1,1,1,1,
  2,2,2,2,2,2,2,
  3,3,3,3,3,3,3,
  4,4,4,4,4,4,4,
  5,5,5,5
};

int Varint::Length32NonInline(uint32 v) {
  return FastLength32(v);
}

int Varint::Length64(uint64 v) {
  uint32 tmp;
  int nb;       // number of bytes we've determined from our tests
  if (v < (1u << 28)) {
    tmp = v;
    nb = 0;
  } else if (v < (1ull << 35)) {
    return 5;
  } else {
    tmp = v >> 35;
    nb = 5;
  }
  return nb + Varint::Length32(tmp);
}

}  // namespace sling

