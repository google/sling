// Copyright 2013 Google Inc. All Rights Reserved.
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

// These routines provide mem versions of standard C string routines,
// such a strpbrk.  They function exactly the same as the str version,
// so if you wonder what they are, replace the word "mem" by
// "str" and check out the man page.  I could return void *, as the
// strutil.h mem*() routines tend to do, but I return char * instead
// since this is by far the most common way these functions are called.
//
// The difference between the mem and str versions is the mem version
// takes a pointer and a length, rather than a nul-terminated string.
// The memcase* routines defined here assume the locale is "C"
// (they use ascii_tolower instead of tolower).
//
// These routines are based on the BSD library.

#ifndef STRING_MEMUTIL_H_
#define STRING_MEMUTIL_H_

#include <string.h>

#include "base/port.h"

namespace sling {

inline char *memcat(char *dest, size_t destlen,
                    const char *src, size_t srclen) {
  return reinterpret_cast<char*>(memcpy(dest + destlen, src, srclen));
}

int memcasecmp(const char *s1, const char *s2, size_t len);
char *memdup(const char *s, size_t slen);
char *memrchr(const char *s, int c, size_t slen);
size_t memspn(const char *s, size_t slen, const char *accept);
size_t memcspn(const char *s, size_t slen, const char *reject);
char *mempbrk(const char *s, size_t slen, const char *accept);

// For internal use only.  Don't call this directly
template<bool case_sensitive>
const char *int_memmatch(const char *phaystack, size_t haylen,
                         const char *pneedle, size_t neelen);

inline const char *memstr(const char *phaystack, size_t haylen,
                          const char *pneedle) {
  return int_memmatch<true>(phaystack, haylen, pneedle, strlen(pneedle));
}

inline const char *memcasestr(const char *phaystack, size_t haylen,
                              const char *pneedle) {
  return int_memmatch<false>(phaystack, haylen, pneedle, strlen(pneedle));
}

inline const char *memmem(const char *phaystack, size_t haylen,
                          const char *pneedle, size_t needlelen) {
  return int_memmatch<true>(phaystack, haylen, pneedle, needlelen);
}

inline const char *memcasemem(const char *phaystack, size_t haylen,
                              const char *pneedle, size_t needlelen) {
  return int_memmatch<false>(phaystack, haylen, pneedle, needlelen);
}

const char *memmatch(const char *phaystack, size_t haylen,
                     const char *pneedle, size_t neelen);

// The ""'s catch people who don't pass in a literal for "str"
#define strliterallen(str) (sizeof("" str "")-1)

// Must use a string literal for prefix.
#define memprefix(str, len, prefix)                         \
  ( (((len) >= strliterallen(prefix))                       \
     && memcmp(str, prefix, strliterallen(prefix)) == 0)    \
    ? str + strliterallen(prefix)                           \
    : nullptr )

#define memcaseprefix(str, len, prefix)                             \
  ( (((len) >= strliterallen(prefix))                               \
     && memcasecmp(str, prefix, strliterallen(prefix)) == 0)        \
    ? str + strliterallen(prefix)                                   \
    : nullptr )

// Must use a string literal for suffix.
#define memsuffix(str, len, suffix)                         \
  ( (((len) >= strliterallen(suffix))                       \
     && memcmp(str + (len) - strliterallen(suffix), suffix, \
               strliterallen(suffix)) == 0)                 \
    ? str + (len) - strliterallen(suffix)                   \
    : nullptr )

#define memcasesuffix(str, len, suffix)                             \
  ( (((len) >= strliterallen(suffix))                               \
     && memcasecmp(str + (len) - strliterallen(suffix), suffix,     \
               strliterallen(suffix)) == 0)                         \
    ? str + (len) - strliterallen(suffix)                           \
    : nullptr )

#define memis(str, len, literal)                               \
  ( (((len) == strliterallen(literal))                         \
     && memcmp(str, literal, strliterallen(literal)) == 0) )

#define memcaseis(str, len, literal)                           \
  ( (((len) == strliterallen(literal))                         \
     && memcasecmp(str, literal, strliterallen(literal)) == 0) )


inline int memcount(const char *buf, size_t len, char c) {
  int num = 0;
  for (int i = 0; i < len; i++) {
    if (buf[i] == c) num++;
  }
  return num;
}

}  // namespace sling

#endif  // STRING_MEMUTIL_H_

