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

#ifndef SLING_BASE_PORT_H_
#define SLING_BASE_PORT_H_

#include <limits.h>
#include <string.h>
#include <stdlib.h>

#if defined(__APPLE__)
#include <unistd.h>
#elif defined(OS_CYGWIN)
#include <malloc.h>
#endif

#include "sling/base/types.h"

namespace sling {

#if defined(__APPLE__)
#ifndef __STDC_FORMAT_MACROS
#define __STDC_FORMAT_MACROS
#endif  // __STDC_FORMAT_MACROS
#endif  // __APPLE__

#if defined(OS_LINUX) || defined(OS_CYGWIN)

// _BIG_ENDIAN
#include <endian.h>

// mysql.h sets _GNU_SOURCE which sets __USE_MISC in <features.h>
// sys/types.h typedefs uint if __USE_MISC
// mysql typedefs uint if HAVE_UINT not set
// The following typedef is carefully considered, and should not cause
// any clashes.
#if !defined(__USE_MISC)
#if !defined(HAVE_UINT)
#define HAVE_UINT 1
typedef unsigned int uint;
#endif
#if !defined(HAVE_USHORT)
#define HAVE_USHORT 1
typedef unsigned short ushort;
#endif
#if !defined(HAVE_ULONG)
#define HAVE_ULONG 1
typedef unsigned long ulong;
#endif
#endif

#if defined(__cplusplus)
#include <cstddef>
#endif

#elif defined(OS_FREEBSD)

// _BIG_ENDIAN
#include <machine/endian.h>

#elif defined(__APPLE__)

// BIG_ENDIAN
#include <machine/endian.h>

#define __BYTE_ORDER  BYTE_ORDER
#define __LITTLE_ENDIAN LITTLE_ENDIAN
#define __BIG_ENDIAN BIG_ENDIAN

#endif

// The following  guarantee declaration of the byte swap functions, and
// define __BYTE_ORDER for MSVC.
#ifdef _MSC_VER

#include <stdlib.h>
#define __BYTE_ORDER __LITTLE_ENDIAN
#define bswap_16(x) _byteswap_ushort(x)
#define bswap_32(x) _byteswap_ulong(x)
#define bswap_64(x) _byteswap_uint64(x)

#elif defined(__APPLE__)

// Mac OS X / Darwin features.
#include <libkern/OSByteOrder.h>
#define bswap_16(x) OSSwapInt16(x)
#define bswap_32(x) OSSwapInt32(x)
#define bswap_64(x) OSSwapInt64(x)

#elif defined(__GLIBC__)

#include <byteswap.h>

#else

static inline uint16 bswap_16(uint16 x) {
  return ((x & 0xFF) << 8) | ((x & 0xFF00) >> 8);
}

#define bswap_16(x) bswap_16(x)

static inline uint32 bswap_32(uint32 x) {
  return (((x & 0xFF) << 24) |
          ((x & 0xFF00) << 8) |
          ((x & 0xFF0000) >> 8) |
          ((x & 0xFF000000) >> 24));
}

#define bswap_32(x) bswap_32(x)

static inline uint64 bswap_64(uint64 x) {
  return (((x & GG_ULONGLONG(0xFF)) << 56) |
          ((x & GG_ULONGLONG(0xFF00)) << 40) |
          ((x & GG_ULONGLONG(0xFF0000)) << 24) |
          ((x & GG_ULONGLONG(0xFF000000)) << 8) |
          ((x & GG_ULONGLONG(0xFF00000000)) >> 8) |
          ((x & GG_ULONGLONG(0xFF0000000000)) >> 24) |
          ((x & GG_ULONGLONG(0xFF000000000000)) >> 40) |
          ((x & GG_ULONGLONG(0xFF00000000000000)) >> 56));
}

#define bswap_64(x) bswap_64(x)

#endif

// Define the macros IS_LITTLE_ENDIAN or IS_BIG_ENDIAN using the above endian
// definitions from endian.h if endian.h was included
#ifdef __BYTE_ORDER
#if __BYTE_ORDER == __LITTLE_ENDIAN
#define IS_LITTLE_ENDIAN
#endif

#if __BYTE_ORDER == __BIG_ENDIAN
#define IS_BIG_ENDIAN
#endif

#else

#if defined(__LITTLE_ENDIAN__)
#define IS_LITTLE_ENDIAN
#elif defined(__BIG_ENDIAN__)
#define IS_BIG_ENDIAN
#endif

#endif  // __BYTE_ORDER

// va_copy portability definitions.
#ifdef _MSC_VER
// MSVC doesn't have va_copy yet.
// This is believed to work for 32-bit MSVC. This may not work at all for
// other platforms.
// If va_list uses the single-element-array trick, you will probably get
// a compiler error here.
//
#include <stdarg.h>
inline void va_copy(va_list &a, va_list &b) {
  a = b;
}

// Nor does it have uid_t.
typedef int uid_t;

#endif

// Mac OS X/Darwin-specific features.
#if defined(__APPLE__)

// For mmap, Linux defines both MAP_ANONYMOUS and MAP_ANON and says MAP_ANON is
// deprecated. In Darwin, MAP_ANON is all there is.
#if !defined MAP_ANONYMOUS
#define MAP_ANONYMOUS MAP_ANON
#endif

namespace std {}  // Avoid error if we didn't see std.
using namespace std;  // Just like VC++, we need a using here.

#endif  // defined(__APPLE__)

// Cygwin-specific behavior.
#if defined(OS_CYGWIN)

// Scans memory for a character.
// memrchr is used in a few places, but it's linux-specific.
inline void *memrchr(const void *bytes, int find_char, size_t len) {
  const unsigned char *cursor =
      reinterpret_cast<const unsigned char *>(bytes) + len - 1;
  unsigned char actual_char = find_char;
  for (; cursor >= bytes; --cursor) {
    if (*cursor == actual_char) {
      return const_cast<void*>(reinterpret_cast<const void *>(cursor));
    }
  }
  return nullptr;
}

#endif  // defined(OS_CYGWIN)

// GCC-specific features.
#if defined(__GNUC__) || defined(__APPLE__)

// Tell the compiler to do printf format string checking if the
// compiler supports it.
#define ABSL_PRINTF_ATTRIBUTE(string_index, first_to_check) \
    __attribute__((__format__ (__printf__, string_index, first_to_check)))
#define ABSL_SCANF_ATTRIBUTE(string_index, first_to_check) \
    __attribute__((__format__ (__scanf__, string_index, first_to_check)))

// Prevent the compiler from padding a structure to natural alignment.
#define ABSL_ATTRIBUTE_PACKED __attribute__ ((packed))

// Prevent the compiler from complaining about or optimizing away variables
// that appear unused.
#undef ABSL_ATTRIBUTE_UNUSED
#define ABSL_ATTRIBUTE_UNUSED __attribute__ ((unused))

// For functions we want to force inline or not inline.
#define ABSL_ATTRIBUTE_ALWAYS_INLINE  __attribute__ ((always_inline))
#define ABSL_HAVE_ATTRIBUTE_ALWAYS_INLINE 1
#define ABSL_ATTRIBUTE_NOINLINE __attribute__ ((noinline))
#define ABSL_HAVE_ATTRIBUTE_NOINLINE 1

// Tell the compiler that some function parameters should be non-null pointers.
#define ABSL_ATTRIBUTE_NONNULL(arg_index) __attribute__((nonnull(arg_index)))

// Tell the compiler that a given function never returns.
#define ABSL_ATTRIBUTE_NORETURN __attribute__((noreturn))

// Tells GCC that a function is hot or cold. GCC can use this information to
// improve static analysis, i.e. a conditional branch to a cold function
// is likely to be not-taken.
#define ABSL_ATTRIBUTE_HOT __attribute__((hot))
#define ABSL_ATTRIBUTE_COLD __attribute__((cold))

// Tell the compiler to warn about unused return values for functions declared
// with this macro. The macro should be used on function declarations following
// the argument list.
#if __GNUC__ > 3 || (__GNUC__ == 3 && __GNUC_MINOR__ >= 4)
#define ABSL_MUST_USE_RESULT __attribute__ ((warn_unused_result))
#else
#define ABSL_MUST_USE_RESULT
#endif

// GCC can be told that a certain branch is not likely to be taken (for
// instance, a CHECK failure), and use that information in static analysis.
// Giving it this information can help it optimize for the common case in
// the absence of better information (ie. -fprofile-arcs).
//
#if defined(__GNUC__)
#define PREDICT_FALSE(x) (__builtin_expect(x, 0))
#define PREDICT_TRUE(x) (__builtin_expect(!!(x), 1))
#else
#define PREDICT_FALSE(x) x
#define PREDICT_TRUE(x) x
#endif

#if !defined(__cplusplus) && !defined(__APPLE__) && !defined(OS_CYGWIN)
// stdlib.h only declares this in C++, not in C, so we declare it here.
// Also make sure to avoid declaring it on platforms which don't support it.
extern int posix_memalign(void **memptr, size_t alignment, size_t size);
#endif

inline void *aligned_malloc(size_t size, int minimum_alignment) {
#if defined(__APPLE__)
  // Mac lacks memalign(), posix_memalign(), however, according to
  // http://stackoverflow.com/questions/196329/osx-lacks-memalign
  // mac allocs are already 16-byte aligned.
  if (minimum_alignment <= 16) return malloc(size);
  // Next, try to return page-aligned memory.
  if (minimum_alignment <= getpagesize()) return valloc(size);
  // Give up
  return nullptr;
#elif defined(OS_CYGWIN)
  return memalign(minimum_alignment, size);
#else  // !__APPLE__ && !OS_CYGWIN
  void *ptr = nullptr;
  if (posix_memalign(&ptr, minimum_alignment, size) != 0)
    return nullptr;
  else
    return ptr;
#endif
}

inline void aligned_free(void *aligned_memory) {
  free(aligned_memory);
}

#else   // not GCC

#define ABSL_PRINTF_ATTRIBUTE(string_index, first_to_check)
#define ABSL_SCANF_ATTRIBUTE(string_index, first_to_check)
#define PACKED
#define ABSL_ATTRIBUTE_UNUSED
#define ABSL_ATTRIBUTE_ALWAYS_INLINE
#define ABSL_ATTRIBUTE_NOINLINE
#define ABSL_ATTRIBUTE_NONNULL(arg_index)
#define ABSL_ATTRIBUTE_NORETURN
#define ABSL_MUST_USE_RESULT
#define PREDICT_FALSE(x) x
#define PREDICT_TRUE(x) x

#endif  // GCC


// Microsoft Visual C++
#ifdef _MSC_VER

// This compiler flag can be easily overlooked on MSVC.
// _CHAR_UNSIGNED gets set with the /J flag.
#ifndef _CHAR_UNSIGNED
#error chars must be unsigned!  Use the /J flag on the compiler command line.
#endif

// Signed vs. unsigned comparison is ok.
#pragma warning(disable : 4018)
// We know casting from a long to a char may lose data.
#pragma warning(disable : 4244)
// Don't need performance warnings about converting ints to bools.
#pragma warning(disable : 4800)
// Integral constant overflow is apparently ok too.
#pragma warning(disable : 4307)
// It's ok to use this in constructor.
#pragma warning(disable : 4355)
// Truncating from double to float is ok.
#pragma warning(disable : 4305)

#include <winsock2.h>
#include <assert.h>
#include <windows.h>
#undef ERROR

namespace std {}
using namespace std;

#ifndef HAVE_UINT
#define HAVE_UINT 1
typedef unsigned int uint;
#endif

#ifndef HAVE_SSIZET
#define HAVE_SSIZET 1
typedef int ssize_t;
#endif

#define strtoq   _strtoi64
#define strtouq  _strtoui64
#define strtoll  _strtoi64
#define strtoull _strtoui64
#define atoll    _atoi64

// You say tomato, I say atotom
#define PATH_MAX MAX_PATH

// You say tomato, I say _tomato
#define vsnprintf _vsnprintf
#define snprintf _snprintf
#define strcasecmp _stricmp
#define strncasecmp _strnicmp

#define hypot _hypot
#define hypotf _hypotf

#define strdup _strdup
#define tempnam _tempnam
#define chdir  _chdir
#define getcwd _getcwd
#define putenv  _putenv

// You say tomato, I say toma
#define random() rand()
#define srandom(x) srand(x)

inline void *aligned_malloc(size_t size, int minimum_alignment) {
  return _aligned_malloc(size, minimum_alignment);
}

inline void aligned_free(void *aligned_memory) {
  _aligned_free(aligned_memory);
}

typedef int pid_t;
typedef unsigned int mode_t;
typedef unsigned short u_int16_t;
typedef short int16_t;

#endif  // _MSC_VER

// Portable handling of unaligned loads, stores, and copies.
// On some platforms, like ARM, the copy functions can be more efficient
// then a load and a store.

#if defined(__i386) || \
    defined(ARCH_ATHLON) || \
    defined(__x86_64__) || \
    defined(_ARCH_PPC)

// x86 and x86-64 can perform unaligned loads/stores directly;
// modern PowerPC hardware can also do unaligned integer loads and stores;
// but note: the FPU still sends unaligned loads and stores to a trap handler!

#define UNALIGNED_LOAD16(_p) (*reinterpret_cast<const uint16 *>(_p))
#define UNALIGNED_LOAD32(_p) (*reinterpret_cast<const uint32 *>(_p))
#define UNALIGNED_LOAD64(_p) (*reinterpret_cast<const uint64 *>(_p))

#define UNALIGNED_STORE16(_p, _val) (*reinterpret_cast<uint16 *>(_p) = (_val))
#define UNALIGNED_STORE32(_p, _val) (*reinterpret_cast<uint32 *>(_p) = (_val))
#define UNALIGNED_STORE64(_p, _val) (*reinterpret_cast<uint64 *>(_p) = (_val))

#elif defined(__arm__) && \
      !defined(__ARM_ARCH_5__) && \
      !defined(__ARM_ARCH_5T__) && \
      !defined(__ARM_ARCH_5TE__) && \
      !defined(__ARM_ARCH_5TEJ__) && \
      !defined(__ARM_ARCH_6__) && \
      !defined(__ARM_ARCH_6J__) && \
      !defined(__ARM_ARCH_6K__) && \
      !defined(__ARM_ARCH_6Z__) && \
      !defined(__ARM_ARCH_6ZK__) && \
      !defined(__ARM_ARCH_6T2__)

// ARMv7 and newer support native unaligned accesses, but only of 16-bit
// and 32-bit values (not 64-bit); older versions either raise a fatal signal,
// do an unaligned read and rotate the words around a bit, or do the reads very
// slowly (trip through kernel mode). There's no simple #define that says just
// “ARMv7 or higher”, so we have to filter away all ARMv5 and ARMv6
// sub-architectures. Newer gcc (>= 4.6) set an __ARM_FEATURE_ALIGNED #define,
// so in time, maybe we can move on to that.

#define UNALIGNED_LOAD16(_p) (*reinterpret_cast<const uint16 *>(_p))
#define UNALIGNED_LOAD32(_p) (*reinterpret_cast<const uint32 *>(_p))

#define UNALIGNED_STORE16(_p, _val) (*reinterpret_cast<uint16 *>(_p) = (_val))
#define UNALIGNED_STORE32(_p, _val) (*reinterpret_cast<uint32 *>(_p) = (_val))

inline uint64 UNALIGNED_LOAD64(const void *p) {
  uint64 t;
  memcpy(&t, p, sizeof t);
  return t;
}

inline void UNALIGNED_STORE64(void *p, uint64 v) {
  memcpy(p, &v, sizeof v);
}

#else

#define NEED_ALIGNED_LOADS

// These functions are provided for architectures that don't support
// unaligned loads and stores.

inline uint16 UNALIGNED_LOAD16(const void *p) {
  uint16 t;
  memcpy(&t, p, sizeof t);
  return t;
}

inline uint32 UNALIGNED_LOAD32(const void *p) {
  uint32 t;
  memcpy(&t, p, sizeof t);
  return t;
}

inline uint64 UNALIGNED_LOAD64(const void *p) {
  uint64 t;
  memcpy(&t, p, sizeof t);
  return t;
}

inline void UNALIGNED_STORE16(void *p, uint16 v) {
  memcpy(p, &v, sizeof v);
}

inline void UNALIGNED_STORE32(void *p, uint32 v) {
  memcpy(p, &v, sizeof v);
}

inline void UNALIGNED_STORE64(void *p, uint64 v) {
  memcpy(p, &v, sizeof v);
}

#endif

#ifdef _LP64
#define UNALIGNED_LOADW(_p) UNALIGNED_LOAD64(_p)
#define UNALIGNED_STOREW(_p, _val) UNALIGNED_STORE64(_p, _val)
#else
#define UNALIGNED_LOADW(_p) UNALIGNED_LOAD32(_p)
#define UNALIGNED_STOREW(_p, _val) UNALIGNED_STORE32(_p, _val)
#endif

#if defined(__cplusplus)

inline void UnalignedCopy16(const void *src, void *dst) {
  UNALIGNED_STORE16(dst, UNALIGNED_LOAD16(src));
}

inline void UnalignedCopy32(const void *src, void *dst) {
  UNALIGNED_STORE32(dst, UNALIGNED_LOAD32(src));
}

inline void UnalignedCopy64(const void *src, void *dst) {
  if (sizeof(void *) == 8) {
    UNALIGNED_STORE64(dst, UNALIGNED_LOAD64(src));
  } else {
    const char *src_char = reinterpret_cast<const char *>(src);
    char *dst_char = reinterpret_cast<char *>(dst);

    UNALIGNED_STORE32(dst_char, UNALIGNED_LOAD32(src_char));
    UNALIGNED_STORE32(dst_char + 4, UNALIGNED_LOAD32(src_char + 4));
  }
}

#endif  // defined(__cpluscplus)

}  // namespace sling

#endif  // SLING_BASE_PORT_H_

