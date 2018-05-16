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

#ifndef SLING_BASE_LOGGING_H_
#define SLING_BASE_LOGGING_H_

#include <sstream>
#include <limits>

#include "sling/base/macros.h"
#include "sling/base/types.h"

namespace sling {

#ifndef LOG

const int INFO = 0;
const int WARNING = 1;
const int ERROR = 2;
const int FATAL = 3;
const int NUM_SEVERITIES = 4;

class LogMessage : public std::basic_ostringstream<char> {
 public:
  LogMessage(const char *fname, int line, int severity);
  ~LogMessage();

  // Minimum severity for LOG statements.
  static int log_level();

  // Minimum log level for VLOG statements.
  static int vlog_level();

 protected:
  void GenerateLogMessage();

 private:
  const char *fname_;
  int line_;
  int severity_;
};

// LogMessageFatal ensures the process will exit in failure after
// logging this message.
class LogMessageFatal : public LogMessage {
 public:
  LogMessageFatal(const char *file, int line) ABSL_ATTRIBUTE_COLD;
  ABSL_ATTRIBUTE_NORETURN ~LogMessageFatal();
};

#define _LOG_INFO ::sling::LogMessage(__FILE__, __LINE__, ::sling::INFO)
#define _LOG_WARNING ::sling::LogMessage(__FILE__, __LINE__, ::sling::WARNING)
#define _LOG_ERROR ::sling::LogMessage(__FILE__, __LINE__, ::sling::ERROR)
#define _LOG_FATAL ::sling::LogMessageFatal(__FILE__, __LINE__)

#define LOG(severity) _LOG_##severity

// Get log level from VLOG environment variable.
#define VLOG_IS_ON(level) ((level) <= ::sling::LogMessage::vlog_level())

#define VLOG(level)                     \
  if (PREDICT_FALSE(VLOG_IS_ON(level))) \
    ::sling::LogMessage(__FILE__, __LINE__, ::sling::INFO)

// CHECK dies with a fatal error if condition is not true.  It is NOT controlled
// by NDEBUG, so the check will be executed regardless of compilation mode.
// Therefore, it is safe to do things like:
//    CHECK(fp->Write(x) == 4)
#define CHECK(condition)           \
  if (PREDICT_FALSE(!(condition))) \
  LOG(FATAL) << "Check failed: " #condition " "

// Function is overloaded for integral types to allow static const integrals
// declared in classes and not defined to be used as arguments to CHECK* macros.
// It's not encouraged though.
template <typename T>
inline const T &GetReferenceableValue(const T &t) { return t; }

inline uint8 GetReferenceableValue(uint8 t) { return t; }
inline uint16 GetReferenceableValue(uint16 t) { return t; }
inline uint32 GetReferenceableValue(uint32 t) { return t; }
inline uint64 GetReferenceableValue(uint64 t) { return t; }

inline int8 GetReferenceableValue(int8 t) { return t; }
inline int16 GetReferenceableValue(int16 t) { return t; }
inline int32 GetReferenceableValue(int32 t) { return t; }
inline int64 GetReferenceableValue(int64 t) { return t; }

// This formats a value for a failing CHECK_XX statement.  Ordinarily, it uses
// the definition for operator<<, with a few special cases below.
template <typename T>
inline void MakeCheckOpValueString(std::ostream* os, const T &v) { (*os) << v; }

// Overrides for char types provide readable values for unprintable characters.
template <>
void MakeCheckOpValueString(std::ostream* os, const char &v);
template <>
void MakeCheckOpValueString(std::ostream* os, const signed char &v);
template <>
void MakeCheckOpValueString(std::ostream* os, const unsigned char &v);

// We need an explicit specialization for std::nullptr_t.
template <>
void MakeCheckOpValueString(std::ostream* os, const std::nullptr_t &p);

// A container for a string pointer which can be evaluated to a bool -
// true iff the pointer is non-NULL.
struct CheckOpString {
  CheckOpString(string *str) : str_(str) {}
  // No destructor: if str_ is non-NULL, we're about to LOG(FATAL),
  // so there's no point in cleaning up str_.
  operator bool() const { return PREDICT_FALSE(str_ != nullptr); }
  string* str_;
};

// Build the error message string. Specify no inlining for code size.
template <typename T1, typename T2>
string *MakeCheckOpString(const T1 &v1, const T2 &v2,
                          const char *exprtext) ABSL_ATTRIBUTE_NOINLINE;

// A helper class for formatting "expr (V1 vs. V2)" in a CHECK_XX statement.
class CheckOpMessageBuilder {
 public:
  // Inserts "exprtext" and " (" to the stream.
  explicit CheckOpMessageBuilder(const char *exprtext);
  ~CheckOpMessageBuilder();
  // For inserting the first variable.
  std::ostream *ForVar1() { return stream_; }
  // For inserting the second variable (adds an intermediate " vs. ").
  std::ostream *ForVar2();
  // Get the result (inserts the closing ")").
  string *NewString();

 private:
  std::ostringstream *stream_;
};

template <typename T1, typename T2>
string *MakeCheckOpString(const T1 &v1, const T2 &v2, const char *exprtext) {
  CheckOpMessageBuilder comb(exprtext);
  MakeCheckOpValueString(comb.ForVar1(), v1);
  MakeCheckOpValueString(comb.ForVar2(), v2);
  return comb.NewString();
}

// Helper functions for CHECK_OP macro.
// The (int, int) specialization works around the issue that the compiler
// will not instantiate the template version of the function on values of
// unnamed enum type - see comment below.
// The (size_t, int) and (int, size_t) specialization are to handle unsigned
// comparison errors while still being thorough with the comparison.
#define DEFINE_CHECK_OP_IMPL(name, op)                                    \
  template <typename T1, typename T2>                                     \
  inline string *name##Impl(const T1 &v1, const T2 &v2,                   \
                            const char *exprtext) {                       \
    if (PREDICT_TRUE(v1 op v2))                                           \
      return nullptr;                                                     \
    else                                                                  \
      return ::sling::MakeCheckOpString(v1, v2, exprtext);                \
  }                                                                       \
  inline string *name##Impl(int v1, int v2, const char *exprtext) {       \
    return name##Impl<int, int>(v1, v2, exprtext);                        \
  }                                                                       \
  inline string *name##Impl(const size_t v1, const int v2,                \
                            const char *exprtext) {                       \
    if (PREDICT_FALSE(v2 < 0)) {                                          \
       return ::sling::MakeCheckOpString(v1, v2, exprtext);               \
    }                                                                     \
    const size_t uval = (size_t) ((unsigned) v1);                         \
    return name##Impl<size_t, size_t>(uval, v2, exprtext);                \
  }                                                                       \
  inline string *name##Impl(const int v1, const size_t v2,                \
                            const char *exprtext) {                       \
    if (PREDICT_FALSE(v2 >= std::numeric_limits<int>::max())) {           \
       return ::sling::MakeCheckOpString(v1, v2, exprtext);               \
    }                                                                     \
    const size_t uval = (size_t) ((unsigned) v2);                         \
    return name##Impl<size_t, size_t>(v1, uval, exprtext);                \
  }

DEFINE_CHECK_OP_IMPL(EQ, == )
DEFINE_CHECK_OP_IMPL(NE, != )
DEFINE_CHECK_OP_IMPL(LE, <= )
DEFINE_CHECK_OP_IMPL(LT, < )
DEFINE_CHECK_OP_IMPL(GE, >= )
DEFINE_CHECK_OP_IMPL(GT, > )
#undef DEFINE_CHECK_OP_IMPL

// In optimized mode, use CheckOpString to hint to compiler that the while
// condition is unlikely.
#define CHECK_OP_LOG(name, op, val1, val2)                            \
  while (::sling::CheckOpString _result =                             \
             ::sling::name##Impl(                                     \
                 ::sling::GetReferenceableValue(val1),                \
                 ::sling::GetReferenceableValue(val2),                \
                 #val1 " " #op " " #val2))                            \
    ::sling::LogMessageFatal(__FILE__, __LINE__) << *(_result.str_)

#define CHECK_OP(name, op, val1, val2) CHECK_OP_LOG(name, op, val1, val2)

// CHECK_EQ/NE/...
#define CHECK_EQ(val1, val2) CHECK_OP(EQ, ==, val1, val2)
#define CHECK_NE(val1, val2) CHECK_OP(NE, !=, val1, val2)
#define CHECK_LE(val1, val2) CHECK_OP(LE, <=, val1, val2)
#define CHECK_LT(val1, val2) CHECK_OP(LT, <, val1, val2)
#define CHECK_GE(val1, val2) CHECK_OP(GE, >=, val1, val2)
#define CHECK_GT(val1, val2) CHECK_OP(GT, >, val1, val2)
#define CHECK_NOTNULL(val)                                 \
  ::sling::CheckNotNull(__FILE__, __LINE__, \
                        "'" #val "' Must be non NULL", (val))

#ifndef NDEBUG
// DCHECK_EQ/NE/...
#define DCHECK(condition) CHECK(condition)
#define DCHECK_EQ(val1, val2) CHECK_EQ(val1, val2)
#define DCHECK_NE(val1, val2) CHECK_NE(val1, val2)
#define DCHECK_LE(val1, val2) CHECK_LE(val1, val2)
#define DCHECK_LT(val1, val2) CHECK_LT(val1, val2)
#define DCHECK_GE(val1, val2) CHECK_GE(val1, val2)
#define DCHECK_GT(val1, val2) CHECK_GT(val1, val2)

#else

#define DCHECK(condition) while (false && (condition)) LOG(FATAL)

// NDEBUG is defined, so DCHECK_EQ(x, y) and so on do nothing. However, we still
// want the compiler to parse x and y, because we don't want to lose potentially
// useful errors and warnings.
#define _DCHECK_NOP(x, y) \
  while (false && ((void) (x), (void) (y), 0)) LOG(FATAL)

#define DCHECK_EQ(x, y) _DCHECK_NOP(x, y)
#define DCHECK_NE(x, y) _DCHECK_NOP(x, y)
#define DCHECK_LE(x, y) _DCHECK_NOP(x, y)
#define DCHECK_LT(x, y) _DCHECK_NOP(x, y)
#define DCHECK_GE(x, y) _DCHECK_NOP(x, y)
#define DCHECK_GT(x, y) _DCHECK_NOP(x, y)

#endif

// These are for when you don't want a CHECK failure to print a verbose
// stack trace.  The implementation of CHECK* in this file already doesn't.
#define QCHECK(condition) CHECK(condition)
#define QCHECK_EQ(x, y) CHECK_EQ(x, y)
#define QCHECK_NE(x, y) CHECK_NE(x, y)
#define QCHECK_LE(x, y) CHECK_LE(x, y)
#define QCHECK_LT(x, y) CHECK_LT(x, y)
#define QCHECK_GE(x, y) CHECK_GE(x, y)
#define QCHECK_GT(x, y) CHECK_GT(x, y)

template <typename T>
T &&CheckNotNull(const char *file, int line, const char *exprtext, T &&t) {
  if (t == nullptr) {
    LogMessageFatal(file, line) << string(exprtext);
  }
  return std::forward<T>(t);
}

#endif  // LOG

}  // namespace sling

#endif  // SLING_BASE_LOGGING_H_

