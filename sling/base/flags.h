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

#ifndef SLING_BASE_FLAGS_H_
#define SLING_BASE_FLAGS_H_

#include <inttypes.h>
#include <string>

#include "sling/base/types.h"

namespace sling {

// Guard against inclusion of of other flags library.
#ifndef DEFINE_VARIABLE

// Command line flag information.
struct Flag {
  // Command line flag types.
  enum Type {BOOL, INT32, UINT32, INT64, UINT64, DOUBLE, STRING};

  // Register command line flag.
  Flag(const char *name,Type type, const char *help,
       const char *filename, void *storage);

  // Get flag value.
  template<typename T> T &value() {
    return *reinterpret_cast<T *>(storage);
  }
  template<typename T> const T &value() const {
    return *reinterpret_cast<const T *>(storage);
  }

  // Look up flag information for command line flag.
  static Flag *Find(const char *name);

  // Set program usage message for help.
  static void SetUsageMessage(const string &usage);

  // Parse command line flags.
  static int ParseCommandLineFlags(int *argc, char **argv);

  // Parse help message.
  static void PrintHelp();

  const char *name;      // flag name
  Type type;             // flag type
  const char *help;      // help message for flag
  const char *filename;  // file where flag is define
  void *storage;         // pointer to flag value
  Flag *next;            // next flag in flag list

  static Flag *head;     // list of all command line flags
  static Flag *tail;     // end of list of all command line flags
};

// Command line flag definitions.
#define DEFINE_VARIABLE(type, fltype, name, value, help) \
  type FLAGS_##name = value; \
  static sling::Flag flags_##name(#name, fltype, help, __FILE__, &FLAGS_##name);

#define DEFINE_bool(name, value, help) \
  DEFINE_VARIABLE(bool, ::sling::Flag::BOOL, name, value, help)

#define DEFINE_int32(name, value, help) \
  DEFINE_VARIABLE(int32_t, ::sling::Flag::INT32, name, value, help)

#define DEFINE_uint32(name, value, help) \
  DEFINE_VARIABLE(uint32_t, ::sling::Flag::UINT32, name, value, help)

#define DEFINE_int64(name, value, help) \
  DEFINE_VARIABLE(int64_t, ::sling::Flag::INT64, name, value, help)

#define DEFINE_uint64(name, value, help) \
  DEFINE_VARIABLE(uint64_t, ::sling::Flag::UINT64, name, value, help)

#define DEFINE_double(name, val, txt) \
  DEFINE_VARIABLE(double, ::sling::Flag::DOUBLE, name, val, txt)

#define DEFINE_string(name, val, txt) \
  DEFINE_VARIABLE(string, ::sling::Flag::STRING, name, val, txt)

// Command line flag declarations.
#define DECLARE_VARIABLE(type, name) extern type FLAGS_##name;

#define DECLARE_bool(name) DECLARE_VARIABLE(bool, name)
#define DECLARE_int32(name) DECLARE_VARIABLE(int32_t, name)
#define DECLARE_uint32(name) DECLARE_VARIABLE(uint32_t, name)
#define DECLARE_int64(name) DECLARE_VARIABLE(int64_t, name)
#define DECLARE_uint64(name) DECLARE_VARIABLE(uint64_t, name)
#define DECLARE_string(name) DECLARE_VARIABLE(string, name)

#endif  // DEFINE_VARIABLE

}  // namespace sling

#endif  // SLING_BASE_FLAGS_H_

