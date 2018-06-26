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

#include "sling/base/flags.h"

#include <string.h>
#include <iostream>

#include "sling/base/logging.h"

DEFINE_bool(help, false, "Print help message");

namespace sling {

// Global list of command line flags.
Flag *Flag::head = nullptr;
Flag *Flag::tail = nullptr;

// Program usage message.
static string usage_message;

// Flag type names.
static const char *flagtype[] = {
  "bool", "int32", "uint32", "int64", "uint64", "double", "string",
};

Flag::Flag(const char *name, Type type, const char *help,
           const char *filename, void *storage)
    : name(name), type(type), help(help), filename(filename), storage(storage) {
  // Link flag into flags list.
  if (head == nullptr) {
    head = tail = this;
  } else {
    tail->next = this;
    tail = this;
  }
  next = nullptr;
}

Flag *Flag::Find(const char *name) {
  Flag *f = head;
  while (f != nullptr) {
    if (strcmp(name, f->name) == 0) return f;
    f = f->next;
  }
  return nullptr;
}

void Flag::SetUsageMessage(const string &usage) {
  usage_message = usage;
}

// Split argument it into a flag name and flag value (or nullptr if they are
// missing). The neg parameter is set if the arg started with "-no" or
// "--no".
static bool SplitArgument(char *arg, const char **name, const char **value) {
  *name = nullptr;
  *value = nullptr;

  // Return false if argument is not a flag.
  if (arg == nullptr || arg[0] != '-') return false;

  // Find the begin of the flag name.
  arg++;  // remove 1st '-'
  if (*arg == '-') {
    arg++;  // remove 2nd '-'
    if (arg[0] == '\0') return true;
  }
  *name = arg;

  // Find the end of the flag name.
  while (*arg != '\0' && *arg != '=') arg++;

  // Get the value if any.
  if (*arg == '=') {
    // NUL-terminate flag name.
    *arg = 0;

    // Get the value.
    *value = arg + 1;
  }

  return true;
}

int Flag::ParseCommandLineFlags(int *argc, char **argv) {
  // Parse all arguments.
  int rc = 0;
  for (int i = 1; i < *argc;) {
    int j = i;
    char *arg = argv[i++];

    // Split arg into flag components.
    const char *name;
    const char *value;
    bool neg = false;
    if (!SplitArgument(arg, &name, &value)) continue;

    // Stop parsing argument if -- is seen.
    if (name == nullptr) break;

    // Look up the flag.
    Flag *flag = Find(name);

    // Try to remove -no prefix.
    if (flag == nullptr && name[0] == 'n' && name[1] == 'o') {
      name += 2;
      neg = true;
      flag = Find(name);
    }

    // Output error for unkown flag.
    if (flag == nullptr) {
      std::cerr << "Error: unrecognized flag " << arg << "\n"
                << "Try --help for options\n";
      rc = j;
      break;
    }

    // If we still need a flag value, use the next argument if available.
    if (value == nullptr && flag->type != BOOL) {
      if (i < *argc) {
        value = argv[i++];
      }
      if (!value) {
        std::cerr << "Error: missing value for flag " << arg << " of type "
                  << flagtype[flag->type] << "\n";
        rc = j;
        break;
      }
    }

    // Parse boolean flag value.
    if (flag->type == BOOL && !neg && value != nullptr) {
      static const char *trueval[]  = {"1", "t", "true", "y", "yes"};
      static const char *falseval[] = {"0", "f", "false", "n", "no"};
      static_assert(sizeof(trueval) == sizeof(falseval), "true/false values");
      int bval = -1;
      for (size_t i = 0; i < sizeof(trueval) / sizeof(*trueval); ++i) {
        if (strcasecmp(value, trueval[i]) == 0) {
          bval = 1;
          break;
        } else if (strcasecmp(value, falseval[i]) == 0) {
          bval = 0;
          break;
        }
      }
      if (bval == -1) {
        std::cerr << "Error: illegal value for boolean flag " << arg << "\n";
        rc = j;
        break;
      }
      neg = (bval == 0);
    }

    // Set the flag.
    char *endptr = nullptr;
    switch (flag->type) {
      case BOOL:
        flag->value<bool>() = !neg;
        break;
      case INT32:
        flag->value<int32>() = strtol(value, &endptr, 10);
        break;
      case UINT32:
        flag->value<uint32>() = strtoul(value, &endptr, 10);
        break;
      case INT64:
        flag->value<int64>() = strtoll(value, &endptr, 10);
        break;
      case UINT64:
        flag->value<uint64>() = strtoull(value, &endptr, 10);
        break;
      case DOUBLE:
        flag->value<double>() = strtod(value, &endptr);
        break;
      case STRING:
        flag->value<string>() = value;
        break;
    }

    // Handle flag value errors.
    if (endptr != nullptr && *endptr != '\0') {
      std::cerr << "Error: illegal value for flag " << arg << " of type "
                << flagtype[flag->type] << "\nTry --help for options\n";
      rc = j;
      break;
    }

    // Remove the flag and value from the command.
    while (j < i) argv[j++] = nullptr;
  }

  // Shrink the argument list
  int j = 1;
  for (int i = 1; i < *argc; i++) {
    if (argv[i] != nullptr) argv[j++] = argv[i];
  }
  *argc = j;

  if (FLAGS_help) {
    PrintHelp();
    exit(0);
  }

  return rc;
}

std::ostream &operator<<(std::ostream &os, const Flag &flag) {
  switch (flag.type) {
    case Flag::BOOL:
      os << (flag.value<bool>() ? "true" : "false");
      break;
    case Flag::INT32:
      os << flag.value<int32>();
      break;
    case Flag::UINT32:
      os << flag.value<uint32>();
      break;
    case Flag::INT64:
      os << flag.value<int64>();
      break;
    case Flag::UINT64:
      os << flag.value<uint64>();
      break;
    case Flag::DOUBLE:
      os << flag.value<double>();
      break;
    case Flag::STRING:
      os << flag.value<string>();
      break;
  }

  return os;
}

void Flag::PrintHelp() {
  if (!usage_message.empty()) {
    std::cout << usage_message << "\n";
  }
  if (head != nullptr) {
    std::cout << "Options:\n";
    Flag *f = head;
    while (f != nullptr) {
      std::cout << "  --" << f->name << " (" << f->help << ")\n"
         << "        type: " << flagtype[f->type] << "  default: " << *f
         << "\n";
      f = f->next;
    }
  }
}

}  // namespace sling

