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

#ifndef SLING_BASE_INIT_H_
#define SLING_BASE_INIT_H_

namespace sling {

// Module initializers are called in order of registration at startup.
struct ModuleInitializer {
  typedef void (*Handler)(void);

  // Add module initializer.
  ModuleInitializer(const char *n, Handler h);

  // Module name.
  const char *name;

  // Handler for initializing module.
  Handler handler;

  // Next initializer.
  ModuleInitializer *next;

  // Linked list of module initializers.
  static ModuleInitializer *first;
  static ModuleInitializer *last;
};

#define REGISTER_INITIALIZER(name, body)                        \
  namespace {                                                   \
    static void init_module_##name () { body; }                 \
    __attribute__((init_priority(1000)))                        \
    sling::ModuleInitializer initializer_module_##name          \
      (#name, init_module_##name);                              \
  }

// Run module initializers for program.
void InitProgram(int *argc, char **argv[]);

// Run module initializers for shared library.
void InitSharedLibrary();

}  // namespace sling

#endif  // SLING_BASE_INIT_H_

