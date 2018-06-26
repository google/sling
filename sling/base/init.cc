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

#include "sling/base/init.h"

#include <string>

#include "sling/base/flags.h"
#include "sling/base/logging.h"
#include "sling/base/types.h"

namespace sling {

// Linked list of module initializers.
ModuleInitializer *ModuleInitializer::first = nullptr;
ModuleInitializer *ModuleInitializer::last = nullptr;

ModuleInitializer::ModuleInitializer(const char *n, Handler h)
    : name(n), handler(h) {
  if (first == nullptr) first = this;
  if (last != nullptr) last->next = this;
  last = this;
}

static void RunModuleInitializers() {
  ModuleInitializer *initializer = ModuleInitializer::first;
  while (initializer != nullptr) {
    VLOG(2) << "Initializing " << initializer->name << " module";
    initializer->handler();
    initializer = initializer->next;
  }
}

void InitProgram(int *argc, char ***argv) {
  // Initialize command line flags.
  if (*argc > 0) {
    string usage;
    usage.append((*argv)[0]);
    usage.append(" [OPTIONS]\n");
    Flag::SetUsageMessage(usage);
    if (Flag::ParseCommandLineFlags(argc, *argv) != 0) exit(1);
  }

  // Run module initializers.
  RunModuleInitializers();
}

void InitSharedLibrary() {
  // Run module initializers.
  RunModuleInitializers();
}

}  // namespace sling

