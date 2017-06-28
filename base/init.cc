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

#include <string>

#include "base/init.h"

#include "base/flags.h"
#include "base/logging.h"
#include "base/types.h"

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

void InitProgram(int *argc, char ***argv) {
  // Install signal handlers for dumping crash information.
  google::InstallFailureSignalHandler();

  // Initialize logging.
  google::InitGoogleLogging((*argv)[0]);

  // Initialize command line flags.
  string usage;
  usage.append((*argv)[0]);
  usage.append(" [OPTIONS]");
  gflags::SetUsageMessage(usage);
  gflags::ParseCommandLineFlags(argc, argv, true);

  // Run module initializers.
  ModuleInitializer *initializer = ModuleInitializer::first;
  while (initializer != nullptr) {
    VLOG(2) << "Initializing " << initializer->name << " module";
    initializer->handler();
    initializer = initializer->next;
  }
}

}  // namespace sling

