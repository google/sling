// Copyright 2018 Google Inc.
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

// Create SLING store snapshot files.

#include <iostream>
#include <string>

#include "sling/base/init.h"
#include "sling/base/flags.h"
#include "sling/base/logging.h"
#include "sling/base/types.h"
#include "sling/file/file.h"
#include "sling/frame/serialization.h"
#include "sling/frame/snapshot.h"
#include "sling/frame/store.h"

DEFINE_bool(check, false, "Check for valid snapshot");
DEFINE_bool(verify, false, "Check snapshot by reading it into memory");

using namespace sling;

int main(int argc, char *argv[]) {
  InitProgram(&argc, &argv);

  // Get files to snapshot.
  std::vector<string> files;
  for (int i = 1; i < argc; ++i) {
    File::Match(argv[i], &files);
  }

  for (const string &file : files) {
    if (FLAGS_check) {
      bool valid = Snapshot::Valid(file);
      std::cout << file << ": " << (valid ? "valid" : "INVALID") << "\n";
    } else if (FLAGS_verify) {
      std::cout << file << ": " << std::flush;
      std::cout << "load " << std::flush;
      Store store;
      CHECK(Snapshot::Read(&store, file));
      std::cout << "done\n" << std::flush;
    } else {
      std::cout << file << ": " << std::flush;
      File::Delete(Snapshot::Filename(file));
      std::cout << "load " << std::flush;
      Store store;
      LoadStore(file, &store);
      std::cout << "freeze " << std::flush;
      store.AllocateSymbolHeap();
      store.Freeze();
      std::cout << "snapshot " << std::flush;
      CHECK(Snapshot::Write(&store, file));
      std::cout << "done\n" << std::flush;
    }
  }

  return 0;
}
