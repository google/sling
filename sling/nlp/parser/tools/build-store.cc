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

#include <iostream>
#include <string>
#include <vector>

#include "sling/base/init.h"
#include "sling/base/logging.h"
#include "sling/base/types.h"
#include "sling/base/flags.h"
#include "sling/frame/object.h"
#include "sling/frame/serialization.h"

DEFINE_string(o, "", "Output for encoded store");

using namespace sling;

int main(int argc, char *argv[]) {
  InitProgram(&argc, &argv);

  // Initialize new store.
  Store store;

  // Load content into store.
  for (int i = 1; i < argc; ++i) {
    LOG(INFO) << "Loading " << argv[i];
    FileReader reader(&store, argv[i]);
    while (!reader.done()) {
      reader.Read();
      if (reader.error()) {
        LOG(FATAL) << "Error reading " << argv[i]
                   << ":" << reader.line() << ":" << reader.column()
                   << ": " << reader.error_message();
      }
    }
  }

  // Compact store.
  store.CoalesceStrings();
  store.GC();
  store.Freeze();

  // Save store to output file.
  LOG(INFO) << "Writing store to " << FLAGS_o;
  FileOutputStream stream(FLAGS_o);
  Output output(&stream);
  Encoder encoder(&store, &output);
  encoder.set_shallow(true);
  encoder.EncodeAll();
  output.Flush();
  CHECK(stream.Close());

  LOG(INFO) << "Done.";
  return 0;
}

