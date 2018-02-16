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

#include "sling/base/logging.h"
#include "sling/frame/decoder.h"
#include "sling/frame/object.h"
#include "sling/frame/reader.h"
#include "sling/frame/store.h"
#include "sling/frame/wire.h"
#include "sling/stream/file.h"
#include "sling/task/process.h"
#include "sling/task/frames.h"
#include "sling/task/task.h"

namespace sling {
namespace task {

// Frame store reader.
class FrameStoreReader : public Process {
 public:
  // Process input file.
  void Run(Task *task) override {
    // Get input file.
    Binding *binding = task->GetInput("input");
    if (binding == nullptr) {
      LOG(ERROR) << "No input resource";
      return;
    }
    Resource *file = binding->resource();

    // Get output channel.
    Channel *output = task->GetSink("output");
    if (output == nullptr) {
      LOG(ERROR) << "No output channel";
      return;
    }

    // Open input file.
    FileInputStream stream(file->name());
    Input input(&stream);

    // Read frames from input and output to output channel.
    Store store;
    if (input.Peek() == WIRE_BINARY_MARKER) {
      Decoder decoder(&store, &input);
      while (!decoder.done()) {
        Object object = decoder.Decode();
        if (object.IsFrame()) {
          output->Send(CreateMessage(object.AsFrame(), true));
        } else {
          output->Send(CreateMessage(Text(), object, true));
        }
      }
    } else {
      Reader reader(&store, &input);
      while (!reader.done()) {
        Object object = reader.Read();
        CHECK(!reader.error()) << reader.GetErrorMessage(file->name());
        if (object.IsFrame()) {
          output->Send(CreateMessage(object.AsFrame(), true));
        } else {
          output->Send(CreateMessage(Text(), object, true));
        }
      }
    }

    // Close output channel.
    output->Close();
  }
};

REGISTER_TASK_PROCESSOR("frame-store-reader", FrameStoreReader);

}  // namespace task
}  // namespace sling

