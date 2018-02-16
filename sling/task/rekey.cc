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

#include <vector>

#include "sling/base/types.h"
#include "sling/frame/store.h"
#include "sling/task/task.h"
#include "sling/task/frames.h"

namespace sling {
namespace task {

// Output messages with new key from frame role.
class RekeyTask : public Processor {
 public:
  void Start(Task *task) override {
    // Get output channel.
    output_ = task->GetSink("output");
    CHECK(output_ != nullptr) << "Output channel missing";

    // Initialize commons.
    role_ = commons_.Lookup(task->Get("key", "id"));
    commons_.Freeze();

    // Statistics.
    num_not_rekeyed_ = task->GetCounter("records_not_rekeyed");
  }

  void Receive(Channel *channel, Message *message) override {
    // Decode frame.
    Store store(&commons_);
    Frame f = DecodeMessage(&store, message);
    CHECK(f.valid());

    // Get key from role.
    Handle key = f.GetHandle(role_);
    if (!key.IsNil()) {
      // Update key in message.
      string keystr = store.DebugString(key);
      message->set_key(keystr);
    } else {
      num_not_rekeyed_->Increment();
    }

    // Output message on output channel.
    output_->Send(message);
  }

 private:
  // Output channel.
  Channel *output_ = nullptr;

  // Commons store.
  Store commons_;

  // Role for re-keying.
  Handle role_;

  // Statistics.
  Counter *num_not_rekeyed_ = nullptr;
};

REGISTER_TASK_PROCESSOR("rekey", RekeyTask);

}  // namespace task
}  // namespace sling

