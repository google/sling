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

#include "sling/base/logging.h"
#include "sling/task/task.h"

namespace sling {
namespace task {

// Mapper that relays all input messages to the output channel.
class IdentityMapper : public Processor {
 public:
  void Start(Task *task) override {
    output_ = task->GetSink("output");
  }

  void Receive(Channel *channel, Message *message) override {
    if (output_ != nullptr) {
      output_->Send(message);
    } else {
      delete message;
    }
  }

 private:
  Channel *output_ = nullptr;
};

REGISTER_TASK_PROCESSOR("identity-mapper", IdentityMapper);

// Reducer that relays all input messages to the corresponding output channel.
class IdentityReducer : public Processor {
 public:
  void Start(Task *task) override {
    outputs_ = task->GetSinks("output");
  }

  void Receive(Channel *channel, Message *message) override {
    int shard = channel->consumer().shard().part();
    CHECK_LT(shard, outputs_.size());
    outputs_[shard]->Send(message);
  }

 private:
  std::vector<Channel *> outputs_;
};

REGISTER_TASK_PROCESSOR("identity-reducer", IdentityReducer);

}  // namespace task
}  // namespace sling

