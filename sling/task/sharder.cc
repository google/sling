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
#include "sling/task/task.h"
#include "sling/util/fingerprint.h"

namespace sling {
namespace task {

// Shard input messages according to key fingerprint.
class SharderTask : public Processor {
 public:
  void Start(Task *task) override {
    // Get output shard channels.
    shards_ = task->GetSinks("output");
  }

  void Receive(Channel *channel, Message *message) override {
    // Compute key fingerprint.
    uint64 fp = Fingerprint(message->key().data(), message->key().size());
    int shard = fp % shards_.size();

    // Output message on output shard channel.
    shards_[shard]->Send(message);
  }

 private:
  // Output shard channels.
  std::vector<Channel *> shards_;
};

REGISTER_TASK_PROCESSOR("sharder", SharderTask);

}  // namespace task
}  // namespace sling

