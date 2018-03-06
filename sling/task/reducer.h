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

#ifndef SLING_TASK_REDUCER_H_
#define SLING_TASK_REDUCER_H_

#include <vector>

#include "sling/base/slice.h"
#include "sling/task/message.h"
#include "sling/task/task.h"
#include "sling/util/mutex.h"

namespace sling {
namespace task {

// Input to reducer with a key and all messages with that key.
class ReduceInput {
 public:
  ReduceInput(int shard, Slice key, std::vector<Message *> &messages)
      : shard_(shard), key_(key), messages_(messages) {}

  // Shard number for messages.
  int shard() const { return shard_; }

  // Key for messages.
  Slice key() const { return key_; }

  // Messages for key.
  const std::vector<Message *> &messages() const { return messages_; }

  // Release message from shard.
  Message *release(int index) const {
    Message *message = messages_[index];
    messages_[index] = nullptr;
    return message;
  }

 private:
  int shard_;
  Slice key_;
  std::vector<Message *> &messages_;
};

// A reducer groups all consecutive messages with the same key together and
// calls the Reduce() method for each key in the input.
class Reducer : public Processor {
 public:
  ~Reducer() override;

  void Start(Task *task) override;
  void Receive(Channel *channel, Message *message) override;
  void Done(Task *task) override;

  // The Reduce() method is called for each key in the input with all the
  // messages for that key.
  virtual void Reduce(const ReduceInput &input) = 0;

  // Output message to output shard.
  void Output(int shard, Message *message);

 private:
  // Reduce messages for a shard.
  void ReduceShard(int shard);

  // Each shard collects messages from a sorted input channel.
  struct Shard {
    Shard() {}
    ~Shard() { clear(); }

    // Clear shard information.
    void clear() {
      for (Message *m : messages) delete m;
      messages.clear();
      key.clear();
    }

    // Current key for input channel.
    Slice key;

    // All collected messages with the current key.
    std::vector<Message *> messages;

    // Mutex for serializing access to shard.
    Mutex mu;
  };
  std::vector<Shard *> shards_;

  // Output channels.
  std::vector<Channel *> outputs_;
};

}  // namespace task
}  // namespace sling

#endif  // SLING_TASK_REDUCER_H_

