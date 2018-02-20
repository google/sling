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

#include "sling/task/mapper.h"

#include "sling/base/logging.h"

namespace sling {
namespace task {

void Mapper::Start(Task *task) {
  // Get output channel.
  output_ = task->GetSink("output");
  if (output_ == nullptr) {
    LOG(ERROR) << "No output channel";
    return;
  }
}

void Mapper::Receive(Channel *channel, Message *message) {
  // Call Map() method on each input message.
  MapInput input(message->key(), message->value());
  Map(input);

  // Delete input message.
  delete message;
}

void Mapper::Done(Task *task) {
  // Close output channel.
  if (output_ != nullptr) output_->Close();
}

void Mapper::Output(Slice key, Slice value) {
  // Ignore if there is no output.
  if (output_ == nullptr) return;

  // Create new message and send it on the output channel.
  Message *message = new Message(key, value);
  output_->Send(message);
}

}  // namespace task
}  // namespace sling

