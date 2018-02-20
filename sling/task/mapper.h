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

namespace sling {
namespace task {

// Input to mapper with a key and a value.
class MapInput {
 public:
  MapInput(Slice key, Slice value)
      : key_(key), value_(value) {}

  // Key for message.
  Slice key() const { return key_; }

  // Value for message.
  Slice value() const { return value_; }

 private:
  Slice key_;
  Slice value_;
};

// A mapper processes all the input message in the Map() method and can output
// new key/value pairs to the output.
class Mapper : public Processor {
 public:
  void Start(Task *task) override;
  void Receive(Channel *channel, Message *message) override;
  void Done(Task *task) override;

  // The Map() method is called for each message in the input and can call the
  // Output() method to produce key/value pairs.
  virtual void Map(const MapInput &input) = 0;

  // Output key/value pair to output.
  void Output(Slice key, Slice value);

  // Return output channel.
  Channel *output() const { return output_; }

 private:
  // Output channel.
  Channel *output_ = nullptr;
};

}  // namespace task
}  // namespace sling

#endif  // SLING_TASK_REDUCER_H_

