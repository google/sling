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

#ifndef SLING_TASK_ENVIRONMENT_H_
#define SLING_TASK_ENVIRONMENT_H_

#include <atomic>
#include <string>

#include "sling/base/types.h"

namespace sling {
namespace task {

class Channel;
class Task;

// Lock-free counter for statistics.
class Counter {
 public:
  // Increment counter.
  void Increment() { ++value_; }
  void Increment(int64 delta) { value_ += delta; }

  // Reset counter.
  void Reset() { value_ = 0; }

  // Set counter value.
  void Set(int64 value) { value_ = value; }

  // Return counter value.
  int64 value() const { return value_; }

 private:
  std::atomic<int64> value_{0};
};

// Container environment interface.
class Environment {
 public:
  virtual ~Environment() = default;

  // Return statistics counter.
  virtual Counter *GetCounter(const string &name) = 0;

  // Notify that channel has completed.
  virtual void ChannelCompleted(Channel *channel) = 0;

  // Notify that task has completed.
  virtual void TaskCompleted(Task *task) = 0;
};

}  // namespace task
}  // namespace sling

#endif  // SLING_TASK_ENVIRONMENT_H_

