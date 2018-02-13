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

#ifndef SLING_TASK_PROCESS_H_
#define SLING_TASK_PROCESS_H_

#include "sling/task/task.h"
#include "sling/util/thread.h"

namespace sling {
namespace task {

// A task process runs the task in a separate thread.
class Process : public Processor {
 public:
  // Delete thread.
  ~Process() override { delete thread_; }

  // Start task thread.
  void Start(Task *task) override;

  // Wait for thread to finish.
  void Done(Task *task) override;

  // This method is run in a separate thread when the task is started.
  virtual void Run(Task *task) = 0;

 private:
  // Thread for running task.
  ClosureThread *thread_ = nullptr;
};

}  // namespace task
}  // namespace sling

#endif  // SLING_TASK_PROCESS_H_

