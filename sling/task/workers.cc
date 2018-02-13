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

#include "sling/task/task.h"
#include "sling/util/threadpool.h"

namespace sling {
namespace task {

// Create a pool of worker threads and distribute the incoming messages to
// the output channel using the worker threads. This adds parallelism to the
// processing of the message stream.
class Workers : public Processor {
 public:
  ~Workers() override { delete pool_; }

  void Start(Task *task) override {
    // Get output port.
    output_ = task->GetSink("output");

    // Get worker pool parameters.
    int num_workers = task->Get("worker_threads", 5);
    int queue_size = task->Get("queue_size", num_workers * 2);

    // Start worker pool.
    pool_ = new ThreadPool(num_workers, queue_size);
    pool_->StartWorkers();
  }

  void Receive(Channel *channel, Message *message) override {
    if (output_ == nullptr) {
      // No receiver.
      delete message;
    } else {
      // Send message to output in one of the worker threads.
      pool_->Schedule([this, message]() {
        output_->Send(message);
      });
    }
  }

  void Done(Task *task) override {
    // Stop all worker threads.
    delete pool_;
    pool_ = nullptr;
  }

 private:
  // Thread pool for dispatching messages.
  ThreadPool *pool_ = nullptr;

  // Output channel.
  Channel *output_;
};

REGISTER_TASK_PROCESSOR("workers", Workers);

}  // namespace task
}  // namespace sling

