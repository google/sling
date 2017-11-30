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

#include "sling/myelin/multi-process.h"

#include <stdlib.h>

#include "sling/base/logging.h"

namespace sling {
namespace myelin {

// Worker thread for multi-processor runtime.
class Worker {
 public:
  // Start worker.
  Worker() : thread_(&Worker::Run, this) {}

  // Stop worker.
  ~Worker() {
    DCHECK_EQ(state_, IDLE);
    mu_.lock();
    state_ = STOP;
    mu_.unlock();
    cv_.notify_all();
    thread_.join();
  }

  // Attach worker to task.
  void Attach(Task *task) {
    DCHECK_EQ(state_, IDLE);
    mu_.lock();
    task->state = this;
    task_ = task;
    state_ = READY;
    mu_.unlock();
    cv_.notify_all();
  }

  // Detach worker from current task.
  void Detach() {
    DCHECK_EQ(state_, READY);
    state_ = IDLE;
  }

  // Start task in worker thread.
  static void Start(Task *task) {
    Worker *worker = reinterpret_cast<Worker *>(task->state);
    DCHECK_EQ(worker->state_, READY);
    worker->state_ = RUN;
  }

  // Wait for worker thread to complete.
  static void Wait(Task *task) {
    Worker *worker = reinterpret_cast<Worker *>(task->state);
    while (worker->state_ != READY);
  }

 private:
  // Worker thread.
  void Run() {
    std::unique_lock<std::mutex> lock(mu_);
    for (;;) {
      switch (state_) {
        case READY:
          // Spin waiting for task;
          break;

        case RUN:
          // Run task.
          task_->func(task_->arg);

          // Signal completion of task.
          state_ = READY;
          break;

        case IDLE:
          // Wait for task to attach to worker (or stop).
          cv_.wait(lock);
          break;

        case STOP:
          // Terminate worker.
          return;
      }
    }
  }

  // Worker states.
  enum State {READY = 0, RUN = 1, IDLE = 2, STOP = 3};

  // Signal for waking up worker.
  std::mutex mu_;
  std::condition_variable cv_;

  // Worker state.
  volatile State state_ = IDLE;

  // Current task for worker.
  Task *task_ = nullptr;

  // Worker thread.
  std::thread thread_;
};

MultiProcessorRuntime::~MultiProcessorRuntime() {
  // Stop all workers.
  for (auto *w : workers_) delete w;
}

void MultiProcessorRuntime::AllocateInstance(Instance *instance) {
  // Allocate memory for instance.
  void *data;
  int rc = posix_memalign(&data, instance->alignment(), instance->size());
  CHECK_EQ(rc, 0);
  instance->set_data(reinterpret_cast<char *>(data));

  // Allocate workers for instance.
  int n = instance->num_tasks();
  if (n > 0) {
    std::lock_guard<std::mutex> lock(mu_);
    for (int i = 0; i < n; ++i) {
      Worker *worker;
      if (workers_.empty()) {
        worker = new Worker();
      } else {
        worker = workers_.back();
        workers_.pop_back();
      }
      worker->Attach(instance->task(i));
    }
  }
}

void MultiProcessorRuntime::FreeInstance(Instance *instance) {
  // Detach instance from workers and return them to worker pool.
  int n = instance->num_tasks();
  if (n > 0) {
    std::lock_guard<std::mutex> lock(mu_);
    for (int i = n - 1; i >= 0; --i) {
      Worker *worker = reinterpret_cast<Worker *>(instance->task(i)->state);
      worker->Detach();
      workers_.push_back(worker);
    }
  }

  // Deallocate instance memory.
  free(instance->data());
}

void MultiProcessorRuntime::ClearInstance(Instance *instance) {
  // Do not clear task data at the start of the instance block.
  memset(instance->data() + instance->cell()->data_start(), 0,
         instance->size() - instance->cell()->data_start());
}

char *MultiProcessorRuntime::AllocateChannel(char *data, size_t old_size,
                                             size_t new_size, size_t alignment,
                                             Placement placement) {
  void *buffer;
  CHECK_EQ(posix_memalign(&buffer, alignment, new_size), 0);
  if (data != nullptr) {
    memcpy(buffer, data, old_size);
    free(data);
  }
  return reinterpret_cast<char *>(buffer);
}

void MultiProcessorRuntime::ClearChannel(char *data, size_t pos, size_t size,
                                         Placement placement) {
  memset(data + pos, 0, size);
}

void MultiProcessorRuntime::FreeChannel(char *data, Placement placement) {
  free(data);
}

Runtime::TaskFunc MultiProcessorRuntime::StartTaskFunc() {
  return Worker::Start;
}

Runtime::TaskFunc MultiProcessorRuntime::WaitTaskFunc() {
  return Worker::Wait;
}

}  // namespace myelin
}  // namespace sling

