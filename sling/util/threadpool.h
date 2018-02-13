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

#ifndef SLING_UTIL_THREADPOOL_H_
#define SLING_UTIL_THREADPOOL_H_

#include <condition_variable>
#include <functional>
#include <mutex>
#include <queue>
#include <vector>

#include "sling/util/thread.h"

namespace sling {

// Thread pool for executing tasks using a pool of worker threads.
class ThreadPool {
 public:
  // Task that can be scheduled for execution.
  typedef std::function<void()> Task;

  // Initialize thread pool.
  ThreadPool(int num_workers, int queue_size);

  // Wait for all workers to complete.
  ~ThreadPool();

  // Start worker threads.
  void StartWorkers();

  // Schedule task to be executed by worker.
  void Schedule(Task &&task);

 private:
  // Fetch next task. Returns false when all tasks have been completed.
  bool FetchTask(Task *task);

  // Shut down workers. This waits until all tasks have been completed.
  void Shutdown();

  // Worker threads.
  int num_workers_;
  std::vector<ClosureThread> workers_;

  // Task queue.
  int queue_size_;
  std::queue<Task> tasks_;

  // Are we done with adding new tasks.
  bool done_ = false;

  // Mutex for serializing access to task queue.
  std::mutex mu_;

  // Signal to notify about new tasks in queue.
  std::condition_variable nonempty_;

  // Signal to notify about available space in queue.
  std::condition_variable nonfull_;
};

}  // namespace sling

#endif  // SLING_UTIL_THREADPOOL_H_

