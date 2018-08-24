// Copyright 2013 Google Inc. All Rights Reserved.
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

#ifndef SLING_UTIL_THREAD_H_
#define SLING_UTIL_THREAD_H_

#include <pthread.h>
#include <functional>
#include <vector>

namespace sling {

// Thread class that executes the Run() method in a new thread.
class Thread {
 public:
  Thread();
  virtual ~Thread();

  // Start thread.
  void Start();

  // Wait until thread terminates.
  void Join();

  // Mark the thread as joinable.
  void SetJoinable(bool joinable);

  // Check if the thread is the currently running thread.
  bool IsSelf() const;

 protected:
  // Entry point for new thread.
  static void *ThreadMain(void *arg);

  // This method is executed in the new thread.
  virtual void Run() = 0;

 private:
  // Thread handle.
  pthread_t thread_;

  // Flag indicating that thread is running.
  bool running_;

  // Joinable thread.
  bool joinable_ = false;
};

// A ClosureThread runs a closure in a new thread.
class ClosureThread : public Thread {
 public:
  // A closure is a void functional.
  typedef std::function<void()> Closure;

  // Initialize closure thread.
  explicit ClosureThread(Closure &&closure)
      : closure_(std::move(closure)) {}
  explicit ClosureThread(const Closure &closure)
      : closure_(closure) {}

 protected:
  // Run the closure in the new thread.
  void Run() override;

 private:
  Closure closure_;
};

// A worker pool runs a closure in a set of worker threads.
class WorkerPool {
 public:
  // A worker is a functional that takes the worker index as an argument.
  typedef std::function<void(int index)> Worker;

  // Start a number of threads and run the functional in each thread.
  void Start(int num_workers, const Worker &worker);

  // Wait for all workers to terminate.
  void Join();

  // Return the number of worker threads in the pool.
  int size() const { return workers_.size(); }

 private:
  // Worker threads.
  std::vector<ClosureThread> workers_;
};

}  // namespace sling

#endif  // SLING_UTIL_THREAD_H_

