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

#include "sling/base/logging.h"
#include "sling/util/threadpool.h"

namespace sling {

ThreadPool::ThreadPool(int num_workers, int queue_size)
    : num_workers_(num_workers), queue_size_(queue_size) {}

ThreadPool::~ThreadPool() {
  // Wait until all tasks have been completed.
  Shutdown();

  // Wait until all workers have terminated.
  for (auto &t : workers_) t.Join();
}

void ThreadPool::StartWorkers() {
  // Create worker threads.
  CHECK(workers_.empty());
  for (int i = 0; i < num_workers_; ++i) {
    workers_.emplace_back([this]() {
      // Keep processing tasks until done.
      Task task;
      while (FetchTask(&task)) task();
    });
  }

  // Start worker threads.
  for (auto &t : workers_) {
    t.SetJoinable(true);
    t.Start();
  }
}

void ThreadPool::Schedule(Task &&task) {
  std::unique_lock<std::mutex> lock(mu_);
  while (tasks_.size() >= queue_size_) {
    nonfull_.wait(lock);
  }
  tasks_.push(std::move(task));
  nonempty_.notify_one();
}

bool ThreadPool::FetchTask(Task *task) {
  std::unique_lock<std::mutex> lock(mu_);
  while (tasks_.empty()) {
    if (done_) return false;
    nonempty_.wait(lock);
  }
  *task = tasks_.front();
  tasks_.pop();
  nonfull_.notify_one();
  return true;
}

void ThreadPool::Shutdown() {
  // Notify all threads that we are done.
  std::lock_guard<std::mutex> lock(mu_);
  done_ = true;
  nonempty_.notify_all();
}

}  // namespace sling

