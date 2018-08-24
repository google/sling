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

#include "sling/task/process.h"

#include "sling/util/thread.h"

namespace sling {
namespace task {

void Process::Start(Task *task) {
  // Add task reference to keep task alive while the task thread is running.
  task->AddRef();

  // Execute the Run() method in a new thread.
  thread_ = new ClosureThread([this, task]() {
    // Run task.
    Run(task);

    // Release task reference to allow task to complete.
    task->Release();
  });

  // Start task thread.
  thread_->SetJoinable(true);
  thread_->Start();
}

void Process::Done(Task *task) {
  // Wait for task thread to finish.
  if (thread_ != nullptr) thread_->Join();
}

Queue *Process::GetQueue(Channel *channel) {
  std::unique_lock<std::mutex> lock(mu_);
  for (;;) {
    // Try to find subscriber.
    auto f = queues_.find(channel);
    if (f != queues_.end()) return f->second;

    // Wait for subscriber.
    subscribe_.wait(lock);
  }
}

void Process::Receive(Channel *channel, Message *message) {
  GetQueue(channel)->Write(message, channel);
}

void Process::Close(Channel *channel) {
  GetQueue(channel)->OnClose(channel);
}

void Process::Subscribe(Queue *queue, Channel *channel) {
  MutexLock lock(&mu_);

  // Add subscriber.
  queues_[channel] = queue;

  // Notify all threads waiting for new subscribers.
  subscribe_.notify_all();
}

void Process::Unsubscribe(Queue *queue) {
  MutexLock lock(&mu_);
  for (auto it = queues_.begin(); it != queues_.end();) {
    if (it->second == queue) {
      it = queues_.erase(it);
    } else {
      ++it;
    }
  }
}

Queue::Queue(Process *owner, Channel *channel, int size) {
  channels_ = 1;
  size_ = size;
  owner_ = owner;
  owner->Subscribe(this, channel);
}

Queue::Queue(Process *owner, const std::vector<Channel *> &channels, int size) {
  channels_ = channels.size();
  size_ = size;
  owner_ = owner;
  for (Channel *channel : channels) owner->Subscribe(this, channel);
}

Queue::~Queue() {
  owner_->Unsubscribe(this);
}

void Queue::Write(Message *message, Channel *channel) {
  std::unique_lock<std::mutex> lock(mu_);
  while (queue_.size() >= size_) {
    nonfull_.wait(lock);
  }
  queue_.emplace_back(message, channel);
  nonempty_.notify_one();
}

bool Queue::Read(Message **message, Channel **channel) {
  std::unique_lock<std::mutex> lock(mu_);
  while (queue_.empty()) nonempty_.wait(lock);
  *message = queue_.front().message;
  if (channel) *channel = queue_.front().channel;
  queue_.pop_front();
  nonfull_.notify_one();
  return *message != nullptr;
}

void Queue::OnClose(Channel *channel) {
  if (--channels_ == 0) {
    // All channels have been closed; notify reader by posting a null message.
    Write(nullptr, nullptr);
  }
}

}  // namespace task
}  // namespace sling

