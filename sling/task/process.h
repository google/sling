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

#include <atomic>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <vector>
#include <unordered_map>

#include "sling/task/task.h"
#include "sling/util/mutex.h"
#include "sling/util/thread.h"

namespace sling {
namespace task {

class Queue;

// A task process runs the task in a separate thread.
class Process : public Processor {
 public:
  // Delete thread.
  ~Process() override { delete thread_; }

  // Start task thread.
  void Start(Task *task) override;

  // Wait for thread to finish.
  void Done(Task *task) override;

  // Receive message on channel and dispatch to queue.
  void Receive(Channel *channel, Message *message) override;

  // Unsubscribe queue from channel when it is closed.
  void Close(Channel *channel) override;

  // This method is run in a separate thread when the task is started.
  virtual void Run(Task *task) = 0;

 private:
  // Get queue for messages arriving on channel. Wait for subscriber if there
  // there are currently no queue subscribing on channel.
  Queue *GetQueue(Channel *channel);

  // Subscribe queue to receive messages from channel.
  void Subscribe(Queue *queue, Channel *channel);

  // Unsubscribe all channels from queue.
  void Unsubscribe(Queue *queue);

  // Thread for running task.
  ClosureThread *thread_ = nullptr;

  // Mapping from channels to queues listening on the channel.
  std::unordered_map<Channel *, Queue *> queues_;

  // Signal to notify about new subscribers.
  std::condition_variable subscribe_;

  // Mutex for serializing access to queues.
  Mutex mu_;

  friend class Queue;
};

// Queue for receiving messages from one or more channels.
class Queue {
 public:
  // Initialize queue for listening on channel(s).
  Queue(Process *owner, Channel *channel, int size = 64);
  Queue(Process *owner, const std::vector<Channel *> &channels, int size = 64);

  // Unsubscribe queue from channel(s).
  ~Queue();

  // Write message from channel to queue.
  void Write(Message *message, Channel *channel);

  // Read message from queue or return false when channel(s) have been closed.
  bool Read(Message **message, Channel **channel);
  bool Read(Message **message) { return Read(message, nullptr); }

 private:
  // Notification about one of the monitored channels begin closed.
  void OnClose(Channel *channel);

  // Queue element.
  struct Element {
    Element(Message *msg, Channel *ch) : message(msg), channel(ch) {}
    Message *message;
    Channel *channel;
  };

  // Process that owns the queue.
  Process *owner_;

  // Number of active channels for queue.
  std::atomic<int> channels_;

  // Maximum number of elements in queue.
  int size_;

  // Message queue.
  std::deque<Element> queue_;

  // Mutex for serializing access to queue.
  Mutex mu_;

  // Signal to notify about new messages in queue.
  std::condition_variable nonempty_;

  // Signal to notify about available space in queue.
  std::condition_variable nonfull_;

  friend class Process;
};

}  // namespace task
}  // namespace sling

#endif  // SLING_TASK_PROCESS_H_

