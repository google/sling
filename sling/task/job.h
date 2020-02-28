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

#ifndef SLING_TASK_JOB_H_
#define SLING_TASK_JOB_H_

#include <condition_variable>
#include <functional>
#include <string>
#include <unordered_map>
#include <vector>

#include "sling/base/flags.h"
#include "sling/base/types.h"
#include "sling/task/environment.h"
#include "sling/task/message.h"
#include "sling/task/task.h"
#include "sling/util/mutex.h"
#include "sling/util/threadpool.h"

DECLARE_int32(event_manager_threads);
DECLARE_int32(event_manager_queue_size);

namespace sling {
namespace task {

class Monitor;

// A stage is a set of tasks that can be run concurrently. A stage can have
// dependencies on other stages, which must have completed before this stage
// can run.
class Stage : public AssetManager {
 public:
  // Stage states.
  enum State {WAITING, READY, RUNNING, DONE};

  Stage(int index) : index_(index) {}

  // Add task to stage.
  void AddTask(Task *task);

  // Add stage dependency.
  void AddDependency(Stage *dependency);

  // Check if stage is ready to run, i.e. all the dependent stages are done.
  bool Ready();

  // Mark stage as ready for running.
  void MarkReady();

  // Start tasks in stage.
  void Start();

  // Notification that task in stage has completed.
  void TaskCompleted(Task *task);

  // Stage index.
  int index() const { return index_; }

  // Stage state.
  State state() const { return state_; }

  // Return list of tasks in stage.
  const std::vector<Task *> &tasks() const { return tasks_; }

  // Number of tasks in stage.
  int num_tasks() const { return tasks_.size(); }

  // Number of completed tasks in stage.
  int num_completed_tasks() const { return num_completed_; }

 private:
  // Stage index.
  int index_;

  // Stage state.
  State state_ = WAITING;

  // Tasks in stage.
  std::vector<Task *> tasks_;

  // Other stages that this stage depend on.
  std::vector<Stage *> dependencies_;

  // Number of tasks in stage that have completed.
  int num_completed_ = 0;
};

// A job manages a set of tasks with inputs and outputs. These tasks are
// connected by channels which allow the tasks to communicate.
class Job : public Environment {
 public:
  // Initialize job.
  Job();

  // Destroy job.
  ~Job();

  // Register monitor for job.
  void RegisterMonitor(Monitor *monitor) { monitor_ = monitor; }

  // Create a new resource for the job.
  Resource *CreateResource(const string &filename,
                           const Format &format,
                           const Shard &shard = Shard());

  // Create new resources for the job. If the filename contains wild cards
  // or has a @n specifier, a set of sharded resources are returned.
  std::vector<Resource *> CreateResources(const string &filename,
                                          const Format &format);

  // Create a set of sharded resources.
  std::vector<Resource *> CreateShardedResources(const string &basename,
                                                 int shards,
                                                 const Format &format);

  // Create a new channel for the job.
  Channel *CreateChannel(const Format &format);
  std::vector<Channel *> CreateChannels(const Format &format, int shards);

  // Create a new task for the job.
  Task *CreateTask(const string &type,
                   const string &name,
                   const Shard &shard = Shard());

  // Create a set of sharded tasks.
  std::vector<Task *> CreateTasks(const string &type,
                                  const string &name,
                                  int shards);

  // Connect producer to consumer with a channel.
  Channel *Connect(const Port &producer,
                   const Port &consumer,
                   const Format &format);
  Channel *Connect(Task *producer,
                   Task *consumer,
                   const string format) {
    return Connect(Port(producer, "output"),
                   Port(consumer, "input"),
                   Format("message", format));
  }

  // Bind resource to input.
  Binding *BindInput(Task *task, Resource *resource,
                     const string &input);

  // Bind resource to output.
  Binding *BindOutput(Task *task, Resource *resource,
                      const string &output);

  // Initialize and start tasks.
  void Start();

  // Wait for all tasks to complete.
  void Wait();

  // Wait for all tasks to complete with timeout. Return false on timeout.
  bool Wait(int ms);

  // Check if all tasks have completed.
  bool Done();

  // Iterate counters.
  void IterateCounters(
      std::function<void(const string &name, Counter *counter)> callback);

  // Dump counters to log.
  void DumpCounters();

  // Task environment interface.
  Counter *GetCounter(const string &name) override;
  void ChannelCompleted(Channel *channel) override;
  void TaskCompleted(Task *task) override;

  // List of stages in job.
  const std::vector<Stage *> stages() const { return stages_; }

  // Job name.
  const string &name() const { return name_; }
  void set_name(const string &name) { name_ = name; }

 private:
  // Build stages for job.
  void BuildStages();

  // Job name.
  string name_;

  // List of tasks in container indexed by id.
  std::vector<Task *> tasks_;

  // List of channels in container indexed by id.
  std::vector<Channel *> channels_;

  // List of resources registered in container.
  std::vector<Resource *> resources_;

  // List of stages for running the tasks.
  std::vector<Stage *> stages_;

  // Statistics counters.
  std::unordered_map<string, Counter *> counters_;

  // Worker queue for event dispatching.
  ThreadPool *event_dispatcher_;

  // Optional monitor for job.
  Monitor *monitor_ = nullptr;

  // Mutex for protecting job state.
  Mutex mu_;

  // Signal that all tasks have completed.
  std::condition_variable completed_;
};

// A workflow can be monitored by a Monitor component. If a monitor is
// registered using Job::RegisterMonitor(), the monitor is notified when the
// job is started and when it completes.
class Monitor {
 public:
  virtual ~Monitor() = default;

  // Callback to notify when a job is started.
  virtual void OnJobStart(Job *job) = 0;

  // Callback to notify when a job completes.
  virtual void OnJobDone(Job *job) = 0;
};

}  // namespace task
}  // namespace sling

#endif  // SLING_TASK_JOB_H_

