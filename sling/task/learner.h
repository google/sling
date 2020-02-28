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

#ifndef SLING_TASK_LEARNER_H_
#define SLING_TASK_LEARNER_H_

#include <atomic>

#include "sling/base/types.h"
#include "sling/myelin/learning.h"
#include "sling/task/process.h"
#include "sling/task/task.h"
#include "sling/util/mutex.h"

namespace sling {
namespace task {

// Task for training models using multiple workers.
class LearnerTask : public Process {
 public:
  // Run training using workers.
  void Train(Task *task, myelin::Network *model);

  // Signal completion of training epoch. Return true when training is done.
  bool EpochCompleted();

  // Worker thread for training model.
  virtual void Worker(int index, myelin::Network *model) = 0;

  // Model evaluation. Return false to end training.
  virtual bool Evaluate(int64 epoch, myelin::Network *model) { return true; }

  // Checkpoint model.
  virtual void Checkpoint(int64 epoch, myelin::Network *model) {}

 private:
  // Total number of training epochs.
  int64 epochs_ = 10000;

  // Number of epochs between model evaluation.
  int64 report_interval_ = 100;

  // Number of seconds between starting up workers.
  int64 rampup_ = 0;

  // Number of seconds between starting the first and the rest of the workers.
  int64 warmup_ = 0;

  // Current number of completed epochs.
  std::atomic<int64> epoch_{0};

  // Flag set when training is done.
  bool done_ = false;

  // Signal model evaluation or completions.
  Mutex eval_mu_;
  std::condition_variable eval_model_;

  // Checkpoint model at regular intervals.
  int checkpoint_interval_ = 100000;
  int last_checkpoint_ = 0;

  // Staticstics.
  Counter *num_workers_ = nullptr;
  Counter *num_epochs_ = nullptr;
};

// Initialize optimizer from task parameters.
myelin::Optimizer *GetOptimizer(Task *task);

}  // namespace task
}  // namespace sling

#endif  // SLING_TASK_LEARNER_H_
