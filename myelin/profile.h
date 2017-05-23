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

#ifndef MYELIN_PROFILE_H_
#define MYELIN_PROFILE_H_

#include <string>

#include "base/clock.h"
#include "base/types.h"
#include "myelin/compute.h"

namespace sling {
namespace myelin {

class Profile {
 public:
  // Initialize profile for cell instance.
  Profile(Instance *instance);

  // Cell for profiled instance.
  const Cell *cell() const { return instance_->cell(); }

  // Check if profiling has been enabled.
  bool enabled() const { return timing_ != nullptr; }

  // Number of steps in the cell.
  int steps() const { return enabled() ? cell()->steps().size() : 0; }

  // Number of tasks in the cell.
  int tasks() const { return enabled() ? cell()->num_tasks() : 0; }

  // Number of invocations of cell.
  int64 invocations() const { return invocations_; }

  // Raw CPU cycle counts for step.
  int64 timing(int idx) const { return timing_[idx]; }

  // Return step in the cell computation.
  const Step *step(int idx) const { return cell()->steps()[idx]; }

  // Number of CPU cycles used per invocation.
  int64 cycles() const {
    return invocations_ ? total_ / invocations_ : 0;
  }

  // Time in microseconds per invocation.
  double time() const {
    return invocations_ ? total_ / (Clock::mhz() * invocations_) : 0;
  }

  // Number of CPU cycles per invocation used by step in cell computation.
  int64 cycles(int idx) const {
    return invocations_ ? timing_[idx] / invocations_ : 0;
  }

  // Time per invocation in microseconds used by step in cell computation.
  double time(int idx) const {
    return invocations_ ? timing_[idx] / (Clock::mhz() * invocations_) : 0;
  }

  // Percentage of time used by step in cell computation.
  double percent(int idx) const {
    return invocations_ ? time(idx) / time() * 100 : 0;
  }

  // Estimated number of operations per invocation of step.
  int64 complexity(int idx) const { return Complexity(step(idx)); }

  // Estimated number of operations per computation.
  int64 complexity() const { return total_complexity_; }

  // Estimated number of operations per second for step.
  double gigaflops(int idx) const {
    int64 ops = complexity(idx);
    double t = time(idx);
    return ops == 0 || t == 0 ? 0 : ops / t / 1e3;
  }

  // Estimated number of operations per second for computation.
  double gigaflops() const {
    return complexity() == 0 || time() == 0 ? 0 : complexity() / time() / 1e3;
  }

  // Number of CPU cycles for starting task.
  int64 start_cycles(int tidx) const {
    return invocations_ ? tasks_[tidx].start / invocations_ : 0;
  }

  // Number of microseconds for starting task.
  double start_time(int tidx) const {
    if (invocations_ == 0) return 0.0;
    return tasks_[tidx].start / (Clock::mhz() * invocations_);
  }

  // Number of CPU cycles for task completion wait.
  int64 wait_cycles(int tidx) const {
    return invocations_ ? tasks_[tidx].wait / invocations_ : 0;
  }

  // Number of microseconds for task completion wait.
  double wait_time(int tidx) const {
    if (invocations_ == 0) return 0.0;
    return tasks_[tidx].wait / (Clock::mhz() * invocations_);
  }

  // Timing profile report in ASCII format.
  string ASCIIReport() const;

  // Estimate the number of operations performed by step.
  static int64 Complexity(const Step *step);

 private:
  // Tasks have start and wait cycle counts.
  struct TaskTiming {
    int64 start;
    int64 wait;
  };

  // Instance for profile.
  Instance *instance_;

  // Number of invocations.
  int64 invocations_;

  // Total number of cycles for computation.
  int64 total_;

  // Total number of operations for computation.
  int64 total_complexity_;

  // Array of clock cycle counts for each step or null if profiling is not
  // enabled.
  int64 *timing_;

  // Array of clock cycle counts for each task.
  TaskTiming *tasks_;
};

}  // namespace myelin
}  // namespace sling

#endif  // MYELIN_PROFILE_H_

