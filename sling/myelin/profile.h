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

#ifndef SLING_MYELIN_PROFILE_H_
#define SLING_MYELIN_PROFILE_H_

#include <string>
#include <vector>

#include "sling/base/clock.h"
#include "sling/base/types.h"
#include "sling/myelin/compute.h"

namespace sling {
namespace myelin {

class Profile {
 public:
  // Sort order for steps.
  enum Order {POSITION, TIME, GFLOPS, COMPLEXITY, KERNEL, NAME, TASK};

  // Initialize profile for summary.
  Profile(ProfileSummary *summary, Order order = POSITION);

  // Initialize profile for cell instance.
  Profile(Instance *instance, Order order = POSITION);

  // Cell for profiled instance.
  const Cell *cell() const { return cell_; }

  // Check if profiling has been enabled.
  bool enabled() const { return timing_ != nullptr; }

  // Number of steps in the cell.
  int steps() const { return steps_.size(); }

  // Number of tasks in the cell.
  int tasks() const { return enabled() ? cell()->num_tasks() : 0; }

  // Number of invocations of cell.
  int64 invocations() const { return invocations_; }

  // Raw CPU cycle counts for step.
  int64 timing(int idx) const { return timing_[steps_[idx].index]; }

  // Return step in the cell computation.
  const Step *step(int idx) const { return steps_[idx].step; }

  // Number of CPU cycles used per invocation.
  int64 cycles() const {
    return invocations_ > 0 ? total_ / invocations_ : 0;
  }

  // Time in microseconds per invocation.
  double time() const {
    return invocations_ > 0 ? total_ / (Clock::mhz() * invocations_) : 0;
  }

  // Number of CPU cycles per invocation used by step in cell computation.
  int64 cycles(int idx) const {
    return invocations_ > 0 ? timing(idx) / invocations_ : 0;
  }

  // Time per invocation in microseconds used by step in cell computation.
  double time(int idx) const {
    return invocations_ > 0 ? timing(idx) / (Clock::mhz() * invocations_) : 0;
  }

  // Percentage of time used by step in cell computation.
  double percent(int idx) const {
    return invocations_ > 0 && total_ > 0 ? time(idx) / time() * 100 : 0;
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

  // Overhead time per invocation.
  double overhead_time() const {
    return invocations_ > 0 ? overhead_ / (Clock::mhz() * invocations_) : 0;
  }

  // Percentage of time used by overhead in cell computation.
  double overhead_percent() const {
    if (invocations_ == 0 || total_ == 0) return 0.0;
    return overhead_time() / time() * 100;
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
  // Initialize profile.
  void Initialize(int64 *data, Order order);

  // Tasks have start and wait cycle counts.
  struct TaskTiming {
    int64 start;
    int64 wait;
  };

  // Step information.
  struct StepInfo {
    int index;              // position in execution order
    const Step *step;       // step in cell
    string sort_name;       // name used for sorting
    float sort_value = 0;   // sort value

    // Comparison operator for sorting steps.
    bool operator <(const StepInfo &other) const {
      if (sort_value < other.sort_value) return true;
      if (sort_value > other.sort_value) return false;
      if (sort_name  < other.sort_name) return true;
      if (sort_name  > other.sort_name) return false;
      return index < other.index;
    }
  };

  // Cell for profile.
  const Cell *cell_ = nullptr;

  // Number of invocations.
  int64 invocations_ = 0;

  // Overhead.
  int64 overhead_ = 0;

  // Total number of cycles for computation.
  int64 total_ = 0;

  // Total number of operations for computation.
  int64 total_complexity_ = 0;

  // Array of clock cycle counts for each step or null if profiling is not
  // enabled.
  int64 *timing_ = nullptr;

  // Array of clock cycle counts for each task.
  TaskTiming *tasks_ = nullptr;

  // Sorted step information.
  std::vector<StepInfo> steps_;
};

class ProfileOverview {
 public:
  // Add profile information for cell.
  void Add(const Profile &profile) {
    cells_.emplace_back(profile);
    total_time_ += profile.time() * profile.invocations();
  }

  // Profile summary report in ASCII format.
  string ASCIIReport() const;

 private:
  // Accumulated profile information for cell.
  struct CellInfo {
    CellInfo(const Profile &profile)
      : cell(profile.cell()),
        invocations(profile.invocations()),
        time(profile.time()) {}

    const Cell *cell;   // network cell
    int64 invocations;  // number of invocation of cell computation
    double time;        // execution time per invocation in microseconds
  };

  std::vector<CellInfo> cells_;  // profile information for each cell
  double total_time_ = 0.0;      // total execution time in microseconds
};

// Data profile for cell instance tensor allocation.
class DataProfile {
 public:
  DataProfile(Cell *cell) : cell_(cell) {}

  string AsSVG();

 private:
  Cell *cell_;
};

// Log profile report if profiling enabled.
void LogProfile(const Network &net);

// Return profile report if profiling enabled.
string ProfileReport(const Network &net);

}  // namespace myelin
}  // namespace sling

#endif  // SLING_MYELIN_PROFILE_H_

