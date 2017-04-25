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

#include "myelin/profile.h"

#include "base/types.h"
#include "string/printf.h"
#include "third_party/jit/cpu.h"

namespace sling {
namespace myelin {

static const char *divider = "+---------+-------------+------------+------"
                             "----------------------+---+-------------------\n";
static const char *header = "| percent |     time    |     cycles | kernel"
                            "                     | t | step\n";

Profile::Profile(Instance *instance) : instance_(instance) {
  if (cell()->profile() != nullptr) {
    // First element is evocation count followed by one cycle counter for each
    // step.
    int64 *data = instance_->Get<int64>(cell()->profile());
    invocations_ = *data;
    timing_ = data + 1;
    total_ = 0;
    for (int i = 0; i < steps(); ++i) total_ += timing_[i];
    tasks_ = reinterpret_cast<TaskTiming *>(timing_ + steps());
  } else {
    invocations_ = 0;
    total_ = 0;
    timing_ = nullptr;
    tasks_ = nullptr;
  }
}

string Profile::ASCIIReport() const {
  // Check if profiling has been enabled.
  if (!enabled()) return "No profile";

  // Output title.
  jit::ProcessorInformation cpu;
  string report;
  StringAppendF(&report,
      "Profile for %lld invocations of %s\n",
      invocations_,
      instance_->cell()->name().c_str());
  StringAppendF(&report, "CPU model: %s\n", cpu.brand());
  StringAppendF(&report,
      "CPU architecture: %s (family %02x model %02x stepping %02x)\n",
      cpu.architecture(),
      cpu.family(), cpu.model(), cpu.stepping());

  report.append("CPU features:");
  if (jit::CPU::Enabled(jit::MMX)) report.append(" MMX");
  if (jit::CPU::Enabled(jit::SSE)) report.append(" SSE");
  if (jit::CPU::Enabled(jit::SSE2)) report.append(" SSE2");
  if (jit::CPU::Enabled(jit::SSE3)) report.append(" SSE3");
  if (jit::CPU::Enabled(jit::SSE4_1)) report.append(" SSE4.1");
  if (jit::CPU::Enabled(jit::SSE4_2)) report.append(" SSE4.2");
  if (jit::CPU::Enabled(jit::AVX)) report.append(" AVX");
  if (jit::CPU::Enabled(jit::AVX2)) report.append(" AVX2");
  if (jit::CPU::Enabled(jit::FMA3)) report.append(" FMA3");
  report.append("\n\n");

  // Output header.
  report.append(divider);
  report.append(header);
  report.append(divider);

  // Output profile for each step.
  for (int i = 0; i < steps(); ++i) {
    string tid;
    if (step(i)->task_index() != -1) {
      tid = StringPrintf("%2d", step(i)->cell()->task(step(i)->task_index()));
    }
    StringAppendF(&report, "| %6.2f%% | %8.3f us | %10lld | %-27s|%-2s | %s\n",
                  percent(i), time(i), cycles(i),
                  step(i)->kernel()->Name().c_str(),
                  tid.c_str(),
                  step(i)->name().c_str());
  }

  // Output totals.
  report.append(divider);
  StringAppendF(&report,
                "| 100.00%% | %8.3f us | %10lld | %-27s|   |\n",
                time(), cycles(), "TOTAL");
  report.append(divider);

  // Output task timing.
  if (tasks() > 0) {
    double total_start = 0.0;
    double total_wait = 0.0;
    report.append("\n");
    report.append("+-------|-------------+-------------+\n");
    report.append("|  task |  start time |   wait time |\n");
    report.append("+-------|-------------+-------------+\n");
    for (int i = 0; i < tasks(); ++i) {
      total_start += start_time(i);
      total_wait += wait_time(i);
      StringAppendF(&report, "| %5d | %8.3f us | %8.3f us |\n",
                    instance_->cell()->task(i),
                    start_time(i),
                    wait_time(i));
    }
    report.append("+-------|-------------+-------------+\n");
    StringAppendF(&report, "| TOTAL | %8.3f us | %8.3f us |\n",
                  total_start, total_wait);
    report.append("+-------|-------------+-------------+\n");

    double compute_time = total_start + total_wait;
    for (int i = 0; i < steps(); ++i) {
      if (step(i)->task_index() == -1) {
        compute_time += time(i);
      }
    }

    double parallelism = time() / compute_time;
    double efficiency = parallelism / (tasks() + 1);
    double rate = 1.0 / (compute_time / 1e6);
    StringAppendF(&report,
                  "\n%.3f us/invocation, %.0f Hz, parallelism %.3f, "
                  "%.2f%% efficiency\n",
                  compute_time,
                  rate,
                  parallelism,
                  efficiency * 100.0);
  }

  return report;
}

}  // namespace myelin
}  // namespace sling

