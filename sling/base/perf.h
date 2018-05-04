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

#ifndef SLING_BASE_PERF_H_
#define SLING_BASE_PERF_H_

#include "sling/base/types.h"

namespace sling {

// Performance and resource usage statistics.
class Perf {
 public:
  // Sample resource status.
  void Sample();

  // User CPU time used (microseconds).
  int64 utime() const { return utime_; }

  // System CPU time used (microseconds).
  int64 stime() const { return stime_; }

  // CPU time used (microseconds).
  int64 cputime() const { return utime_ + stime_; }

  // Memory usage (bytes).
  int64 memory() const { return memory_; }

  // I/O read rate (bytes/sec).
  int64 ioread() const { return ioread_; }

  // I/O write rate (bytes/sec).
  int64 iowrite() const { return iowrite_; }

  // I/O rate (bytes/sec).
  int64 io() const { return ioread_ + iowrite_; }

  // Number of floating-point operations.
  int64 flops() const { return flops_; }

  // CPU temperature (celsius).
  float cputemp() const { return cputemp_; }

  // Peak memory usage.
  static int64 peak_memory_usage() { return peak_memory; }

  // Peak CPU temperature.
  static float peak_cpu_temperature() { return peak_cputemp; }

  // Address of FLOP counter (used by Myelin instrumentation).
  static int64 *flopptr() { return &flop; }

 private:
  int64 utime_;        // user CPU time used (microseconds)
  int64 stime_;        // system CPU time used (microseconds)
  int64 memory_;       // memory used (bytes)
  int64 ioread_;       // I/O read rate (bytes/sec)
  int64 iowrite_;      // I/O write rate (bytes/sec)
  int64 flops_;        // floating-point operations
  float cputemp_;      // CPU temperature (celsius)

  // Peak memory usage (bytes).
  static int64 peak_memory;

  // Peak CPU temperature (celsius).
  static float peak_cputemp;

  // FLOP counter (updated by Myelin).
  static int64 flop;
};

}  // namespace sling

#endif  // SLING_BASE_PERF_H_

