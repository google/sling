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

#ifndef SLING_BASE_CLOCK_H_
#define SLING_BASE_CLOCK_H_

#include "sling/base/types.h"

namespace sling {

// Cycle-counting clock for performance measurements.
class Clock {
 public:
  // TSC timestamp.
  typedef int64_t Timestamp;

  // Return timestamp from cycle counter.
  static inline Timestamp now() {
    uint64_t low, high;
    __asm__ volatile("rdtsc" : "=a"(low), "=d"(high));
    return (high << 32) | low;
  }

  // Return clock speed in Hz.
  static double hz();

  // Return clock speed in MHz.
  static double mhz();

  // Start clock.
  void start() { start_ = now(); }

  // Stop clock.
  void stop() { end_ = now(); }

  // Return clock cycles elapsed since start.
  Timestamp elapsed() const { return now() - start_; }

  // Return clock cycles between start and stop.
  Timestamp cycles() const { return end_ - start_; }

  // Return time in seconds.
  double secs() const;

  // Return time in milliseconds.
  double ms() const;

  // Return time in microseconds.
  double us() const;

  // Return time in nanoseconds.
  double ns() const;

 private:
  Timestamp start_;  // start timestamp
  Timestamp end_;    // end timestamp
};

}  // namespace sling

#endif  // SLING_BASE_CLOCK_H_

