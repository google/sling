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

#include "sling/base/clock.h"

#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include "sling/base/init.h"
#include "sling/base/types.h"

namespace sling {

static double cycles_per_second;
static double cycles_per_millisec;
static double cycles_per_microsec;
static double cycles_per_nanosec;

double Clock::hz() { return cycles_per_second; }
double Clock::mhz() { return cycles_per_microsec; }

double Clock::secs() const { return cycles() / cycles_per_second; }
double Clock::ms() const { return cycles() / cycles_per_millisec; }
double Clock::us() const { return cycles() / cycles_per_microsec; }
double Clock::ns() const { return cycles() / cycles_per_nanosec; }

static void InitClock() {
  // Try to get clock speed from OS.
  static const char *mhz_files[] = {
    "/sys/devices/system/cpu/cpu0/tsc_freq_khz",
    "/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq",
    nullptr,
  };
  bool found = false;
  for (const char **fn = mhz_files; !found && *fn; ++fn) {
    int fd = open(*fn, O_RDONLY);
    if (fd == -1) continue;
    char line[1024];
    memset(line, '\0', sizeof(line));
    if (read(fd, line, sizeof(line) - 1) > 4) {
      cycles_per_second = atoi(line) * 1000.0;
      if (cycles_per_second > 0) found = true;
    }
    close(fd);
  }

  // Estimate cycles per second with a timed loop.
  if (!found) {
    int64_t start = Clock::now();
    clock_t in_one_sec = clock() + CLOCKS_PER_SEC;
    while (clock() < in_one_sec);
    int64_t end = Clock::now();
    cycles_per_second = end - start;
  }

  // Compute cycles per millisecond, microsecond, and nanosecond.
  cycles_per_millisec = cycles_per_second / 1000.0;
  cycles_per_microsec = cycles_per_second / 1000000.0;
  cycles_per_nanosec = cycles_per_second / 1000000000.0;
}

REGISTER_INITIALIZER(clock, {
  InitClock();
});

}  // namespace sling

