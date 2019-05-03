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

#include "sling/base/perf.h"

#include <glob.h>
#include <stdio.h>
#include <unistd.h>
#include <sys/resource.h>
#include <string>
#include <vector>

#include "sling/base/init.h"
#include "sling/base/types.h"

namespace sling {

// Global performance status.
int64 Perf::flop = 0;
int64 Perf::peak_memory = 0;
float Perf::peak_cputemp = 0;

// Thermal devices for measuring CPU temperature.
static std::vector<string> thermal_devices;

// Page size.
static int page_size = 4096;

// Get memory usage for process.
static int64 GetMemoryUsage() {
  FILE *f = fopen("/proc/self/statm", "r");
  if (!f) return 0;
  char buffer[64];
  int64 ram = atoi(fgets(buffer, 64, f));
  fclose(f);
  return ram * page_size;
}

void Perf::Sample() {
  // Get resource usage for process.
  struct rusage ru;
  if (getrusage(RUSAGE_SELF, &ru) == 0) {
    utime_ = ru.ru_utime.tv_sec * 1000000LL + ru.ru_utime.tv_usec;
    stime_ = ru.ru_stime.tv_sec * 1000000LL + ru.ru_stime.tv_usec;
    ioread_ = ru.ru_inblock;
    iowrite_ = ru.ru_oublock;
  } else {
    utime_ = stime_ = ioread_ = iowrite_ = 0;
  }

  // Get memory usage.
  memory_ = GetMemoryUsage();
  if (memory_ > peak_memory) peak_memory = memory_;

  // Get flops_FLOP counter value.
  flops_ = flop;

  // Get CPU temperature (warmest thermal zone).
  cputemp_ = 0.0;
  for (const string &dev : thermal_devices) {
    char buffer[32];
    FILE *f = fopen(dev.c_str(), "r");
    if (!f) continue;
    float temp = atof(fgets(buffer, 32, f)) / 1000.0;
    fclose(f);
    if (temp > cputemp_) cputemp_ = temp;
  }
  if (cputemp_ > peak_cputemp) peak_cputemp = cputemp_;
}

static void InitPerf() {
  // Get memory page size.
  page_size = sysconf(_SC_PAGESIZE);

  // Get proc files for thermal zones.
  glob_t globbuf;
  glob("/sys/class/thermal/thermal_zone*/temp", 0, nullptr, &globbuf);
  for (int i = 0; i < globbuf.gl_pathc; ++i) {
    thermal_devices.push_back(globbuf.gl_pathv[i]);
  }
  globfree(&globbuf);
}

REGISTER_INITIALIZER(perf, { InitPerf(); });

}  // namespace sling

