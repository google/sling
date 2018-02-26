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

#include "sling/task/dashboard.h"

#include <time.h>
#include <unistd.h>
#include <sys/resource.h>
#include <sstream>

namespace sling {
namespace task {

Dashboard::Dashboard() {
  start_time_ = time(0);
  File::Match("/sys/class/thermal/thermal_zone*/temp", &thermal_devices_);
}

Dashboard::~Dashboard() {
  for (auto *j : jobs_) delete j;
}

void Dashboard::Register(HTTPServer *http) {
  http->Register("/status", this, &Dashboard::HandleStatus);
  app_.Register(http);
}

string Dashboard::GetStatus() {
  MutexLock lock(&mu_);
  std::stringstream out;

  // Output current time and status.
  bool running = (status_ < FINAL);
  out << "{\"time\":" << (running ? time(0) : end_time_);
  out << ",\"started\":" << start_time_;
  out << ",\"finished\":" << (running ? 0 : 1);

  // Output jobs.
  out << ",\"jobs\":[";
  bool first_job = true;
  for (JobStatus *status : jobs_) {
    if (!first_job) out << ",";
    first_job = false;
    out << "{\"name\":\"" << status->name << "\"";
    out << ",\"started\":" << status->started;
    if (status->ended != 0) out << ",\"ended\":" << status->ended;

    if (status->job != nullptr) {
      // Output stages for running job.
      out << ",\"stages\":[";
      bool first_stage = true;
      for (Stage *stage : status->job->stages()) {
        if (!first_stage) out << ",";
        first_stage = false;
        out << "{\"tasks\":" << stage->num_tasks()
            << ",\"done\":" << stage->num_completed_tasks() << "}";
      }
      out << "],";

      // Output counters for running job.
      out << "\"counters\":{";
      bool first_counter = true;
      status->job->IterateCounters(
        [&out, &first_counter](const string &name, Counter *counter) {
          if (!first_counter) out << ",";
          first_counter = false;
          out << "\"" << name << "\":" << counter->value();
        }
      );
      out << "}";
    } else {
      // Output counters for completed job.
      out << ",\"counters\":{";
      bool first_counter = true;
      for (auto &counter : status->counters) {
        if (!first_counter) out << ",";
        first_counter = false;
        out << "\"" << counter.first << "\":" << counter.second;
      }
      out << "}";
    }
    out << "}";
  }
  out << "]";

  // Output resource usage.
  struct rusage ru;
  if (getrusage(RUSAGE_SELF, &ru) == 0) {
    int64 utime = ru.ru_utime.tv_sec * 1000000LL + ru.ru_utime.tv_usec;
    int64 stime = ru.ru_stime.tv_sec * 1000000LL + ru.ru_stime.tv_usec;
    int64 mem = ru.ru_maxrss * 1024LL;
    if (mem > peak_memory_) peak_memory_ = mem;
    if (!running) mem = peak_memory_;
    int64 ioread = ru.ru_inblock;
    int64 iowrite = ru.ru_oublock;
    out << ",\"utime\":" << utime;
    out << ",\"stime\":" << stime;
    out << ",\"mem\":" << mem;
    out << ",\"ioread\":" << ioread;
    out << ",\"iowrite\":" << iowrite;
  }

  // Get CPU temperature.
  float temperature = GetCPUTemperature();
  if (temperature > 0.0) {
    if (temperature > peak_temp_) peak_temp_ = temperature;
    if (!running) temperature = peak_temp_;
    out << ",\"temperature\":" << temperature;
  }

  out << "}";
  return out.str();
}

void Dashboard::HandleStatus(HTTPRequest *request, HTTPResponse *response) {
  response->SetContentType("text/json; charset=utf-8");
  response->Append(GetStatus());
  if (status_ == IDLE) status_ = MONITORED;
  if (status_ == FINAL) status_ = SYNCHED;
}

void Dashboard::OnJobStart(Job *job) {
  MutexLock lock(&mu_);

  // Add job to job list.
  JobStatus *status = new JobStatus(job);
  jobs_.push_back(status);
  status->name = job->name();
  status->started = time(0);
  active_jobs_[job] = status;
}

void Dashboard::OnJobDone(Job *job) {
  MutexLock lock(&mu_);

  // Get job status.
  JobStatus *status = active_jobs_[job];
  CHECK(status != nullptr);

  // Record job completion time.
  status->ended = time(0);

  // Update final counter values.
  job->IterateCounters([status](const string &name, Counter *counter) {
    status->counters.emplace_back(name, counter->value());
  });

  // Remove job from active job list.
  active_jobs_.erase(job);
  status->job = nullptr;
}

void Dashboard::Finalize(int timeout) {
  if (status_ == MONITORED) {
    // Signal that all jobs are done.
    end_time_ = time(0);
    status_ = FINAL;

    // Wait until final status has been sent back.
    for (int wait = 0; wait < timeout && status_ != SYNCHED; ++wait) sleep(1);
  }
  status_ = TERMINAL;
}

float Dashboard::GetCPUTemperature() {
  float temp = 0.0;
  for (const string &dev : thermal_devices_) {
    string data;
    if (File::ReadContents(dev, &data)) {
      float value = std::stoi(data) / 1000.0;
      if (value > temp) temp = value;
    }
  }
  return temp;
}

}  // namespace task
}  // namespace sling

