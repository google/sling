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

#ifndef SLING_TASK_DASHBOARD_H_
#define SLING_TASK_DASHBOARD_H_

#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "sling/base/types.h"
#include "sling/http/http-server.h"
#include "sling/http/static-content.h"
#include "sling/task/job.h"
#include "sling/util/mutex.h"

namespace sling {
namespace task {

// Dashboard for monitoring jobs.
class Dashboard : public Monitor {
 public:
  enum Status {
    IDLE,        // dashboard is idle when jobs are not being monitored
    MONITORED,   // job status has been requested by client
    FINAL,       // all jobs has completed
    SYNCHED,     // final status has been sent to client
    TERMINAL,    // dashboard ready for shutdown
  };

  // List of counters.
  typedef std::vector<std::pair<string, int64>> CounterList;

  Dashboard();
  ~Dashboard();

  // Register job status service.
  void Register(HTTPServer *http);

  // Wait until final status has been reported or timeout (in seconds).
  void Finalize(int timeout);

  // Get job status in JSON format.
  string GetStatus();

  // Handle job status queries.
  void HandleStatus(HTTPRequest *request, HTTPResponse *response);

  // Job monitor interface.
  void OnJobStart(Job *job) override;
  void OnJobDone(Job *job) override;

 private:
  // Status for active or completed job.
  struct JobStatus {
    JobStatus(Job *job) : job(job) {}
    Job *job;               // job object or null if job has completed
    string name;            // job name
    int64 started = 0;      // job start time
    int64 ended = 0;        // job completion time
    CounterList counters;   // final job counters
  };

  // List of active and completed jobs.
  std::vector<JobStatus *> jobs_;

  // Map of active jobs.
  std::unordered_map<Job *, JobStatus *> active_jobs_;

  // Dashboard monitoring status. This is used for the final handshake to
  // report the final status of the jobs before termination.
  Status status_ = IDLE;

  // Dashboard app.
  StaticContent app_{"/", "sling/task/app"};

  // Start time.
  int64 start_time_;

  // Completion time.
  int64 end_time_;

  // Mutex for serializing access to dashboard.
  Mutex mu_;
};

}  // namespace task
}  // namespace sling

#endif  // SLING_TASK_DASHBOARD_H_

