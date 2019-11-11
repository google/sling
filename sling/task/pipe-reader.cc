// Copyright 2019 Google Inc.
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

#include <stdio.h>
#include <string>

#include "sling/base/logging.h"
#include "sling/base/types.h"
#include "sling/stream/input.h"
#include "sling/stream/unix-file.h"
#include "sling/task/process.h"
#include "sling/task/task.h"

namespace sling {
namespace task {

// Run command and output lines to channel.
class PipeReader : public Process {
 public:
  // Process input file.
  void Run(Task *task) override {
    // Get command.
    string command = task->Get("command", "");

    // Get output channel.
    Channel *output = task->GetSink("output");
    if (output == nullptr) {
      LOG(ERROR) << "No output channel";
      return;
    }

    // Run command.
    int buffer_size = task->Get("buffer_size", 1 << 16);
    FILE *pipe = popen(command.c_str(), "r");
    if (pipe == nullptr) {
      LOG(ERROR) << "Error running command: " << command;
      return;
    }
    StdFileInputStream stream(pipe, false, buffer_size);
    Input input(&stream);

    // Read lines from output of program and output to output channel.
    string line;
    while (input.ReadLine(&line)) {
      // Send message with line to output channel.
      output->Send(new Message(Slice(), Slice(line)));
    }

    // Close pipe and output channel.
    int status = pclose(pipe);
    CHECK(WIFEXITED(status)) << status;
    CHECK_EQ(WEXITSTATUS(status), 0);
    output->Close();
  }
};

REGISTER_TASK_PROCESSOR("pipe-reader", PipeReader);

}  // namespace task
}  // namespace sling

