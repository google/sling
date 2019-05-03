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

#include <string>

#include "sling/base/logging.h"
#include "sling/file/recordio.h"
#include "sling/task/process.h"
#include "sling/task/task.h"

namespace sling {
namespace task {

// Read records from record file and output to channel.
class RecordFileReader : public Process {
 public:
  // Process input file.
  void Run(Task *task) override {
    // Get input file.
    Binding *input = task->GetInput("input");
    if (input == nullptr) {
      LOG(ERROR) << "No input resource";
      return;
    }

    // Get output channel.
    Channel *output = task->GetSink("output");
    if (output == nullptr) {
      LOG(ERROR) << "No output channel";
      return;
    }

    // Open input file.
    RecordFileOptions options;
    options.buffer_size = task->Get("buffer_size", options.buffer_size);
    RecordReader reader(input->resource()->name(), options);

    // Statistics counters.
    Counter *records_read = task->GetCounter("records_read");
    Counter *key_bytes_read = task->GetCounter("key_bytes_read");
    Counter *value_bytes_read = task->GetCounter("value_bytes_read");

    // The "limit" parameter can be used to limit the number of records read.
    int64 limit = -1;
    task->Fetch("limit", &limit);

    // Read records from file and output to output channel.
    Record record;
    while (!reader.Done()) {
      // Read record.
      CHECK(reader.Read(&record))
          << ", file: " << input->resource()->name()
          << ", position: " << reader.Tell();

      // Update stats.
      records_read->Increment();
      key_bytes_read->Increment(record.key.size());
      value_bytes_read->Increment(record.value.size());

      // Send message with record to output channel.
      Message *message = new Message(record.key, record.value);
      output->Send(message);

      // Check for early stopping.
      if (limit != -1 && records_read->value() >= limit) break;
    }

    // Close reader.
    CHECK(reader.Close());

    // Close output channel.
    output->Close();
  }
};

REGISTER_TASK_PROCESSOR("record-file-reader", RecordFileReader);

}  // namespace task
}  // namespace sling

