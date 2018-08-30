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

#include "sling/base/logging.h"
#include "sling/file/recordio.h"
#include "sling/task/task.h"
#include "sling/util/mutex.h"

namespace sling {
namespace task {

// Write incoming messages to record file.
class RecordFileWriter : public Processor {
 public:
  ~RecordFileWriter() override { delete writer_; }

  void Init(Task *task) override {
    // Get output file.
    Binding *output = task->GetOutput("output");
    if (output == nullptr) {
      LOG(ERROR) << "Output missing";
      return;
    }

    // Open record file writer.
    RecordFileOptions options;
    if (task->Get("indexed", false)) options.indexed = true;
    writer_ = new RecordWriter(output->resource()->name(), options);
  }

  void Receive(Channel *channel, Message *message) override {
    MutexLock lock(&mu_);

    // Write message to record file.
    CHECK(writer_->Write(message->key(), message->value()));
    delete message;
  }

  void Done(Task *task) override {
    MutexLock lock(&mu_);

    // Close writer.
    if (writer_ != nullptr) {
      CHECK(writer_->Close());
      delete writer_;
      writer_ = nullptr;
    }
  }

 private:
  // Record writer for writing to output.
  RecordWriter *writer_ = nullptr;

  // Mutex for record writer.
  Mutex mu_;
};

REGISTER_TASK_PROCESSOR("record-file-writer", RecordFileWriter);

}  // namespace task
}  // namespace sling

