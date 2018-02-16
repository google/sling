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
#include "sling/stream/file.h"
#include "sling/stream/output.h"
#include "sling/task/task.h"
#include "sling/util/mutex.h"

namespace sling {
namespace task {

// Write incoming messages to text file.
class TextFileWriter : public Processor {
 public:
  ~TextFileWriter() override {
    delete writer_;
    delete stream_;
  }

  void Init(Task *task) override {
    // Get output file.
    Binding *output = task->GetOutput("output");
    if (output == nullptr) {
      LOG(ERROR) << "Output missing";
      return;
    }

    // Open file output stream.
    stream_ = new FileOutputStream(
        output->resource()->name(),
        task->Get("buffer_size", 1 << 16));
    writer_ = new Output(stream_);

    // Statistics.
    lines_written_ = task->GetCounter("text_lines_written");
    bytes_written_ = task->GetCounter("text_bytes_written");
  }

  void Receive(Channel *channel, Message *message) override {
    MutexLock lock(&mu_);

    // Write message value to text file.
    const Slice &line = message->value();
    writer_->Write(line.data(), line.size());
    writer_->WriteChar('\n');
    lines_written_->Increment();
    bytes_written_->Increment(line.size() + 1);
    delete message;
  }

  void Done(Task *task) override {
    MutexLock lock(&mu_);

    // Close writer.
    delete writer_;
    writer_ = nullptr;

    // Close output file.
    if (stream_ != nullptr) {
      CHECK(stream_->Close());
      delete stream_;
      stream_ = nullptr;
    }
  }

 private:
  // File output stream for writing to output.
  FileOutputStream *stream_ = nullptr;

  // Output buffer.
  Output *writer_ = nullptr;

  // Mutex for serializing writes.
  Mutex mu_;

  // Statistics.
  Counter *lines_written_ = nullptr;
  Counter *bytes_written_ = nullptr;
};

REGISTER_TASK_PROCESSOR("text-file-writer", TextFileWriter);

}  // namespace task
}  // namespace sling

