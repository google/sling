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

#include <algorithm>
#include <queue>
#include <string>
#include <vector>

#include "sling/base/logging.h"
#include "sling/base/types.h"
#include "sling/file/recordio.h"
#include "sling/string/printf.h"
#include "sling/task/task.h"
#include "sling/util/mutex.h"

namespace sling {
namespace task {

// Element in the merge sort queue.
struct MergeItem {
  Record record;          // current record
  RecordReader *reader;   // record reader for reading records from merge file
};

// Comparator for elements in merge sort queue.
struct ItemComparator {
  bool operator ()(const MergeItem *a, const MergeItem *b) {
    return a->record.key > b->record.key;
  }
};

// Message comparator.
struct MessageComparator {
  bool operator ()(const Message *a, const Message *b) const {
    return a->key() < b->key();
  }
};

// Sorts all the input messages by key and output these in sorted order on the
// output channel.
class Sorter : public Processor {
 public:
  Sorter() {}
  ~Sorter() override {
    for (auto *m : messages_) delete m;
  }

  void Start(Task *task) override {
    // Get output port.
    output_ = task->GetSink("output");
    CHECK(output_ != nullptr) << "Output channel missing";
    task->Fetch("sort_buffer_size", &max_buffer_size_);
  }

  void Receive(Channel *channel, Message *message) override {
    MutexLock lock(&mu_);

    // Add message to buffer.
    messages_.push_back(message);
    buffer_bytes_ += message->key().size() + message->value().size();

    // Sort and write buffer when buffer is full.
    if (buffer_bytes_ > max_buffer_size_) {
      // Sort messages in sort buffer.
      SortMessages();

      // Flush messages to merge file.
      Flush();
    }
  }

  void Done(Task *task) override {
    MutexLock lock(&mu_);

    // Send sorted messages to output channel.
    if (next_merge_file_ == 0) {
      // All messages are in the sort buffer.
      SendMessageBuffer();
    } else {
      // Sort and flush remaining messages to merge file.
      SortMessages();
      Flush();

      // Send messages from merge files to output channel.
      SendMergedMessages();

      // Remove temporary files.
      RemoveTempFiles();
    }

    // Close output channel.
    output_->Close();
  }

  // Remove temporary files.
  void RemoveTempFiles() {
    // Remove temporary merge files.
    for (int i = 0; i < next_merge_file_; ++i) {
      File::Delete(MergeFileName(i));
    }

    // Remove directory.
    if (!tmpdir_.empty()) File::Rmdir(tmpdir_);
  }

  // Return file name of merge file.
  string MergeFileName(int index) const {
    return StringPrintf("%s/%05d", tmpdir_.c_str(), index);
  }

  // Flush sort buffer to new merge file.
  void Flush() {
    // Check if there are any messages to flush.
    if (messages_.empty()) return;

    // Create temp dir if not already done.
    if (tmpdir_.empty()) {
      CHECK(File::CreateTempDir(&tmpdir_));
    }

    // Write messages to next merge file.
    int fileno = next_merge_file_++;
    VLOG(3) << "Flush " << buffer_bytes_ << " bytes and "
            << messages_.size() << " messages to " << MergeFileName(fileno);
    RecordFileOptions options;
    RecordWriter writer(MergeFileName(fileno), options);
    for (Message *message : messages_) {
      CHECK(writer.Write(message->key(), message->value()));
      delete message;
    }
    CHECK(writer.Close());

    // Clear sort buffer.
    messages_.clear();
    buffer_bytes_ = 0;
  }

  // Sort messages in sort buffer.
  void SortMessages() {
    VLOG(3) << "Sort " << messages_.size() << " messages";
    MessageComparator comparator;
    std::sort(messages_.begin(), messages_.end(), comparator);
  }

  // Send messages in sort buffer to output channel.
  void SendMessageBuffer() {
    // Sort the messages in the buffer.
    SortMessages();

    // Send messages to output.
    VLOG(3) << "Output " << messages_.size() << " messages";
    for (Message *message : messages_) {
      output_->Send(message);
    }
    messages_.clear();
  }

  // Send messages in merge files to output channel.
  void SendMergedMessages() {
    // Priority queue for merging files.
    typedef std::vector<MergeItem *> MergeItemArray;
    std::priority_queue<MergeItem *, MergeItemArray, ItemComparator> merger;

    // Open merge files.
    int num_files = next_merge_file_;
    std::vector<MergeItem> items(num_files);
    for (int i = 0; i < num_files; ++i) {
      // Open reader for merge file.
      MergeItem &item = items[i];
      item.reader = new RecordReader(MergeFileName(i));

      // Add first record to sort queue.
      if (!item.reader->Done()) {
        CHECK(item.reader->Read(&item.record));
        merger.push(&item);
      }
    }

    // Merge files and output sorted messages to output channel.
    VLOG(3) << "Merge " << num_files << " files";
    while (!merger.empty()) {
      // Get next item from queue.
      MergeItem *item = merger.top();
      merger.pop();

      // Send message to output channel.
      Message *message = new Message(item->record.key, item->record.value);
      output_->Send(message);

      // Get next item from merge file and add it to queue.
      if (!item->reader->Done()) {
        CHECK(item->reader->Read(&item->record));
        merger.push(item);
      }
    }

    // Close merge file readers.
    VLOG(3) << "Close merge files";
    for (auto &item : items) {
      CHECK(item.reader->Close());
      delete item.reader;
    }
  }

 private:
  // Temporary local directory for sort-merge files.
  string tmpdir_;

  // Buffer of messages that have not yet been sorted and written to merge file.
  std::vector<Message *> messages_;

  // Maximum size of messages in the sort buffer.
  int64 max_buffer_size_ = 64 * 1024 * 1024;

  // Size of messages in the sort buffer.
  uint64 buffer_bytes_ = 0;

  // Next merge file number.
  int next_merge_file_ = 0;

  // Output channel.
  Channel *output_;

  // Mutex for serializing access.
  Mutex mu_;
};

REGISTER_TASK_PROCESSOR("sorter", Sorter);

}  // namespace task
}  // namespace sling

