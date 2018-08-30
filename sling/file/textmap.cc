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

#include "sling/file/textmap.h"

#include <stdlib.h>
#include <algorithm>

#include "sling/base/logging.h"
#include "sling/base/status.h"
#include "sling/base/types.h"
#include "sling/file/file.h"

namespace sling {

TextMapInput::TextMapInput(const std::vector<string> &filenames,
                           int buffer_size)
    : filenames_(filenames), buffer_size_(buffer_size) {
  // Allocate input buffer.
  buffer_ = static_cast<char *>(malloc(buffer_size));
  next_ = end_ = buffer_;
}

TextMapInput::~TextMapInput() {
  // Deallocate input buffer.
  free(buffer_);

  // Close current input file.
  if (file_ != nullptr) CHECK(file_->Close());
}

bool TextMapInput::Next() {
  while (current_file_ < filenames_.size()) {
    if (file_ != nullptr) {
      // Read next line from the current file.
      key_.clear();
      value_.clear();

      // Read key.
      int c;
      while ((c = NextChar()) != -1) {
        if (c == '\t' || c == '\n') break;
        key_.push_back(c);
      }

      // Read value.
      if (c == '\t') {
        while ((c = NextChar()) != -1) {
          if (c == '\n') break;
          value_.push_back(c);
        }
      }

      if (c == -1 && key_.empty() && value_.empty()) {
        // No more lines in file. Switch to next file.
        CHECK(file_->Close());
        file_ = nullptr;
        current_file_++;
      } else {
        // Return next entry.
        id_++;
        return true;
      }
    } else {
      // Open the next file.
      file_ = File::OpenOrDie(filenames_[current_file_], "r");
    }
  }

  // No more entries.
  return false;
}

int TextMapInput::Fill() {
  DCHECK(next_ == end_);
  DCHECK(file_ != nullptr);
  uint64 bytes;
  if (!file_->Read(buffer_, buffer_size_, &bytes)) return -1;
  if (bytes == 0) return -1;
  next_ = buffer_;
  end_ = buffer_ + bytes;
  return *next_++;
}

bool TextMapInput::Read(int *index, string *name, int64 *count) {
  if (!Next()) return false;
  if (index != nullptr) *index = id();
  if (name != nullptr) *name = key();
  if (count != nullptr) *count = atoll(value().c_str());
  return true;
}

TextMapOutput::TextMapOutput(const string &filename, int buffer_size) {
  // Open output file.
  file_ = File::OpenOrDie(filename, "w");

  // Allocate output buffer.
  buffer_ = static_cast<char *>(malloc(buffer_size));
  next_ = buffer_;
  end_ =  buffer_ + buffer_size;
}

TextMapOutput::~TextMapOutput() {
  // Close output file.
  Close();

  // Deallocate output buffer.
  free(buffer_);
}

void TextMapOutput::Close() {
  if (file_ != nullptr) {
    // Flush output buffer.
    Flush();

    // Close output file.
    CHECK(file_->Close());
    file_ = nullptr;
  }
}

void TextMapOutput::Flush() {
  if (next_ != buffer_) {
    file_->WriteOrDie(buffer_, next_ - buffer_);
  }
  next_ = buffer_;
}

void TextMapOutput::Output(const char *data, size_t size) {
  // Flush buffer if there is not enough room for data.
  if (size > end_ - next_) Flush();

  // Add data to buffer or directly to file if output buffer is too small.
  if (size <= end_ - next_) {
    memcpy(next_, data, size);
    next_ += size;
  } else {
    DCHECK(next_ == buffer_);
    file_->WriteOrDie(data, size);
  }
}

void TextMapOutput::Write(Text key, Text value) {
  Output(key.data(), key.size());
  Output("\t", 1);
  Output(value.data(), value.size());
  Output("\n", 1);
}

void TextMapOutput::Write(Text key, int64 value) {
  Write(key, std::to_string(value));
}

}  // namespace sling
