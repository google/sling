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

#include "sling/stream/output.h"

#include "sling/base/logging.h"
#include "sling/stream/stream.h"
#include "sling/util/varint.h"

namespace sling {

Output::Output(OutputStream *stream) : stream_(stream) {
  Next();
}

void Output::Flush() {
  // Check if there is any data that need to be backed up.
  if (buffer_ == nullptr) return;
  int unused = limit_ - current_;
  if (unused == 0) return;

  // Back up in the output stream and reset buffer.
  stream_->BackUp(unused);
  buffer_ = current_ = limit_ = nullptr;
}

void Output::Write(const char *data, int size) {
  if (limit_ - current_ >= size) {
    // Copy data directly to output buffer.
    memcpy(current_, data, size);
    current_ += size;
  } else {
    // Copy data buffer in stages.
    while (size > 0) {
      if (current_ != limit_) {
        // Write to output buffer.
        int bytes = limit_ - current_;
        if (bytes > size) bytes = size;
        memcpy(current_, data, bytes);
        current_ += bytes;
        data += bytes;
        size -= bytes;
      } else {
        // Request a new output buffer
        Next();
      }
    }
  }
}

void Output::WriteVarint32(uint32 value) {
  // Optimize the case where we are certain that we have enough space in the
  // current output buffer.
  if (limit_ - current_ >= Varint::kMax32) {
    current_ = Varint::Encode32Inline(current_, value);
  } else {
    char buffer[Varint::kMax32];
    char *end = Varint::Encode32(buffer, value);
    Write(buffer, end - buffer);
  }
}

void Output::WriteVarint64(uint64 value) {
  // Optimize the case where we are certain that we have enough space in the
  // current output buffer.
  if (limit_ - current_ >= Varint::kMax64) {
    current_ = Varint::Encode64(current_, value);
  } else {
    char buffer[Varint::kMax64];
    char *end = Varint::Encode64(buffer, value);
    Write(buffer, end - buffer);
  }
}

void Output::Next() {
  for (;;) {
    void *data;
    int size;
    if (stream_->Next(&data, &size)) {
      if (size > 0) {
        // Got new output buffer. This implicitly commits the old buffer.
        buffer_ = reinterpret_cast<char *>(data);
        current_ = buffer_;
        limit_ = buffer_ + size;
        return;
      }
    } else {
      // Error occurred.
      LOG(FATAL) << "Unable to write to output stream";
    }
  }
}

}  // namespace sling

