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

#include "sling/stream/input.h"

#include "sling/base/logging.h"
#include "sling/base/types.h"
#include "sling/stream/stream.h"
#include "sling/util/varint.h"

namespace sling {

Input::Input(InputStream *stream) : stream_(stream) {
  current_ = nullptr;
  limit_ = nullptr;
  done_ = false;
}

Input::~Input() {
  // Return unread data to underlying stream.
  if (!done_) stream_->BackUp(limit_ - current_);
}

bool Input::Read(char *data, int size) {
  // Handle simple case where we have all the data in the buffer.
  const char *end = current_ + size;
  if (end <= limit_) {
    memcpy(data, current_, size);
    current_ = end;
    return true;
  }

  // Keep reading until we have read all bytes or reached the end of the input.
  while (size > 0) {
    if (!empty()) {
      // Get data from the buffer.
      int bytes = limit_ - current_;
      if (bytes > size) bytes = size;
      memcpy(data, current_, bytes);
      current_ += bytes;
      data += bytes;
      size -= bytes;
    } else {
      // Read more data from the input.
      if (!Fill()) return false;
    }
  }

  return true;
}

void Input::Skip(int bytes) {
  const char *end = current_ + bytes;
  if (end <= limit_) {
    // All the skipped bytes are in the buffer, so we can just advance the
    // current input pointer.
    current_ = end;
  } else {
    // Skip all the remaining data in the current input buffer.
    int left = end - limit_;
    current_ = limit_;

    // Skip any remaining bytes in the underlying input stream.
    if (left > 0) {
      if (!stream_->Skip(left)) done_ = true;
    }
  }
}

bool Input::ReadString(int size, string *output) {
  while (size > 0) {
    if (!empty()) {
      // Get data from the buffer.
      int bytes = limit_ - current_;
      if (bytes > size) bytes = size;
      output->append(current_, bytes);
      current_ += bytes;
      size -= bytes;
    } else {
      // Read more data from the input.
      if (!Fill()) return false;
    }
  }
  return true;
}

bool Input::ReadLine(string *output) {
  output->clear();
  for (;;) {
    // Fill buffer if empty.
    if (empty()) {
      if (!Fill()) return !output->empty();
    }

    // Find end of line.
    const char *p = current_;
    while (p < limit_) {
      if (*p++ == '\n') {
        // Line end found.
        output->append(current_, p - current_);
        current_ = p;
        return !output->empty();
      }
    }

    // Consume the rest of the input chunk.
    output->append(current_, limit_ - current_);
    current_ = limit_;
  }
}

bool Input::ReadVarint32Fallback(uint32 *value) {
  uint32 result = 0;
  for (int i = 0; i < 64; i += 7) {
    if (empty() && !Fill()) return false;
    uint32 byte = *current_++;
    result |= (byte & 127) << i;
    if (byte < 128) {
      *value = result;
      return true;
    }
  }
  return false;
}

bool Input::ReadVarint64Fallback(uint64 *value) {
  uint64 result = 0;
  for (int i = 0; i < 64; i += 7) {
    if (empty() && !Fill()) return false;
    uint32 byte = *current_++;
    result |= static_cast<uint64>(byte & 127) << i;
    if (byte < 128) {
      *value = result;
      return true;
    }
  }
  return false;
}

int Input::Peek() {
  if (current_ == limit_) Fill();
  if (current_ == limit_) return -1;
  return static_cast<uint8>(*current_);
}

bool Input::Fill() {
  // Check if we have already reached the end of the input.
  if (done_) return false;

  // Keep reading until we get some data or we have reached the end.
  for (;;) {
    const void *data;
    int size;
    if (stream_->Next(&data, &size)) {
      if (size > 0) {
        // More data read, set up start and end of buffer.
        current_ = reinterpret_cast<const char *>(data);
        limit_ = current_ + size;
        return true;
      }
    } else {
      // No more data.
      current_ = limit_ = nullptr;
      done_ = true;
      return false;
    }
  }
}

}  // namespace sling

