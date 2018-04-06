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

#include "sling/http/http-stream.h"

#include "sling/http/http-server.h"
#include "sling/stream/stream.h"

namespace sling {

HTTPInputStream::HTTPInputStream(HTTPBuffer *buffer) : buffer_(buffer) {}

bool HTTPInputStream::Next(const void **data, int *size) {
  int n = buffer_->size();
  if (n > 0) {
    *data = buffer_->start;
    *size = n;
    buffer_->start = buffer_->end;
    return true;
  } else {
    return false;
  }
}

void HTTPInputStream::BackUp(int count) {
  buffer_->start -= count;
}

bool HTTPInputStream::Skip(int count) {
  int left = buffer_->size();
  if (count > left) {
    buffer_->start = buffer_->end;
    return false;
  } else {
    buffer_->start += count;
    return true;
  }
}

int64 HTTPInputStream::ByteCount() const {
  return buffer_->start - buffer_->floor;
}

HTTPOutputStream::HTTPOutputStream(HTTPBuffer *buffer, int block_size)
    : buffer_(buffer), block_size_(block_size) {}

bool HTTPOutputStream::Next(void **data, int *size) {
  if (buffer_->full()) buffer_->ensure(block_size_);

  int n = buffer_->remaining();
  if (n > block_size_) n = block_size_;
  *data = buffer_->end;
  *size = n;
  buffer_->end += n;
  return true;
}

void HTTPOutputStream::BackUp(int count) {
  buffer_->end -= count;
}

int64 HTTPOutputStream::ByteCount() const {
  return buffer_->size();
}

}  // namespace sling

