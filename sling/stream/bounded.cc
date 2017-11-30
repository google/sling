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

#include "sling/stream/bounded.h"

namespace sling {

BoundedInputStream::BoundedInputStream(InputStream *input, int64 limit)
    : input_(input), left_(limit) {
  start_ = input_->ByteCount();
}

BoundedInputStream::~BoundedInputStream() {
  // Back up if we overshot the size of the stream.
  if (left_ < 0) input_->BackUp(-left_);
}

bool BoundedInputStream::Next(const void **data, int *size) {
  // Check if we have reached the limit of the stream.
  if (left_ <= 0) return false;

  // Read next chunk from the underlying stream.
  if (!input_->Next(data, size)) return false;

  // Adjust size of we overshot the limit.
  left_ -= *size;
  if (left_ < 0) *size += left_;

  return true;
}

void BoundedInputStream::BackUp(int count) {
  if (left_ < 0) {
    // Include the overshoot when backing up in the underlying stream.
    input_->BackUp(count - left_);
    left_ = count;
  } else {
    // Back up in the underlying stream.
    input_->BackUp(count);
    left_ += count;
  }
}

bool BoundedInputStream::Skip(int count) {
  if (count > left_) {
    // Skip to end.
    if (left_ < 0) return false;
    input_->Skip(left_);
    left_ = 0;
    return false;
  } else {
    // Skip within limit.
    if (!input_->Skip(count)) return false;
    left_ -= count;
    return true;
  }
}

int64 BoundedInputStream::ByteCount() const {
  if (left_ < 0) {
    return input_->ByteCount() + left_ - start_;
  } else {
    return input_->ByteCount() - start_;
  }
}

}  // namespace sling

