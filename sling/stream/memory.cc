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

#include "sling/stream/memory.h"

#include <stdlib.h>
#include <string>

#include "sling/base/logging.h"

namespace sling {

ArrayInputStream::ArrayInputStream(const void *data, int size, int block_size)
    : data_(reinterpret_cast<const char *>(data)),
      size_(size),
      block_size_(block_size > 0 ? block_size : size),
      position_(0) {}

ArrayInputStream::~ArrayInputStream() {
}

bool ArrayInputStream::Next(const void **data, int *size) {
  int n = size_ - position_;
  if (n > 0) {
    if (n > block_size_) n = block_size_;
    *data = data_ + position_;
    *size = n;
    position_ += n;
    return true;
  } else {
    return false;
  }
}

void ArrayInputStream::BackUp(int count) {
  CHECK_LE(count, position_);
  CHECK_GE(count, 0);
  position_ -= count;
}

bool ArrayInputStream::Skip(int count) {
  CHECK_GE(count, 0);
  int left = size_ - position_;
  if (count > left) {
    position_ = size_;
    return false;
  } else {
    position_ += count;
    return true;
  }
}

int64 ArrayInputStream::ByteCount() const {
  return position_;
}

ArrayOutputStream::ArrayOutputStream(int block_size)
  : data_(nullptr),
    size_(0),
    position_(0),
    block_size_(block_size) {}

ArrayOutputStream::~ArrayOutputStream() {
  free(data_);
}

bool ArrayOutputStream::Next(void **data, int *size) {
  int left = size_ - position_;
  if (left == 0) {
    size_ = size_ > 0 ? size_ * 2 : block_size_;
    data_ = static_cast<char *>(realloc(data_, size_));
    if (data_ == nullptr) return false;
    left = size_ - position_;
  }

  int n = left;
  if (n > block_size_) n = block_size_;
  *data = data_ + position_;
  *size = n;
  position_ += n;
  return true;
}

void ArrayOutputStream::BackUp(int count) {
  CHECK_LE(count, position_);
  CHECK_GE(count, 0);
  position_ -= count;
}

int64 ArrayOutputStream::ByteCount() const {
  return position_;
}

StringOutputStream::StringOutputStream(string *buffer) : buffer_(buffer) {}

StringOutputStream::~StringOutputStream() {}

bool StringOutputStream::Next(void **data, int *size) {
  int position = buffer_->size();
  if (position < buffer_->capacity()) {
    // Resize the string to match its capacity.
    buffer_->resize(buffer_->capacity());
  } else {
    // Buffer has reached capacity.
    int new_size = position > 0 ? position * 2 : 8192;
    buffer_->resize(new_size);
  }

  *data = &(*buffer_)[position];
  *size = buffer_->size() - position;
  return true;
}

void StringOutputStream::BackUp(int count) {
  CHECK_GE(count, 0);
  CHECK_LE(count, buffer_->size());
  buffer_->resize(buffer_->size() - count);
}

int64 StringOutputStream::ByteCount() const {
  return buffer_->size();
}

}  // namespace sling

