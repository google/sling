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

#include "sling/stream/unix-file.h"

#include <stdio.h>

#include "sling/base/logging.h"
#include "sling/base/types.h"

namespace sling {

StdFileInputStream::StdFileInputStream(FILE *file, int block_size) {
  file_ = file;
  size_ = block_size;
  buffer_ = new uint8[size_];
  used_ = 0;
  backup_ = 0;
  position_ = ftell(file_);
}

StdFileInputStream::StdFileInputStream(FILE *file,
                                       bool take_ownership,
                                       int block_size) {
  file_ = file;
  owned_ = take_ownership;
  size_ = block_size;
  buffer_ = new uint8[size_];
  used_ = 0;
  backup_ = 0;
  position_ = ftell(file_);
}

StdFileInputStream::~StdFileInputStream() {
  if (!owned_ && backup_ > 0) fseek(file_, -backup_, SEEK_CUR);
  if (owned_ && file_ != nullptr) CHECK(fclose(file_) == 0);
  delete [] buffer_;
}

bool StdFileInputStream::Next(const void **data, int *size) {
  // Return backed up data if we have any.
  if (backup_ > 0) {
    *data = buffer_ + used_ - backup_;
    *size = backup_;
    backup_ = 0;
    return true;
  }

  // Read data into buffer.
  uint64 bytes = fread(buffer_, 1, size_, file_);
  if (bytes == 0) return false;
  used_ = bytes;
  position_ += bytes;

  // Return buffer read from file.
  *data = buffer_;
  *size = used_;
  return true;
}

void StdFileInputStream::BackUp(int count) {
  backup_ += count;
  CHECK(backup_ <= used_);
}

bool StdFileInputStream::Skip(int count) {
  // Skip over backed up data.
  if (backup_ >= count) {
    backup_ -= count;
    return true;
  } else if (backup_ > 0) {
    count -= backup_;
    backup_ = 0;
  }

  // Advance file position.
  position_ += count;
  if (count > 0) CHECK(fseek(file_, count, SEEK_CUR) >= 0);
  return true;
}

int64 StdFileInputStream::ByteCount() const {
  return position_ - backup_;
}

StdFileOutputStream::StdFileOutputStream(FILE *file, int block_size) {
  file_ = file;
  size_ = block_size;
  buffer_ = new uint8[size_];
  used_ = 0;
  position_ = 0;
}

StdFileOutputStream::StdFileOutputStream(FILE *file,
                                         bool take_ownership,
                                         int block_size) {
  file_ = file;
  owned_ = take_ownership;
  size_ = block_size;
  buffer_ = new uint8[size_];
  used_ = 0;
  position_ = 0;
}

StdFileOutputStream::~StdFileOutputStream() {
  CHECK(Close());
  delete [] buffer_;
}

bool StdFileOutputStream::Close() {
  if (file_ != nullptr) {
    // Flush buffer.
    if (used_ > 0) {
      if (fwrite(buffer_, used_, 1, file_) != used_) return false;
      position_ += used_;
      used_ = 0;
    }

    // Close file.
    if (owned_) {
      if (fclose(file_) != 0) return false;
      file_ = nullptr;
    }
  }

  return true;
}

bool StdFileOutputStream::Next(void **data, int *size) {
  // Flush buffer.
  if (used_ > 0) {
    if (fwrite(buffer_, 1, used_, file_) != used_) return false;
    position_ += used_;
  }

  // Return write buffer to caller.
  used_ = size_;
  *data = buffer_;
  *size = size_;
  return true;
}

void StdFileOutputStream::BackUp(int count) {
  CHECK_LE(count, used_);
  used_ -= count;
}

int64 StdFileOutputStream::ByteCount() const {
  return position_ + used_;
}

}  // namespace sling

