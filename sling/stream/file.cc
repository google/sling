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

#include "sling/stream/file.h"

#include <string>

#include "sling/base/logging.h"
#include "sling/base/types.h"

namespace sling {

FileInputStream::FileInputStream(const string &filename, int block_size) {
  CHECK(File::Open(filename, "r", &file_));
  size_ = block_size;
  buffer_ = new uint8[size_];
  used_ = 0;
  backup_ = 0;
  position_ = 0;
}

FileInputStream::FileInputStream(File *file, int block_size) {
  file_ = file;
  size_ = block_size;
  buffer_ = new uint8[size_];
  used_ = 0;
  backup_ = 0;
  position_ = file->Tell();
}

FileInputStream::FileInputStream(File *file,
                                 bool take_ownership,
                                 int block_size) {
  file_ = file;
  owned_ = take_ownership;
  size_ = block_size;
  buffer_ = new uint8[size_];
  used_ = 0;
  backup_ = 0;
  position_ = file->Tell();
}

FileInputStream::~FileInputStream() {
  if (owned_ && file_ != nullptr) CHECK(file_->Close());
  delete [] buffer_;
}

bool FileInputStream::Next(const void **data, int *size) {
  // Return backed up data if we have any.
  if (backup_ > 0) {
    *data = buffer_ + used_ - backup_;
    *size = backup_;
    backup_ = 0;
    return true;
  }

  // Read data into buffer.
  uint64 bytes;
  if (!file_->PRead(position_, buffer_, size_, &bytes).ok()) return false;
  if (bytes <= 0) {
    used_ = 0;
    return false;
  }
  used_ = bytes;
  position_ += bytes;

  // Return buffer read from file.
  *data = buffer_;
  *size = used_;
  return true;
}

void FileInputStream::BackUp(int count) {
  backup_ += count;
  CHECK(backup_ <= used_);
}

bool FileInputStream::Skip(int count) {
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

  // Check if we moved past the end of the file.
  int64 file_size = file_->Size();
  if (file_size < 0) return false;
  if (position_ > file_size) {
    position_ = file_size;
    return false;
  }
  return true;
}

int64 FileInputStream::ByteCount() const {
  return position_ - backup_;
}

FileOutputStream::FileOutputStream(const string &filename, int block_size) {
  CHECK(File::Open(filename, "w", &file_));
  size_ = block_size;
  buffer_ = new uint8[size_];
  used_ = 0;
  position_ = 0;
}

FileOutputStream::FileOutputStream(File *file, int block_size) {
  file_ = file;
  size_ = block_size;
  buffer_ = new uint8[size_];
  used_ = 0;
  position_ = 0;
}

FileOutputStream::~FileOutputStream() {
  CHECK(Close());
  delete [] buffer_;
}

bool FileOutputStream::Close() {
  if (file_ != nullptr) {
    // Flush buffer.
    if (used_ > 0) {
      if (!file_->Write(buffer_, used_).ok()) return false;
      position_ += used_;
      used_ = 0;
    }

    // Close file.
    if (!file_->Close().ok()) return false;
    file_ = nullptr;
  }

  return true;
}

bool FileOutputStream::Next(void **data, int *size) {
  // Flush buffer.
  if (used_ > 0) {
    if (!file_->Write(buffer_, used_).ok()) return false;
    position_ += used_;
  }

  // Return write buffer to caller.
  used_ = size_;
  *data = buffer_;
  *size = size_;
  return true;
}

void FileOutputStream::BackUp(int count) {
  CHECK_LE(count, used_);
  used_ -= count;
}

int64 FileOutputStream::ByteCount() const {
  return position_ + used_;
}

}  // namespace sling

