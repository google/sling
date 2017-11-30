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

#ifndef SLING_STREAM_MEMORY_H_
#define SLING_STREAM_MEMORY_H_

#include <string>

#include "sling/base/slice.h"
#include "sling/base/types.h"
#include "sling/stream/stream.h"

namespace sling {

// An InputStream backed by an in-memory array of bytes.
class ArrayInputStream : public InputStream {
 public:
  ArrayInputStream(const void *data, int size, int block_size = -1);
  ArrayInputStream(const Slice &buffer, int block_size = -1)
      : ArrayInputStream(buffer.data(), buffer.size(), block_size) {}
  ~ArrayInputStream() override;

  // InputStream interface.
  bool Next(const void **data, int *size) override;
  void BackUp(int count) override;
  bool Skip(int count) override;
  int64 ByteCount() const override;

 private:
  const char *data_;
  int size_;
  int block_size_;
  int position_;
};

// An OutputStream backed by an in-memory array of bytes.
class ArrayOutputStream : public OutputStream {
 public:
  ArrayOutputStream(int block_size = 8192);
  ~ArrayOutputStream() override;

  // Return buffer with output data.
  Slice data() const { return Slice(data_, position_); }

  // Release data buffer to caller. Must be deallocated with free().
  char *release() {
    char *data = data_;
    data_ = nullptr;
    size_ = position_ = 0;
    return data;
  }

  // OutputStream interface.
  bool Next(void **data, int *size) override;
  void BackUp(int count) override;
  int64 ByteCount() const override;

 private:
  char *data_;
  int size_;
  int position_;
  int block_size_;
};

// An InputStream backed by string.
class StringInputStream : public ArrayInputStream {
 public:
  StringInputStream(const string &str, int block_size = -1)
      : ArrayInputStream(str.data(), str.size(), block_size) {}
};

// An OutputStream which appends bytes to a string.
class StringOutputStream : public OutputStream {
 public:
  explicit StringOutputStream(string *buffer);
  ~StringOutputStream() override;

  // OutputStream interface
  bool Next(void **data, int *size) override;
  void BackUp(int count) override;
  int64 ByteCount() const override;

 private:
  string *buffer_;
};

}  // namespace sling

#endif  // SLING_STREAM_MEMORY_H_

