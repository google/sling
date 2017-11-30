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

#ifndef SLING_STREAM_FILE_H_
#define SLING_STREAM_FILE_H_

#include <string>

#include "sling/base/types.h"
#include "sling/file/file.h"
#include "sling/stream/stream.h"

namespace sling {

// File-based input stream.
class FileInputStream : public InputStream {
 public:
  // Opens file.
  explicit FileInputStream(const string &filename, int block_size = 1 << 20);

  // Takes ownership of an existing file.
  explicit FileInputStream(File *file, int block_size = 1 << 20);

  // Use existing file.
  FileInputStream(File *file, bool take_ownership, int block_size = 1 << 20);

  // Closes file.
  ~FileInputStream() override;

  // Implementation of InputStream interface.
  bool Next(const void **data, int *size) override;
  void BackUp(int count) override;
  bool Skip(int count) override;
  int64 ByteCount() const override;

 private:
  File *file_ = nullptr;  // underlying file to read from
  bool owned_ = true;     // ownership of underlying file
  uint8 *buffer_;         // file buffer
  int size_;              // size of file buffer
  int used_;              // number of current used bytes in buffer
  int backup_;            // number of bytes currently backed up
  int64 position_;        // current file position
};

// File-based output stream.
class FileOutputStream : public OutputStream {
 public:
  // Opens file.
  explicit FileOutputStream(const string &filename, int block_size = 1 << 20);

  // Takes ownership of an existing file.
  explicit FileOutputStream(File *file, int block_size = 1 << 20);

  // Closes file.
  ~FileOutputStream() override;

  // Closes file and returns true if successful. This will also flush any
  // remaining data in the buffer.
  bool Close();

  // Implementation of OutputStream interface.
  bool Next(void **data, int *size) override;
  void BackUp(int count) override;
  int64 ByteCount() const override;

 private:
  File *file_ = nullptr;  // underlying file to read from
  uint8 *buffer_;         // file buffer
  int size_;              // size of file buffer
  int used_;              // number of current used bytes in buffer
  int64 position_;        // current file position
};

}  // namespace sling

#endif  // SLING_STREAM_FILE_H_

