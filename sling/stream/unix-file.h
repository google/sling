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

#ifndef SLING_STREAM_UNIX_FILE_H_
#define SLING_STREAM_UNIX_FILE_H_

#include <stdio.h>

#include "sling/base/types.h"
#include "sling/stream/stream.h"

namespace sling {

// Input stream based on FILE * from stdio.h.
class StdFileInputStream : public InputStream {
 public:
  // Takes ownership of an existing file.
  explicit StdFileInputStream(FILE *file, int block_size = 1 << 20);

  // Use existing file.
  StdFileInputStream(FILE *file, bool take_ownership, int block_size = 1 << 20);

  // Closes file.
  ~StdFileInputStream() override;

  // Implementation of InputStream interface.
  bool Next(const void **data, int *size) override;
  void BackUp(int count) override;
  bool Skip(int count) override;
  int64 ByteCount() const override;

 private:
  FILE *file_ = nullptr;  // underlying file to read from
  bool owned_ = true;     // ownership of underlying file
  uint8 *buffer_;         // file buffer
  int size_;              // size of file buffer
  int used_;              // number of current used bytes in buffer
  int backup_;            // number of bytes currently backed up
  int64 position_;        // current file position
};

// Output stream based on FILE * from stdio.h.
class StdFileOutputStream : public OutputStream {
 public:
  // Takes ownership of an existing file.
  explicit StdFileOutputStream(FILE *file, int block_size = 1 << 20);

  // Use existing file.
  StdFileOutputStream(FILE *file, bool take_ownership,
                      int block_size = 1 << 20);

  // Closes file.
  ~StdFileOutputStream() override;

  // Closes file and returns true if successful. This will also flush any
  // remaining data in the buffer.
  bool Close();

  // Implementation of OutputStream interface.
  bool Next(void **data, int *size) override;
  void BackUp(int count) override;
  int64 ByteCount() const override;

 private:
  FILE *file_ = nullptr;  // underlying file to read from
  bool owned_ = true;     // ownership of underlying file
  uint8 *buffer_;         // file buffer
  int size_;              // size of file buffer
  int used_;              // number of current used bytes in buffer
  int64 position_;        // current file position
};

}  // namespace sling

#endif  // SLING_STREAM_UNIX_FILE_H_

