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

#ifndef SLING_STREAM_BZIP2_H_
#define SLING_STREAM_BZIP2_H_

#include "sling/base/types.h"
#include "sling/stream/stream.h"
#include "third_party/bz2lib/bzlib.h"

namespace sling {

// BZIP2 stream compression.
class BZip2Compressor : public OutputStream {
 public:
  // Initialize compressor.
  BZip2Compressor(OutputStream *sink,
                  int block_size = 1 << 20,
                  int compression_level = 9);
  ~BZip2Compressor() override;

  // Implementation of OutputStream interface.
  bool Next(void **data, int *size) override;
  void BackUp(int count) override;
  int64 ByteCount() const override;

 private:
  // Compressor.
  bz_stream stream_;
};

// BZIP2 stream decompression.
class BZip2Decompressor : public InputStream {
 public:
  // Initialize decompressor.
  BZip2Decompressor(InputStream *source,
                    int block_size = 1 << 20);
  ~BZip2Decompressor() override;

  // Implementation of InputStream interface.
  bool Next(const void **data, int *size) override;
  void BackUp(int count) override;
  bool Skip(int count) override;
  int64 ByteCount() const override;

 private:
  // Decompress next chunk.
  bool NextChunk();

  // Source for compressed input.
  InputStream *source_;

  // Compression buffer.
  char *buffer_;
  int block_size_;

  // Decompressor.
  bz_stream stream_;

  // Number of bytes uncompressed.
  uint64 total_bytes_;

  // Reset decompressor on next chunk (for multi stream bzip2 files).
  bool reset_;

  // Number of bytes to back up.
  int backup_;
};

}  // namespace sling

#endif  // SLING_STREAM_BZIP2_H_

