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

#ifndef SLING_STREAM_GZIP_H_
#define SLING_STREAM_GZIP_H_

#include "sling/base/types.h"
#include "sling/stream/stream.h"
#include "third_party/zlib/zlib.h"

namespace sling {

// GZIP stream compression.
class GZipCompressor : public OutputStream {
 public:
  // Initialize compressor.
  GZipCompressor(OutputStream *sink,
                  int block_size = 1 << 20,
                  int compression_level = 9);
  ~GZipCompressor() override;

  // Implementation of OutputStream interface.
  bool Next(void **data, int *size) override;
  void BackUp(int count) override;
  int64 ByteCount() const override;

 private:
  // Compressor.
  z_stream stream_;
};

// GZIP stream decompression.
class GZipDecompressor : public InputStream {
 public:
  // Initialize decompressor.
  GZipDecompressor(InputStream *source,
                   int block_size = 1 << 20,
                   int window_bits = 15 + 16);
  ~GZipDecompressor() override;

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

  // Decompression buffer.
  char *buffer_;
  int block_size_;

  // Decompressor.
  z_stream stream_;

  // Number of bytes uncompressed.
  uint64 total_bytes_;

  // Reset decompressor on next chunk (for multi stream gzip files).
  bool reset_;

  // Number of bytes to back up.
  int backup_;
};

}  // namespace sling

#endif  // SLING_STREAM_GZIP_H_

