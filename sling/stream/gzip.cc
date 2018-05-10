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

#include "sling/stream/gzip.h"

#include <string.h>

#include "sling/base/logging.h"
#include "third_party/zlib/zlib.h"

namespace sling {

GZipCompressor::GZipCompressor(OutputStream *sink,
                               int block_size,
                               int compression_level) {
  memset(&stream_, 0, sizeof(stream_));
  CHECK(deflateInit(&stream_, compression_level) == Z_OK);
}

GZipCompressor::~GZipCompressor() {
  CHECK(deflateEnd(&stream_) == Z_OK);
}

bool GZipCompressor::Next(void **data, int *size) {
  LOG(FATAL) << "Not yet implemented";
  *size = 0;
  return false;
}

void GZipCompressor::BackUp(int count) {
  LOG(FATAL) << "Not yet implemented";
}

int64 GZipCompressor::ByteCount() const {
  LOG(FATAL) << "Not yet implemented";
  return -1;
}

GZipDecompressor::GZipDecompressor(InputStream *source,
                                   int block_size,
                                   int window_bits)
    : source_(source), block_size_(block_size) {
  memset(&stream_, 0, sizeof(stream_));
  CHECK(inflateInit2(&stream_, window_bits) == Z_OK);
  buffer_ = new char[block_size_];
  total_bytes_ = 0;
  reset_ = false;
  backup_ = 0;
}

GZipDecompressor::~GZipDecompressor() {
  CHECK(inflateEnd(&stream_) == Z_OK);
  delete [] buffer_;
}

bool GZipDecompressor::Next(const void **data, int *size) {
  // Check if there is any backed up data.
  if (backup_ > 0) {
    *data = stream_.next_out - backup_;
    *size = backup_;
    backup_ = 0;
    return true;
  }

  // Read next chunk from source.
  while (stream_.avail_in == 0) {
    const void *chunk;
    int bytes;
    if (!source_->Next(&chunk, &bytes)) return false;
    stream_.next_in = static_cast<Bytef *>(const_cast<void *>(chunk));
    stream_.avail_in = bytes;
  }

  // Check for reset.
  if (reset_) {
    // Keep the remaining input chunk.
    Bytef *next = stream_.next_in;
    int avail = stream_.avail_in;

    // Reset decompressor.
    CHECK(inflateReset(&stream_) == Z_OK);

    // Initialize decompressor with remaining input chunk.
    stream_.next_in = next;
    stream_.avail_in = avail;
    reset_ = false;
  }

  // Decompress chunk.
  stream_.next_out = reinterpret_cast<Bytef *>(buffer_);
  stream_.avail_out = block_size_;
  int rc = inflate(&stream_, Z_NO_FLUSH);
  if (rc == Z_STREAM_END) {
    reset_ = true;
  } else {
    CHECK(rc == Z_OK) << "GZIP input error " << rc << ": " << stream_.msg;
  }

  // Return uncompressed data.
  int uncompressed = reinterpret_cast<char *>(stream_.next_out) - buffer_;
  *data = buffer_;
  *size = uncompressed;
  total_bytes_ += uncompressed;
  return true;
}

void GZipDecompressor::BackUp(int count) {
  backup_ += count;
  CHECK_LE(backup_, reinterpret_cast<char *>(stream_.next_out) - buffer_);
}

bool GZipDecompressor::Skip(int count) {
  while (count > 0) {
    const void *chunk;
    int bytes;
    if (!Next(&chunk, &bytes)) return false;
    if (count >= bytes) {
      count -= bytes;
    } else {
      BackUp(bytes - count);
      count = 0;
    }
  }
  return true;
}

int64 GZipDecompressor::ByteCount() const {
  return total_bytes_ - backup_;
}

}  // namespace sling

