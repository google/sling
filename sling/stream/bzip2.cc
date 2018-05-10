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

#include "sling/stream/bzip2.h"

#include <string.h>

#include "sling/base/logging.h"
#include "third_party/bz2lib/bzlib.h"

extern "C" {
// BZip2 library error handler.
void bz_internal_error(int errcode) {
  LOG(FATAL) << "BZip2 internal error " << errcode;
}
}

namespace sling {

BZip2Compressor::BZip2Compressor(OutputStream *sink,
                                 int block_size,
                                 int compression_level) {
  memset(&stream_, 0, sizeof(stream_));
  CHECK(BZ2_bzCompressInit(&stream_, compression_level, 0, 0) == BZ_OK);
}

BZip2Compressor::~BZip2Compressor() {
  CHECK(BZ2_bzCompressEnd(&stream_) == BZ_OK);
}

bool BZip2Compressor::Next(void **data, int *size) {
  LOG(FATAL) << "Not yet implemented";
  *size = 0;
  return false;
}

void BZip2Compressor::BackUp(int count) {
  LOG(FATAL) << "Not yet implemented";
}

int64 BZip2Compressor::ByteCount() const {
  LOG(FATAL) << "Not yet implemented";
  return -1;
}

BZip2Decompressor::BZip2Decompressor(InputStream *source, int block_size)
    : source_(source), block_size_(block_size) {
  memset(&stream_, 0, sizeof(stream_));
  CHECK(BZ2_bzDecompressInit(&stream_, 0, 0) == BZ_OK);
  buffer_ = new char[block_size_];
  total_bytes_ = 0;
  reset_ = false;
  backup_ = 0;
}

BZip2Decompressor::~BZip2Decompressor() {
  CHECK(BZ2_bzDecompressEnd(&stream_) == BZ_OK);
  delete [] buffer_;
}

bool BZip2Decompressor::Next(const void **data, int *size) {
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
    stream_.next_in = static_cast<char *>(const_cast<void *>(chunk));
    stream_.avail_in = bytes;
  }

  // Check for reset.
  if (reset_) {
    // Keep the remaining input chunk.
    char *next = stream_.next_in;
    int avail = stream_.avail_in;

    // Reset decompressor.
    CHECK(BZ2_bzDecompressEnd(&stream_) == BZ_OK);
    CHECK(BZ2_bzDecompressInit(&stream_, 0, 0) == BZ_OK);

    // Initialize decompressor with remaining input chunk.
    stream_.next_in = next;
    stream_.avail_in = avail;
    reset_ = false;
  }

  // Decompress chunk.
  stream_.next_out = buffer_;
  stream_.avail_out = block_size_;
  int rc = BZ2_bzDecompress(&stream_);
  if (rc == BZ_STREAM_END) {
    reset_ = true;
  } else {
    CHECK(rc == BZ_OK) << "Corrupt BZIP2 input, error " << rc;
  }

  // Return uncompressed data.
  int uncompressed = stream_.next_out - buffer_;
  *data = buffer_;
  *size = uncompressed;
  total_bytes_ += uncompressed;
  return true;
}

void BZip2Decompressor::BackUp(int count) {
  backup_ += count;
  CHECK_LE(stream_.next_out - buffer_, backup_);
}

bool BZip2Decompressor::Skip(int count) {
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

int64 BZip2Decompressor::ByteCount() const {
  return total_bytes_ - backup_;
}

}  // namespace sling

