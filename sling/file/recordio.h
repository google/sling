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

#ifndef SLING_FILE_RECORDIO_H_
#define SLING_FILE_RECORDIO_H_

#include "sling/base/slice.h"
#include "sling/base/status.h"
#include "sling/base/types.h"
#include "sling/file/file.h"
#include "third_party/snappy/snappy-sinksource.h"

namespace sling {

// Record with key and value.
struct Record {
  Record() {}
  Record(const Slice &k, const Slice &v) : key(k), value(v) {}

  Slice key;
  Slice value;
};

// Record buffer.
class RecordBuffer : public snappy::Sink, public snappy::Source {
 public:
  ~RecordBuffer();

  // Check if the buffer is empty.
  bool empty() { return begin_ == end_; }

  // Returns the number of used bytes in the buffer.
  size_t size() { return end_ - begin_; }

  // Returns the capacity of the buffer.
  size_t capacity() { return ceil_ - floor_; }

  // Returns number of unused bytes remaining in the buffer.
  size_t remaining() { return ceil_ - end_; }

  // Clear buffer.
  void clear() { begin_ = end_ = floor_; }

  // Change buffer capacity.
  void resize(size_t bytes);

  // Ensure space is available for writing.
  void ensure(size_t bytes);

  // Flush buffer so the used portion is at the beginning.
  void flush();

  // Sink interface for decompression.
  void Append(const char *bytes, size_t n) override;
  char *GetAppendBuffer(size_t length, char *scratch) override;
  char *GetAppendBufferVariable(size_t min_size, size_t desired_size_hint,
                                char *scratch, size_t scratch_size,
                                size_t *allocated_size) override;

  // Source interface for compression.
  size_t Available() const override;
  const char *Peek(size_t *len) override;
  void Skip(size_t n) override;

  // Data appended to buffer.
  void appended(size_t n) { end_ += n; }

  // Data consumed from buffer.
  void consumed(size_t n) { begin_ += n; }

  // Buffer access.
  char *floor() const { return floor_; }
  char *ceil() const { return ceil_; }
  char *begin() const { return begin_; }
  char *end() const { return end_; }

 private:
  char *floor_ = nullptr;  // start of allocated memory
  char *ceil_ = nullptr;   // end of allocated memory
  char *begin_ = nullptr;  // start of used part of buffer
  char *end_ = nullptr;    // end of used part of buffer
};

class RecordFile {
 public:
  // Maximum record header length.
  static const int MAX_HEADER_LEN = 21;

  // Maximum skip record length.
  static const int MAX_SKIP_LEN = 12;

  // Magic number for identifying record files.
  static const uint32 MAGIC = 0x46434552;

  // Record types.
  enum RecordType {
    DATA_RECORD = 1,
    FILLER_RECORD = 2,
  };

  // Compression types.
  enum CompressionType {
    UNCOMPRESSED = 0,
    SNAPPY = 1,
  };

  // File header information.
  struct FileHeader {
    uint32 magic;
    uint8 hdrlen;
    uint8 compression;
    uint16 flags;
    uint64 index;
    uint64 chunk_size;
  };

  // Record header information.
  struct Header {
    RecordType record_type;
    uint64 record_size;
    uint64 key_size;
  };

  // Parse header from data. Returns the number of bytes read or -1 on error.
  static int ReadHeader(const char *data, Header *header);

  // Write header to data. Returns number of bytes written.
  static int WriteHeader(const Header &header, char *data);
};

// Configuration options for record file.
struct RecordFileOptions {
  // Input/output buffer size.
  int buffer_size = 1 << 20;

  // Chunk size. Records never overlap chunk boundaries.
  int chunk_size = 64 * (1 << 20);

  // Record compression.
  RecordFile::CompressionType compression = RecordFile::SNAPPY;
};

// Reader for reading records from a record file.
class RecordReader : public RecordFile {
 public:
  // Open record file for reading.
  RecordReader(File *file, const RecordFileOptions &options);
  RecordReader(const string &filename, const RecordFileOptions &options);
  explicit RecordReader(File *file);
  explicit RecordReader(const string &filename);
  ~RecordReader();

  // Close record file.
  Status Close();

  // Return true if we have read all records in the file.
  bool Done() { return position_ == size_; }

  // Read next record from record file.
  Status Read(Record *record);

  // Return current position in record file.
  uint64 Tell() { return position_; }

  // Seek to new position in record file.
  Status Seek(uint64 pos);

  // Skip bytes in input. The offset can be negative.
  Status Skip(int64 n);

 private:
  // Fill input buffer.
  Status Fill();

  // Input file.
  File *file_;

  // File size.
  uint64 size_;

  // Current position in record file.
  uint64 position_;

  // Record file meta information.
  FileHeader info_;

  // Input buffer.
  RecordBuffer input_;

  // Buffer for decompressed record data.
  RecordBuffer decompressed_data_;
};

// Writer for writing records to record file.
class RecordWriter : public RecordFile {
 public:
  // Open record file for writing.
  RecordWriter(File *file, const RecordFileOptions &options);
  RecordWriter(const string &filename, const RecordFileOptions &options);
  explicit RecordWriter(File *file);
  explicit RecordWriter(const string &filename);
  ~RecordWriter();

  // Close record file.
  Status Close();

  // Write record to record file.
  Status Write(const Record &record);

  // Write key/value pair to file.
  Status Write(const Slice &key, const Slice &value) {
    return Write(Record(key, value));
  }

  // Write record with empty key.
  Status Write(const Slice &value) {
    return Write(Record(Slice(), value));
  }

  // Return current position in record file.
  uint64 Tell() const { return position_; }

 private:
  // Flush output buffer to disk.
  Status Flush();

  // Output file.
  File *file_;

  // Current position in record file.
  uint64 position_;

  // Record file meta information.
  FileHeader info_;

  // Output buffer.
  RecordBuffer output_;

  // Buffer for compressed record data.
  RecordBuffer compressed_data_;
};

}  // namespace sling

#endif  // SLING_FILE_RECORDIO_H_

