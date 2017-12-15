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

#include "sling/file/recordio.h"

#include "sling/base/logging.h"
#include "sling/base/types.h"
#include "sling/util/varint.h"
#include "third_party/snappy/snappy.h"
#include "third_party/snappy/snappy-sinksource.h"

namespace sling {

namespace {

// Default record file options.
RecordFileOptions default_options;

// Slice compression source.
class SliceSource : public snappy::Source {
 public:
  SliceSource(const Slice &slice) : slice_(slice) {}

  size_t Available() const override {
    return slice_.size() - pos_;
  }

  const char *Peek(size_t *len) override {
    *len = slice_.size() - pos_;
    return slice_.data() + pos_;
  }

  void Skip(size_t n) override {
    pos_ += n;
  }

 private:
  Slice slice_;
  int pos_ = 0;
};

}  // namespace

RecordBuffer::~RecordBuffer() {
  free(floor_);
}

void RecordBuffer::resize(size_t bytes) {
  size_t size = ceil_ - floor_;
  if (size != bytes) {
    size_t offset = begin_ - floor_;
    size_t used = end_ - begin_;
    floor_ = static_cast<char *>(realloc(floor_, bytes));
    CHECK(floor_ != nullptr);
    ceil_ = floor_ + bytes;
    begin_ = floor_ + offset;
    end_ = begin_ + used;
  }
}

void RecordBuffer::ensure(size_t bytes) {
  size_t minsize = end_ - floor_ + bytes;
  size_t newsize = ceil_ - floor_;
  if (newsize == 0) newsize = 4096;
  while (newsize < minsize) newsize *= 2;
  resize(newsize);
}

void RecordBuffer::flush() {
  if (begin_ > floor_) {
    size_t used = end_ - begin_;
    memmove(floor_, begin_, used);
    begin_ = floor_;
    end_ = begin_ + used;
  }
}

void RecordBuffer::Append(const char *bytes, size_t n) {
  ensure(n);
  if (bytes != end_) memcpy(end_, bytes, n);
  end_ += n;
}

char *RecordBuffer::GetAppendBuffer(size_t length, char *scratch) {
  ensure(length);
  return end_;
}

char *RecordBuffer::GetAppendBufferVariable(
    size_t min_size, size_t desired_size_hint,
    char *scratch, size_t scratch_size,
    size_t *allocated_size) {
  if (size() < min_size) {
    ensure(desired_size_hint > 0 ? desired_size_hint : min_size);
  }
  *allocated_size = remaining();
  return end_;
}

size_t RecordBuffer::Available() const {
  return end_ - begin_;
}

const char *RecordBuffer::Peek(size_t *len) {
  *len = end_ - begin_;
  return begin_;
}

void RecordBuffer::Skip(size_t n) {
  DCHECK_LE(n, end_ - begin_);
  begin_ += n;
}

int RecordFile::ReadHeader(const char *data, Header *header) {
  // Read record type.
  const char *p = data;
  header->record_type = static_cast<RecordType>(*p++);
  if (header->record_type > FILLER_RECORD) return -1;

  // Read record length.
  p = Varint::Parse64(p, &header->record_size);
  if (!p) return -1;

  // Read key length for data records.
  if (header->record_type == DATA_RECORD) {
    p = Varint::Parse64(p, &header->key_size);
  } else {
    header->key_size = 0;
  }

  // Return number of bytes consumed.
  return p - data;
}

int RecordFile::WriteHeader(const Header &header, char *data) {
  // Write record type.
  char *p = data;
  *p++ = header.record_type;

  // Write record length.
  p = Varint::Encode64(p, header.record_size);

  // Write key length for data records.
  if (header.record_type == DATA_RECORD) {
    p = Varint::Encode64(p, header.key_size);
  }

  // Return number of bytes written.
  return p - data;
}

RecordReader::RecordReader(File *file, const RecordFileOptions &options)
    : file_(file) {
  // Allocate input buffer.
  CHECK_GE(options.buffer_size, sizeof(FileHeader));
  input_.resize(options.buffer_size);

  // Read record file header.
  CHECK(Fill());
  CHECK_GE(input_.size(), sizeof(FileHeader))
      << "Record file truncated: " << file->filename();
  memcpy(&info_, input_.begin(), sizeof(FileHeader));
  CHECK_EQ(info_.magic, MAGIC)
      << "Not a record file: " << file->filename();
  input_.consumed(sizeof(FileHeader));
  position_ = sizeof(FileHeader);
  CHECK(file_->GetSize(&size_));
}

RecordReader::RecordReader(const string &filename,
                           const RecordFileOptions &options)
    : RecordReader(File::OpenOrDie(filename, "r"), options) {}

RecordReader::RecordReader(File *file)
    : RecordReader(file, default_options) {}

RecordReader::RecordReader(const string &filename)
    : RecordReader(filename, default_options) {}

RecordReader::~RecordReader() {
  CHECK(Close());
}

Status RecordReader::Close() {
  if (file_) {
    Status s = file_->Close();
    file_ = nullptr;
    if (!s.ok()) return s;
  }
  return Status::OK;
}

Status RecordReader::Fill() {
  input_.flush();
  uint64 bytes;
  Status s = file_->Read(input_.end(), input_.remaining(), &bytes);
  if (!s.ok()) return s;
  input_.appended(bytes);
  return Status::OK;
}

Status RecordReader::Read(Record *record) {
  // Keep reading until we read a data record.
  for (;;) {
    // Fill input buffer if it is nearly empty.
    if (input_.size() < MAX_HEADER_LEN) {
      Status s = Fill();
      if (!s.ok()) return s;
    }

    // Read record header.
    Header hdr;
    int hdrsize = ReadHeader(input_.begin(), &hdr);
    if (hdrsize < 0) return Status(1, "Corrupt record header");

    // Skip filler records.
    if (hdr.record_type == FILLER_RECORD) {
      Status s = Skip(hdr.record_size);
      if (!s.ok()) return s;
      continue;
    } else {
      input_.consumed(hdrsize);
      position_ += hdrsize;
    }

    // Read record into input buffer.
    if (hdr.record_size > input_.size()) {
      // Expand input buffer if needed.
      if (hdr.record_size > input_.capacity()) {
        input_.resize(hdr.record_size);
      }

      // Read more data into input buffer.
      Status s = Fill();
      if (!s.ok()) return s;

      // Make sure we have enough data.
      if (hdr.record_size > input_.size()) {
        return Status(1, "Record truncated");
      }
    }

    // Get record key.
    if (hdr.key_size > 0) {
      record->key = Slice(input_.begin(), hdr.key_size);
      input_.consumed(hdr.key_size);
    } else {
      record->key = Slice();
    }

    // Get record value.
    size_t value_size = hdr.record_size - hdr.key_size;
    if (info_.compression == SNAPPY) {
      // Decompress record value.
      decompressed_data_.clear();
      snappy::ByteArraySource source(input_.begin(), value_size);
      CHECK(snappy::Uncompress(&source, &decompressed_data_));
      input_.consumed(value_size);
      record->value =
          Slice(decompressed_data_.begin(), decompressed_data_.end());
    } else if (info_.compression == UNCOMPRESSED) {
      record->value = Slice(input_.begin(), value_size);
      input_.consumed(value_size);
    } else {
      return Status(1, "Unknown compression type");
    }

    position_ += hdr.record_size;
    return Status::OK;
  }
}

Status RecordReader::Skip(int64 n) {
  // Check if we can skip to position in input buffer.
  position_ += n;
  char *ptr = input_.begin() + n;
  if (ptr >= input_.floor() && ptr < input_.end()) {
    input_.consumed(n);
    return Status::OK;
  }

  // Clear input buffer and seek to new position.
  int64 offset = n - input_.size();
  input_.clear();
  return file_->Skip(offset);
}

Status RecordReader::Seek(uint64 pos) {
  // Check if we can skip to position in input buffer.
  int64 offset = pos - position_;
  position_ = pos;
  char *ptr = input_.begin() + offset;
  if (ptr >= input_.floor() && ptr < input_.end()) {
    input_.consumed(offset);
    return Status::OK;
  }

  // Clear input buffer and seek to new position.
  input_.clear();
  return file_->Seek(pos);
}

RecordWriter::RecordWriter(File *file, const RecordFileOptions &options)
    : file_(file) {
  // Allocate output buffer.
  output_.resize(options.buffer_size);
  position_ = 0;

  // Write file header.
  memset(&info_, 0, sizeof(info_));
  info_.magic = MAGIC;
  info_.hdrlen = sizeof(info_);
  info_.compression = options.compression;
  info_.chunk_size = options.chunk_size;
  memcpy(output_.end(), &info_, sizeof(info_));
  output_.appended(sizeof(info_));
  position_ += sizeof(info_);
}

RecordWriter::RecordWriter(const string &filename,
                           const RecordFileOptions &options)
    : RecordWriter(File::OpenOrDie(filename, "w"), options) {}

RecordWriter::RecordWriter(File *file)
    : RecordWriter(file, default_options) {}

RecordWriter::RecordWriter(const string &filename)
    : RecordWriter(filename, default_options) {}

RecordWriter::~RecordWriter() {
  CHECK(Close());
}

Status RecordWriter::Close() {
  // Check if file has already been closed.
  if (file_ == nullptr) return Status::OK;

  // Flush output buffer.
  Status s = Flush();
  if (!s.ok()) return s;

  // Close output file.
  s = file_->Close();
  file_ = nullptr;

  return s;
}

Status RecordWriter::Flush() {
  if (output_.empty()) return Status::OK;
  Status s = file_->Write(output_.begin(), output_.size());
  if (!s.ok()) return s;
  output_.clear();
  return Status::OK;
}

Status RecordWriter::Write(const Record &record) {
  // Compress record value if requested.
  Slice value;
  if (info_.compression == SNAPPY) {
    // Compress record value.
    SliceSource source(record.value);
    compressed_data_.clear();
    snappy::Compress(&source, &compressed_data_);
    value = Slice(compressed_data_.begin(), compressed_data_.end());
  } else if (info_.compression == UNCOMPRESSED) {
    // Store uncompressed record value.
    value = record.value;
  } else {
    return Status(1, "Unknown compression type");
  }

  // Compute on-disk record size estimate.
  size_t maxsize = MAX_HEADER_LEN + record.key.size() + value.size();

  // Records cannot be bigger than the chunk size.
  size_t size_with_skip = maxsize + MAX_SKIP_LEN;
  CHECK_LE(size_with_skip, info_.chunk_size)
      << "Record too big (" << size_with_skip << " bytes), "
      << "maximum is " << info_.chunk_size << " bytes";

  // Flush output buffer if it does not have room for record.
  if (maxsize > output_.remaining()) {
    Status s = Flush();
    if (!s.ok()) return s;
  }

  // Check if record will cross chunk boundary.
  if (info_.chunk_size != 0) {
    uint64 chunk_used = position_ % info_.chunk_size;
    if (chunk_used + size_with_skip > info_.chunk_size) {
      // Write filler record. For a filler record, the record size includes
      // the header.
      Header filler;
      filler.record_type = FILLER_RECORD;
      filler.record_size = info_.chunk_size - chunk_used;
      filler.key_size = 0;
      int hdrlen = WriteHeader(filler, output_.end());
      output_.appended(hdrlen);

      // Flush output buffer.
      Status s = Flush();
      if (!s.ok()) return s;

      // Skip to next chunk boundary.
      position_ += filler.record_size;
      s = file_->Seek(position_);
      if (!s.ok()) return s;
    }
  }

  // Write record header.
  Header hdr;
  hdr.record_type = DATA_RECORD;
  hdr.record_size = record.key.size() + value.size();
  hdr.key_size = record.key.size();
  output_.ensure(maxsize);
  int hdrlen = WriteHeader(hdr, output_.end());
  output_.appended(hdrlen);
  position_ += hdrlen;

  // Write record key.
  if (record.key.size() > 0) {
    memcpy(output_.end(), record.key.data(), record.key.size());
    output_.appended(record.key.size());
    position_ += record.key.size();
  }

  // Write record value.
  memcpy(output_.end(), value.data(), value.size());
  output_.appended(value.size());
  position_ += value.size();

  return Status::OK;
}

}  // namespace sling

