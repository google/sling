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

#include "sling/stream/zipfile.h"

#include "sling/base/logging.h"
#include "sling/stream/bounded.h"
#include "sling/stream/file.h"
#include "sling/stream/file-input.h"
#include "sling/stream/gzip.h"

namespace sling {

ZipFileReader::ZipFileReader(const string &filename, int block_size) {
  // Open ZIP file for reading.
  file_ = File::OpenOrDie(filename, "r");
  block_size_ = block_size;

  // Read EOCD record.
  uint64 size = file_->Size();
  CHECK_GE(size, sizeof(EOCDRecord));
  CHECK(file_->Seek(size - sizeof(EOCDRecord)));
  EOCDRecord eocd;
  file_->ReadOrDie(&eocd, sizeof(EOCDRecord));
  CHECK_EQ(eocd.signature, 0x06054b50);
  CHECK_LE(eocd.dirofs + eocd.dirsize, size);
  uint64 num_records = eocd.numrecs;

  // Read the 64-bit version of the record, if any. If found, this will
  // supersede the ordinary record read above.
  int locator_size = sizeof(EOCD64Locator);
  int64 locator_offset = size - sizeof(EOCDRecord) - locator_size;
  if (locator_offset >= 0) {
    CHECK(file_->Seek(locator_offset));
    EOCD64Locator locator;
    file_->ReadOrDie(&locator, locator_size);
    if (locator.signature == 0x07064b50) {
      // 64-bit locator is present. Get the offset of the EOCD64Record.
      CHECK_EQ(locator.disknum, 0);
      CHECK_EQ(locator.totaldisks, 1);
      CHECK(file_->Seek(locator.eocd64offset));

      // Read the 64-bit record.
      EOCD64Record eocd64;
      uint32 eocd64size = sizeof(EOCD64Record);
      file_->ReadOrDie(&eocd64, eocd64size);
      CHECK_EQ(eocd64.signature, 0x06064b50);
      CHECK_EQ(eocd64.eocd64size, eocd64size - sizeof(uint32) - sizeof(uint64));
      CHECK_EQ(eocd64.disknum, 0);
      CHECK_EQ(eocd64.dirdisk, 0);
      CHECK_EQ(eocd64.dirofs, eocd.dirofs);
      CHECK_EQ(eocd64.dirsize, eocd.dirsize);
      CHECK_EQ(eocd64.diskrecs, eocd64.numrecs);

      // Override the number of entries.
      num_records = eocd64.numrecs;
    }
  }

  // Read file directory.
  char *directory = new char[eocd.dirsize];
  CHECK(file_->Seek(eocd.dirofs));
  file_->ReadOrDie(directory, eocd.dirsize);
  char *dirptr = directory;
  char *dirend = dirptr + eocd.dirsize;
  files_.resize(num_records);
  for (int i = 0; i < num_records; ++i) {
    // Get next entry in directory.
    CHECK_LE(dirptr + sizeof(CDFile), dirend);
    CDFile *entry = reinterpret_cast<CDFile *>(dirptr);
    CHECK_EQ(entry->signature, 0x02014b50);
    dirptr += sizeof(CDFile);

    // Get filename.
    size_t fnlen = entry->fnlen;
    CHECK_LE(dirptr + fnlen, dirend);
    string filename(dirptr, fnlen);

    // Add file to file directory list.
    files_[i].filename = filename;
    files_[i].size = entry->uncompressed;
    files_[i].compressed = entry->compressed;
    files_[i].offset = entry->offset;
    switch (entry->method) {
      case 0: files_[i].method = STORED; break;
      case 8: files_[i].method = DEFLATE; break;
      default: files_[i].method = UNSUPPORTED;
    }

    // Move to next directory entry.
    dirptr += entry->fnlen + entry->extralen + entry->commentlen;
  }
  delete [] directory;
}

ZipFileReader::~ZipFileReader() {
  CHECK(file_->Close());
}

InputStream *ZipFileReader::Read(const Entry &entry) {
  // Read file header.
  FileHeader header;
  CHECK(file_->Seek(entry.offset));
  file_->ReadOrDie(&header, sizeof(header));
  CHECK_EQ(header.signature, 0x04034b50);
  CHECK(file_->Skip(header.fnlen + header.extralen));

  // Set up input pipeline.
  InputPipeline *pipeline = new InputPipeline();
  pipeline->Add(new FileInputStream(file_, false, block_size_));
  pipeline->Add(new BoundedInputStream(pipeline->last(), entry.compressed));
  switch (entry.method) {
    case STORED:
      break;
    case DEFLATE:
      pipeline->Add(new GZipDecompressor(pipeline->last(), block_size_, -15));
      break;
    case UNSUPPORTED:
      LOG(FATAL) << "Unsupported compression type";
      break;
  }

  return pipeline;
}

}  // namespace sling

