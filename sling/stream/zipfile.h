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

#ifndef SLING_STREAM_ZIPFILE_H_
#define SLING_STREAM_ZIPFILE_H_

#include <string>
#include <vector>

#include "sling/base/types.h"
#include "sling/base/port.h"
#include "sling/file/file.h"
#include "sling/stream/stream.h"

namespace sling {

// ZIP file reader.
class ZipFileReader {
 public:
  // Compression methods.
  enum CompressionMethod {STORED, DEFLATE, UNSUPPORTED};

  // File entry information.
  struct Entry {
    string filename;           // filename
    uint32 offset;             // offset of file in archive
    uint32 size;               // uncompressed size
    uint32 compressed;         // compressed size
    CompressionMethod method;  // compression method
  };

  // Open and read ZIP file catalog.
  explicit ZipFileReader(const string &filename, int block_size = 1 << 16);
  ~ZipFileReader();

  // Return list of files in archive.
  const std::vector<Entry> &files() const { return files_; }

  // Return stream for reading file from archive.
  InputStream *Read(const Entry &entry);

 private:
  // End of central directory record (EOCD).
  struct EOCDRecord {
    uint32 signature;   // end of central directory signature = 0x06054b50
    uint16 disknum;     // number of this disk
    uint16 dirdisk;     // disk where central directory starts
    uint16 diskrecs;    // number of central directory records on this disk
    uint16 numrecs;     // total number of central directory records
    uint32 dirsize;     // size of central directory (bytes)
    uint32 dirofs;      // offset of start of central directory
    uint16 commentlen;  // comment length
  } ABSL_ATTRIBUTE_PACKED;

  // Locator for 64-bit EOCD.
  struct EOCD64Locator {
    uint32 signature;       // locator signature = 0x07064b50
    uint32 disknum;         // number of this disk
    uint64 eocd64offset;    // offset of EOCD64Record
    uint32 totaldisks;      // total number of disks
  } ABSL_ATTRIBUTE_PACKED;

  // 64-bit EOCDRecord.
  struct EOCD64Record {
    uint32 signature;   // end of central directory signature = 0x06064b50
    uint64 eocd64size;  // size in bytes of subsequent fields of EOCD64Record
    uint16 creator;     // creator version
    uint16 extractor;   // extractor version
    uint32 disknum;     // number of this disk
    uint32 dirdisk;     // disk where central directory starts
    uint64 diskrecs;    // number of central directory records on this disk
    uint64 numrecs;     // total number of central directory records
    uint64 dirsize;     // number of central directory (bytes)
    uint64 dirofs;      // offset of start of central directory
  } ABSL_ATTRIBUTE_PACKED;

  // Central directory file record.
  struct CDFile {
    uint32 signature;     // central directory file signature = 0x02014b50
    uint16 version;       // creator version
    uint16 minversion;    // version needed to extract
    uint16 flags;         // general purpose bit flag
    uint16 method;        // compression method
    uint16 mtime;         // file last modification time
    uint16 mdate;         // file last modification date
    uint32 crc32;         // CRC-32 checksum
    uint32 compressed;    // compressed size
    uint32 uncompressed;  // uncompressed size
    uint16 fnlen;         // file name length
    uint16 extralen;      // extra field length
    uint16 commentlen;    // file comment length
    uint16 disknum;       // disk number where file starts
    uint16 intattrs;      // internal file attributes
    uint32 extattrs;      // external file attributes
    uint32 offset;        // relative offset of local file header
  } ABSL_ATTRIBUTE_PACKED;

  // File header record.
  struct FileHeader {
    uint32 signature;     // file header signature = 0x04034b50
    uint16 minversion;    // version needed to extract
    uint16 flags;         // general purpose bit flag
    uint16 method;        // compression method
    uint16 mtime;         // file last modification time
    uint16 mdate;         // file last modification date
    uint32 crc32;         // CRC-32 checksum
    uint32 compressed;    // compressed size
    uint32 uncompressed;  // uncompressed size
    uint16 fnlen;         // file name length
    uint16 extralen;      // extra field length
  } ABSL_ATTRIBUTE_PACKED;

  // ZIP file.
  File *file_;

  // Block size.
  int block_size_;

  // List of files in ZIP archive.
  std::vector<Entry> files_;
};

}  // namespace sling

#endif  // SLING_STREAM_ZIPFILE_H_

