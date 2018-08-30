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

// Record file indexing tool.

#include <iostream>
#include <string>

#include "sling/base/init.h"
#include "sling/base/flags.h"
#include "sling/base/logging.h"
#include "sling/base/types.h"
#include "sling/file/file.h"
#include "sling/file/recordio.h"

sling::RecordFileOptions options;

DEFINE_bool(check, false, "Check record file options");

DEFINE_int32(buffer_size, options.buffer_size,
             "Input/output buffer size");
DEFINE_int32(chunk_size, options.chunk_size,
             "Size of each record chunk");
DEFINE_int32(compression, options.compression,
             "Record file compression");
DEFINE_int32(index_page_size, options.index_page_size,
             "Number of entries in each index record");
DEFINE_int32(index_cache_size, options.index_cache_size,
             "Size of index page cache");

using namespace sling;

int main(int argc, char *argv[]) {
  InitProgram(&argc, &argv);

  // Set record file options.
  options.buffer_size = FLAGS_buffer_size;
  options.chunk_size = FLAGS_chunk_size;
  options.compression =
      static_cast<RecordFile::CompressionType>(FLAGS_compression);
  options.index_page_size = FLAGS_index_page_size;
  options.index_cache_size = FLAGS_index_cache_size;

  // Get files to index.
  std::vector<string> files;
  for (int i = 1; i < argc; ++i) {
    File::Match(argv[i], &files);
  }

  if (FLAGS_check) {
    // Output information for each record file.
    for (const string &file : files) {
      RecordReader reader(file, options);
      auto &info = reader.info();
      const char *version = "?";
      if (info.magic == RecordFile::MAGIC1) version = "1";
      if (info.magic == RecordFile::MAGIC2) version = "2";

      std::cout << file << ":"
                << " version " << version
                << " data size: " << reader.size()
                << " compression: " << static_cast<int>(info.compression)
                << " chunk size: " << info.chunk_size
                << " indexed: " << (info.index_root != 0 ? "yes" : "no");
      if (info.index_root != 0) {
        size_t size = reader.file()->Size();
        std::cout << " index size: " << (size - reader.size())
                  << " index page size: " << info.index_page_size
                  << " index depth: " << info.index_depth;
      }
      std::cout << "\n";
    }
  } else {
    // Add index to files.
    for (const string &file : files) {
      std::cout << "Indexing " << file << "\n";
      CHECK(RecordWriter::AddIndex(file, options));
    }
    std::cout << "Done.\n";
  }

  return 0;
}
