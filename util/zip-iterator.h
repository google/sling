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

#ifndef UTIL_ZIP_ITERATOR_H_
#define UTIL_ZIP_ITERATOR_H_

#include <string>

#include "third_party/zlib/upstream/contrib/minizip/unzip.h"

namespace sling {

// Iterates over files in a zip archive.
class ZipIterator {
 public:
  ZipIterator(const std::string &zip_filename) : zip_filename_(zip_filename) {}
  ~ZipIterator();

  // Outputs the name and contents of the next entry in the zip archive.
  // Returns true if successful and false otherwise.
  bool Next(std::string *filename, std::string *contents);

 private:
  void Init();

  // Name of the zip file.
  std::string zip_filename_;

  // Underlying pointer to the current file.
  unzFile file_ = NULL;

  // Index of the current file.
  int index_ = 0;

  // Zip archive info.
  unz_global_info64 global_info_;

  // Buffer array where the files are read.
  char *buffer_ = nullptr;

  // Max size of the buffer.
  static constexpr int kBufferSize = 2000000;
};

}  // namespace sling

#endif // UTIL_ZIP_ITERATOR_H_
