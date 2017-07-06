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

#include "util/zip-iterator.h"

#include <iostream>

#include "base/logging.h"
#include "base/macros.h"

namespace sling {

void ZipIterator::Init() {
  file_ = unzOpen64(zip_filename_.data());
  CHECK(file_ != NULL);
  CHECK_EQ(unzGetGlobalInfo64(file_, &global_info_), UNZ_OK) << zip_filename_;
  buffer_ = (char *)malloc(kBufferSize);
  std::cout << global_info_.number_entry << " entries\n";
}

ZipIterator::~ZipIterator() {
  if (file_ != NULL) unzClose(file_);
  free(buffer_);
}

bool ZipIterator::Next(string *filename, string *contents) {
  if (file_ == NULL) Init();
  if (index_ == global_info_.number_entry) return false;  // no more files

  if (filename != nullptr) {
    unz_file_info64 file_info;
    char filenamebuffer[1024];
    CHECK_EQ(unzGetCurrentFileInfo64(file_, &file_info,
                                     filenamebuffer, sizeof(filenamebuffer),
                                     NULL, 0, NULL, 0), UNZ_OK) << index_;
    *filename = filenamebuffer;
  }

  CHECK_EQ(unzOpenCurrentFile(file_), UNZ_OK) << index_;

  int size = 0;
  contents->clear();
  do {
    size = unzReadCurrentFile(file_, static_cast<void *>(buffer_), kBufferSize);
    CHECK_GE(size, 0) << index_;
    if (size > 0) contents->append(buffer_, size);
  } while(size > 0);

  CHECK_EQ(unzCloseCurrentFile(file_), UNZ_OK) << index_;

  index_++;
  if (index_ < global_info_.number_entry) {
    CHECK_EQ(unzGoToNextFile(file_), UNZ_OK) << index_;
  }

  return true;
}

}  // namespace sling

