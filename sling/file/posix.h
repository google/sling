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

#ifndef SLING_FILE_POSIX_H_
#define SLING_FILE_POSIX_H_

#include <string>

#include "sling/base/types.h"
#include "sling/file/file.h"

namespace sling {

// Create file from POSIX file descriptor. The returned file takes ownership
// of the file descriptor.
File *NewFileFromDescriptor(const string &name, int fd);

// Create file for standard output.
File *NewStdoutFile();

}  // namespace sling

#endif  // SLING_FILE_POSIX_H_

