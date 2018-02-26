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

#ifndef SLING_FILE_EMBED_H_
#define SLING_FILE_EMBED_H_

#include <stdint.h>
#include <string>

#include "sling/base/types.h"

namespace sling {

// File information for embedded files created with the embed-data tool.
struct EmbeddedFile {
  const char *name;    // file name
  uint64_t size;       // file size
  const char *data;    // file content
  uint64_t mtime;      // file modification time
};

// Find embedded file.
const EmbeddedFile *GetEmbeddedFile(const string &name);

// Return contents of embedded file.
const char *GetEmbeddedFileContent(const string &name);

}  // namespace sling

#endif  // SLING_FILE_EMBED_H_

