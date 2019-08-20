// Copyright 2018 Google Inc.
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

#ifndef SLING_FRAME_SNAPSHOT_H_
#define SLING_FRAME_SNAPSHOT_H_

#include <string>

#include "sling/base/status.h"
#include "sling/base/types.h"
#include "sling/frame/store.h"

namespace sling {

// Global frame stores can be snapshot and saved to .snap files. These can then
// be loaded into a new empty global store. For large stores, this is faster
// than reading the frame store in encoded format.
class Snapshot {
 public:
  // Filename for snapshot.
  static string Filename(const string &filename);

  // Check if there is a valid snapshot file for the store.
  static bool Valid(const string &filename);

  // Read snapshot into empty global store.
  static Status Read(Store *store, const string &filename);

  // Write store to snapshot file.
  static Status Write(Store *store, const string &filename);

 private:
  // Current magic and version for snapshots.
  static const int MAGIC = 0x50414e53;
  static const int VERSION = 2;

  // Snapshot file header.
  struct Header {
    int magic;      // magic number for identifying snapshot file
    int version;    // snapshot file format version
    int heaps;      // number of heaps in snapshot
    int handles;    // size of handle table
    Word symtab;    // symbol table handle
    int symbols;    // number of symbols in symbol table
    int buckets;    // number of hash buckets in the symbol table
    int symheap;    // heap for symbol table (-1 means no separate heap)
  };
};

}  // namespace sling

#endif  // SLING_FRAME_SNAPSHOT_H_

