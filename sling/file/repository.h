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

#ifndef SLING_FILE_REPOSITORY_H_
#define SLING_FILE_REPOSITORY_H_

#include <string>
#include <vector>

#include "sling/base/types.h"
#include "sling/base/logging.h"
#include "sling/file/file.h"

namespace sling {

class RepositoryMapItem;

// A repository consists of a number of named data blocks that can be stored
// on disk.
class Repository {
 public:
  // Current repository file format version and "magic" marker.
  static const uint32 kRepositoryVersion = 1;
  static const uint32 kRepositoryMagic = 1330660690;

  // This header record is located at the start of the repository file.
  struct Header {
    // Magic number for identifying this as an entity repository file.
    uint32 magic;

    // Entity repository format version.
    uint32 version;

    // Number of blocks in repository.
    uint32 blocks;

    // File position and size of the directory in the repository.
    int64 directory_position;
    int64 directory_size;
  };

  // Create an empty repository.
  Repository();
  ~Repository();

  // Open repository and read directory.
  void Open(const string &filename);

  // Close repository file.
  void Close();

  // Load all data blocks into repository.
  void LoadAll();

  // Load a data block from file into the repository.
  bool LoadBlock(const string &name);

  // Read repository from file.
  void Read(const string &filename);

  // Write repository to file.
  void Write(const string &filename);

  // Add memory block to repository store.
  void AddBlock(const string &name, const void *data, size_t size);

  // Add block and return a temporary file for writing the block.
  File *AddBlock(const string &name);

  // Return raw memory data blocks in repository.
  const char *GetBlock(const string &name) const;
  char *GetMutableBlock(const string &name);
  size_t GetBlockSize(const string &name) const;
  template<class T> void FetchBlock(const string &name,
                                    const T **data) const {
    *data = reinterpret_cast<const T *>(GetBlock(name));
  }

  // Write hash map to repository. This creates two blocks in the repository,
  // a map item block ("<name>Items"), which contains all the items, and a
  // bucket block ("<name>Buckets") with indices into the item block to the
  // start of each bucket. The bucket array contains an extra entry to mark
  // the end of the item block. The items array will be sorted in bucket order.
  void WriteMap(const string &name,
                std::vector<RepositoryMapItem *> *items,
                int num_buckets);

 private:
  // Data block in repository.
  struct Block {
    string name;
    char *data = nullptr;
    size_t size = 0;
    File *file = nullptr;
    uint64 position = 0;
    bool mmaped = false;
  };

  // Entry in repository directory.
  struct Entry {
    uint64 block_position;
    uint64 block_size;
    uint32 name_offset;
    uint32 name_size;
  };

  // Repository file.
  File *file_ = nullptr;

  // Data blocks for repository.
  std::vector<Block> blocks_;
};

// A repository index uses an index and a data block from the repository to
// implement indexed lookup of objects. The index block is an array (of type
// OFFSET). The offsets in the index tables are used for locating objects (of
// type OBJ) in the data block.
template<class OFFSET, class OBJ> class RepositoryIndex {
 public:
  RepositoryIndex() : index_(nullptr), data_(nullptr) {}

  RepositoryIndex(const Repository &repository,
                  const string &index_block,
                  const string &data_block) {
    Init(repository, index_block, data_block, false);
  }

  // Initialize repository index from index and data blocks. If optional is
  // true this method returns false if the blocks are not in the repository.
  // Otherwise we CHECK fault if the blocks are missing.
  bool Init(const Repository &repository,
            const string &index_block,
            const string &data_block,
            bool optional) {
    // Get memory blocks for index and data tables.
    repository.FetchBlock(index_block, &index_);
    data_ = repository.GetBlock(data_block);
    if (index_ != nullptr) {
      size_ = repository.GetBlockSize(index_block) / sizeof(OFFSET);
    }
    if (optional) {
      return index_ != nullptr && data_ != nullptr;
    } else {
      CHECK(index_ != nullptr) << "No repository index block " << index_block;
      CHECK(data_ != nullptr) << "No repository data block " << data_block;
      return true;
    }
  }

  // Return the number of items in the index.
  int64 size() const { return size_; }

 protected:
  // Return object by id from index.
  const OBJ *GetObject(int id) const {
    OFFSET offset = index_[id];
    return offset != -1 ? GetObjectAt(offset) : nullptr;
  }

  OBJ *GetMutableObject(int id) {
    OFFSET offset = index_[id];
    return offset != -1 ? GetMutableObjectAt(offset) : nullptr;
  }

  // Return object at a certain offset in the data block.
  const OBJ *GetObjectAt(OFFSET offset) const {
    return reinterpret_cast<const OBJ *>(data_ + offset);
  }

  OBJ *GetMutableObjectAt(OFFSET offset) {
    return const_cast<OBJ *>(reinterpret_cast<const OBJ *>(data_ + offset));
  }

 private:
  // Object index.
  const OFFSET *index_ = nullptr;

  // Object data.
  const char *data_ = nullptr;

  // Number of elements in index.
  int64 size_ = 0;
};

// Abstract repository map item class for writing hash maps to a repository.
// Sub-classes must implement the Write() and Hash() methods.
class RepositoryMapItem {
 public:
  RepositoryMapItem() : bucket_(0) {}
  virtual ~RepositoryMapItem() {}

  // Write item to file and return the number of bytes written.
  virtual int Write(File *file) const = 0;

  // Return hash value for item.
  virtual uint64 Hash() const = 0;

  // Compute bucket for item.
  void ComputeBucket(int num_buckets) { bucket_ = Hash() % num_buckets; }

  int bucket() const { return bucket_; }

 private:
  // Hash bucket number for item.
  int bucket_;
};

// Repository index used for implementing hash maps.
template<class OBJ> class RepositoryMap : public RepositoryIndex<uint64, OBJ> {
 public:
  using RepositoryIndex<uint64, OBJ>::Init;
  using RepositoryIndex<uint64, OBJ>::size;

  // Initialize buckets and items for map.
  void Init(const Repository &repository, const string &name) {
    string buckets = name + "Buckets";
    string items = name + "Items";
    if (RepositoryIndex<uint64, OBJ>::Init(repository, buckets, items, true)) {
      num_buckets_ = size() - 1;
    } else {
      num_buckets_ = 0;
    }
  }

  // Return the number of buckets.
  int num_buckets() const { return num_buckets_; }

 private:
  // Number of buckets. It is assumed that the bucket block contains an extra
  // entry for marking the end of the table.
  int num_buckets_ = 0;
};

// Base class for all objects in the repository. These objects have customized
// layouts and cannot be instantiated as normal objects using the new operator.
// Repository objects can have dynamically sized fields while still having a
// sequential memory layout.
class RepositoryObject {
 protected:
  // Return base pointer to the object. This is a char pointer to make it
  // easier to do pointer arithmetics.
  const char *base() const {
    return reinterpret_cast<const char *>(this);
  }
};

// This macro defines methods for accessing a field in a repository object.
// The following methods are defined:
// <name>_size(): Return the size of the field.
// <name>_offset(): Return the base offset of the field.
// <name>_ptr: Return a const pointer to the field.
// mutable_<name>_ptr(): Return a non-const pointer to the field.
#define REPOSITORY_FIELD(type, name, elements, offset) \
  int name##_size() const { \
    return sizeof(type) * (elements); \
  } \
  int name##_offset() const { \
    return offset; \
  } \
  const type *name##_ptr() const { \
    return reinterpret_cast<const type *>(base() + name##_offset()); \
  } \
  type *mutable_##name##_ptr() { \
    return const_cast<type *>(name##_ptr()); \
  }

// Return the offset after a field, i.e. its offset plus its size.
#define AFTER(name) name##_offset() + name##_size()

}  // namespace sling

#endif  // SLING_FILE_REPOSITORY_H_

