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

#include "sling/file/repository.h"

#include <stdlib.h>
#include <algorithm>
#include <string>
#include <vector>

#include "sling/base/types.h"
#include "sling/base/logging.h"
#include "sling/base/status.h"
#include "sling/base/types.h"
#include "sling/file/file.h"

namespace sling {

Repository::Repository()  {}

Repository::~Repository()  {
  // Delete all memory blocks and temporary files.
  for (Block &block : blocks_) {
    if (block.mmaped) {
      File::FreeMappedMemory(block.data, block.size);
    } else {
      free(block.data);
    }
    if (block.file != nullptr) {
      string tmpfile = block.file->filename();
      block.file->Close();
      File::Delete(tmpfile);
    }
  }

  // Close repository file.
  if (file_ != nullptr) {
    CHECK(file_->Close());
  }
}

void Repository::Open(const string &filename) {
  // Make sure that repository has not already been opened.
  CHECK(file_ == nullptr) << "Repository already opened";

  // Open repository file.
  file_ = File::OpenOrDie(filename, "r");

  // Read header from file.
  Header header;
  uint64 bytes;
  CHECK(file_->PRead(0, &header, sizeof(Header), &bytes));
  CHECK_EQ(bytes, sizeof(Header))
      << "Unable to read repository file header: " << filename;

  // Check magic signature and version.
  CHECK_EQ(header.magic, kRepositoryMagic)
      << "Invalid repository file: " << filename;
  CHECK(header.version == kRepositoryVersion)
      << "Unsupported repository file version: " <<  header.version;

  // Read repository directory.
  char *directory = static_cast<char *>(malloc(header.directory_size));
  CHECK(file_->PRead(header.directory_position,
                     directory,
                     header.directory_size, &bytes));
  CHECK_EQ(bytes, header.directory_size)
      << "Unable to read repository directory: " << filename;

  // Add blocks from directory.
  Entry *entries = reinterpret_cast<Entry *>(directory);
  for (int i = 0; i < header.blocks; ++i) {
    Entry &entry = entries[i];
    Block block;
    block.position = entry.block_position;
    block.size = entry.block_size;
    block.name.assign(directory + entry.name_offset, entry.name_size);
    blocks_.push_back(block);
  }
  free(directory);
}

void Repository::Close() {
  if (file_ != nullptr) {
    CHECK(file_->Close());
    file_ = nullptr;
  }
}

void Repository::LoadAll() {
  for (const Block &block : blocks_) LoadBlock(block.name);
}

bool Repository::LoadBlock(const string &name) {
  // Find block.
  int block_index = -1;
  for (int i = 0; i < blocks_.size(); ++i) {
    if (name == blocks_[i].name) {
      block_index = i;
      break;
    }
  }

  // Return false if block was not found.
  if (block_index == -1) return false;

  // Check if block has already been loaded.
  Block &block = blocks_[block_index];
  if (block.data != nullptr) return true;

  // Try to memory-map block.
  void *mapping = file_->MapMemory(block.position, block.size);
  if (mapping != nullptr) {
    VLOG(3) << "Mapped block " << name << " (" << block.size << " bytes)";
    block.data = static_cast<char *>(mapping);
    block.mmaped = true;
  } else {
    // Allocate memory block.
    block.data = reinterpret_cast<char *>(malloc(block.size));
    CHECK(block.data != nullptr)
         << "Unable to allocate " << block.size << " bytes for block " << name;

    // Read data.
    VLOG(3) << "Reading block " << name << " (" << block.size << " bytes)";
    CHECK(file_ != nullptr);
    uint64 bytes;
    CHECK(file_->PRead(block.position, block.data, block.size, &bytes));
    CHECK_EQ(bytes, block.size)
        << "Could not read block " << name << " from repository";
  }

  return true;
}

void Repository::Read(const string &filename) {
  // Open repository.
  Open(filename);

  // Load all blocks.
  LoadAll();

  // Close repository file.
  Close();
}

void Repository::Write(const string &filename) {
  // Open output file.
  File *output = File::OpenOrDie(filename, "w");

  // Setup positions and sizes of each block.
  size_t pagesize = File::PageSize();
  int64 position = sizeof(Header);
  int names_size = 0;
  for (Block &block : blocks_) {
    // Get block size for file-based blocks.
    if (block.file != nullptr) {
      block.size = block.file->Size();
    }

    // Try to align large blocks on page boundaries.
    if (position % pagesize != 0 && block.size >= pagesize) {
      position += pagesize - position % pagesize;
    }

    block.position = position;
    position += block.size;
    names_size += block.name.size();
  }

  // Set up repository header.
  int directory_size = blocks_.size() * sizeof(Entry) + names_size;
  Header header;
  header.magic = kRepositoryMagic;
  header.version = kRepositoryVersion;
  header.blocks = blocks_.size();
  header.directory_position = position;
  header.directory_size = directory_size;

  // Build directory.
  char *directory = static_cast<char *>(malloc(directory_size));
  int name_offset = blocks_.size() * sizeof(Entry);
  Entry *entries = reinterpret_cast<Entry *>(directory);
  for (int i = 0; i < blocks_.size(); ++i) {
    Block &block = blocks_[i];
    Entry &entry = entries[i];
    entry.block_position = block.position;
    entry.block_size = block.size;
    entry.name_offset = name_offset;
    entry.name_size = block.name.size();
    memcpy(directory + name_offset, block.name.data(), block.name.size());
    name_offset += block.name.size();
  }

  // Write header record.
  output->WriteOrDie(&header, sizeof(Header));

  // Write each block.
  for (Block &block : blocks_) {
    CHECK(output->Seek(block.position));
    if (block.data != nullptr) {
      // Write data block.
      output->WriteOrDie(block.data, block.size);
    } else if (block.file != nullptr) {
      // Copy block from temporary file.
      static const int kBufferSize = 8192;
      CHECK(block.file->Seek(0));
      char buffer[kBufferSize];
      uint64 bytes;
      for (;;) {
        CHECK(block.file->Read(buffer, kBufferSize, &bytes));
        if (bytes == 0) break;
        output->WriteOrDie(buffer, bytes);
      }
    }
  }

  // Write directory.
  output->WriteOrDie(directory, directory_size);
  free(directory);

  // Close output file.
  CHECK(output->Close());
}

void Repository::AddBlock(const string &name,
                          const void *data,
                          size_t size) {
  Block block;
  block.name = name;
  block.size = size;
  block.data = reinterpret_cast<char *>(malloc(block.size));
  CHECK(block.data != nullptr);
  memcpy(block.data, data, size);
  blocks_.push_back(block);
}

File *Repository::AddBlock(const string &name) {
  Block block;
  block.name = name;
  block.file = File::TempFile();
  blocks_.push_back(block);
  return block.file;
}

const char *Repository::GetBlock(const string &name) const {
  for (const Block &block : blocks_) {
    if (block.name == name) return block.data;
  }
  return nullptr;
}

char *Repository::GetMutableBlock(const string &name) {
  for (Block &block : blocks_) {
    if (block.name == name) return block.data;
  }
  return nullptr;
}

size_t Repository::GetBlockSize(const string &name) const {
  for (const Block &block : blocks_) {
    if (block.name == name) {
      return block.file != nullptr ? block.file->Size() : block.size;
    }
  }
  return 0;
}

void Repository::WriteMap(const string &name,
                          std::vector<RepositoryMapItem *> *items,
                          int num_buckets) {
  // Compute bucket for each item.
  for (RepositoryMapItem *item : *items) item->ComputeBucket(num_buckets);

  // Sort items in bucket order.
  std::sort(items->begin(), items->end(),
      [](const RepositoryMapItem *a, const RepositoryMapItem *b) {
        return a->bucket() < b->bucket();
      });

  // Create item block.
  File *data = AddBlock(name + "Items");

  // Allocate bucket array. We allocate one extra bucket to mark the end of the
  // map items. This ensures that all items in a bucket b are in the range
  // from bucket[b] to bucket[b + 1], even for the last bucket.
  std::vector<uint64> buckets(num_buckets + 1);
  uint64 offset = 0;
  int bucket = -1;
  for (const RepositoryMapItem *item : *items) {
    // Write item to block.
    int size = item->Write(data);

    // Update bucket table.
    while (bucket < item->bucket()) buckets[++bucket] = offset;

    // Move data item position forward to next item.
    offset += size;
  }
  while (bucket < num_buckets) buckets[++bucket] = offset;

  // Write bucket table to repository.
  int64 size = (num_buckets + 1) * sizeof(uint64);
  AddBlock(name + "Buckets", buckets.data(), size);
}

}  // namespace sling

