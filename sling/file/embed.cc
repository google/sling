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

#include "sling/file/embed.h"

#include <fnmatch.h>
#include <stdio.h>
#include <string>
#include <unordered_map>

#include "sling/base/logging.h"
#include "sling/base/status.h"
#include "sling/base/types.h"
#include "sling/file/file.h"

namespace sling {

// Registered embedded files.
static std::vector<EmbeddedFile *> *registered_files = nullptr;

extern "C" {
void register_embedded_files(EmbeddedFile *files, int count);
}

// This function is called from the init function in each embedded data object
// file. It is called with an array of embedded file structures.
void register_embedded_files(EmbeddedFile *files, int count) {
  if (registered_files == nullptr) {
    registered_files = new std::vector<EmbeddedFile *>;
  }
  for (int i = 0; i < count; ++i) {
    registered_files->push_back(&files[i]);
  }
}

// Internal file.
class InternalFile : public File {
 public:
  InternalFile(EmbeddedFile *file) : file_(file) {}

  Status PRead(uint64 pos, void *buffer, size_t size, uint64 *read) override {
    ssize_t bytes = size;
    if (pos + bytes > file_->size) bytes = file_->size - pos;
    if (bytes < 0) bytes = 0;
    if (bytes > 0) memcpy(buffer, file_->data + pos, bytes);
    if (read) *read = bytes;
    return Status::OK;
  }

  Status Read(void *buffer, size_t size, uint64 *read) override {
    uint64 bytes;
    Status st = PRead(position_, buffer, size, &bytes);
    if (!st.ok()) return st;
    position_ += bytes;
    if (read) *read = bytes;
    return Status::OK;
  }

  Status PWrite(uint64 pos, const void *buffer, size_t size) override {
    return Status(EACCES, "File read-only", file_->name);
  }

  Status Write(const void *buffer, size_t size) override {
    return Status(EACCES, "File read only", file_->name);
  }

  Status Seek(uint64 pos) override {
    position_ = pos;
    return Status::OK;
  }

  Status Skip(uint64 n) override {
    position_ += n;
    return Status::OK;
  }

  Status GetPosition(uint64 *pos) override {
    *pos = position_;
    return Status::OK;
  }

  Status GetSize(uint64 *size) override {
    *size = file_->size;
    return Status::OK;
  }

  Status Stat(FileStat *stat) override {
    stat->size = file_->size;
    stat->mtime = file_->mtime;
    stat->is_file = true;
    stat->is_directory = false;
    return Status::OK;
  }

  Status Close() override {
    delete this;
    return Status::OK;
  }

  Status Flush() override {
    return Status::OK;
  }

  string filename() const override { return file_->name; }

 private:
  // Embedded file information and content.
  EmbeddedFile *file_;

  // Current position.
  ssize_t position_ = 0;
};

// Internal directory.
class InternalDirectory : public File {
 public:
  InternalDirectory(const string &name) : name_(name) {}

  Status PRead(uint64 pos, void *buffer, size_t size, uint64 *read) override {
    if (read) *read = 0;
    return Status::OK;
  }

  Status Read(void *buffer, size_t size, uint64 *read) override {
    if (read) *read = 0;
    return Status::OK;
  }

  Status PWrite(uint64 pos, const void *buffer, size_t size) override {
    return Status(EISDIR, "Is directory", name_);
  }

  Status Write(const void *buffer, size_t size) override {
    return Status(EISDIR, "Is directory", name_);
  }

  Status Seek(uint64 pos) override {
    return Status::OK;
  }

  Status Skip(uint64 n) override {
    return Status::OK;
  }

  Status GetPosition(uint64 *pos) override {
    *pos = 0;
    return Status::OK;
  }

  Status GetSize(uint64 *size) override {
    *size = 0;
    return Status::OK;
  }

  Status Stat(FileStat *stat) override {
    stat->size = 0;
    stat->mtime = 0;
    stat->is_file = false;
    stat->is_directory = true;
    return Status::OK;
  }

  Status Close() override {
    delete this;
    return Status::OK;
  }

  Status Flush() override {
    return Status::OK;
  }

  string filename() const override { return name_; }

 private:
  // Directory name.
  string name_;
};

// Internal file system interface.
class InternalFileSystem : public FileSystem {
 public:
  InternalFileSystem() {
    if (instance == nullptr) instance = this;
  }

  ~InternalFileSystem() override {
    if (instance == this) instance = nullptr;
  }

  void Init() override {
    // Add registered files to file system.
    if (registered_files != nullptr) {
      string filename;
      string dirname;
      for (EmbeddedFile *f : *registered_files) {
        VLOG(9) << "Register embedded file: " << f->name;

        // Add file to file map.
        filename = f->name;
        files_[filename] = f;

        // Add directories.
        int pos = 0;
        int slash;
        while ((slash = filename.find('/', pos)) != -1) {
          dirname = filename.substr(0, slash);
          auto f = files_.find(dirname);
          if (f != files_.end()) {
            if (f->second != nullptr) {
              LOG(FATAL) << "Embedded file " << f->second->name
                         << " shadows directory for " << filename;
            }
          } else {
            // Add new directory.
            files_[dirname] = nullptr;
          }
          pos = slash + 1;
        }
      }
    }
  }

  bool IsDefaultFileSystem() override {
    return false;
  }

  Status Open(const string &name, const char *mode, File **f) override {
    File *file = LookupFile(name);
    if (file == nullptr) return Status(ENOENT, "File not found", name);
    *f = file;
    return Status::OK;
  }

  Status CreateTempFile(File **f) override {
    return Status(ENOSYS, "CreateTempFile not supported");
  }

  Status CreateTempDir(string *dir) override {
    return Status(ENOSYS, "CreateTempDir not supported");
  }

  bool FileExists(const string &filename) override {
    File *file = LookupFile(filename);
    if (file != nullptr) {
      CHECK(file->Close());
      return true;
    } else {
      return false;
    }
  }

  Status GetFileSize(const string &filename, uint64 *size) override {
    File *file = LookupFile(filename);
    if (file == nullptr) return Status(ENOENT, "File not found", filename);
    CHECK(file->GetSize(size));
    CHECK(file->Close());
    return Status::OK;
  }

  Status DeleteFile(const string &filename) override {
    return Status(ENOSYS, "DeleteFile not supported", filename);
  }

  Status Stat(const string &filename, FileStat *stat) override {
    File *file = LookupFile(filename);
    if (file == nullptr) return Status(ENOENT, "File not found", filename);
    CHECK(file->Stat(stat));
    CHECK(file->Close());
    return Status::OK;
  }

  Status RenameFile(const string &source, const string &target) override {
    return Status(ENOSYS, "RenameFile not supported", source);
  }

  Status CreateDir(const string &dirname) override {
    return Status(ENOSYS, "CreateDir not supported", dirname);
  }

  Status DeleteDir(const string &dirname) override {
    return Status(ENOSYS, "DeleteDir not supported", dirname);
  }

  Status Match(const string &pattern, std::vector<string> *filenames) override {
    string filename;
    if (pattern.find('*') != -1 || pattern.find('?') != -1) {
      // Find files matching pattern.
      for (const auto it : files_) {
        if (fnmatch(pattern.c_str(), it.first.c_str(), 0) == 0) {
          filename = "/intern/";
          filename.append(it.first);
          filenames->push_back(filename);
        }
      }
    } else {
      // Look up file.
      auto f = files_.find(pattern);
      if (f != files_.end()) {
        filename = "/intern/";
        filename.append(f->first);
        filenames->push_back(filename);
      }
    }
    return Status::OK;
  }

  // Look up file or directory object for file name.
  File *LookupFile(const string &name) {
    auto f = files_.find(name);
    if (f == files_.end()) return nullptr;
    if (f->second == nullptr) {
      return new InternalDirectory(name);
    } else {
      return new InternalFile(f->second);
    }
  }

  // Look up embedded file.
  static const EmbeddedFile *Lookup(const string &name) {
    if (instance == nullptr) return nullptr;
    auto f = instance->files_.find(name);
    if (f == instance->files_.end()) return nullptr;
    return f->second;
  }

 private:
  // Mapping from file names to embedded files.
  std::unordered_map<string, EmbeddedFile *> files_;

  // Instance of internal file system.
  static InternalFileSystem *instance;
};

REGISTER_FILE_SYSTEM_TYPE("intern", InternalFileSystem);

InternalFileSystem *InternalFileSystem::instance = nullptr;

const EmbeddedFile *GetEmbeddedFile(const string &name) {
  return InternalFileSystem::Lookup(name);
}

const char *GetEmbeddedFileContent(const string &name) {
  const EmbeddedFile *file = InternalFileSystem::Lookup(name);
  if (file == nullptr) return nullptr;
  return file->data;
}

}  // namespace sling

