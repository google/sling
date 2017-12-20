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

#include "sling/file/file.h"

#include <pthread.h>
#include <string>
#include <unordered_map>

#include "sling/base/init.h"
#include "sling/base/logging.h"
#include "sling/base/registry.h"
#include "sling/base/status.h"
#include "sling/base/types.h"

// Registry for file systems.
REGISTER_SINGLETON_REGISTRY("file system", sling::FileSystem);

namespace sling {
namespace {

// The file systems need to be initialized at most once.
pthread_once_t file_systems_initialized = PTHREAD_ONCE_INIT;

// Registered file systems.
std::unordered_map<string, FileSystem *> file_systems;
FileSystem *default_file_system = nullptr;

// Initialize all registered file systems.
void InitializeFileSystems() {
  auto *registry = FileSystem::registry();
  for (auto *fs = registry->components; fs != nullptr; fs = fs->next()) {
    VLOG(2) << "Initializing " << fs->type() << " file system";
    fs->object()->Init();
    if (fs->object()->IsDefaultFileSystem()) {
      default_file_system = fs->object();
    } else {
      file_systems[fs->type()] = fs->object();
    }
  }
}

// Find file system for file name. If no matching file system is found, the
// default file system is returned.
FileSystem *FindFileSystem(const string &filename, string *rest) {
  // Initialize file systems if not already done.
  if (default_file_system == nullptr) File::Init();

  // Match the first component in the path.
  if (!filename.empty() && filename[0] == '/') {
    int slash = filename.find('/', 1);
    if (slash != -1) {
      auto f = file_systems.find(filename.substr(1, slash - 1));
      if (f != file_systems.end()) {
        *rest = filename.substr(slash + 1);
        return f->second;
      }
    }
  }

  // Fall back on the default file system.
  *rest = filename;
  return default_file_system;
}

Status NoFileSystem(const string &filename) {
  return Status(1, "No file system", filename);
}

}  // namespace

void File::Init() {
  // Allow this method to be called multiple times.
  pthread_once(&file_systems_initialized, InitializeFileSystems);
}

Status File::Open(const string &name, const char *mode, File **f) {
  // Find file system.
  string rest;
  FileSystem *fs = FindFileSystem(name, &rest);
  if (fs == nullptr) return NoFileSystem(name);

  // Open new file.
  return fs->Open(rest, mode, f);
}

File *File::Open(const string &name, const char *mode) {
  File *f;
  if (!Open(name, mode, &f).ok()) return nullptr;
  return f;
}

File *File::OpenOrDie(const string &name, const char *mode) {
  File *f;
  CHECK(Open(name, mode, &f));
  return f;
}

File *File::TempFile() {
  CHECK(default_file_system != nullptr) << "No filesystems";
  File *f;
  CHECK(default_file_system->CreateTempFile(&f));
  return f;
}

Status File::Delete(const string &name) {
  // Find file system.
  string rest;
  FileSystem *fs = FindFileSystem(name, &rest);
  if (fs == nullptr) return NoFileSystem(name);

  // Delete file.
  return fs->DeleteFile(rest);
}

bool File::Exists(const string &name) {
  // Find file system.
  string rest;
  FileSystem *fs = FindFileSystem(name, &rest);
  if (fs == nullptr) return false;

  // Check if file exists.
  return fs->FileExists(rest);
}

Status File::GetSize(const string &name, uint64 *size) {
  // Find file system.
  string rest;
  FileSystem *fs = FindFileSystem(name, &rest);
  if (fs == nullptr) return NoFileSystem(name);

  // Get file size.
  return fs->GetFileSize(rest, size);
}

Status File::Stat(const string &name, FileStat *stat) {
  // Find file system.
  string rest;
  FileSystem *fs = FindFileSystem(name, &rest);
  if (fs == nullptr) return NoFileSystem(name);

  // Get file size.
  return fs->Stat(rest, stat);
}

Status File::Rename(const string &source, const string &target) {
  // Find file system.
  string src;
  string tgt;
  FileSystem *srcfs = FindFileSystem(source, &src);
  FileSystem *tgtfs = FindFileSystem(target, &tgt);
  if (srcfs == nullptr) return NoFileSystem(source);
  if (tgtfs == nullptr) return NoFileSystem(target);
  if (srcfs != tgtfs) return Status(1, "Cross file system rename", source);

  // Rename file.
  return srcfs->RenameFile(src, tgt);
}

Status File::Mkdir(const string &dir) {
  // Find file system.
  string rest;
  FileSystem *fs = FindFileSystem(dir, &rest);
  if (fs == nullptr) return NoFileSystem(dir);

  // Create directory.
  return fs->CreateDir(rest);
}

Status File::Rmdir(const string &dir) {
  // Find file system.
  string rest;
  FileSystem *fs = FindFileSystem(dir, &rest);
  if (fs == nullptr) return NoFileSystem(dir);

  // Create directory.
  return fs->DeleteDir(rest);
}

Status File::CreateLocalTempDir(string *dir) {
  char tmpldir[] = "/tmp/local.XXXXXX";
  char *tmpname = mkdtemp(tmpldir);
  if (tmpname == nullptr) {
    return Status(errno, "mkdtemp", strerror(errno));
  }
  *dir = tmpname;
  return Status::OK;
}

Status File::CreateGlobalTempDir(string *dir) {
  char tmpldir[] = "/var/data/tmp/global.XXXXXX";
  char *tmpname = mkdtemp(tmpldir);
  if (tmpname == nullptr) {
    return Status(errno, "mkdtemp", strerror(errno));
  }
  *dir = tmpname;
  return Status::OK;
}

Status File::Match(const string &pattern, std::vector<string> *filenames) {
  // Find file system.
  string rest;
  FileSystem *fs = FindFileSystem(pattern, &rest);
  if (fs == nullptr) return NoFileSystem(pattern);

  // Convert sharded file pattern.
  int at = rest.find('@');
  if (at != string::npos) {
    string shards = rest.substr(at + 1);
    rest.resize(at);
    rest.append("-\?\?\?\?\?-of-");
    rest.append(5 - shards.size(), '0');
    rest.append(shards);
  }

  // Find files.
  return fs->Match(rest, filenames);
}

Status File::ReadContents(const string &filename, string *data) {
  // Open file for reading.
  File *f;
  Status st = Open(filename, "r", &f);
  if (!st.ok()) return st;

  // Read contents.
  st = f->ReadToString(data);
  if (!st.ok()) {
    f->Close();
    return st;
  }

  // Close file.
  return f->Close();
}

Status File::WriteContents(const string &filename,
                           const void *data,
                           size_t size) {
  // Open file for writing.
  File *f;
  Status st = Open(filename, "w", &f);
  if (!st.ok()) return st;

  // Write contents.
  st = f->Write(data, size);
  if (!st.ok()) {
    f->Close();
    return st;
  }

  // Close file.
  return f->Close();
}

size_t File::ReadOrDie(void *buffer, size_t size) {
  uint64 read;
  CHECK(Read(buffer, size, &read));
  return read;
}

void File::WriteOrDie(const void *buffer, size_t size) {
  CHECK(Write(buffer, size));
}

Status File::ReadToString(string *contents) {
  // Get current position and size.
  uint64 pos, size;
  Status st = GetPosition(&pos);
  if (!st.ok()) return st;
  st = GetSize(&size);
  if (!st.ok()) return st;
  size -= pos;

  // Read file data into string buffer.
  contents->resize(size);
  char *data = &(*contents)[0];
  uint64 read;
  st = Read(data, size, &read);
  if (st.ok() && read < contents->size()) contents->resize(read);
  return st;
}

Status File::WriteString(const string &str) {
  return Write(str.data(), str.size());
}

Status File::WriteLine(const string &line) {
  Status st = WriteString(line);
  if (st.ok()) st = Write("\n", 1);
  return st;
}

uint64 File::Tell() {
  uint64 pos;
  if (!GetPosition(&pos).ok()) return -1;
  return pos;
}

uint64 File::Size() {
  uint64 size;
  if (!GetSize(&size).ok()) return -1;
  return size;
}

REGISTER_INITIALIZER(filesystem, {
  File::Init();
});

}  // namespace sling

