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

#ifndef SLING_FILE_FILE_H_
#define SLING_FILE_FILE_H_

#include <string>
#include <vector>

#include "sling/base/logging.h"
#include "sling/base/registry.h"
#include "sling/base/status.h"
#include "sling/base/types.h"

namespace sling {

// File information.
struct FileStat {
  uint64 size;
  time_t mtime;
  bool is_file;
  bool is_directory;
};

// Abstract file interface.
class File {
 protected:
  // Use Close() to close and delete the file object.
  virtual ~File() = default;

 public:
  // Read up to "size" bytes from the file at position.
  virtual Status PRead(uint64 pos, void *buffer, size_t size, uint64 *read) = 0;

  // Read up to "size" bytes from the file at the current position.
  virtual Status Read(void *buffer, size_t size, uint64 *read) = 0;

  // Reads "size" bytes to buffer from file. Returns errors if less than "size"
  // bytes read.
  Status Read(void *buffer, size_t size);

  // Reads "size" bytes to buffer from file. Fails on read errors or if less
  // than "size" bytes read.
  void ReadOrDie(void *buffer, size_t size);

  // Reads the whole file to a string.
  Status ReadToString(string *contents);

  // Write data to the file at position.
  virtual Status PWrite(uint64 pos, const void *buffer, size_t size) = 0;

  // Write data to the file at the current position.
  virtual Status Write(const void *buffer, size_t size) = 0;

  // Write buffer to file. Fails on write errors.
  void WriteOrDie(const void *buffer, size_t size);

  // Write string to file.
  Status WriteString(const string &str);

  // Write string to file and append a newline.
  Status WriteLine(const string &line);

  // Map file region into memory. Return null on error or if not supported.
  virtual void *MapMemory(uint64 pos, size_t size, bool writable = false);

  // Set the current file position.
  virtual Status Seek(uint64 pos) = 0;

  // Skip bytes in the file.
  virtual Status Skip(uint64 n) = 0;

  // Get current file position.
  virtual Status GetPosition(uint64 *pos) = 0;

  // Return the current file position.
  uint64 Tell();

  // Get file size.
  virtual Status GetSize(uint64 *size) = 0;

  // Return the size of the file.
  uint64 Size();

  // Get file information.
  virtual Status Stat(FileStat *stat) = 0;

  // Close the file and delete the file object.
  virtual Status Close() = 0;

  // Flush unwritten data.
  virtual Status Flush() = 0;

  // Return the file name.
  virtual string filename() const = 0;

  // Initialize file systems. This can be called multiple times.
  static void Init();

  // Open file. Modes are "r", "r+", "w", "w+", "a", and "a+".
  static Status Open(const string &name, const char *mode, File **f);

  // Open file. Return null if the file cannot be opened.
  static File *Open(const string &name, const char *mode);

  // Open file and fail if file cannot be opened.
  static File *OpenOrDie(const string &name, const char *mode);

  // Delete a file.
  static Status Delete(const string &name);

  // Tests if a file exists.
  static bool Exists(const string &name);

  // Get size of named file.
  static Status GetSize(const string &name, uint64 *size);

  // Rename file.
  static Status Rename(const string &source, const string &target);

  // Get file information.
  static Status Stat(const string &name, FileStat *stat);

  // Create directory.
  static Status Mkdir(const string &dir);

  // Remove directory.
  static Status Rmdir(const string &dir);

  // Create temporary file.
  static File *TempFile();

  // Create temporary directory.
  static Status CreateTempDir(string *dir);

  // Find file names matching pattern.
  static Status Match(const string &pattern,
                      std::vector<string> *filenames);
  static std::vector<string> Match(const string &pattern) {
    std::vector<string> filenames;
    CHECK(Match(pattern, &filenames));
    return filenames;
  }

  // Read contents of file.
  static Status ReadContents(const string &filename, string *data);

  // Write contents of file.
  static Status WriteContents(const string &filename,
                              const void *data, size_t size);
  static Status WriteContents(const string &filename, const string &data) {
    return WriteContents(filename, data.data(), data.size());
  }

  // Return page size for memory mapping.
  static size_t PageSize();

  // Flush mapped memory to disk.
  static Status FlushMappedMemory(void *data, size_t size);

  // Free memory mapping.
  static Status FreeMappedMemory(void *data, size_t size);
};

// Abstract file system interface.
class FileSystem : public Singleton<FileSystem> {
 public:
  virtual ~FileSystem() = default;

  // Initialize file system.
  virtual void Init() = 0;

  // Returns true if this is the default file system.
  virtual bool IsDefaultFileSystem() = 0;

  // Open file.
  virtual Status Open(const string &name, const char *mode, File **f) = 0;

  // Returns true iff the named file exists.
  virtual bool FileExists(const string &filename) = 0;

  // Get size of file without opening it.
  virtual Status GetFileSize(const string &filename, uint64 *size) = 0;

  // Delete the named file.
  virtual Status DeleteFile(const string &filename) = 0;

  // Rename file source to target.
  virtual Status RenameFile(const string &source, const string &target) = 0;

  // Create temporary file.
  virtual Status CreateTempFile(File **f) = 0;

  // Create temporary directory.
  virtual Status CreateTempDir(string *dir) = 0;

  // Get file information.
  virtual Status Stat(const string &name, FileStat *stat) = 0;

  // Create the specified directory.
  virtual Status CreateDir(const string &dirname) = 0;

  // Delete the specified directory.
  virtual Status DeleteDir(const string &dirname) = 0;

  // Find file names matching pattern.
  virtual Status Match(const string &pattern,
                       std::vector<string> *filenames) = 0;

  // Flish mapped memory to disk.
  virtual Status FlushMappedMemory(void *data, size_t size);

  // Release mapped memory.
  virtual Status FreeMappedMemory(void *data, size_t size);
};

}  // namespace sling

#define REGISTER_FILE_SYSTEM_TYPE(name, component) \
  REGISTER_SINGLETON_TYPE(sling::FileSystem, name, component)

#endif  // SLING_FILE_FILE_H_

