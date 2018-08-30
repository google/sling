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

#ifndef SLING_FILE_TEXTMAP_H_
#define SLING_FILE_TEXTMAP_H_

#include <string>
#include <vector>

#include "sling/base/status.h"
#include "sling/base/types.h"
#include "sling/file/file.h"
#include "sling/string/text.h"

namespace sling {

// A text map file is a text file with one entry per line. The key and
// value are separated by a tab character.
class TextMapInput {
 public:
  TextMapInput(const std::vector<string> &filenames, int buffer_size = 1 << 16);
  TextMapInput(const string &filename) : TextMapInput({filename}, 1 << 16) {}
  ~TextMapInput();

  // Read next entry from file. Return false if there are more entries.
  bool Next();

  // Return current entry id.
  int id() const { return id_; }

  // Return current key and value.
  const string &key() const { return key_; }
  const string &value() const { return value_; }

  // Read the next label from input. Return false if there are no more labels.
  // The value is parsed as a string and returned as count. All parameters can
  // be omitted by passing a null value.
  bool Read(int *index, string *name, int64 *count);

 private:
  // Get next character from input. Returns -1 on end of current file.
  int NextChar() {
    if (next_ < end_) {
      return *next_++;
    } else {
      return Fill();
    }
  }

  // Fill buffer and return first character or -1 if end of current file.
  int Fill();

  // Current file.
  File *file_ = nullptr;

  // Input files.
  std::vector<string> filenames_;

  // Current file number.
  int current_file_ = 0;

  // Input buffer.
  int buffer_size_;
  char *buffer_;
  char *next_;
  char *end_;

  // Current entry. First entry, i.e. line, is zero.
  int id_ = -1;

  // Current key and value.
  string key_;
  string value_;
};

// Write text map to output file.
class TextMapOutput {
 public:
  // Open text map file for writing.
  TextMapOutput(const string &filename, int buffer_size = 1 << 16);
  ~TextMapOutput();

  // Flush and close text map.
  void Close();

  // Write entry to text map.
  void Write(Text key, Text value);
  void Write(Text key, int64 value);

 private:
  // Output buffered data to file.
  void Output(const char *data, size_t size);

  // Flush output buffer.
  void Flush();

  // Output file.
  File *file_;

  // Output buffer.
  char *buffer_;
  char *next_;
  char *end_;
};

}  // namespace sling

#endif  // SLING_FILE_TEXTMAP_H_
