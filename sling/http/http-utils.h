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

#ifndef SLING_HTTP_HTTP_UTILS_H_
#define SLING_HTTP_HTTP_UTILS_H_

#include <string.h>
#include <string>

#include "sling/base/types.h"

namespace sling {

// HTTP memory buffer.
struct HTTPBuffer {
 public:
  ~HTTPBuffer() { free(floor); }

  // Buffer size.
  int size() const { return end - start; }

  // Buffer capacity.
  int capacity() const { return ceil - floor; }

  // Number of bytes left in buffer.
  int remaining() const { return ceil - end; }

  // Whether buffer is empty.
  bool empty() const { return start == end; }

  // Whether buffer is full.
  bool full() const { return end == ceil; }

  // Clear buffer and allocate space.
  void reset(int size);

  // Flush buffer by moving the used part to the beginning of the buffer.
  void flush();

  // Make room in buffer.
  void ensure(int minfree);

  // Clear buffer;
  void clear();

  // Get next line from buffer and nul terminate it. Returns null if no newline
  // is found. White space and HTTP header continuations are replaced with
  // spaces and trailing whitespace is removed.
  char *gets();

  // Append string to buffer.
  void append(const char *data, int size);
  void append(const char *str) { if (str) append(str, strlen(str)); }

  char *floor = nullptr;  // start of allocated memory
  char *ceil = nullptr;   // end of allocated memory
  char *start = nullptr;  // start of used part of buffer
  char *end = nullptr;    // end of used part of buffer
};

// HTTP header.
struct HTTPHeader {
  HTTPHeader(char *n, char *v) : name(n), value(v) {}
  char *name;
  char *value;
};

// Decode URL component and append to output.
bool DecodeURLComponent(const char *url, int length, string *output);
bool DecodeURLComponent(const char *url, string *output);

// Escape text for HTML.
string HTMLEscape(const char *text, int size);

inline string HTMLEscape(const char *text) {
  return HTMLEscape(text, strlen(text));
}

inline string HTMLEscape(const string &text) {
  return HTMLEscape(text.data(), text.size());
}

}  // namespace sling

#endif  // SLING_HTTP_HTTP_UTILS_H_

