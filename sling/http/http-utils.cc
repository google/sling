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

#include "sling/http/http-utils.h"

#include "sling/base/logging.h"
#include "sling/string/ctype.h"

namespace sling {

// Returns value for ASCII hex digit.
static int HexDigit(int c) {
  return (c <= '9') ? c - '0' : (c & 7) + 9;
}

// Decode URL component.
bool DecodeURLComponent(const char *url, int length, string *output) {
  const char *end = url + length;
  while (url < end) {
    char c = *url++;
    if (c == '%') {
      if (url + 2 >= end) return false;
      char x1 = *url++;
      if (!ascii_isxdigit(x1)) return false;
      char x2 = *url++;
      if (!ascii_isxdigit(x2)) return false;
      output->push_back((HexDigit(x1) << 4) + HexDigit(x2));
    } else if (c == '+') {
      output->push_back(' ');
    } else {
      output->push_back(c);
    }
  }

  return true;
}

bool DecodeURLComponent(const char *url, string *output) {
  if (url == nullptr) return true;
  return DecodeURLComponent(url, strlen(url), output);
}

string HTMLEscape(const char *text, int size) {
  string escaped;
  const char *p = text;
  const char *end = text + size;
  while (p < end) {
    char ch = *p++;
    switch (ch) {
      case '&':  escaped.append("&amp;"); break;
      case '<':  escaped.append("&lt;"); break;
      case '>':  escaped.append("&gt;"); break;
      case '"':  escaped.append("&quot;"); break;
      case '\'': escaped.append("&#39;");  break;
      default: escaped.push_back(ch);
    }
  }
  return escaped;
}

void HTTPBuffer::reset(int size) {
  if (size != capacity()) {
    if (size == 0) {
      free(floor);
      floor = ceil = start = end = nullptr;
    } else {
      floor = static_cast<char *>(realloc(floor, size));
      CHECK(floor != nullptr) << "Out of memory, " << size << " bytes";
      ceil = floor + size;
    }
  }
  start = end = floor;
}

void HTTPBuffer::flush() {
  if (start > floor) {
    int size = end - start;
    memcpy(floor, start, size);
    start = floor;
    end = start + size;
  }
}

void HTTPBuffer::ensure(int minfree) {
  // Check if there is enough free space in buffer.
  if (ceil - end >= minfree) return;

  // Compute new size of buffer.
  int size = ceil - floor;
  int minsize = end + minfree - floor;
  while (size < minsize) {
    if (size == 0) {
      size = 1024;
    } else {
      size *= 2;
    }
  }

  // Expand buffer.
  char *p = static_cast<char *>(realloc(floor, size));
  CHECK(p != nullptr) << "Out of memory, " << size << " bytes";

  // Adjust pointers.
  start += p - floor;
  end += p - floor;
  floor = p;
  ceil = p + size;
}

void HTTPBuffer::clear() {
  free(floor);
  floor = ceil = start = end = nullptr;
}

char *HTTPBuffer::gets() {
  char *line = start;
  char *s = line;
  while (s < end) {
    switch (*s) {
      case '\n':
        if (s + 1 < end && (s[1] == ' ' || s[1] == '\t')) {
          // Replace HTTP header continuation with space.
          *s++ = ' ';
        } else {
          //  End of line found. Strip trailing whitespace.
          *s = 0;
          start = s + 1;
          while (s > line) {
            s--;
            if (*s != ' ' && *s != '\t') break;
            *s = 0;
          }
          return line;
        }
        break;

      case '\r':
      case '\t':
        // Replace whitespace with space.
        *s++ = ' ';
        break;

      default:
        s++;
    }
  }

  return nullptr;
}

void HTTPBuffer::append(const char *data, int size) {
  ensure(size);
  memcpy(end, data, size);
  end += size;
}

}  // namespace sling

