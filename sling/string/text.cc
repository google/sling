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

// Derived from Google StringPiece.

#include "sling/string/text.h"

#include <string.h>
#include <algorithm>
#include <string>

#include "sling/base/logging.h"
#include "sling/base/types.h"

namespace sling {

const Text::size_type Text::npos = size_type(-1);

static const char *memmatch(const char *haystack, size_t hlen,
                            const char *needle, size_t nlen) {
  if (nlen == 0) return haystack;  // even if haylen is 0
  if (hlen < nlen) return nullptr;

  const char *match;
  const char *hayend = haystack + hlen - nlen + 1;
  while ((match = static_cast<const char *>(memchr(haystack, needle[0],
                                                   hayend - haystack)))) {
    if (memcmp(match, needle, nlen) == 0) return match;
    haystack = match + 1;
  }
  return nullptr;
}

Text::Text(Text other, ssize_t pos)
    : ptr_(other.ptr_ + pos), length_(other.length_ - pos) {
  DCHECK_LE(0, pos);
  DCHECK_LE(pos, other.length_);
}

Text::Text(Text other, ssize_t pos, ssize_t len)
    : ptr_(other.ptr_ + pos), length_(std::min(len, other.length_ - pos)) {
  DCHECK_LE(0, pos);
  DCHECK_LE(pos, other.length_);
  DCHECK_GE(len, 0);
}

void Text::CopyToString(string *target) const {
  target->assign(ptr_, length_);
}

void Text::AppendToString(string *target) const {
  target->append(ptr_, length_);
}

ssize_t Text::copy(char *buf, size_type n, size_type pos) const {
  ssize_t ret = std::min(length_ - pos, n);
  memcpy(buf, ptr_ + pos, ret);
  return ret;
}

bool Text::contains(Text t) const {
  return find(t, 0) != npos;
}

ssize_t Text::find(Text t, size_type pos) const {
  if (length_ <= 0 || pos > static_cast<size_type>(length_)) {
    if (length_ == 0 && pos == 0 && t.length_ == 0) return 0;
    return npos;
  }
  const char *result = memmatch(ptr_ + pos, length_ - pos, t.ptr_, t.length_);
  return result ? result - ptr_ : npos;
}

ssize_t Text::find(char c, size_type pos) const {
  if (length_ <= 0 || pos >= static_cast<size_type>(length_)) {
    return npos;
  }
  const char *result =
    static_cast<const char*>(memchr(ptr_ + pos, c, length_ - pos));
  return result != nullptr ? result - ptr_ : npos;
}

ssize_t Text::rfind(Text t, size_type pos) const {
  if (length_ < t.length_) return npos;
  const size_t ulen = length_;
  if (t.length_ == 0) return std::min(ulen, pos);

  const char *last = ptr_ + std::min(ulen - t.length_, pos) + t.length_;
  const char *result = std::find_end(ptr_, last, t.ptr_, t.ptr_ + t.length_);
  return result != last ? result - ptr_ : npos;
}

// Search range is [0..pos] inclusive.  If pos == npos, search everything.
ssize_t Text::rfind(char c, size_type pos) const {
  if (length_ <= 0) return npos;
  ssize_t end = std::min(pos, static_cast<size_type>(length_ - 1));
  for (ssize_t i = end; i >= 0; --i) {
    if (ptr_[i] == c) return i;
  }
  return npos;
}

// For each character in characters_wanted, sets the index corresponding
// to the ASCII code of that character to 1 in table.  This is used by
// the find_.*_of methods below to tell whether or not a character is in
// the lookup table in constant time.
// The argument `table' must be an array that is large enough to hold all
// the possible values of an unsigned char.  Thus it should be be declared
// as follows:
//   bool table[UCHAR_MAX + 1]
static inline void BuildLookupTable(Text characters, bool *table) {
  const ssize_t length = characters.length();
  const char * const data = characters.data();
  for (ssize_t i = 0; i < length; ++i) {
    table[static_cast<unsigned char>(data[i])] = true;
  }
}

ssize_t Text::find_first_of(Text t, size_type pos) const {
  if (length_ <= 0 || t.length_ <= 0) return npos;

  // Avoid the cost of BuildLookupTable() for a single-character search.
  if (t.length_ == 1) return find_first_of(t.ptr_[0], pos);

  bool lookup[UCHAR_MAX + 1] = { false };
  BuildLookupTable(t, lookup);
  for (ssize_t i = pos; i < length_; ++i) {
    if (lookup[static_cast<unsigned char>(ptr_[i])]) {
      return i;
    }
  }
  return npos;
}

ssize_t Text::find_first_not_of(Text t, size_type pos) const {
  if (length_ <= 0) return npos;
  if (t.length_ <= 0) return 0;

  // Avoid the cost of BuildLookupTable() for a single-character search.
  if (t.length_ == 1) return find_first_not_of(t.ptr_[0], pos);

  bool lookup[UCHAR_MAX + 1] = { false };
  BuildLookupTable(t, lookup);
  for (ssize_t i = pos; i < length_; ++i) {
    if (!lookup[static_cast<unsigned char>(ptr_[i])]) {
      return i;
    }
  }
  return npos;
}

ssize_t Text::find_first_not_of(char c, size_type pos) const {
  if (length_ <= 0) return npos;

  for (; pos < static_cast<size_type>(length_); ++pos) {
    if (ptr_[pos] != c) return pos;
  }
  return npos;
}

ssize_t Text::find_last_of(Text t, size_type pos) const {
  if (length_ <= 0 || t.length_ <= 0) return npos;

  // Avoid the cost of BuildLookupTable() for a single-character search.
  if (t.length_ == 1) return find_last_of(t.ptr_[0], pos);

  bool lookup[UCHAR_MAX + 1] = { false };
  BuildLookupTable(t, lookup);
  ssize_t end = std::min(pos, static_cast<size_type>(length_ - 1));
  for (ssize_t i = end; i >= 0; --i) {
    if (lookup[static_cast<unsigned char>(ptr_[i])]) {
      return i;
    }
  }
  return npos;
}

ssize_t Text::find_last_not_of(Text t, size_type pos) const {
  if (length_ <= 0) return npos;

  ssize_t i = std::min(pos, static_cast<size_type>(length_ - 1));
  if (t.length_ <= 0) return i;

  // Avoid the cost of BuildLookupTable() for a single-character search.
  if (t.length_ == 1) return find_last_not_of(t.ptr_[0], pos);

  bool lookup[UCHAR_MAX + 1] = { false };
  BuildLookupTable(t, lookup);
  for (; i >= 0; --i) {
    if (!lookup[static_cast<unsigned char>(ptr_[i])]) {
      return i;
    }
  }
  return npos;
}

ssize_t Text::find_last_not_of(char c, size_type pos) const {
  if (length_ <= 0) return npos;

  ssize_t end = std::min(pos, static_cast<size_type>(length_ - 1));
  for (ssize_t i = end; i >= 0; --i) {
    if (ptr_[i] != c) return i;
  }
  return npos;
}

Text Text::substr(size_type pos, size_type n) const {
  if (pos > length_) pos = length_;
  if (n > length_ - pos) n = length_ - pos;
  return Text(ptr_ + pos, n);
}

std::ostream &operator <<(std::ostream &o, Text t) {
  o.write(t.data(), t.size());
  return o;
}

}  // namespace sling

