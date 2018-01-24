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

#ifndef SLING_STRING_TEXT_H_
#define SLING_STRING_TEXT_H_

#include <string.h>
#include <iosfwd>
#include <limits>
#include <string>

#include "sling/base/logging.h"
#include "sling/base/port.h"
#include "sling/base/slice.h"
#include "sling/base/types.h"
#include "sling/util/city.h"

namespace sling {

class Text {
 private:
  const char *ptr_;
  ssize_t length_;

 public:
  // Constructors.
  Text() : ptr_(nullptr), length_(0) {}
  Text(const char *str) : ptr_(str), length_(str ? strlen(str) : 0) {}
  Text(const string &str) : ptr_(str.data()), length_(str.size()) {}
  Text(const char *str, ssize_t len) : ptr_(str), length_(len) {}
  Text(const Slice &slice) : ptr_(slice.data()), length_(slice.size()) {}

  // Substring of another text.
  // pos must be non-negative and <= x.length().
  Text(Text other, ssize_t pos);

  // Substring of another text.
  // pos must be non-negative and <= x.length().
  // len must be non-negative and will be pinned to at most x.length() - pos.
  Text(Text other, ssize_t pos, ssize_t len);

  // Access to string buffer.
  const char *data() const { return ptr_; }
  ssize_t size() const { return length_; }
  ssize_t length() const { return length_; }
  bool empty() const { return length_ == 0; }

  // Clear text.
  void clear() {
    ptr_ = nullptr;
    length_ = 0;
  }

  // Set contents.
  void set(const char *data, ssize_t len) {
    DCHECK_GE(len, 0);
    ptr_ = data;
    length_ = len;
  }

  void set(const char *str) {
    ptr_ = str;
    length_ = str ? strlen(str) : 0;
  }

  void set(const void *data, ssize_t len) {
    ptr_ = reinterpret_cast<const char *>(data);
    length_ = len;
  }

  // Index operator.
  char operator[](ssize_t index) const {
    DCHECK_GE(index, 0);
    DCHECK_LT(index, length_);
    return ptr_[index];
  }

  // Remove prefix from text.
  void remove_prefix(ssize_t n) {
    DCHECK_GE(length_, n);
    ptr_ += n;
    length_ -= n;
  }

  // Remove suffix from text.
  void remove_suffix(ssize_t n) {
    DCHECK_GE(length_, n);
    length_ -= n;
  }

  // Compare text to another text. Returns {-1, 0, 1}
  int compare(Text t) const {
    const ssize_t min_size = length_ < t.length_ ? length_ : t.length_;
    int r = memcmp(ptr_, t.ptr_, min_size);
    if (r < 0) return -1;
    if (r > 0) return 1;
    if (length_ < t.length_) return -1;
    if (length_ > t.length_) return 1;
    return 0;
  }

  // Return text as string.
  string str() const {
    if (ptr_ == nullptr) return string();
    return string(data(), size());
  }

  string as_string() const {
    if (ptr_ == nullptr) return string();
    return string(data(), size());
  }

  string ToString() const {
    if (ptr_ == nullptr) return string();
    return string(data(), size());
  }

  // Copy text to string.
  void CopyToString(string *target) const;

  // Append text to string.
  void AppendToString(string *target) const;

  // Return text as slice.
  Slice slice() const { return Slice(ptr_, length_); }

  // Prefix check.
  bool starts_with(Text t) const {
    return (length_ >= t.length_) && (memcmp(ptr_, t.ptr_, t.length_) == 0);
  }

  // Suffix check.
  bool ends_with(Text t) const {
    return ((length_ >= t.length_) &&
            (memcmp(ptr_ + (length_ - t.length_), t.ptr_, t.length_) == 0));
  }

  // STL container definitions.
  typedef char value_type;
  typedef const char *pointer;
  typedef const char &reference;
  typedef const char &const_reference;
  typedef size_t size_type;
  typedef ptrdiff_t difference_type;
  static const size_type npos;

  // Iterators.
  typedef const char *const_iterator;
  typedef const char *iterator;
  typedef std::reverse_iterator<const_iterator> const_reverse_iterator;
  typedef std::reverse_iterator<iterator> reverse_iterator;

  iterator begin() const { return ptr_; }
  iterator end() const { return ptr_ + length_; }
  const_reverse_iterator rbegin() const {
    return const_reverse_iterator(ptr_ + length_);
  }
  const_reverse_iterator rend() const {
    return const_reverse_iterator(ptr_);
  }
  ssize_t max_size() const { return length_; }
  ssize_t capacity() const { return length_; }
  ssize_t copy(char *buf, size_type n, size_type pos = 0) const;

  // Checks if text contains another text.
  bool contains(Text t) const;

  // Find operations.
  ssize_t find(Text t, size_type pos = 0) const;
  ssize_t find(char c, size_type pos = 0) const;
  ssize_t rfind(Text t, size_type pos = npos) const;
  ssize_t rfind(char c, size_type pos = npos) const;

  ssize_t find_first_of(Text t, size_type pos = 0) const;
  ssize_t find_first_of(char c, size_type pos = 0) const {
    return find(c, pos);
  }
  ssize_t find_first_not_of(Text t, size_type pos = 0) const;
  ssize_t find_first_not_of(char c, size_type pos = 0) const;
  ssize_t find_last_of(Text t, size_type pos = npos) const;
  ssize_t find_last_of(char c, size_type pos = npos) const {
    return rfind(c, pos);
  }
  ssize_t find_last_not_of(Text t, size_type pos = npos) const;
  ssize_t find_last_not_of(char c, size_type pos = npos) const;

  // Substring.
  Text substr(size_type pos, size_type n = npos) const;
};

// Comparison operators.
inline bool operator ==(Text x, Text y) {
  return x.size() == y.size() && memcmp(x.data(), y.data(), x.size()) == 0;
}

inline bool operator !=(Text x, Text y) {
  return !(x == y);
}

inline bool operator <(Text x, Text y) {
  const ssize_t min_size = x.size() < y.size() ? x.size() : y.size();
  const int r = memcmp(x.data(), y.data(), min_size);
  return (r < 0) || (r == 0 && x.size() < y.size());
}

inline bool operator >(Text x, Text y) {
  return y < x;
}

inline bool operator <=(Text x, Text y) {
  return !(x > y);
}

inline bool operator >=(Text x, Text y) {
  return !(x < y);
}

// Allow text to be streamed.
extern std::ostream &operator <<(std::ostream &o, Text t);

}  // namespace sling

namespace std {

template<> struct hash<sling::Text> {
  size_t operator()(sling::Text t) const {
    return sling::CityHash64(t.data(), t.size());
  }
};

}  // namespace std

#endif  // STRING_TEXT_H_

