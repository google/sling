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

#ifndef SLING_BASE_SLICE_H_
#define SLING_BASE_SLICE_H_

#include <iosfwd>
#include <stddef.h>
#include <string.h>
#include <string>

#include "sling/base/logging.h"
#include "sling/base/types.h"

namespace sling {

// Slice is a simple structure pointing to a region of memory. The user of a
// Slice must ensure that the slice is not used after the memory has been
// deallocated.
class Slice {
 public:
  // Create an empty slice.
  Slice() : data_(nullptr), size_(0) {}

  // Create a slice that refers to d[0,n-1].
  Slice(const char *d, size_t n) : data_(d), size_(n) {}
  Slice(const void *d, size_t n)
      : data_(static_cast<const char *>(d)), size_(n) {}

  // Create a slice that refers to the contents of "s".
  Slice(const string &s) : data_(s.data()), size_(s.size()) {}

  // Create a slice that refers to s[0,strlen(s)-1].
  Slice(const char *s) : data_(s), size_(strlen(s)) {}

  // Create a slice that refers to [begin,end-1].
  Slice(const char *begin, const char *end)
      : data_(begin), size_(end - begin) {}

  // Return a pointer to the beginning of the referenced data.
  const char *data() const { return data_; }

  // Return the length (in bytes) of the referenced data.
  size_t size() const { return size_; }

  // Return true iff the length of the referenced data is zero.
  bool empty() const { return size_ == 0; }

  // Return the ith byte in the referenced data.
  char operator [](size_t i) const {
    DCHECK_LT(i, size_);
    return data_[i];
  }

  // Change this slice to refer to an empty array.
  void clear() { data_ = nullptr; size_ = 0; }

  // Drop the first "n" bytes from this slice.
  void remove_prefix(size_t n) {
    DCHECK_LE(n, size_);
    data_ += n;
    size_ -= n;
  }

  // Return a string that contains a copy of the referenced data.
  string str() const { return string(data_, size_); }

  // Three-way comparison.  Returns value:
  //   <  0 iff this <  other,
  //   == 0 iff this == other,
  //   >  0 iff this >  other
  int compare(const Slice &other) const {
    const size_t smallest = (size_ < other.size_) ? size_ : other.size_;
    int r = memcmp(data_, other.data_, smallest);
    if (r == 0) {
      if (size_ < other.size_) {
        r = -1;
      } else if (size_ > other.size_) {
        r = +1;
      }
    }
    return r;
  }

  // Prefix check.
  bool starts_with(const Slice &x) const {
    return size_ >= x.size_ &&
           memcmp(data_, x.data_, x.size_) == 0;
  }

  // Suffix check.
  bool ends_with(const Slice &x) const {
    return size_ >= x.size_ &&
           memcmp(data_ + (size_ - x.size_), x.data_, x.size_) == 0;
  }

 private:
  const char *data_;
  size_t size_;
};

inline bool operator ==(const Slice &x, const Slice &y) {
  return x.size() == y.size() &&
         memcmp(x.data(), y.data(), x.size()) == 0;
}

inline bool operator !=(const Slice &x, const Slice &y) {
  return !(x == y);
}

inline bool operator <(const Slice &x, const Slice &y) {
  const size_t smallest = x.size() < y.size() ? x.size() : y.size();
  const int r = memcmp(x.data(), y.data(), smallest);
  return (r < 0) || (r == 0 && x.size() < y.size());
}

inline bool operator >(const Slice &x, const Slice &y) {
  return y < x;
}

inline bool operator <=(const Slice &x, const Slice &y) {
  return !(x > y);
}

inline bool operator >=(const Slice &x, const Slice &y) {
  return !(x < y);
}

inline std::ostream &operator <<(std::ostream &o, Slice s) {
  o.write(s.data(), s.size());
  return o;
}

}  // namespace sling

#endif  // SLING_BASE_SLICE_H_

