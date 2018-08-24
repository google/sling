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

#ifndef SLING_UTIL_TOP_H_
#define SLING_UTIL_TOP_H_

#include <algorithm>
#include <vector>

namespace sling {

// Vector that keeps track of the top-n elements of an incrementally provided
// set of elements added one at a time. If the number of elements exceeds the
// limit, n, the lowest elements are incrementally dropped. If p elements are
// pushed in total, the runtime is O(p log n). This means that for constant n
// the runtime is linear in p.
template <class T, class Compare = std::greater<T>>
class Top : public std::vector<T> {
 public:
  // Initialize top vector with limit, i.e. the maximum number of elements.
  explicit Top(size_t limit) : limit_(limit), cmp_(Compare()) {}
  Top(size_t limit, const Compare &cmp) : limit_(limit), cmp_(cmp) {}

  // Push new element to vector. If the limit has been reached, the lowest
  // element is dropped.
  void push(const T &element) {
    if (limit_ == 0) return;
    if (this->size() < limit_) {
      // Push element to heap.
      this->push_back(element);
      std::push_heap(this->begin(), this->end(), cmp_);
    } else {
      // Only insert element if it is greater than the smallest elements.
      if (cmp_(element, this->front())) {
        std::pop_heap(this->begin(), this->end(), cmp_);
        this->back() = element;
        std::push_heap(this->begin(), this->end(), cmp_);
      }
    }
  }

  // Prepare vector for incrementally inserting elements.
  void prepare() {
    std::make_heap(this->begin(), this->end(), cmp_);
  }

  // Sort vector in descending order. This destoys the heap structure so
  // elements can no longer be inserted before prepare() has been called.
  void sort() {
    std::sort(this->begin(), this->end(), cmp_);
  }

 private:
  size_t limit_;  // maximum number of elements in vector
  Compare cmp_;   // greater-than comparison function
};

}  // namespace sling

#endif  // SLING_UTIL_TOP_H_

