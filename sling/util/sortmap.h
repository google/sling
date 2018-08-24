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

#ifndef SLING_UTIL_SORTMAP_H_
#define SLING_UTIL_SORTMAP_H_

#include <algorithm>
#include <vector>
#include <unordered_map>

namespace sling {

// A hash map which can be sorted by value. This implementation is space
// efficient since the sorted array just keeps pointers to the internal
// nodes in the hash map.
template<typename K, typename V, class H = std::hash<K>> struct SortableMap {
 public:
  typedef std::unordered_map<K, V, H> Map;
  typedef typename Map::value_type Node;
  typedef std::vector<Node *> Array;

  // Look up value in hash map.
  V &operator[](const K &key) { return map[key]; }

  // Sort hash map.
  void sort() {
    array.clear();
    array.reserve(map.size());
    for (Node &node : map) array.emplace_back(&node);
    std::sort(array.begin(), array.end(), [](const Node *n1, const Node *n2) {
      return n1->second < n2->second;
    });
  }

  Map map;
  Array array;
};

}  // namespace sling

#endif  // SLING_UTIL_SORTMAP_H_
