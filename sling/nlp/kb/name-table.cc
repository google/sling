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

#include "sling/nlp/kb/name-table.h"

#include <algorithm>
#include <string>
#include <unordered_map>
#include <vector>

#include "sling/util/unicode.h"

namespace sling {
namespace nlp {

void NameTable::Load(const string &filename) {
  // Load name repository from file.
  repository_.Read(filename);

  // Initialize name table.
  name_index_.Initialize(repository_);

  // Initialize entity table.
  repository_.FetchBlock("Entities", &entity_table_);

  // Get text normalization flags.
  const char *norm = repository_.GetBlock("normalization");
  if (norm) {
    string normalization(norm, repository_.GetBlockSize("normalization"));
    normalization_ = ParseNormalization(normalization);
  }
}

void NameTable::LookupPrefix(Text prefix,
                             int limit, int boost,
                             std::vector<Text> *matches) const {
  // Normalize prefix.
  string normalized;
  UTF8::Normalize(prefix.data(), prefix.size(), normalization_, &normalized);
  Text normalized_prefix(normalized);

  // Find first name that is greater than or equal to the prefix.
  int lo = 0;
  int hi = name_index_.size() - 1;
  while (lo < hi) {
    int mid = (lo + hi) / 2;
    const NameItem *item = name_index_.GetName(mid);
    if (item->name() < normalized_prefix) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }

  // Find all names matching the prefix. Stop if we hit the limit.
  std::unordered_map<const EntityItem *, int> entities;
  int index = lo;
  while (index < name_index_.size()) {
    // Check if we have reached the limit.
    if (entities.size() > limit) break;

    // Stop if the current name does not match the prefix.
    const NameItem *item = name_index_.GetName(index);
    if (!item->name().starts_with(normalized_prefix)) break;

    // Add boost for exact match.
    int extra = 0;
    if (item->name().size() == normalized_prefix.size()) extra = boost;

    // Add matching entities.
    const EntityName *entity_names =  item->entities();
    for (int i = 0; i < item->num_entities(); ++i) {
      const EntityItem *entity = GetEntity(entity_names[i].offset);
      entities[entity] += entity_names[i].count + extra;
    }

    index++;
  }

  // Sort matching entities by decreasing frequency.
  std::vector<std::pair<uint32, const EntityItem *>> matching_entities;
  for (auto it : entities) {
    matching_entities.emplace_back(it.second, it.first);
  }
  std::sort(matching_entities.rbegin(), matching_entities.rend());

  // Copy matching entity ids to output.
  matches->clear();
  for (const auto &item : matching_entities) {
    matches->push_back(item.second->id());
  }
}

}  // namespace nlp
}  // namespace sling

