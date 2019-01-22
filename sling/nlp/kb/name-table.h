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

#ifndef SLING_NLP_WIKI_NAME_TABLE_H_
#define SLING_NLP_WIKI_NAME_TABLE_H_

#include <string>
#include <vector>

#include "sling/base/types.h"
#include "sling/file/repository.h"
#include "sling/string/text.h"
#include "sling/util/unicode.h"

namespace sling {
namespace nlp {

// Name table for looking up entities based on name prefix.
class NameTable {
 public:
  // Load name repository from file.
  void Load(const string &filename);

  // Look up entities with names matching a prefix. The matches are sorted
  // by decreasing entity frequency.
  void LookupPrefix(Text prefix, int limit, int boost,
                    std::vector<Text> *matches) const;

 private:
  // Entity name with offset and frequency.
  struct EntityName {
    uint32 offset;
    uint32 count;
  };

  // Entity item in repository.
  class EntityItem : public RepositoryObject {
   public:
    // Entity id.
    Text id() const { return Text(id_ptr(), *idlen_ptr()); }

    // Entity frequency.
    uint32 count() const { return *count_ptr(); }

   private:
    // Entity frequency.
    REPOSITORY_FIELD(uint32, count, 1, 0);

    // Entity id.
    REPOSITORY_FIELD(uint8, idlen, 1, AFTER(count));
    REPOSITORY_FIELD(char, id, *idlen_ptr(), AFTER(idlen));
  };

  // Name item in repository.
  class NameItem : public RepositoryObject {
   public:
    // Return name.
    Text name() const { return Text(name_ptr(), *namelen_ptr()); }

    // Return number of entities matching name.
    int num_entities() const { return *entlen_ptr(); }

    // Return array of entity names.
    const EntityName *entities() const { return entities_ptr(); }

   private:
    // Name.
    REPOSITORY_FIELD(uint8, namelen, 1, 0);
    REPOSITORY_FIELD(char, name, *namelen_ptr(), AFTER(namelen));

    // Entity list.
    REPOSITORY_FIELD(uint32, entlen, 1, AFTER(name));
    REPOSITORY_FIELD(EntityName, entities, num_entities(), AFTER(entlen));
  };

  // Name index in repository.
  class NameIndex : public RepositoryIndex<uint32, NameItem> {
   public:
    // Initialize name index.
    void Initialize(const Repository &repository) {
      Init(repository, "Index", "Names", false);
    }

    // Return name from name index.
    const NameItem *GetName(int index) const {
      return GetObject(index);
    }
  };

  // Get entity from entity table.
  const EntityItem *GetEntity(uint32 offset) const {
    return reinterpret_cast<const EntityItem *>(entity_table_ + offset);
  }

  // Repository with name table.
  Repository repository_;

  // Name index.
  NameIndex name_index_;

  // Entity table.
  const char *entity_table_ = nullptr;

  // Text normalization flags.
  Normalization normalization_ = NORMALIZE_DEFAULT;
};

}  // namespace nlp
}  // namespace sling

#endif  // SLING_NLP_WIKI_NAME_TABLE_H_

