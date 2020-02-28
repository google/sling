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

#ifndef SLING_NLP_WIKI_PHRASE_TABLE_H_
#define SLING_NLP_WIKI_PHRASE_TABLE_H_

#include <string>
#include <vector>
#include <utility>

#include "sling/base/types.h"
#include "sling/file/repository.h"
#include "sling/frame/store.h"
#include "sling/frame/object.h"
#include "sling/string/text.h"
#include "sling/util/asset.h"

namespace sling {
namespace nlp {

// Phrase table for looking up entities based on name fingerprints.
class PhraseTable : public Asset {
 public:
  struct Match {
    Text id;        // entity id of matching item
    Handle item;    // matching item
    int count;      // frequency of matching item
    int form;       // majority case form for item
    bool reliable;  // matching alias from reliable source
  };

  typedef std::vector<Match> MatchList;

  ~PhraseTable() override { delete entity_table_; }

  // Load phrase repository from file.
  void Load(Store *store, const string &filename);

  // Find all entities matching a phrase fingerprint.
  void Lookup(uint64 fp, Handles *matches) const;

  // Find all entities matching a phrase fingerprint and return list of matches.
  void Lookup(uint64 fp, MatchList *matches) const;

  // Text normalization flags.
  const string &normalization() const { return normalization_; }

  // Acquire shared phrase table.
  static const PhraseTable *Acquire(AssetManager *assets,
                                    Store *store,
                                    const string &filename);

 private:
  // Get handle for entity.
  Handle GetEntityHandle(int index) const;

  // Entity phrase with entity index and frequency. The count_and_flags field
  // contains the count in the lower 29 bit. Bit 29 and 30 contain the case
  // form, and bit 31 contains the reliable source flag.
  struct EntityPhrase {
    uint32 index;
    uint32 count_and_flags;
    int count() const { return count_and_flags & ((1 << 29) - 1); }
    int form() const { return (count_and_flags >> 29) & 3; }
    bool reliable() const { return count_and_flags & (1 << 31); }
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

  // Phrase item in repository.
  class PhraseItem : public RepositoryObject {
   public:
    // Return fingerprint.
    uint64 fingerprint() const { return *fingerprint_ptr(); }

    // Return number of entities matching phrase fingerprint.
    int num_entities() const { return *entlen_ptr(); }

    // Return array of entities matching phrase fingerprint.
    const EntityPhrase *entities() const { return entities_ptr(); }

    // Return next item in list.
    const PhraseItem *next() const {
      int size = sizeof(uint64) + sizeof(uint32) +
                 num_entities() * sizeof(EntityPhrase);
      const char *self = reinterpret_cast<const char *>(this);
      return reinterpret_cast<const PhraseItem *>(self + size);
    }

   private:
    // Phrase fingerprint.
    REPOSITORY_FIELD(uint64, fingerprint, 1, 0);

    // Entity list.
    REPOSITORY_FIELD(uint32, entlen, 1, AFTER(fingerprint));
    REPOSITORY_FIELD(EntityPhrase, entities, num_entities(), AFTER(entlen));
  };

  // Phrase index in repository.
  class PhraseIndex : public RepositoryMap<PhraseItem> {
   public:
    // Initialize phrase index.
    void Initialize(const Repository &repository) {
      Init(repository, "Phrase");
    }

    // Return first element in bucket.
    const PhraseItem *GetBucket(int bucket) const { return GetObject(bucket); }
  };

  // Entity index in repository.
  class EntityIndex : public RepositoryIndex<uint32, EntityItem> {
   public:
    // Initialize name index.
    void Initialize(const Repository &repository) {
      Init(repository, "EntityIndex", "EntityItems", false);
    }

    // Return entity from entity index.
    const EntityItem *GetEntity(int index) const {
      return GetObject(index);
    }

    // Return id for entity.
    Text GetEntityId(int index) const {
      return GetEntity(index)->id();
    }
  };

 public:
  // Opaque public type for phrase items in the phrase table.
  typedef PhraseItem Phrase;

  // Find matching phrase in phrase table. Return null if phrase is not found.
  const Phrase *Find(uint64 fp) const;

  // Get matching handles for phrase.
  void GetMatches(const Phrase *phrase, Handles *matches) const;

  // Get matching items for phrase.
  void GetMatches(const Phrase *phrase, MatchList *matches) const;

 private:
  // Repository with name table.
  Repository repository_;

  // Phrase index.
  PhraseIndex phrase_index_;

  // Entity index.
  EntityIndex entity_index_;

  // Store for resolving entity ids.
  Store *store_ = nullptr;

  // Entities resolved to frame handles.
  Handles *entity_table_ = nullptr;

  // Text normalization flags.
  string normalization_ = "lcn";
};

}  // namespace nlp
}  // namespace sling

#endif  // SLING_NLP_WIKI_PHRASE_TABLE_H_

