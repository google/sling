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

#include "sling/nlp/wiki/phrase-table.h"

namespace sling {
namespace nlp {

void PhraseTable::Load(Store *store, const string &filename) {
  // Load name repository from file.
  repository_.Read(filename);

  // Initialize phrase table.
  phrase_index_.Initialize(repository_);

  // Initialize entity table.
  entity_index_.Initialize(repository_);

  // Get text normalization flags.
  const char *norm = repository_.GetBlock("normalization");
  if (norm) {
    normalization_.assign(norm, repository_.GetBlockSize("normalization"));
  }

  // Allocate handle array for resolved entities.
  store_ = store;
  entity_table_ = new Handles(store);
  entity_table_->resize(entity_index_.size());
}

Handle PhraseTable::GetEntityHandle(int index) {
  Handle handle = (*entity_table_)[index];
  if (handle.IsNil()) {
    const EntityItem *entity = entity_index_.GetEntity(index);
    handle = store_->LookupExisting(entity->id());
    if (handle.IsNil()) {
      VLOG(1) << "Cannot resolve " << entity->id() << " in phrase table";
    }
    (*entity_table_)[index] = handle;
  }
  return handle;
}

void PhraseTable::Lookup(uint64 fp, Handles *matches) {
  matches->clear();
  int bucket = fp % phrase_index_.num_buckets();
  const PhraseItem *phrase = phrase_index_.GetBucket(bucket);
  const PhraseItem *end = phrase_index_.GetBucket(bucket + 1);
  while (phrase < end) {
    if (phrase->fingerprint() == fp) {
      const EntityPhrase *entities = phrase->entities();
      for (int i = 0; i < phrase->num_entities(); ++i) {
        int index = entities[i].index;
        Handle handle = GetEntityHandle(index);
        matches->push_back(handle);
      }
      break;
    }
    phrase = phrase->next();
  }
}

void PhraseTable::Lookup(uint64 fp, MatchList *matches) {
  matches->clear();
  int bucket = fp % phrase_index_.num_buckets();
  const PhraseItem *phrase = phrase_index_.GetBucket(bucket);
  const PhraseItem *end = phrase_index_.GetBucket(bucket + 1);
  while (phrase < end) {
    if (phrase->fingerprint() == fp) {
      const EntityPhrase *entities = phrase->entities();
      for (int i = 0; i < phrase->num_entities(); ++i) {
        int index = entities[i].index;
        Text id = entity_index_.GetEntityId(index);
        Handle handle = GetEntityHandle(index);
        int count = entities[i].count();
        int form = entities[i].form();
        bool reliable = entities[i].reliable();
        matches->emplace_back(id, handle, count, form, reliable);
      }
      break;
    }
    phrase = phrase->next();
  }
}

}  // namespace nlp
}  // namespace sling
