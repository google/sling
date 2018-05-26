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

  // Resolve all the entity ids in the store.
  entity_table_ = new Handles(store);
  entity_table_->resize(entity_index_.size());
  for (int i = 0; i < entity_index_.size(); ++i) {
    const EntityItem *entity = entity_index_.GetEntity(i);
    Handle handle = store->LookupExisting(entity->id());
    (*entity_table_)[i] = handle;
  }
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
        matches->push_back((*entity_table_)[entities[i].index]);
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
        Handle entity = (*entity_table_)[entities[i].index];
        int count = entities[i].count;
        matches->emplace_back(entity, count);
      }
      break;
    }
    phrase = phrase->next();
  }
}

}  // namespace nlp
}  // namespace sling

