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

#include "sling/nlp/kb/phrase-table.h"

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

Handle PhraseTable::GetEntityHandle(int index) const {
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

const PhraseTable::Phrase *PhraseTable::Find(uint64 fp) const {
  int bucket = fp % phrase_index_.num_buckets();
  const PhraseItem *phrase = phrase_index_.GetBucket(bucket);
  const PhraseItem *end = phrase_index_.GetBucket(bucket + 1);
  while (phrase < end) {
    if (phrase->fingerprint() == fp) return phrase;
    phrase = phrase->next();
  }
  return nullptr;
}

void PhraseTable::GetMatches(const Phrase *phrase, Handles *matches) const {
  if (phrase == nullptr) {
    matches->clear();
    return;
  }
  const EntityPhrase *entities = phrase->entities();
  matches->resize(phrase->num_entities());
  for (int i = 0; i < phrase->num_entities(); ++i) {
    int index = entities[i].index;
    (*matches)[i] = GetEntityHandle(index);
  }
}

void PhraseTable::GetMatches(const Phrase *phrase, MatchList *matches) const {
  if (phrase == nullptr) {
    matches->clear();
    return;
  }
  const EntityPhrase *entities = phrase->entities();
  matches->resize(phrase->num_entities());
  for (int i = 0; i < phrase->num_entities(); ++i) {
    int index = entities[i].index;
    Match &match = (*matches)[i];
    match.id = entity_index_.GetEntityId(index);
    match.item = GetEntityHandle(index);
    auto &entity = entities[i];
    match.count = entity.count();
    match.form = entity.form();
    match.reliable = entity.reliable();
  }
}

void PhraseTable::Lookup(uint64 fp, Handles *matches) const {
  GetMatches(Find(fp), matches);
}

void PhraseTable::Lookup(uint64 fp, MatchList *matches) const {
  GetMatches(Find(fp), matches);
}

const PhraseTable *PhraseTable::Acquire(AssetManager *assets,
                                        Store *store,
                                        const string &filename) {
  return assets->Acquire<PhraseTable>(filename, [&]() {
    PhraseTable *aliases = new PhraseTable();
    aliases->Load(store, filename);
    return aliases;
  });
}

}  // namespace nlp
}  // namespace sling
