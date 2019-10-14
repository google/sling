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

#include <algorithm>
#include <string>
#include <unordered_map>

#include "sling/base/logging.h"
#include "sling/base/types.h"
#include "sling/file/repository.h"
#include "sling/frame/object.h"
#include "sling/nlp/document/phrase-tokenizer.h"
#include "sling/nlp/kb/facts.h"
#include "sling/nlp/wiki/wiki.h"
#include "sling/task/frames.h"
#include "sling/task/task.h"
#include "sling/util/mutex.h"

namespace sling {
namespace nlp {

// Build phrase table repository from aliases.
class PhraseTableBuilder : public task::FrameProcessor {
 public:
  void Startup(task::Task *task) override {
    // Get language for names.
    string lang = task->Get("language", "en");
    language_ = commons_->Lookup("/lang/" + lang);
    task->Fetch("reliable_alias_sources", &reliable_alias_sources_);
    task->Fetch("transfer_aliases", &transfer_aliases_);

    // Set phrase normalization.
    tokenizer_.set_normalization(
        ParseNormalization(task->Get("normalization", "lcn")));

    // Initialize alias transfer.
    if (transfer_aliases_) InitAliasTransfer();

    // Statistics.
    num_aliases_ = task->GetCounter("aliases");
    num_phrases_ = task->GetCounter("phrases");
    num_entities_ = task->GetCounter("entities");
    num_instances_ = task->GetCounter("instances");
    num_transfers_ = task->GetCounter("alias_transfers");
    num_zero_transfers_ = task->GetCounter("alias_zero_transfers");
    num_instance_transfers_ = task->GetCounter("alias_instance_transfers");
  }

  void Process(Slice key, const Frame &frame) override {
    MutexLock lock(&mu_);

    // Get index for entity.
    int index;
    string id(key.data(), key.size());
    auto fe = entity_mapping_.find(id);
    if (fe == entity_mapping_.end()) {
      index = entity_table_.size();
      entity_table_.emplace_back(id);
      num_entities_->Increment();
      entity_mapping_[id] = index;
    } else {
      index = fe->second;
    }

    // Add aliases.
    for (const Slot &s : frame) {
      if (s.name == n_alias_) {
        // Check language.
        Frame alias(frame.store(), s.value);
        if (alias.GetHandle(n_lang_) != language_) continue;
        num_aliases_->Increment();

        // Compute phrase fingerprint.
        Text name = alias.GetText(n_name_);
        int count = alias.GetInt(n_count_, 1);
        int sources = alias.GetInt(n_sources_, 0);
        int form = alias.GetInt(n_form_, 0);
        uint64 fp = tokenizer_.Fingerprint(name);
        if (fp == 1) continue;

        // Look up or add phrase for entity to phrase table.
        Phrase *&phrase = phrase_table_[fp];
        if (phrase == nullptr) {
          phrase = new Phrase(fp);
          num_phrases_->Increment();
        }

        // Add entity to phrase.
        bool reliable = (sources & reliable_alias_sources_);
        phrase->entities.emplace_back(index, count, form, reliable);

        // Add alias count to entity frequency.
        entity_table_[index].count += count;
        num_instances_->Increment(count);
      }
    }
  }

  void Flush(task::Task *task) override {
    // Prune phrase table by transfering unreliable aliases to reliable
    // aliases for related items.
    if (transfer_aliases_) {
      LOG(INFO) << "Transfer aliases";
      TransferAliases();
    }

    // Build phrase repository.
    Repository repository;

    // Add normalization flags to repository.
    string norm = NormalizationString(tokenizer_.normalization());
    repository.AddBlock("normalization", norm.data(), norm.size());

    // Write entity map.
    LOG(INFO) << "Build entity map";
    File *entity_index_block = repository.AddBlock("EntityIndex");
    File *entity_item_block = repository.AddBlock("EntityItems");
    uint32 offset = 0;
    for (Entity &entity : entity_table_) {
      // Write entity index entry.
      entity_index_block->WriteOrDie(&offset, sizeof(uint32));

      // Write count and id to entity entry.
      CHECK_LT(entity.id.size(), 256);
      uint8 idlen = entity.id.size();
      entity_item_block->WriteOrDie(&entity.count, sizeof(uint32));
      entity_item_block->WriteOrDie(&idlen, sizeof(uint8));
      entity_item_block->WriteOrDie(entity.id.data(), idlen);

      // Compute offset of next entry.
      offset += sizeof(uint32) + sizeof(uint8) + idlen;
    }

    // Write phrase map.
    LOG(INFO) << "Build phrase map";
    int num_phrases = phrase_table_.size();
    int num_buckets = (num_phrases + 32) / 32;
    std::vector<RepositoryMapItem *> items;
    items.reserve(num_phrases);
    for (auto &it : phrase_table_) {
      Phrase *phrase = it.second;
      items.push_back(phrase);

      // Sort entities in decreasing order.
      std::sort(phrase->entities.begin(), phrase->entities.end(),
          [](const EntityPhrase &a, const EntityPhrase &b) {
            return a.count() > b.count();
          });
    }
    repository.WriteMap("Phrase", &items, num_buckets);

    // Write repository to file.
    const string &filename = task->GetOutput("repository")->resource()->name();
    CHECK(!filename.empty());
    LOG(INFO) << "Write phrase repository to " << filename;
    repository.Write(filename);
    LOG(INFO) << "Repository done";

    // Clear collected data.
    for (auto &it : phrase_table_) delete it.second;
    phrase_table_.clear();
    entity_table_.clear();
    entity_mapping_.clear();
  }

 private:
  // Entity with id and frequency.
  struct Entity {
    Entity(const string &id) : id(id) {}
    string id;
    uint32 count = 0;
  };

  // Entity phrase with index and frequency. The count_and_flags field contains
  // the count in the lower 29 bit. Bit 29 and 30 contain the case form, and
  // bit 31 contains the reliable source flag.
  struct EntityPhrase {
    EntityPhrase() = default;
    EntityPhrase(int index, uint32 count, uint32 form, bool reliable)
        : index(index),
          count_and_flags(count | (form << 29) | (reliable ? (1 << 31) : 0)) {}
    uint32 index;
    uint32 count_and_flags;

    // Phrase frequency.
    int count() const { return count_and_flags & ((1 << 29) - 1); }
    void set_count(uint32 count) {
      count_and_flags = (count_and_flags & ~((1 << 29) - 1)) | count;
    }

    // Alias reliability.
    bool reliable() const { return count_and_flags & (1 << 31); }

    // Phrase form.
    int form() const { return (count_and_flags >> 29) & 3; }
  };

  // Phrase with fingerprint and entity distribution.
  struct Phrase : public RepositoryMapItem {
    // Initialize new phrase.
    Phrase(uint64 fingerprint) : fingerprint(fingerprint) {}

    // Write phrase to repository.
    int Write(File *file) const override {
      file->WriteOrDie(&fingerprint, sizeof(uint64));
      uint32 count = entities.size();
      file->WriteOrDie(&count, sizeof(uint32));
      for (const EntityPhrase &ep : entities) {
        file->WriteOrDie(&ep, sizeof(EntityPhrase));
      }
      return sizeof(uint64) + sizeof(uint32) + count * sizeof(EntityPhrase);
    }

    // Use phrase fingerprint as the hash code.
    uint64 Hash() const override { return fingerprint; }

    uint64 fingerprint;                  // phrase fingerprint
    std::vector<EntityPhrase> entities;  // list of entities for name phrase
  };

  // Transfer alias counts from source to target.
  bool Transfer(EntityPhrase *source, EntityPhrase *target) {
    // Check for conflicting case forms.
    int source_form = source->form();
    int target_form = target->form();
    if (source_form != CASE_NONE &&
        target_form != CASE_NONE &&
        source_form != target_form) {
      return false;
    }

    // Check for zero transfers.
    int source_count = source->count();
    int target_count = target->count();
    if (source_count == 0) {
      num_zero_transfers_->Increment();
      return false;
    }

    // Transfer alias counts from source to target.
    target->set_count(target_count + source_count);
    source->set_count(0);
    num_transfers_->Increment();
    num_instance_transfers_->Increment(source_count);
    return true;
  }

  // Exchange aliases between items.
  bool Exchange(EntityPhrase *a, EntityPhrase *b) {
    if (a->reliable() && !b->reliable()) {
      return Transfer(b, a);
    } else if (b->reliable() && !a->reliable()) {
      return Transfer(a, b);
    } else {
      return false;
    }
  }

  void TransferAliases() {
    // Run over all phrases in phrase table.
    for (auto &it : phrase_table_) {
      Phrase *phrase = it.second;

      // There must be more than one entity for any transfers to take place.
      if (phrase->entities.size() < 2) continue;

      // Build mappings between entity items and entity indices.
      int num_items = phrase->entities.size();
      Handles entity_item(commons_);
      HandleMap<int> entity_index;
      entity_item.resize(num_items);
      for (int i = 0; i < num_items; ++i) {
        const EntityPhrase &e = phrase->entities[i];
        const Entity &entity = entity_table_[e.index];
        Handle item = commons_->Lookup(entity.id);
        CHECK(!item.IsNil()) << entity.id;
        entity_item[i] = item;
        entity_index[item] = i;
      }

      // Find potential targets for alias transfer.
      bool pruned = false;
      std::vector<int> numbers;
      std::vector<int> years;
      for (int source = 0; source < num_items; ++source) {
        // Get set of facts for item.
        Facts facts(&catalog_);
        facts.Extract(entity_item[source]);
        for (int i = 0; i < facts.size(); ++i) {
          // Get base property and target value.
          Handle p = facts.first(i);
          Handle t = facts.last(i);

          // Collect numbers and years.
          if (p == n_instance_of_) {
            if (t == n_natural_number_) {
              numbers.push_back(source);
            }
            if (t == n_year_ || t == n_year_bc_ || t == n_decade_) {
              years.push_back(source);
            }
          }

          // Check for property exceptions.
          if (transfer_exceptions_.count(p) > 0) continue;

          // Check if target has the phrase as an alias.
          auto f = entity_index.find(t);
          if (f == entity_index.end()) continue;
          int target = f->second;
          if (target == source) continue;

          // Transfer alias from unreliable to reliable alias.
          auto &src = phrase->entities[source];
          auto &tgt = phrase->entities[target];
          if (Exchange(&src, &tgt)) pruned = true;
        }
      }

      // Transfer aliases for years.
      if (!years.empty()) {
        for (int source = 0; source < years.size(); ++source) {
          for (int target = 0; target < years.size(); ++target) {
            if (source == target) continue;
            auto &src = phrase->entities[years[source]];
            auto &tgt = phrase->entities[years[target]];
            if (Exchange(&src, &tgt)) pruned = true;
          }
        }
      }

      // Transfer aliases for numbers.
      if (!numbers.empty()) {
        for (int source = 0; source < numbers.size(); ++source) {
          for (int target = 0; target < numbers.size(); ++target) {
            if (source == target) continue;
            auto &src = phrase->entities[numbers[source]];
            auto &tgt = phrase->entities[numbers[target]];
            if (Exchange(&src, &tgt)) pruned = true;
          }
        }
      }

      // Prune aliases with zero count.
      if (pruned) {
        int j = 0;
        for (int i = 0; i < num_items; ++i) {
          if (phrase->entities[i].count() == 0) continue;
          if (i != j) phrase->entities[j] = phrase->entities[i];
          j++;
        }
        phrase->entities.resize(j);
      }
    }
  }

  void InitAliasTransfer() {
    // Initialize alias transfer exceptions.
    static const char *exceptions[] = {
      "P1889",  // different from
      "P460",   // said to be the same as
      "P1533",  // identical to this given name
      "P138",   // named after
      "P2959",  // permanent duplicated item
      "P734",   // family name
      "P735",   // given name
      "P112",   // founded by
      "P115",   // home venue
      "P144",   // based on
      "P1950",  // second family name in Spanish name
      "P2359",  // Roman nomen gentilicium
      "P2358",  // Roman praenomen
      "P2365",  // Roman cognomen
      "P2366",  // Roman agnomen
      "P941",   // inspired by
      "P629",   // edition or translation of
      "P37",    // official language
      "P103",   // native language
      "P566",   // basionym
      nullptr
    };
    for (const char **p = exceptions; *p != nullptr; ++p) {
      transfer_exceptions_.insert(commons_->LookupExisting(*p));
    }

    // Initialize fact catalog.
    catalog_.Init(commons_);
  }

  // Symbols.
  Name n_lang_{names_, "lang"};
  Name n_name_{names_, "name"};
  Name n_alias_{names_, "alias"};
  Name n_count_{names_, "count"};
  Name n_form_{names_, "form"};
  Name n_sources_{names_, "sources"};

  Name n_instance_of_{names_, "P31"};
  Name n_natural_number_{names_, "Q21199"};
  Name n_year_{names_, "Q577"};
  Name n_year_bc_{names_, "Q29964144"};
  Name n_decade_{names_, "Q39911"};

  // Language for aliases.
  Handle language_;

  // Reliable alias sources.
  int reliable_alias_sources_ =
    (1 << SRC_WIKIDATA_LABEL) |
    (1 << SRC_WIKIDATA_ALIAS) |
    (1 << SRC_WIKIDATA_NAME) |
    (1 << SRC_WIKIDATA_DEMONYM) |
    (1 << SRC_WIKIPEDIA_NAME);

  // Phrase tokenizer.
  PhraseTokenizer tokenizer_;

  // Sorted name table mapping phrase fingerprints to entities.
  std::unordered_map<uint64, Phrase *> phrase_table_;

  // Entity table with id and frequency count.
  std::vector<Entity> entity_table_;

  // Mapping of entity id to entity index in entity table.
  std::unordered_map<string, int> entity_mapping_;

  // Alias transfer.
  bool transfer_aliases_ = false;

  // Fact catalog for alias transfer.
  FactCatalog catalog_;

  // Property exceptions for alias transfer.
  HandleSet transfer_exceptions_;

  // Statistics.
  task::Counter *num_phrases_ = nullptr;
  task::Counter *num_entities_ = nullptr;
  task::Counter *num_aliases_ = nullptr;
  task::Counter *num_instances_ = nullptr;
  task::Counter *num_transfers_ = nullptr;
  task::Counter *num_zero_transfers_ = nullptr;
  task::Counter *num_instance_transfers_ = nullptr;

  // Mutex for serializing access to repository.
  Mutex mu_;
};

REGISTER_TASK_PROCESSOR("phrase-table-builder", PhraseTableBuilder);

}  // namespace nlp
}  // namespace sling

