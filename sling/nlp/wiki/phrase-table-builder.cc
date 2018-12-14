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
#include "sling/nlp/document/phrase-tokenizer.h"
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
    task->Fetch("noisy_alias_sources", &noisy_alias_sources_);

    // Set phrase normalization.
    tokenizer_.set_normalization(
        ParseNormalization(task->Get("normalization", "lcp")));

    // Statistics.
    num_aliases_ = task->GetCounter("aliases");
    num_phrases_ = task->GetCounter("phrases");
    num_entities_ = task->GetCounter("entities");
    num_instances_ = task->GetCounter("instances");
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
        bool reliable = (sources & ~noisy_alias_sources_);
        uint32 count_and_flags = count | (form << 29);
        if (reliable) count_and_flags |= (1 << 31);
        phrase->entities.emplace_back(index, count_and_flags);

        // Add alias count to entity frequency.
        entity_table_[index].count += count;
        num_instances_->Increment(count);
      }
    }
  }

  void Flush(task::Task *task) override {
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
    EntityPhrase(int index, uint32 count_and_flags)
        : index(index), count_and_flags(count_and_flags) {}
    uint32 index;
    uint32 count_and_flags;
    int count() const { return count_and_flags & ((1 << 29) - 1); }
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

  // Symbols.
  Name n_lang_{names_, "lang"};
  Name n_name_{names_, "name"};
  Name n_alias_{names_, "alias"};
  Name n_count_{names_, "count"};
  Name n_form_{names_, "form"};
  Name n_sources_{names_, "sources"};

  // Language for aliases.
  Handle language_;

  // Noisy alias sources. Aliases that are only backed by noisy alias sources
  // are not marked as reliable.
  int noisy_alias_sources_ =
    (1 << SRC_WIKIPEDIA_ANCHOR) |
    (1 << SRC_WIKIPEDIA_LINK) |
    (1 << SRC_WIKIPEDIA_DISAMBIGUATION);

  // Phrase tokenizer.
  PhraseTokenizer tokenizer_;

  // Sorted name table mapping phrase fingerprints to entities.
  std::unordered_map<uint64, Phrase *> phrase_table_;

  // Entity table with id and frequency count.
  std::vector<Entity> entity_table_;

  // Mapping of entity id to entity index in entity table.
  std::unordered_map<string, int> entity_mapping_;

  // Statistics.
  task::Counter *num_phrases_ = nullptr;
  task::Counter *num_entities_ = nullptr;
  task::Counter *num_aliases_ = nullptr;
  task::Counter *num_instances_ = nullptr;

  // Mutex for serializing access to repository.
  Mutex mu_;
};

REGISTER_TASK_PROCESSOR("phrase-table-builder", PhraseTableBuilder);

}  // namespace nlp
}  // namespace sling

