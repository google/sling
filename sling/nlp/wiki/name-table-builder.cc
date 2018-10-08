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

#include <map>
#include <string>
#include <unordered_map>

#include "sling/base/logging.h"
#include "sling/base/types.h"
#include "sling/file/repository.h"
#include "sling/task/frames.h"
#include "sling/task/task.h"
#include "sling/util/mutex.h"
#include "sling/util/unicode.h"

namespace sling {
namespace nlp {

// Build name table repository from aliases.
class NameTableBuilder : public task::FrameProcessor {
 public:
  void Startup(task::Task *task) override {
    // Get language for names.
    string lang = task->Get("language", "en");
    language_ = commons_->Lookup("/lang/" + lang);

    // Set name normalization.
    normalization_ = ParseNormalization(task->Get("normalization", "lcp"));

    // Statistics.
    num_aliases_ = task->GetCounter("aliases");
    num_names_ = task->GetCounter("names");
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

        // Normalize name.
        Text name = alias.GetText(n_name_);
        int count = alias.GetInt(n_count_, 1);
        string normalized;
        UTF8::Normalize(name.data(), name.size(), normalization_, &normalized);
        if (normalized.empty()) continue;
        if (normalized.size() > 127) continue;

        // Add alias for entity to name table.
        std::vector<EntityName> &entities = name_table_[normalized];
        if (entities.empty()) num_names_->Increment();
        entities.emplace_back(index, count);

        // Add alias count to entity frequency.
        entity_table_[index].count += count;
        num_instances_->Increment(count);
      }
    }
  }

  void Flush(task::Task *task) override {
    // Build name repository.
    Repository repository;

    // Add normalization flags to repository.
    string norm = NormalizationString(normalization_);
    repository.AddBlock("normalization", norm.data(), norm.size());

    // Get name repository blocks.
    File *index_block = repository.AddBlock("Index");
    File *name_block = repository.AddBlock("Names");
    File *entity_block = repository.AddBlock("Entities");

    // Write entity block.
    LOG(INFO) << "Build entity block";
    uint32 offset = 0;
    for (Entity &entity : entity_table_) {
      entity.offset = offset;

      // Write count and id to entity entry.
      CHECK_LT(entity.id.size(), 256);
      uint8 idlen = entity.id.size();
      entity_block->WriteOrDie(&entity.count, sizeof(uint32));
      entity_block->WriteOrDie(&idlen, sizeof(uint8));
      entity_block->WriteOrDie(entity.id.data(), idlen);

      // Compute offset of next entry.
      offset += sizeof(uint32) + sizeof(uint8) + idlen;
    }

    // Write name and index blocks. The names in the map are already sorted.
    LOG(INFO) << "Build name and index blocks";
    offset = 0;
    std::vector<uint32> entity_array;
    for (const auto &it : name_table_) {
      const string &name = it.first;
      const std::vector<EntityName> &entities = it.second;

      // Write name offset to index.
      index_block->WriteOrDie(&offset, sizeof(uint32));

      // Write name to name block.
      CHECK_LT(name.size(), 256);
      uint8 namelen = name.size();
      name_block->WriteOrDie(&namelen, sizeof(uint8));
      name_block->WriteOrDie(name.data(), namelen);

      // Write entity list to name block.
      uint32 entlen = entities.size();
      name_block->WriteOrDie(&entlen, sizeof(uint32));
      entity_array.clear();
      for (const EntityName &entity : entities) {
        entity_array.push_back(entity_table_[entity.index].offset);
        entity_array.push_back(entity.count);
      }
      int entity_array_size = entity_array.size() * sizeof(uint32);
      name_block->WriteOrDie(entity_array.data(), entity_array_size);

      // Compute offset of next entry.
      offset += sizeof(uint8) + namelen + sizeof(uint32) + entity_array_size;
    }

    // Write repository to file.
    const string &filename = task->GetOutput("repository")->resource()->name();
    LOG(INFO) << "Write name repository to " << filename;
    repository.Write(filename);
    LOG(INFO) << "Repository done";

    // Clear collected data.
    name_table_.clear();
    entity_table_.clear();
    entity_mapping_.clear();
  }

 private:
  // Entity with id and frequency.
  struct Entity {
    Entity(const string &id) : id(id) {}
    string id;
    uint32 count = 0;
    uint32 offset;
  };

  // Entity name with index and frequency.
  struct EntityName {
    EntityName(int index, uint32 count) : index(index), count(count) {}
    uint32 index;
    uint32 count;
  };

  // Symbols.
  Name n_lang_{names_, "lang"};
  Name n_name_{names_, "name"};
  Name n_alias_{names_, "alias"};
  Name n_count_{names_, "count"};

  // Language for aliases.
  Handle language_;

  // Text normalization flags.
  Normalization normalization_;

  // Sorted name table mapping normalized strings to entities with that name.
  std::map<string, std::vector<EntityName>> name_table_;

  // Entity table with id and frequency count.
  std::vector<Entity> entity_table_;

  // Mapping of entity id to entity index in entity table.
  std::unordered_map<string, int> entity_mapping_;

  // Statistics.
  task::Counter *num_names_ = nullptr;
  task::Counter *num_entities_ = nullptr;
  task::Counter *num_aliases_ = nullptr;
  task::Counter *num_instances_ = nullptr;

  // Mutex for serializing access to repository.
  Mutex mu_;
};

REGISTER_TASK_PROCESSOR("name-table-builder", NameTableBuilder);

}  // namespace nlp
}  // namespace sling

