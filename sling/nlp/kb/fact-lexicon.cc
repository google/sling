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

#include <string>
#include <unordered_map>
#include <utility>

#include "sling/file/textmap.h"
#include "sling/frame/object.h"
#include "sling/frame/store.h"
#include "sling/frame/serialization.h"
#include "sling/nlp/kb/facts.h"
#include "sling/task/frames.h"
#include "sling/task/process.h"
#include "sling/util/bloom.h"
#include "sling/util/sortmap.h"

namespace sling {
namespace nlp {

using namespace task;

// Extract fact and category lexicons from items.
class FactLexiconExtractor : public Process {
 public:
  void Run(Task *task) override {
    // Get parameters.
    int64 bloom_size = task->Get("bloom_size", 4000000000L);
    int bloom_hashes = task->Get("bloom_hashes", 4);
    int fact_threshold = task->Get("fact_threshold", 30);
    int category_threshold = task->Get("category_threshold", 30);

    // Set up counters.
    Counter *num_items = task->GetCounter("items");
    Counter *num_facts = task->GetCounter("facts");
    Counter *num_fact_types = task->GetCounter("fact_types");
    Counter *num_filtered = task->GetCounter("filtered_facts");
    Counter *num_facts_selected = task->GetCounter("facts_selected");
    Counter *num_categories_selected = task->GetCounter("categories_selected");

    // Load knowledge base.
    Store commons;
    LoadStore(task->GetInputFile("kb"), &commons);

    // Resolve symbols.
    Names names;
    Name p_item_category(names, "/w/item/category");
    Name n_item(names, "/w/item");
    Name p_instance_of(names, "P31");
    Name n_wikimedia_category(names, "Q4167836");
    Name n_wikimedia_disambiguation(names, "Q4167410");

    names.Bind(&commons);

    // Initialize fact catalog.
    FactCatalog catalog;
    catalog.Init(&commons);
    commons.Freeze();

    // A Bloom filter is used for checking for singleton facts. It is used as
    // a fast and compact check for detecting if a fact is a new fact. The
    // probabilistic nature of the Bloom filter means that the fact instance
    // counts can be off by one.
    BloomFilter filter(bloom_size, bloom_hashes);

    // The categories are collected in a sortable hash map so the most frequent
    // categories can be selected.
    SortableMap<Handle, int64, HandleHash> category_lexicon;

    // The facts are collected in a sortable hash map where the key is the
    // fact fingerprint. The name of the fact is stored in a nul-terminated
    // dynamically allocated string.
    SortableMap<int64, std::pair<int64, char *>> fact_lexicon;

    // Extract facts from all items in the knowledge base.
    commons.ForAll([&](Handle handle) {
      Frame item(&commons, handle);
      if (!item.IsA(n_item)) return;

      // Skip categories and disambiguation page items.
      Handle cls = item.GetHandle(p_instance_of);
      if (cls == n_wikimedia_category) return;
      if (cls == n_wikimedia_disambiguation) return;

      // Extract facts from item.
      Store store(&commons);
      Facts facts(&catalog);
      facts.Extract(handle);
      Handles fact_list(&store);
      facts.AsArrays(&store, &fact_list);

      // Add facts to fact lexicon.
      for (Handle fact : fact_list) {
        int64 fp = store.Fingerprint(fact);
        if (filter.add(fp)) {
          auto &entry = fact_lexicon[fp];
          if (entry.second == nullptr) {
            entry.second = strdup(ToText(&store, fact).c_str());
            num_fact_types->Increment();
          }
          entry.first++;
        } else {
          num_filtered->Increment();
        }
      }
      num_facts->Increment(facts.list().size());

      // Extract categories from item.
      for (const Slot &s : item) {
        if (s.name == p_item_category) {
          category_lexicon[s.value]++;
        }
      }

      num_items->Increment();
    });
    task->GetCounter("categories")->Increment(category_lexicon.map.size());

    // Write fact lexicon to text map.
    fact_lexicon.sort();
    TextMapOutput factout(task->GetOutputFile("factmap"));
    for (int i = fact_lexicon.array.size() - 1; i >= 0; --i) {
      Text fact(fact_lexicon.array[i]->second.second);
      int64 count = fact_lexicon.array[i]->second.first;
      if (count < fact_threshold) break;
      factout.Write(fact, count);
      num_facts_selected->Increment();
    }
    factout.Close();

    // Write category lexicon to text map.
    category_lexicon.sort();
    TextMapOutput catout(task->GetOutputFile("catmap"));
    for (int i = category_lexicon.array.size() - 1; i >= 0; --i) {
      Frame cat(&commons, category_lexicon.array[i]->first);
      int64 count = category_lexicon.array[i]->second;
      if (count < category_threshold) break;
      catout.Write(cat.Id(), count);
      num_categories_selected->Increment();
    }
    catout.Close();

    // Clean up.
    for (auto &it : fact_lexicon.map) free(it.second.second);
  }
};

REGISTER_TASK_PROCESSOR("fact-lexicon-extractor", FactLexiconExtractor);

// Extract facts items.
class FactExtractor : public Process {
 public:
  void Run(Task *task) override {
    // Set up counters.
    Counter *num_items = task->GetCounter("items");
    Counter *num_facts = task->GetCounter("facts");
    Counter *num_groups = task->GetCounter("groups");
    Counter *num_facts_extracted = task->GetCounter("facts_extracted");
    Counter *num_facts_skipped = task->GetCounter("facts_skipped");
    Counter *num_no_facts = task->GetCounter("items_without_facts");
    Counter *num_cats = task->GetCounter("categories");
    Counter *num_cats_extracted = task->GetCounter("categories_extracted");
    Counter *num_cats_skipped = task->GetCounter("categories_skipped");
    Counter *num_no_cats = task->GetCounter("items_without_categories");

    // Load knowledge base.
    LoadStore(task->GetInputFile("kb"), &commons_);

    // Resolve symbols.
    names_.Bind(&commons_);

    // Initialize fact catalog.
    FactCatalog catalog;
    catalog.Init(&commons_);
    commons_.Freeze();

    // Read fact and category lexicons.
    ReadFactLexicon(task->GetInputFile("factmap"));
    ReadCategoryLexicon(task->GetInputFile("catmap"));

    // Get output channel for resolved fact frames.
    Channel *output = task->GetSink("output");

    // Extract facts from all items in the knowledge base.
    commons_.ForAll([&](Handle handle) {
      Frame item(&commons_, handle);
      if (!item.IsA(n_item_)) return;

      // Skip categories and disambiguation page items.
      Handle cls = item.GetHandle(p_instance_of_);
      if (cls == n_wikimedia_category_) return;
      if (cls == n_wikimedia_disambiguation_) return;

      // Extract facts from item.
      Store store(&commons_);
      Facts facts(&catalog);
      facts.Extract(handle);
      Handles fact_list(&store);
      facts.AsArrays(&store, &fact_list);

      // Add all facts for item found in fact lexicon.
      Handles fact_indices(&store);
      Handles group_indices(&store);
      int start = 0;
      for (int g = 0; g < facts.groups().size(); ++g) {
        int end = facts.groups()[g];
        int prev = fact_indices.size();
        for (int i = start; i < end; ++i) {
          Handle fact = fact_list[i];
          uint64 fp = store.Fingerprint(fact);
          auto f = fact_lexicon_.find(fp);
          if (f != fact_lexicon_.end()) {
            fact_indices.push_back(Handle::Integer(f->second));
          }
        }
        int curr = fact_indices.size();
        if (curr > prev) {
          group_indices.push_back(Handle::Integer(curr));
        }
        start = end;
      }

      int total = facts.list().size();
      int extracted = fact_indices.size();
      int skipped = total - extracted;
      num_facts->Increment(total);
      num_facts_extracted->Increment(extracted);
      num_facts_skipped->Increment(skipped);
      num_groups->Increment(group_indices.size());
      if (extracted == 0) num_no_facts->Increment();

      // Extract categories from item.
      Handles category_indices(&store);
      for (const Slot &s : item) {
        if (s.name == p_item_category_) {
          auto f = category_lexicon_.find(s.value);
          if (f != category_lexicon_.end()) {
            category_indices.push_back(Handle::Integer(f->second));
            num_cats_extracted->Increment();
          } else {
            num_cats_skipped->Increment();
          }
          num_cats->Increment();
        }
      }
      if (category_indices.empty()) num_no_cats->Increment();

      // Build frame with resolved facts.
      Builder builder(&store);
      builder.Add(p_item_, item);
      builder.Add(p_facts_, Array(&store, fact_indices));
      builder.Add(p_groups_, Array(&store, group_indices));
      builder.Add(p_categories_, Array(&store, category_indices));

      // Output frame with resolved facts on output channel.
      output->Send(CreateMessage(item.Id(), builder.Create()));
      num_items->Increment();
    });
  }

 private:
  // Read fact lexicon.
  void ReadFactLexicon(const string &filename) {
    Store store(&commons_);
    TextMapInput factmap(filename);
    string key;
    int index;
    while (factmap.Read(&index, &key, nullptr)) {
      uint64 fp = FromText(&store, key).Fingerprint();
      fact_lexicon_[fp] = index;
    }
  }

  // Read category lexicon.
  void ReadCategoryLexicon(const string &filename) {
    TextMapInput catmap(filename);
    string key;
    int index;
    while (catmap.Read(&index, &key, nullptr)) {
      Handle cat = commons_.Lookup(key);
      category_lexicon_[cat] = index;
    }
  }

  // Commons store with knowledge base.
  Store commons_;

  // Fact lexicon mapping from fact fingerprint to fact index.
  std::unordered_map<uint64, int> fact_lexicon_;

  // Category lexicon mapping from category handle to category index.
  HandleMap<int> category_lexicon_;

  // Symbols.
  Names names_;
  Name p_item_category_{names_, "/w/item/category"};
  Name n_item_{names_, "/w/item"};
  Name p_instance_of_{names_, "P31"};
  Name n_wikimedia_category_{names_, "Q4167836"};
  Name n_wikimedia_disambiguation_{names_, "Q4167410"};

  Name p_item_{names_, "item"};
  Name p_facts_{names_, "facts"};
  Name p_groups_{names_, "groups"};
  Name p_categories_{names_, "categories"};
};

REGISTER_TASK_PROCESSOR("fact-extractor", FactExtractor);

}  // namespace nlp
}  // namespace sling
