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

#include "sling/base/logging.h"
#include "sling/base/types.h"
#include "sling/frame/encoder.h"
#include "sling/frame/object.h"
#include "sling/frame/reader.h"
#include "sling/frame/serialization.h"
#include "sling/frame/store.h"
#include "sling/nlp/wiki/wiki.h"
#include "sling/nlp/wiki/wikidata-converter.h"
#include "sling/stream/input.h"
#include "sling/stream/memory.h"
#include "sling/string/strcat.h"
#include "sling/string/numbers.h"
#include "sling/string/text.h"
#include "sling/task/frames.h"
#include "sling/task/reducer.h"
#include "sling/task/task.h"

namespace sling {
namespace nlp {

// Parse Wikidata items and convert to SLING profiles.
class WikidataImporter : public task::Processor {
 public:
  ~WikidataImporter() override {
    delete converter_;
    delete commons_;
  }

  // Initialize Wikidata importer.
  void Init(task::Task *task) override {
    // Get output channels.
    item_channel_ = task->GetSink("items");
    CHECK(item_channel_ != nullptr);
    property_channel_ = task->GetSink("properties");
    CHECK(property_channel_ != nullptr);

    // Initialize counters.
    num_items_ = task->GetCounter("items");
    num_lexemes_ = task->GetCounter("lexemes");
    num_properties_ = task->GetCounter("properties");

    // Initialize Wikidata converter.
    string lang = task->Get("primary_language", "");
    commons_ = new Store();
    converter_ = new WikidataConverter(commons_, lang);
    bool only_primary = task->Get("only_primary_language", false);
    bool only_known = task->Get("only_known_languages", false);
    converter_->set_only_primary_language(only_primary);
    converter_->set_only_known_languages(only_known);

    names_.Bind(commons_);
    commons_->Freeze();
  }

  // Convert Wikidata item from JSON to SLING.
  void Receive(task::Channel *channel, task::Message *message) override {
    // Discard header and footers.
    if (message->value().size() < 3) {
      delete message;
      return;
    }

    // Read Wikidata item in JSON format into local SLING store.
    Store store(commons_);
    ArrayInputStream stream(message->value());
    Input input(&stream);
    Reader reader(&store, &input);
    reader.set_json(true);
    Object obj = reader.Read();
    delete message;
    CHECK(obj.valid());
    CHECK(obj.IsFrame()) << message->value();

    // Create SLING frame for item.
    Frame profile = converter_->Convert(obj.AsFrame());
    bool is_property = profile.IsA(n_property_);
    bool is_lexeme = profile.IsA(n_lexeme_);

    // Output property or item.
    if (is_lexeme) {
      // Discard lexemes for now since lexicographic data is still in beta.
      num_lexemes_->Increment();
    } else if (is_property) {
      property_channel_->Send(task::CreateMessage(profile));
      num_properties_->Increment();
    } else {
      item_channel_->Send(task::CreateMessage(profile));
      num_items_->Increment();
    }
  }

  // Clean up.
  void Done(task::Task *task) override {
    delete converter_;
    converter_ = nullptr;
    delete commons_;
    commons_ = nullptr;
  }

 private:
  // Output channel for items and properties.
  task::Channel *item_channel_ = nullptr;
  task::Channel *property_channel_ = nullptr;

  // Commons store.
  Store *commons_ = nullptr;

  // Wikidata converter.
  WikidataConverter *converter_ = nullptr;

  // Statistics.
  task::Counter *num_items_ = nullptr;
  task::Counter *num_lexemes_ = nullptr;
  task::Counter *num_properties_ = nullptr;

  // Symbols.
  Names names_;
  Name n_lexeme_{names_, "/w/lexeme"};
  Name n_property_{names_, "/w/property"};
};

REGISTER_TASK_PROCESSOR("wikidata-importer", WikidataImporter);

// Build Wikidata to Wikipedia id mapping.
class WikipediaMapping : public task::FrameProcessor {
 public:
  void Startup(task::Task *task) override {
    // Get language for mapping.
    string lang = task->Get("language", "en");
    language_ = commons_->Lookup(StrCat("/lang/" + lang));
    wikitypes_.Init(commons_);

    // Statistics.
    num_skipped_ = task->GetCounter("items_skipped");
    num_items_ = task->GetCounter("total_items_mapped");
    num_articles_ = task->GetCounter("article_pages_mapped");
    num_disambiguations_ = task->GetCounter("disambiguation_pages_mapped");
    num_categories_ = task->GetCounter("category_pages_mapped");
    num_lists_ = task->GetCounter("list_pages_mapped");
    num_templates_ = task->GetCounter("template_pages_mapped");
    num_infoboxes_ = task->GetCounter("infobox_pages_mapped");
  }

  void Process(Slice key, const Frame &frame) override {
    // Get Wikipedia id.
    Frame wikipedia = frame.GetFrame(n_wikipedia_);
    if (wikipedia.invalid()) {
      num_skipped_->Increment();
      return;
    }
    num_items_->Increment();
    Frame article = wikipedia.GetFrame(language_);
    if (article.invalid()) return;

    // Determine page type.
    bool is_category = false;
    bool is_disambiguation = false;
    bool is_list = false;
    bool is_infobox = false;
    bool is_template = false;
    for (const Slot &s : frame) {
      if (s.name == n_instance_of_) {
        if (wikitypes_.IsCategory(s.value)) {
          is_category = true;
        } else if (wikitypes_.IsDisambiguation(s.value)) {
          is_disambiguation = true;
        } else if (wikitypes_.IsList(s.value)) {
          is_list = true;
        } else if (wikitypes_.IsInfobox(s.value)) {
          is_infobox = true;
        } else if (wikitypes_.IsTemplate(s.value)) {
          is_template = true;
        }
      }
    }

    // Output mapping.
    Builder builder(frame.store());
    builder.AddId(article.id());
    builder.Add(n_qid_, frame);
    if (is_list) {
      builder.Add(n_kind_, n_kind_list_);
      num_lists_->Increment();
    } else if (is_category) {
      builder.Add(n_kind_, n_kind_category_);
      num_categories_->Increment();
    } else if (is_disambiguation) {
      builder.Add(n_kind_, n_kind_disambiguation_);
      num_disambiguations_->Increment();
    } else if (is_infobox) {
      builder.Add(n_kind_, n_kind_infobox_);
      num_infoboxes_->Increment();
    } else if (is_template) {
      builder.Add(n_kind_, n_kind_template_);
      num_templates_->Increment();
    } else {
      builder.Add(n_kind_, n_kind_article_);
      num_articles_->Increment();
    }

    OutputShallow(builder.Create());
  }

 private:
  // Language.
  Handle language_;

  // Wiki page types.
  WikimediaTypes wikitypes_;

  // Names.
  Name n_instance_of_{names_, "P31"};
  Name n_wikipedia_{names_, "/w/item/wikipedia"};
  Name n_qid_{names_, "/w/item/qid"};
  Name n_kind_{names_, "/w/item/kind"};
  Name n_kind_article_{names_, "/w/item/kind/article"};
  Name n_kind_disambiguation_{names_, "/w/item/kind/disambiguation"};
  Name n_kind_category_{names_, "/w/item/kind/category"};
  Name n_kind_list_{names_, "/w/item/kind/list"};
  Name n_kind_template_{names_, "/w/item/kind/template"};
  Name n_kind_infobox_{names_, "/w/item/kind/infobox"};

  // Statistics.
  task::Counter *num_skipped_ = nullptr;
  task::Counter *num_items_ = nullptr;
  task::Counter *num_articles_ = nullptr;
  task::Counter *num_disambiguations_ = nullptr;
  task::Counter *num_categories_ = nullptr;
  task::Counter *num_lists_ = nullptr;
  task::Counter *num_templates_ = nullptr;
  task::Counter *num_infoboxes_ = nullptr;
};

REGISTER_TASK_PROCESSOR("wikipedia-mapping", WikipediaMapping);

// Merge items with the same qid.
class ItemMerger : public task::Reducer {
 public:
  void Reduce(const task::ReduceInput &input) override {
    if (input.messages().size() == 1) {
      // Only one frame for item; output directly.
      Output(input.shard(), input.release(0));
    } else {
      // Merge all items.
      Store store;
      Builder builder(&store);
      for (task::Message *message : input.messages()) {
        Frame item = DecodeMessage(&store, message);
        builder.AddFrom(item);
      }

      // Output merged categories for item.
      Frame merged = builder.Create();
      Output(input.shard(), task::CreateMessage(input.key(), merged));
    }
  }
};

REGISTER_TASK_PROCESSOR("item-merger", ItemMerger);

// Prune Wikidata items for knowledge base repository.
class WikidataPruner : public task::FrameProcessor {
 public:
  void Startup(task::Task *task) override {
    // Get parameters.
    task->Fetch("prune_aliases", &prune_aliases_);
    task->Fetch("prune_wiki_links", &prune_wiki_links_);
    task->Fetch("prune_wiki_maps", &prune_wiki_maps_);
    task->Fetch("prune_category_members", &prune_category_members_);

    // Initialize aux filter.
    filter_.Init(commons_);
    aux_output_ = task->GetSink("aux");

    // Initialize counters.
    num_kb_items_ = task->GetCounter("kb_items");
    num_aux_items_ = task->GetCounter("aux_items");
  }

  void Process(Slice key, const Frame &frame) override {
    // Check if item is an auxiliary item. This need to be checked before the
    // item is pruned.
    bool aux = filter_.IsAux(frame);

    // Optionally, remove aliases, wikilinks, and categories from item.
    Builder item(frame);
    if (prune_aliases_) item.Delete(n_alias_);
    if (prune_wiki_links_) item.Delete(n_links_);
    if (prune_wiki_maps_) item.Delete(n_wikipedia_);
    if (prune_category_members_) item.Delete(n_member_);
    item.Update();

    // Filter out aux items.
    if (aux) {
      // Output aux items to separate channel.
      num_aux_items_->Increment();
      if (aux_output_ != nullptr) {
        aux_output_->Send(task::CreateMessage(frame));
      }
    } else {
      // Output item.
      num_kb_items_->Increment();
      Output(frame);
    }
  }

 private:
  // Symbols.
  Name n_alias_{names_, "alias"};
  Name n_wikipedia_{names_, "/w/item/wikipedia"};
  Name n_links_{names_, "/w/item/links"};
  Name n_member_{names_, "/w/item/member"};

  // Item filter.
  AuxFilter filter_;

  // Optional output channel for aux items.
  task::Channel *aux_output_;

  // Parameters.
  bool prune_aliases_ = true;
  bool prune_wiki_links_ = true;
  bool prune_wiki_maps_ = true;
  bool prune_category_members_ = true;

  // Statistics.
  task::Counter *num_kb_items_;
  task::Counter *num_aux_items_;
};

REGISTER_TASK_PROCESSOR("wikidata-pruner", WikidataPruner);

// Collect Wikidata properties.
class WikidataPropertyCollector : public task::FrameProcessor {
 public:
  void Process(Slice key, const Frame &frame) override {
    // Save property id.
    properties_.push_back(frame.Id().str());

    // Output property.
    Output(frame);
  }

  // Output property catalog.
  void Flush(task::Task *task) override {
    Store store;
    Builder catalog(&store);
    catalog.AddId("/w/entity");
    catalog.AddIs("schema");
    catalog.Add("name", "Wikidata entity");
    catalog.AddLink("family", "/schema/wikidata");
    for (const string &id : properties_) {
      catalog.AddLink("role", id);
    }
    Output(catalog.Create());
  }

 private:
  // Property ids.
  std::vector<string> properties_;
};

REGISTER_TASK_PROCESSOR("wikidata-property-collector",
                        WikidataPropertyCollector);

}  // namespace nlp
}  // namespace sling

