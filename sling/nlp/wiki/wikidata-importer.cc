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
#include "sling/nlp/kb/calendar.h"
#include "sling/nlp/wiki/wiki.h"
#include "sling/stream/input.h"
#include "sling/stream/output.h"
#include "sling/string/strcat.h"
#include "sling/string/numbers.h"
#include "sling/string/text.h"
#include "sling/task/frames.h"
#include "sling/task/reducer.h"
#include "sling/task/task.h"

namespace sling {
namespace nlp {

// Conversion table for Wikidata JSON date precision.
static Date::Precision date_precision[] = {
  Date::NONE,       // 0: 1 Gigayear
  Date::NONE,       // 1: 100 Megayears
  Date::NONE,       // 2: 10 Megayears
  Date::NONE,       // 3: Megayear
  Date::NONE,       // 4: 100 Kiloyears
  Date::NONE,       // 5: 10 Kiloyears
  Date::MILLENNIUM, // 6: Kiloyear
  Date::CENTURY,    // 7: 100 years
  Date::DECADE,     // 8: 10 years
  Date::YEAR,       // 9: years
  Date::MONTH,      // 10: months
  Date::DAY,        // 11: days
  Date::NONE,       // 12: hours (unused)
  Date::NONE,       // 13: minutes (unused)
  Date::NONE,       // 14: seconds (unused)
};

// Parse Wikidata items and convert to SLING profiles.
class WikidataImporter : public task::Processor {
 public:
  ~WikidataImporter() override { delete commons_; }

  // Initialize Wikidata importer.
  void Init(task::Task *task) override {
    // Get output channels.
    item_channel_ = task->GetSink("items");
    CHECK(item_channel_ != nullptr);
    property_channel_ = task->GetSink("properties");
    CHECK(property_channel_ != nullptr);

    // Create commons store.
    commons_ = new Store();

    // Initialize global symbols.
    names_.Bind(commons_);
    AddTypeMapping(s_string_.name(), "/w/string");
    AddTypeMapping(s_time_.name(), "/w/time");
    AddTypeMapping(s_quantity_.name(), "/w/quantity");
    AddTypeMapping(s_monolingualtext_.name(), "/w/text");
    AddTypeMapping(s_wikibase_item_.name(), "/w/item");
    AddTypeMapping(s_commons_media_.name(), "/w/media");
    AddTypeMapping(s_external_id_.name(), "/w/xref");
    AddTypeMapping(s_wikibase_property_.name(), "/w/property");
    AddTypeMapping(s_url_.name(), "/w/url");
    AddTypeMapping(s_globe_coordinate_.name(), "/w/geo");
    AddTypeMapping(s_math_.name(), "/w/math");
    AddTypeMapping(s_tabular_data_.name(), "/w/table");
    AddTypeMapping(s_geo_shape_.name(), "/w/shape");

    // Get primary language.
    primary_language_name_ = Wiki::language_priority[0];
    task->Fetch("primary_language", &primary_language_name_);
    primary_language_ = commons_->Lookup(primary_language_name_);

    // Initialize per-language information.
    const char **lang = Wiki::language_priority;
    int priority = 1;
    while (*lang != 0) {
      LanguageInfo info;
      info.priority = priority;
      info.language = commons_->Lookup(StrCat("/lang/", *lang));
      info.wikisite = commons_->Lookup(StrCat(*lang, "wiki"));
      languages_[commons_->Lookup(*lang)] = info;
      lang++;
      priority++;
    }

    commons_->Freeze();
  }

  // Add property data type mapping.
  void AddTypeMapping(Text datatype, Text type) {
    datatypes_[datatype] = commons_->Lookup(type);
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

    // Get top-level item attributes.
    Frame item = obj.AsFrame();
    string id = item.GetString(s_id_);
    string type = item.GetString(s_type_);
    Frame labels = item.GetFrame(s_labels_);
    Frame descriptions = item.GetFrame(s_descriptions_);

    // Create builder for constructing the frame for the item.
    Builder builder(&store);
    if (!id.empty()) builder.AddId(id);
    bool is_property = (type == "property");
    builder.AddIsA(is_property ? n_property_ : n_item_);

    // Get label and description based on language.
    Handle label = PickName(labels);
    if (!label.IsNil()) builder.Add(n_name_, label);
    Handle description = PickName(descriptions);
    if (!description.IsNil()) builder.Add(n_description_, description);

    // Add data type for property.
    if (is_property) {
      String datatype = item.Get(s_datatype_).AsString();
      CHECK(!datatype.IsNil());
      auto f = datatypes_.find(datatype.text());
      CHECK(f != datatypes_.end()) << datatype.text();
      builder.Add(n_source_, n_entity_);
      builder.Add(n_target_, f->second);
    }

    // Parse labels and aliases.
    Frame aliases = item.GetFrame(s_aliases_);
    if (!is_property) {
      for (auto &it : languages_) {
        // Get label for language.
        if (labels.valid()) {
          Frame label = labels.Get(it.first).AsFrame();
          if (label.valid()) {
            Builder alias(&store);
            alias.Add(n_name_, label.GetHandle(s_value_));
            alias.Add(n_lang_, it.second.language);
            alias.Add(n_alias_sources_, 1 << SRC_WIKIDATA_LABEL);
            builder.Add(n_profile_alias_, alias.Create());
          }
        }

        // Get aliases for language.
        if (aliases.valid()) {
          Array alias_list = aliases.Get(it.first).AsArray();
          if (alias_list.valid()) {
            for (int i = 0; i < alias_list.length(); ++i) {
              Handle name =
                  Frame(&store, alias_list.get(i)).GetHandle(s_value_);
              Builder alias(&store);
              alias.Add(n_name_, name);
              alias.Add(n_lang_, it.second.language);
              alias.Add(n_alias_sources_, 1 << SRC_WIKIDATA_ALIAS);
              builder.Add(n_profile_alias_, alias.Create());
            }
          }
        }
      }
    }

    // Parse claims.
    Frame claims = item.GetFrame(s_claims_);
    if (claims.valid()) {
      for (const Slot &property : claims) {
        Array statement_list(&store, property.value);
        for (int i = 0; i < statement_list.length(); ++i) {
          // Parse statement.
          Frame statement(&store, statement_list.get(i));
          Frame snak = statement.GetFrame(s_mainsnak_);
          CHECK(snak.valid());
          Handle property = snak.GetHandle(s_property_);
          CHECK(!property.IsNil());
          Frame datavalue = snak.GetFrame(s_datavalue_);
          if (datavalue.invalid()) continue;
          Object value(&store, ConvertValue(datavalue));

          // Add qualifiers.
          Frame qualifiers = statement.GetFrame(s_qualifiers_);
          if (qualifiers.valid()) {
            Builder qualified(&store);
            qualified.AddIs(value);
            for (const Slot &qproperty : qualifiers) {
              Array qstatement_list(&store, qproperty.value);
              for (int j = 0; j < qstatement_list.length(); ++j) {
                Frame qstatement(&store, qstatement_list.get(j));
                Handle qproperty = qstatement.GetHandle(s_property_);
                CHECK(!qproperty.IsNil());
                Frame qdatavalue = qstatement.GetFrame(s_datavalue_);
                if (qdatavalue.invalid()) continue;
                Object qvalue(&store, ConvertValue(qdatavalue));
                qualified.Add(Property(&store, qproperty), qvalue);
              }
            }

            value = qualified.Create();
          }

          // Add property with value.
          builder.Add(Property(&store, property), value);
        }
      }
    }

    // Add Wikipedia links.
    Frame sitelinks = item.GetFrame(s_sitelinks_);
    if (sitelinks.valid()) {
      Builder sites(&store);
      for (auto &it : languages_) {
        Frame site = sitelinks.GetFrame(it.second.wikisite);
        if (site.valid()) {
          string title = site.GetString(s_title_);
          string lang = Frame(&store, it.first).Id().str();
          if (!title.empty()) {
            string wiki_id = Wiki::Id(lang, title);
            sites.AddLink(it.second.language, wiki_id);
          }
        }
      }
      builder.Add(n_wikipedia_, sites.Create());
    }

    // Create SLING frame for item.
    Frame profile = builder.Create();

    // Output property or item.
    if (is_property) {
      property_channel_->Send(task::CreateMessage(profile));
    } else {
      item_channel_->Send(task::CreateMessage(profile));
    }
  }

  // Clean up.
  void Done(task::Task *task) override {
    delete commons_;
    commons_ = nullptr;
  }

 private:
  // Return symbol for Wikidata item.
  static Handle Item(Store *store, int id) {
    return store->Lookup(StrCat("Q", id));
  }

  // Return symbol for Wikidata property.
  static Handle Property(Store *store, int id) {
    return store->Lookup(StrCat("P", id));
  }
  static Handle Property(Store *store, Handle property) {
    return store->Lookup(store->GetString(property)->str());
  }

  // Convert number.
  Handle ConvertNumber(Text str) {
    // Try to convert as integer.
    int integer;
    if (safe_strto32(str.data(), str.size(), &integer) &&
        integer >= Handle::kMinInt && integer <= Handle::kMaxInt) {
      return Handle::Integer(integer);
    }

    // Try to convert as floating point number.
    float number;
    if (safe_strtof(str.str(), &number)) {
      return Handle::Float(number);
    }

    return Handle::nil();
  }

  Handle ConvertNumber(Store *store, Handle value) {
    if (value.IsNil()) return Handle::nil();
    if (value.IsInt() || value.IsFloat()) return value;
    if (store->IsString(value)) {
      Handle converted = ConvertNumber(store->GetString(value)->str());
      if (!converted.IsNil()) return converted;
    }
    return value;
  }

  // Convert Wikidata quantity.
  Handle ConvertQuantity(const Frame &value) {
    // Get quantity amount, unit, and bounds.
    Store *store = value.store();
    Handle amount = ConvertNumber(store, value.GetHandle(s_amount_));
    Handle unit = value.GetHandle(s_unit_);
    Handle lower = ConvertNumber(store, value.GetHandle(s_lowerbound_));
    Handle upper = ConvertNumber(store, value.GetHandle(s_upperbound_));
    Handle precision = Handle::nil();

    // Convert unit.
    if (store->IsString(unit)) {
      Text unitstr = store->GetString(unit)->str();
      if (unitstr == "1") {
        unit = Handle::nil();
      } else if (unitstr.starts_with(entity_prefix_)) {
        unitstr.remove_prefix(entity_prefix_.size());
        unit = store->Lookup(unitstr);
      } else {
        LOG(WARNING) << "Unknown unit: " << unitstr;
      }
    }

    // Discard empty bounds.
    if (lower == amount && upper == amount) {
      lower = Handle::nil();
      upper = Handle::nil();
    } else if (amount.IsInt() && lower.IsInt() && upper.IsInt()) {
      int upper_precision = upper.AsInt() - amount.AsInt();
      int lower_precision = amount.AsInt() - lower.AsInt();
      if (upper_precision == 1 && lower_precision == 1) {
        lower = Handle::nil();
        upper = Handle::nil();
      } else if (upper_precision == lower_precision) {
        precision = Handle::Integer(upper_precision);
      }
    } else if (amount.IsFloat() && lower.IsFloat() && upper.IsFloat()) {
      float upper_precision = upper.AsFloat() - amount.AsFloat();
      float lower_precision = amount.AsFloat() - lower.AsFloat();
      float ratio = upper_precision / lower_precision;
      if (ratio > 0.999 && ratio < 1.001) {
        precision = Handle::Float(upper_precision);
      }
    }

    // Create quantity frame if needed.
    if (!unit.IsNil() || !lower.IsNil() || !upper.IsNil()) {
      Builder quantity(store);
      quantity.Add(n_amount_, amount);
      if (!unit.IsNil()) quantity.Add(n_unit_, unit);
      if (!precision.IsNil()) {
        quantity.Add(n_precision_, precision);
      } else {
        if (!lower.IsNil()) quantity.Add(n_low_, lower);
        if (!upper.IsNil()) quantity.Add(n_high_, upper);
      }
      amount = quantity.Create().handle();
    }

    return amount;
  }

  // Convert Wikidata timestamp.
  Handle ConvertTime(const Frame &value) {
    // Convert ISO date string and precision to date.
    Store *store = value.store();
    Object timestamp = value.Get(s_time_);
    Date date(timestamp);
    date.precision = date_precision[value.GetInt(s_precision_, 11)];

    // Convert timestamp to simplified integer or string format.
    int number = Calendar::DateNumber(date);
    if (number != -1) return Handle::Integer(number);
    string ts = Calendar::DateString(date);
    if (!ts.empty()) return store->AllocateString(ts);
    return timestamp.handle();
  }

  // Convert Wikidata entity id.
  Handle ConvertEntity(const Frame &value) {
    String type = value.Get(s_entity_type_).AsString();
    Handle id = value.GetHandle(s_numeric_id_);
    if (type.equals("property")) {
      return Property(value.store(), id.AsInt());
    } else if (type.equals("item")) {
      return Item(value.store(), id.AsInt());
    } else {
      LOG(FATAL) << "Unknown entity type: " << ToText(value);
      return Handle::nil();
    }
  }

  // Convert Wikidata globe coordinate.
  Handle ConvertCoordinate(const Frame &value) {
    // Get fields.
    Store *store = value.store();
    Handle lat = ConvertNumber(store, value.GetHandle(s_latitude_));
    Handle lng = ConvertNumber(store, value.GetHandle(s_longitude_));
    Handle prec = ConvertNumber(store, value.GetHandle(s_precision_));
    Handle globe = value.GetHandle(s_globe_);

    // Determine globe for coordinate, default to Earth.
    if (store->IsString(globe)) {
      Text globestr = store->GetString(globe)->str();
      if (globestr.starts_with(entity_prefix_)) {
        globestr.remove_prefix(entity_prefix_.size());
      }
      if (globestr == "Q2") {
        globe = Handle::nil();
      } else {
        globe = store->Lookup(globestr);
      }
    }

    // Cap precision.
    if (prec.IsFloat() && prec.AsFloat() < 0.0001) prec = Handle::nil();

    // Create geo frame.
    Builder geo(store);
    geo.AddIsA(n_geo_);
    geo.Add(n_lat_, lat);
    geo.Add(n_lng_, lng);
    if (!prec.IsNil()) geo.Add(n_precision_, prec);
    if (!globe.IsNil()) geo.Add(n_globe_, globe);

    return geo.Create().handle();
  }

  // Convert Wikidata value.
  Handle ConvertValue(const Frame &datavalue) {
    String type = datavalue.Get(s_type_).AsString();
    if (type.IsNil()) return Handle::nil();
    if (type.equals("string")) {
      return datavalue.GetHandle(s_value_);
    } else {
      Frame value = datavalue.GetFrame(s_value_);
      if (value.invalid()) return Handle::nil();

      if (type.equals("wikibase-entityid")) {
        return ConvertEntity(value);
      } else if (type.equals("time")) {
        return ConvertTime(value);
      } else if (type.equals("quantity")) {
        return ConvertQuantity(value);
      } else if (type.equals("monolingualtext")) {
        return value.GetHandle(s_text_);
      } else if (type.equals("globecoordinate")) {
        return ConvertCoordinate(value);
      } else {
        LOG(FATAL) << "Unknown data type: " << type.text();
        return Handle::nil();
      }
    }
  }

  // Pick name based on language priority.
  Handle PickName(const Frame &names) {
    Store *store = names.store();
    if (names.invalid()) return Handle::nil();
    int priority = 999;
    Handle name = Handle::nil();
    for (const Slot &l : names) {
      auto f = languages_.find(l.name);
      if (f != languages_.end()) {
        if (f->second.priority < priority) {
          name = Frame(store, l.value).GetHandle(s_value_);
          priority = f->second.priority;
        }
      } else if (name.IsNil()) {
        name = Frame(store, l.value).GetHandle(s_value_);
      }
    }
    return name;
  }

  // Output channel for items and properties.
  task::Channel *item_channel_ = nullptr;
  task::Channel *property_channel_ = nullptr;

  // Commons store.
  Store *commons_ = nullptr;

  // Wikidata property data types.
  std::unordered_map<Text, Handle> datatypes_;

  // Primary language.
  string primary_language_name_;
  Handle primary_language_;

  // Per-language information.
  struct LanguageInfo {
    int priority;
    Handle language;
    Handle wikisite;
  };
  HandleMap<LanguageInfo> languages_;

  // Symbols.
  Names names_;

  Name n_name_{names_, "name"};
  Name n_description_{names_, "description"};
  Name n_lang_{names_, "lang"};
  Name n_source_{names_, "source"};
  Name n_target_{names_, "target"};

  Name n_entity_{names_, "/w/entity"};
  Name n_item_{names_, "/w/item"};
  Name n_property_{names_, "/w/property"};
  Name n_wikipedia_{names_, "/w/item/wikipedia"};
  Name n_low_{names_, "/w/low"};
  Name n_high_{names_, "/w/high"};
  Name n_precision_{names_, "/w/precision"};
  Name n_amount_{names_, "/w/amount"};
  Name n_unit_{names_, "/w/unit"};
  Name n_geo_{names_, "/w/geo"};
  Name n_lat_{names_, "/w/lat"};
  Name n_lng_{names_, "/w/lng"};
  Name n_globe_{names_, "/w/globe"};

  Name n_profile_alias_{names_, "/s/profile/alias"};
  Name n_alias_sources_{names_, "/s/alias/sources"};

  // Wikidata attribute names.
  Name s_id_{names_, "_id"};
  Name s_type_{names_, "type"};
  Name s_datatype_{names_, "datatype"};
  Name s_labels_{names_, "labels"};
  Name s_descriptions_{names_, "descriptions"};
  Name s_value_{names_, "value"};
  Name s_aliases_{names_, "aliases"};
  Name s_claims_{names_, "claims"};
  Name s_sitelinks_{names_, "sitelinks"};
  Name s_datavalue_{names_, "datavalue"};
  Name s_entity_type_{names_, "entity-type"};
  Name s_numeric_id_{names_, "numeric-id"};
  Name s_latitude_{names_, "latitude"};
  Name s_longitude_{names_, "longitude"};
  Name s_precision_{names_, "precision"};
  Name s_globe_{names_, "globe"};
  Name s_mainsnak_{names_, "mainsnak"};
  Name s_text_{names_, "text"};
  Name s_amount_{names_, "amount"};
  Name s_unit_{names_, "unit"};
  Name s_upperbound_{names_, "upperBound"};
  Name s_lowerbound_{names_, "lowerBound"};
  Name s_qualifiers_{names_, "qualifiers"};
  Name s_property_{names_, "property"};
  Name s_title_{names_, "title"};

  // Wikidata types.
  Name s_string_{names_, "string"};
  Name s_time_{names_, "time"};
  Name s_wikibase_entityid_{names_, "wikibase-entityid"};
  Name s_globecoordinate_{names_, "globecoordinate"};
  Name s_monolingualtext_{names_, "monolingualtext"};
  Name s_quantity_{names_, "quantity"};

  // Wikidata property data types.
  Name s_wikibase_item_{names_, "wikibase-item"};
  Name s_commons_media_{names_, "commonsMedia"};
  Name s_external_id_{names_, "external-id"};
  Name s_wikibase_property_{names_, "wikibase-property"};
  Name s_url_{names_, "url"};
  Name s_globe_coordinate_{names_, "globe-coordinate"};
  Name s_math_{names_, "math"};
  Name s_tabular_data_{names_, "tabular-data"};
  Name s_geo_shape_{names_, "geo-shape"};
  // ... plus string, time, quantity, monolingualtext

  // Entity prefix.
  Text entity_prefix_ = "http://www.wikidata.org/entity/";
};

REGISTER_TASK_PROCESSOR("wikidata-importer", WikidataImporter);

// Build Wikidata to Wikipedia id mapping.
class WikipediaMapping : public task::FrameProcessor {
 public:
  void Startup(task::Task *task) override {
    // Get language for mapping.
    string lang = task->Get("language", "en");
    language_ = commons_->Lookup(StrCat("/lang/" + lang));

    // Statistics.
    num_skipped_ = task->GetCounter("skipped_pages");
    num_pages_ = task->GetCounter("total_pages");
    num_articles_ = task->GetCounter("article_pages");
    num_disambiguations_ = task->GetCounter("disambiguation_pages");
    num_categories_ = task->GetCounter("category_pages");
    num_lists_ = task->GetCounter("list_pages");
    num_templates_ = task->GetCounter("template_pages");
    num_infoboxes_ = task->GetCounter("infobox_pages");
  }

  void Process(Slice key, const Frame &frame) override {
    // Get Wikipedia id.
    Frame wikipedia = frame.GetFrame(n_wikipedia_);
    if (wikipedia.invalid()) {
      num_skipped_->Increment();
      return;
    }
    num_pages_->Increment();
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
        if (s.value == n_category_) {
          is_category = true;
        } else if (s.value == n_disambiguation_) {
          is_disambiguation = true;
        } else if (s.value == n_list_) {
          is_list = true;
        } else if (s.value == n_infobox_) {
          is_infobox = true;
        } else if (s.value == n_template_) {
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

  // Names.
  Name n_wikipedia_{names_, "/w/item/wikipedia"};
  Name n_instance_of_{names_, "P31"};
  Name n_category_{names_, "Q4167836"};
  Name n_disambiguation_{names_, "Q4167410"};
  Name n_list_{names_, "Q13406463"};
  Name n_template_{names_, "Q11266439"};
  Name n_infobox_{names_, "Q19887878"};

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
  task::Counter *num_pages_ = nullptr;
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
    // Initialize aux filter.
    filter_.Init(commons_);
    aux_output_ = task->GetSink("aux");

    // Initialize counters.
    num_kb_items_ = task->GetCounter("num_kb_items");
    num_aux_items_ = task->GetCounter("num_aux_items");
  }

  void Process(Slice key, const Frame &frame) override {
    // Check if item is an auxiliary item. This need to be checked before the
    // item is pruned.
    bool aux = filter_.IsAux(frame);

    // Remove aliases and wikilinks from item.
    Builder item(frame);
    item.Delete(n_profile_alias_);
    item.Delete(n_wikipedia_);
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
  Name n_profile_alias_{names_, "/s/profile/alias"};
  Name n_wikipedia_{names_, "/w/item/wikipedia"};

  // Item filter.
  AuxFilter filter_;

  // Optional output channel for aux items.
  task::Channel *aux_output_;

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

