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

#ifndef SLING_NLP_WIKI_WIKIDATA_CONVERTER_H_
#define SLING_NLP_WIKI_WIKIDATA_CONVERTER_H_

#include <string>
#include <unordered_map>

#include "sling/frame/object.h"
#include "sling/frame/store.h"
#include "sling/string/strcat.h"
#include "sling/string/text.h"

namespace sling {
namespace nlp {

// Conversion of Wikidata JSON encoding to SLING item encoding.
class WikidataConverter {
 public:
  // Initialize WikiData converter.
  WikidataConverter(Store *commons, const string &language);

  // Convert Wikidata JSON item to frame representation.
  Frame Convert(const Frame &item);

  // Only import labels and description in primary language.
  bool only_primary_language() const { return only_primary_language_; }
  void set_only_primary_language(bool b) { only_primary_language_ = b; }

  // Only import labels and description in known languages.
  bool only_known_languages() const { return only_known_languages_; }
  void set_only_known_languages(bool b) { only_known_languages_ = b; }

 private:
  // Return symbol for Wikidata item.
  static Handle Item(Store *store, int id) {
    return store->Lookup(StrCat("Q", id));
  }

  // Return symbol for Wikidata lexeme.
  static Handle Lexeme(Store *store, int id) {
    return store->Lookup(StrCat("L", id));
  }

  // Return symbol for Wikidata form.
  static Handle Form(Store *store, Text id) {
    return store->Lookup(id);
  }

  // Return symbol for Wikidata sense.
  static Handle Sense(Store *store, Text id) {
    return store->Lookup(id);
  }

  // Return symbol for Wikidata property.
  static Handle Property(Store *store, int id) {
    return store->Lookup(StrCat("P", id));
  }
  static Handle Property(Store *store, Handle property) {
    return store->Lookup(store->GetString(property)->str());
  }

  // Convert number.
  Handle ConvertNumber(Text str);
  Handle ConvertNumber(Store *store, Handle value);

  // Convert Wikidata quantity.
  Handle ConvertQuantity(const Frame &value);

  // Convert Wikidata monolingual text.
  Handle ConvertText(const Frame &value);

  // Convert Wikidata timestamp.
  Handle ConvertTime(const Frame &value);

  // Convert Wikidata entity id.
  Handle ConvertEntity(const Frame &value);

  // Convert Wikidata globe coordinate.
  Handle ConvertCoordinate(const Frame &value);

  // Convert Wikidata value.
  Handle ConvertValue(const Frame &datavalue);

  // Symbols.
  Names names_;

  // Wikidata property data types.
  std::unordered_map<Text, Handle> datatypes_;

  // Primary language.
  string primary_language_name_;
  Handle primary_language_;
  bool only_primary_language_ = false;
  bool only_known_languages_ = false;

  // Per-language information.
  struct LanguageInfo {
    int priority;
    Handle language;
    Handle wikisite;
  };
  HandleMap<LanguageInfo> languages_;
  std::unordered_map<string, Handle> language_map_;

  Name n_name_{names_, "name"};
  Name n_description_{names_, "description"};
  Name n_lang_{names_, "lang"};
  Name n_source_{names_, "source"};
  Name n_target_{names_, "target"};

  Name n_entity_{names_, "/w/entity"};
  Name n_item_{names_, "/w/item"};
  Name n_lexeme_{names_, "/w/lexeme"};
  Name n_form_{names_, "/w/form"};
  Name n_sense_{names_, "/w/sense"};
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
  Name n_lang_mul_{names_, "/lang/mul"};
  Name n_lang_none_{names_, "/lang/zxx"};

  Name n_alias_{names_, "alias"};
  Name n_sources_{names_, "sources"};

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
  Name s_language_{names_, "language"};
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
  Name s_wikibase_lexeme_{names_, "wikibase-lexeme"};
  Name s_wikibase_form_{names_, "wikibase-form"};
  Name s_wikibase_sense_{names_, "wikibase-sense"};
  Name s_commons_media_{names_, "commonsMedia"};
  Name s_external_id_{names_, "external-id"};
  Name s_wikibase_property_{names_, "wikibase-property"};
  Name s_url_{names_, "url"};
  Name s_globe_coordinate_{names_, "globe-coordinate"};
  Name s_math_{names_, "math"};
  Name s_tabular_data_{names_, "tabular-data"};
  Name s_geo_shape_{names_, "geo-shape"};
  Name s_musical_notation_{names_, "musical-notation"};
  // ... plus string, time, quantity, monolingualtext

  // Entity prefix.
  Text entity_prefix_ = "http://www.wikidata.org/entity/";
};

}  // namespace nlp
}  // namespace sling

#endif  // SLING_NLP_WIKI_WIKIDATA_CONVERTER_H_

