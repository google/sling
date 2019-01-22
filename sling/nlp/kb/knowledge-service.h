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

#ifndef NLP_KB_KNOWLEDGE_SERVICE_H_
#define NLP_KB_KNOWLEDGE_SERVICE_H_

#include <string>

#include "sling/base/types.h"
#include "sling/frame/object.h"
#include "sling/frame/store.h"
#include "sling/http/http-server.h"
#include "sling/http/static-content.h"
#include "sling/nlp/kb/calendar.h"
#include "sling/nlp/kb/name-table.h"

namespace sling {
namespace nlp {

class KnowledgeService {
 public:
  // Information collected for item.
  struct Item {
    Item(Store *store)
        : properties(store),
          xrefs(store),
          categories(store),
          image(Handle::nil()),
          alternate_image(Handle::nil()) {}

    Handles properties;
    Handles xrefs;
    Handles categories;
    Handle image = Handle::nil();
    Handle alternate_image;
  };

  // Load and initialize knowledge base.
  void Load(Store *kb, const string &name_table);

  // Register knowledge base service.
  void Register(HTTPServer *http);

  // Handle KB name queries.
  void HandleQuery(HTTPRequest *request, HTTPResponse *response);

  // Handle KB item requests.
  void HandleGetItem(HTTPRequest *request, HTTPResponse *response);

  // Handle KB frame requests.
  void HandleGetFrame(HTTPRequest *request, HTTPResponse *response);

 private:
  // Fetch properties.
  void FetchProperties(const Frame &item, Item *info);

  // Get standard properties (ref, name, and description).
  void GetStandardProperties(const Frame &item, Builder *builder) const;

  // Get unit name.
  string UnitName(const Frame &unit);

  // Convert value to readable text.
  string AsText(Handle value);

  // Property information.
  struct Property {
    Handle id;
    Handle name;
    Handle datatype;
    string url;
    bool image;
    bool alternate_image;
  };

  // Knowledge base store.
  Store *kb_ = nullptr;

  // Property map.
  HandleMap<Property> properties_;

  // Calendar.
  Calendar calendar_;

  // Name table.
  NameTable aliases_;

  // Knowledge base browser app.
  StaticContent app_{"/kb", "sling/nlp/kb/app"};

  // Symbols.
  Names names_;
  Name n_name_{names_, "name"};
  Name n_description_{names_, "description"};
  Name n_role_{names_, "role"};
  Name n_target_{names_, "target"};
  Name n_properties_{names_, "properties"};
  Name n_qualifiers_{names_, "qualifiers"};
  Name n_xrefs_{names_, "xrefs"};
  Name n_property_{names_, "property"};
  Name n_values_{names_, "values"};
  Name n_categories_{names_, "categories"};
  Name n_type_{names_, "type"};
  Name n_text_{names_, "text"};
  Name n_ref_{names_, "ref"};
  Name n_url_{names_, "url"};
  Name n_thumbnail_{names_, "thumbnail"};
  Name n_matches_{names_, "matches"};
  Name n_lang_{names_, "lang"};

  Name n_xref_type_{names_, "/w/xref"};
  Name n_item_type_{names_, "/w/item"};
  Name n_property_type_{names_, "/w/property"};
  Name n_url_type_{names_, "/w/url"};
  Name n_text_type_{names_, "/w/text"};
  Name n_quantity_type_{names_, "/w/quantity"};
  Name n_geo_type_{names_, "/w/geo"};
  Name n_media_type_{names_, "/w/media"};
  Name n_time_type_{names_, "/w/time"};
  Name n_string_type_{names_, "/w/string"};
  Name n_lat_{names_, "/w/lat"};
  Name n_lng_{names_, "/w/lng"};
  Name n_amount_{names_, "/w/amount"};
  Name n_unit_{names_, "/w/unit"};
  Name n_category_{names_, "/w/item/category"};

  Name n_instance_of_{names_, "P31"};
  Name n_formatter_url_{names_, "P1630"};
  Name n_representative_image_{names_, "Q26940804"};
  Name n_image_{names_, "P18"};

  Name n_unit_symbol_{names_, "P558"};
  Name n_writing_system_{names_, "P282"};
  Name n_latin_script_{names_, "Q8229"};
  Name n_language_{names_, "P2439"};
  Name n_name_language_{names_, "P407"};
};

}  // namespace nlp
}  // namespace sling

#endif  // NLP_KB_KNOWLEDGE_SERVICE_H_

