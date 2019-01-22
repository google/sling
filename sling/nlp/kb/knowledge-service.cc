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

#include "sling/nlp/kb/knowledge-service.h"

#include <math.h>
#include <string>
#include <unordered_map>
#include <vector>

#include "sling/base/logging.h"
#include "sling/base/types.h"
#include "sling/frame/object.h"
#include "sling/frame/serialization.h"
#include "sling/frame/store.h"
#include "sling/http/http-server.h"
#include "sling/http/web-service.h"
#include "sling/nlp/kb/calendar.h"
#include "sling/string/text.h"
#include "sling/string/strcat.h"

namespace sling {
namespace nlp {

// Convert geo coordinate from decimal to minutes and seconds.
static string ConvertGeoCoord(double coord, bool latitude) {
  // Compute direction.
  const char *sign;
  if (coord < 0) {
    coord = -coord;
    sign = latitude ? "S" : "W";
  } else {
    sign = latitude ? "N" : "E";
  }

  // Compute degrees.
  double integer;
  double remainder = modf(coord, &integer);
  int degrees = static_cast<int>(integer);

  // Compute minutes.
  remainder = modf(remainder * 60, &integer);
  int minutes = static_cast<int>(integer);

  // Compute seconds.
  remainder = modf(remainder * 60, &integer);
  int seconds = static_cast<int>(integer + 0.5);

  // Build coordinate string.
  return StrCat(degrees, "°", minutes, "′", seconds, "″", sign);
}

void KnowledgeService::Load(Store *kb, const string &name_table) {
  // Bind names and freeze store.
  kb_ = kb;
  CHECK(names_.Bind(kb_));

  // Get meta data for properties.
  for (const Slot &s : Frame(kb, kb->Lookup("/w/entity"))) {
    if (s.name != n_role_) continue;
    Frame property(kb, s.value);
    Property p;

    // Get property id and name.
    p.id = s.value;
    p.name = property.GetHandle(n_name_);

    // Property data type.
    p.datatype = property.GetHandle(n_target_);

    // Get URL formatter for property.
    Handle formatter = property.Resolve(n_formatter_url_);
    if (kb->IsString(formatter)) {
      p.url = String(kb, formatter).value();
    }

    // Check if property is a representative image for the item.
    p.image = false;
    p.alternate_image = false;
    for (const Slot &ps : property) {
      if (ps.name == n_instance_of_ && ps.value == n_representative_image_) {
        if (property == n_image_) {
          p.image = true;
        } else {
          p.alternate_image = true;
        }
      }
    }

    // Add property.
    properties_[p.id] = p;
  }

  // Initialize calendar.
  calendar_.Init(kb);

  // Load name table.
  if (!name_table.empty()) {
    LOG(INFO) << "Loading name table from " << name_table;
    aliases_.Load(name_table);
  }
}

void KnowledgeService::Register(HTTPServer *http) {
  http->Register("/kb/query", this, &KnowledgeService::HandleQuery);
  http->Register("/kb/item", this, &KnowledgeService::HandleGetItem);
  http->Register("/kb/frame", this, &KnowledgeService::HandleGetFrame);
  app_.Register(http);
}

void KnowledgeService::HandleQuery(HTTPRequest *request,
                                   HTTPResponse *response) {
  WebService ws(kb_, request, response);

  // Get query
  Text query = ws.Get("q");
  int window = ws.Get("window", 5000);
  int limit = ws.Get("limit", 30);
  int boost = ws.Get("boost", 1000);
  LOG(INFO) << "Name query: " << query;

  // Lookup name in name table.
  std::vector<Text> matches;
  if (!query.empty()) {
    aliases_.LookupPrefix(query, window, boost, &matches);
  }

  // Check for exact match with id.
  Handles results(ws.store());
  Handle idmatch = kb_->Lookup(query);
  if (!idmatch.IsNil()) {
    Frame item(kb_, idmatch);
    if (item.valid()) {
      Builder match(ws.store());
      GetStandardProperties(item, &match);
      results.push_back(match.Create().handle());
    }
  }

  // Generate response.
  Builder b(ws.store());
  for (Text id : matches) {
    if (results.size() >= limit) break;
    Frame item(kb_, kb_->Lookup(id));
    if (item.invalid()) continue;
    Builder match(ws.store());
    GetStandardProperties(item, &match);
    results.push_back(match.Create().handle());
  }
  b.Add(n_matches_,  Array(ws.store(), results));

  // Return response.
  ws.set_output(b.Create());
}

void KnowledgeService::HandleGetItem(HTTPRequest *request,
                                     HTTPResponse *response) {
  WebService ws(kb_, request, response);

  // Look up item in knowledge base.
  Text itemid = ws.Get("id");
  LOG(INFO) << "Look up item '" << itemid << "'";
  Handle handle = kb_->LookupExisting(itemid);
  if (handle.IsNil()) {
    response->SendError(404, nullptr, "Item not found");
    return;
  }

  // Generate response.
  Frame item(ws.store(), handle);
  if (!item.valid()) {
    response->SendError(404, nullptr, "Invalid item");
    return;
  }
  Builder b(ws.store());
  GetStandardProperties(item, &b);
  Handle datatype = item.GetHandle(n_target_);
  if (!datatype.IsNil()) {
    Frame dt(kb_, datatype);
    if (dt.valid()) {
      b.Add(n_type_, dt.GetHandle(n_name_));
    }
  }

  // Fetch properties.
  Item info(ws.store());
  FetchProperties(item, &info);
  b.Add(n_properties_, Array(ws.store(), info.properties));
  b.Add(n_xrefs_, Array(ws.store(), info.xrefs));
  b.Add(n_categories_, Array(ws.store(), info.categories));

  // Set item image.
  if (!info.image.IsNil()) {
    b.Add(n_thumbnail_, info.image);
  } else if (!info.alternate_image.IsNil()) {
    b.Add(n_thumbnail_, info.alternate_image);
  }

  // Return response.
  ws.set_output(b.Create());
}

void KnowledgeService::FetchProperties(const Frame &item, Item *info) {
  // Collect properties and values.
  HandleMap<Handles *> property_map;
  for (const Slot &s : item) {
    // Collect categories.
    if (s.name == n_category_) {
      Builder b(item.store());
      Frame cat(item.store(), s.value);
      GetStandardProperties(cat, &b);
      info->categories.push_back(b.Create().handle());
      continue;
    }

    // Skip non-property slots.
    if (properties_.find(s.name) == properties_.end()) continue;

    // Get property list for property.
    Handles *property_list = nullptr;
    auto f = property_map.find(s.name);
    if (f != property_map.end()) {
      property_list = f->second;
    } else {
      property_list = new Handles(item.store());
      property_map[s.name] = property_list;
    }

    // Add property value.
    property_list->push_back(s.value);
  }

  // Build property lists.
  for (auto it : property_map) {
    const auto f = properties_.find(it.first);
    CHECK(f != properties_.end());
    const Property &property = f->second;

    // Add property information.
    Builder p(item.store());
    p.Add(n_property_, property.name);
    p.Add(n_ref_, property.id);
    p.Add(n_type_, property.datatype);

    // Add property values.
    Handles values(item.store());
    for (Handle h : *it.second) {
      // Resolve value.
      Handle value = h;
      bool qualified = false;
      if (kb_->IsFrame(h)) {
        // Handle the ambiguity between qualified frames and mono-lingual text
        // by checking for the presence of a language slot.
        Frame f(kb_, h);
        Handle qua = f.GetHandle(Handle::is());
        Handle lang = f.GetHandle(n_lang_);
        if (!qua.IsNil() && lang.IsNil()) {
          value = qua;
          qualified = true;
        }
      }

      // Add property value based on property type.
      Builder v(item.store());
      if (property.datatype == n_item_type_) {
        // Add reference to other item.
        Frame ref(kb_, value);
        if (ref.valid()) {
          GetStandardProperties(ref, &v);
        }
      } else if (property.datatype == n_xref_type_) {
        // Add external reference.
        String identifier(kb_, value);
        v.Add(n_text_, identifier);
      } else if (property.datatype == n_property_type_) {
        // Add reference to property.
        Frame ref(kb_, value);
        if (ref.valid()) {
          GetStandardProperties(ref, &v);
        }
      } else if (property.datatype == n_string_type_) {
        // Add string value.
        v.Add(n_text_, value);
      } else if (property.datatype == n_text_type_) {
        // Add text value with language.
        if (kb_->IsFrame(value)) {
          Frame monotext(kb_, value);
          v.Add(n_text_, monotext.GetHandle(Handle::is()));
          Frame lang = monotext.GetFrame(n_lang_);
          if (lang.valid()) {
            v.Add(n_lang_, lang.GetHandle(n_name_));
          }
        } else {
          v.Add(n_text_, value);
        }
      } else if (property.datatype == n_url_type_) {
        // Add URL value.
        v.Add(n_text_, value);
        v.Add(n_url_, value);
      } else if (property.datatype == n_media_type_) {
        // Add image.
        v.Add(n_text_, value);

        // Set representative image for item.
        if (property.image && info->image.IsNil()) {
          info->image = value;
        }
        if (property.alternate_image && info->alternate_image.IsNil()) {
          info->alternate_image = value;
        }
      } else if (property.datatype == n_geo_type_) {
        // Add coordinate value.
        Frame coord(kb_, value);
        double lat = coord.GetFloat(n_lat_);
        double lng = coord.GetFloat(n_lng_);
        v.Add(n_text_, StrCat(ConvertGeoCoord(lat, true), ", ",
                              ConvertGeoCoord(lng, false)));
        v.Add(n_url_, StrCat("http://maps.google.com/maps?q=",
                              lat, ",", lng));
      } else if (property.datatype == n_quantity_type_) {
        // Add quantity value.
        string text;
        if (kb_->IsFrame(value)) {
          Frame quantity(kb_, value);
          text = AsText(quantity.GetHandle(n_amount_));

          // Get unit symbol, preferably in latin script.
          Frame unit = quantity.GetFrame(n_unit_);
          text.append(" ");
          text.append(UnitName(unit));
        } else {
          text = AsText(value);
        }
        v.Add(n_text_, text);
      } else if (property.datatype == n_time_type_) {
        // Add time value.
        Object time(kb_, value);
        v.Add(n_text_, calendar_.DateAsString(time));
      }

      // Add URL if property has URL formatter.
      if (!property.url.empty() && kb_->IsString(value)) {
        String identifier(kb_, value);
        string url = property.url;
        int pos = url.find("$1");
        if (pos != -1) {
          Text replacement = identifier.text();
          url.replace(pos, 2, replacement.data(), replacement.size());
        }
        if (!url.empty()) v.Add(n_url_, url);
      }

      // Get qualifiers.
      if (qualified) {
        Item qualifiers(item.store());
        FetchProperties(Frame(item.store(), h), &qualifiers);
        if (!qualifiers.properties.empty()) {
          v.Add(n_qualifiers_, Array(item.store(), qualifiers.properties));
        }
      }

      values.push_back(v.Create().handle());
    }
    p.Add(n_values_, Array(item.store(), values));

    // Add property to property list.
    if (property.datatype == n_xref_type_) {
      info->xrefs.push_back(p.Create().handle());
    } else {
      info->properties.push_back(p.Create().handle());
    }
    delete it.second;
  }
}

void KnowledgeService::GetStandardProperties(const Frame &item,
                                             Builder *builder) const {
  builder->Add(n_ref_, item.Id());
  Handle name = item.GetHandle(n_name_);
  if (!name.IsNil()) {
    builder->Add(n_text_, name);
  } else {
    builder->Add(n_text_, item.Id());
  }
  Handle description = item.GetHandle(n_description_);
  if (!description.IsNil()) builder->Add(n_description_, description);
}

string KnowledgeService::AsText(Handle value) {
  value = kb_->Resolve(value);
  if (value.IsInt()) {
    char buf[32];
    snprintf(buf, sizeof(buf), "%d", value.AsInt());
    return buf;
  } else if (value.IsFloat()) {
    float number = value.AsFloat();
    char buf[32];
    if (floorf(number) == number) {
      snprintf(buf, sizeof(buf), "%.f", number);
    } else if (number > 0.001) {
      snprintf(buf, sizeof(buf), "%.3f", number);
    } else {
      snprintf(buf, sizeof(buf), "%g", number);
    }
    return buf;
  } else {
    return ToText(kb_, value);
  }
}

string KnowledgeService::UnitName(const Frame &unit) {
  // Check for valid unit.
  if (!unit.valid()) return "";

  // Find best unit symbol, preferably in latin script.
  Handle best = Handle::nil();
  Handle fallback = Handle::nil();
  for (const Slot &s : unit) {
    if (s.name != n_unit_symbol_) continue;
    Frame symbol(kb_, s.value);
    if (!symbol.valid()) {
      if (fallback.IsNil()) fallback = s.value;
      continue;
    }

    // Prefer latin script.
    Handle script = symbol.GetHandle(n_writing_system_);
    if (script == n_latin_script_ && best.IsNil()) {
      best = s.value;
    } else {
      // Skip language specific names.
      if (symbol.Has(n_language_) || symbol.Has(n_name_language_)) continue;

      // Fall back to symbols with no script.
      if (script == Handle::nil() && fallback.IsNil()) {
        fallback = s.value;
      }
    }
  }
  if (best.IsNil()) best = fallback;

  // Try to get name of best unit symbol.
  if (!best.IsNil()) {
    Handle unit_name = kb_->Resolve(best);
    if (kb_->IsString(unit_name)) {
      return String(kb_, unit_name).value();
    }
  }

  // Fall back to item name of unit.
  return unit.GetString(n_name_);
}

void KnowledgeService::HandleGetFrame(HTTPRequest *request,
                                      HTTPResponse *response) {
  WebService ws(kb_, request, response);

  // Look up frame in knowledge base.
  Text id = ws.Get("id");
  Handle handle = kb_->LookupExisting(id);
  if (handle.IsNil()) {
    response->SendError(404, nullptr, "Frame not found");
    return;
  }

  // Return frame as response.
  ws.set_output(Object(kb_, handle));
}

}  // namespace nlp
}  // namespace sling

