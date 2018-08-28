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

#include "sling/nlp/kb/facts.h"

#include "sling/base/logging.h"

namespace sling {
namespace nlp {

void FactCatalog::Init(Store *store) {
  // Initialize names.
  store_ = store;
  CHECK(names_.Bind(store));

  // Initialize calendar.
  calendar_.Init(store);

  // Determine extraction method for each property.
  for (const Slot &s : Frame(store, store->Lookup("/w/entity"))) {
    if (s.name != p_role_) continue;
    Frame property(store, s.value);
    Handle target = property.GetHandle(p_target_);
    if (target == n_item_) {
      Handle baseprop = property.GetHandle(p_subproperty_of_);
      if (baseprop == p_location_) {
        SetExtractor(property, &Facts::ExtractLocation);
      } else {
        SetExtractor(property, &Facts::ExtractSimple);
      }
    } else if (target == n_time_) {
      SetExtractor(property, &Facts::ExtractDate);
    }
  }

  // Set extraction method for specific properties.
  SetExtractor(p_instance_of_, &Facts::ExtractType);
  SetExtractor(p_educated_at_, &Facts::ExtractAlmaMater);
  SetExtractor(p_employer_, &Facts::ExtractEmployer);
  SetExtractor(p_occupation_, &Facts::ExtractOccupation);
  SetExtractor(p_position_, &Facts::ExtractPosition);
}

void Facts::Extract(Handle item) {
  // Extract facts from the properties of the item.
  auto &extractors = catalog_->property_extractors_;
  for (const Slot &s : Frame(store_, item)) {
    // Look up extractor for property.
    auto f = extractors.find(s.name);
    if (f == extractors.end()) continue;

    // Extract facts for property.
    FactCatalog::Extractor extractor = f->second;
    push(s.name);
    (this->*extractor)(s.value);
    pop();
  }
}

void Facts::ExtractSimple(Handle value) {
  AddFact(store_->Resolve(value));
}

void Facts::ExtractClosure(Handle item, Handle relation) {
  Handles closure(store_);
  closure.push_back(store_->Resolve(item));
  int current = 0;
  while (current < closure.size()) {
    Frame f(store_, closure[current++]);
    AddFact(f.handle());
    for (const Slot &s : f) {
      if (s.name != relation) continue;

      // Check if new item is already known.
      Handle newitem = store_->Resolve(s.value);
      bool known = false;
      for (Handle h : closure) {
        if (newitem == h) {
          known = true;
          break;
        }
      }
      if (!known) closure.push_back(newitem);
    }
  }
}

void Facts::ExtractType(Handle type) {
  ExtractClosure(type, catalog_->p_subclass_of_.handle());
}

void Facts::ExtractClass(Handle item) {
  for (const Slot &s : Frame(store_, store_->Resolve(item))) {
    if (s.name == catalog_->p_instance_of_) {
      push(catalog_->p_instance_of_);
      ExtractType(s.value);
      pop();
    }
  }
}

void Facts::ExtractProperty(Handle item, const Name &property) {
  Frame f(store_, store_->Resolve(item));
  Handle value = f.GetHandle(property);
  if (!value.IsNil()) {
    push(property);
    ExtractSimple(value);
    pop();
  }
}

void Facts::ExtractQualifier(Handle item, const Name &qualifier) {
  Frame f(store_, item);
  if (!f.Has(Handle::is())) return;
  Handle value = f.GetHandle(qualifier);
  if (!value.IsNil()) {
    push(qualifier);
    ExtractSimple(value);
    pop();
  }
}

void Facts::ExtractDate(Handle value) {
  // Convert value to date.
  Date date(Object(store_, store_->Resolve(value)));

  // Add facts for year, decade, and century.
  AddFact(catalog_->calendar_.Year(date));
  AddFact(catalog_->calendar_.Decade(date));
  AddFact(catalog_->calendar_.Century(date));
}

void Facts::ExtractLocation(Handle location) {
  ExtractClosure(location, catalog_->p_located_in_.handle());
}

void Facts::ExtractPlacement(Handle item) {
  Frame f(store_, store_->Resolve(item));
  Handle value = f.GetHandle(catalog_->p_located_in_);
  if (!value.IsNil()) {
    push(catalog_->p_located_in_);
    ExtractLocation(value);
    pop();
  }
}

void Facts::ExtractAlmaMater(Handle institution) {
  ExtractSimple(institution);
  ExtractClass(institution);
  ExtractPlacement(institution);
  ExtractQualifier(institution, catalog_->p_academic_degree_);
}

void Facts::ExtractEmployer(Handle employer) {
  ExtractSimple(employer);
  ExtractClass(employer);
}

void Facts::ExtractOccupation(Handle occupation) {
  ExtractType(occupation);
}

void Facts::ExtractPosition(Handle position) {
  ExtractType(position);
  ExtractProperty(position, catalog_->p_jurisdiction_);
}

void Facts::AddFact(Handle value) {
  if (value.IsNil()) return;
  push(value);
  Handle *begin = path_.data();
  Handle *end = begin + path_.size();
  Handle fact = store_->AllocateArray(begin, end);
  list_.push_back(fact);
  pop();
}

}  // namespace nlp
}  // namespace sling

