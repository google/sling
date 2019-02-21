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

#ifndef SLING_NLP_KB_FACTS_H_
#define SLING_NLP_KB_FACTS_H_

#include "sling/frame/store.h"
#include "sling/frame/object.h"
#include "sling/nlp/kb/calendar.h"

namespace sling {
namespace nlp {

class Facts;
class Taxonomy;

// A fact catalog holds the configuration information for extracting facts
// from items.
class FactCatalog {
 public:
  // Fact extractor method.
  typedef void (Facts::*Extractor)(Handle value);

  // Intialize fact catalog.
  void Init(Store *store);

  // Initialize and return a default taxonomy. Caller takes ownership.
  Taxonomy *CreateDefaultTaxonomy();

 private:
  // Set extractor for property type.
  void SetExtractor(Handle property, Extractor extractor) {
    property_extractors_[property] = extractor;
  }
  void SetExtractor(const Frame &property, Extractor extractor) {
    SetExtractor(property.handle(), extractor);
  }
  void SetExtractor(const Name &property, Extractor extractor) {
    SetExtractor(property.handle(), extractor);
  }

  // Check if item is a base item.
  bool IsBaseItem(Handle item) const { return base_items_.count(item) != 0; }

  // Knowledge base store.
  Store *store_ = nullptr;

  // Extraction mapping for properties.
  HandleMap<Extractor> property_extractors_;

  // Calendar.
  Calendar calendar_;

  // Items that stop closure expansion.
  HandleSet base_items_;

  // Date and location valued properties.
  HandleSet date_properties_;
  HandleSet location_properties_;

  // Symbols.
  Names names_;
  Name p_role_{names_, "role"};
  Name p_target_{names_, "target"};
  Name p_located_in_{names_, "P131"};
  Name p_location_{names_, "P276"};
  Name p_instance_of_{names_, "P31"};
  Name p_subclass_of_{names_, "P279"};
  Name p_subproperty_of_{names_, "P1647"};
  Name p_educated_at_{names_, "P69"};
  Name p_occupation_{names_, "P106"};
  Name p_employer_{names_, "P108"};
  Name p_jurisdiction_{names_, "P1001"};
  Name p_position_{names_, "P39"};
  Name p_academic_degree_{names_, "P512"};
  Name p_member_of_sports_team_{names_, "P54"};
  Name p_league_{names_, "P118"};
  Name p_time_period_{names_, "P2348"};
  Name p_start_time_{names_, "P580"};
  Name p_end_time_{names_, "P582"};

  Name n_time_{names_, "/w/time"};
  Name n_item_{names_, "/w/item"};

  friend class Facts;
  friend class Taxonomy;
};

// Set of facts. A fact is represented as a list properties followed by a
// value, e.g. [P69 P31 Q3918] means "educated at: instance of: university".
// A fact can be seen as a path through the frame graph from an unspecified
// starting frame.
class Facts {
 public:
  Facts(const FactCatalog *catalog, Store *store)
      : catalog_(catalog), store_(store), list_(store), path_(store),
      closure_(true) {}

  // Accessor/mutator for whether closure will be performed on certain facts.
  bool closure() const { return closure_; }
  void set_closure(bool c) { closure_ = c; }

  // Extract facts for item.
  void Extract(Handle item);

  // Extract facts for item only for properties in 'properties'.
  void ExtractFor(Handle item, const HandleSet &properties);

  // Extract item types (P31) with closure over subclass of (P279).
  void ExtractItemTypes(Handle item, Handles *types);

  // Extract simple fact with no backoff.
  void ExtractSimple(Handle value);

  // Extract simple property from item.
  void ExtractProperty(Handle item, const Name &property);

  // Extract qualified property from item.
  void ExtractQualifier(Handle item, const Name &qualifier);

  // Extract fact with backoff through transitive property relation.
  void ExtractClosure(Handle item, Handle relation);

  // Extract type with super-class backoff.
  void ExtractType(Handle type);

  // Extract class using instance-of with super-class backoff.
  void ExtractClass(Handle item);

  // Extract date-valued fact with backoff to year, decade and century.
  void ExtractDate(Handle value);

  // Extract time period.
  void ExtractTimePeriod(Handle period);

  // Extract location of item with containment backoff.
  void ExtractPlacement(Handle item);

  // Extract location with containment backoff.
  void ExtractLocation(Handle location);

  // Extract alma mater.
  void ExtractAlmaMater(Handle institution);

  // Extract occupation.
  void ExtractOccupation(Handle occupation);

  // Extract employer.
  void ExtractEmployer(Handle employer);

  // Extract position.
  void ExtractPosition(Handle position);

  // Extract team.
  void ExtractTeam(Handle team);

  // Add fact based on current path.
  void AddFact(Handle value);

  // Returns true if 'coarse' subsumes 'fine' as a value of 'property'.
  // E.g. 'US' subsumes 'New York' for place of birth property.
  // Returns true in the base case where coarse = fine.
  // Only works for location and date-valued properties.
  bool Subsumes(Handle property, Handle coarse, Handle fine);

  // Add value to current fact path.
  void push(Handle value) { path_.push_back(value); }
  void push(const Name &value) { path_.push_back(value.handle()); }

  // Remove last value from current fact path.
  void pop() { path_.pop_back(); }

  // Fact list.
  const Handles &list() const { return list_; }

 private:
  // Catalog for facts.
  const FactCatalog *catalog_;

  // Store for facts.
  Store *store_;

  // List of facts in the form of [P1,..,Pn,Q] values.
  Handles list_;

  // Current fact path [P1,...,Pn].
  Handles path_;

  // Whether closure is enabled.
  bool closure_;
};

// A taxonomy is a type system for classifying items into a list of types.
// The types are specified in order from the most specific type to the most
// general type.
class Taxonomy {
 public:
  // Initialize taxonomy from a ranked type list.
  Taxonomy(const FactCatalog *catalog, const std::vector<Text> &types);
  Taxonomy(const FactCatalog *catalog, const char **types);

  // Classify item according to taxonomy.
  Handle Classify(const Frame &item);

 private:
  // Fact catalog.
  const FactCatalog *catalog_;

  // Mapping from type to priority.
  HandleMap<int> typemap_;
};

}  // namespace nlp
}  // namespace sling

#endif  // SLING_NLP_KB_FACTS_H_
