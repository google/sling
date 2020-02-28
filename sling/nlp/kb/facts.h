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

  // Initialize and return an entity taxonomy. Caller takes ownership.
  Taxonomy *CreateEntityTaxonomy();

  // Returns true if 'coarse' is in the closure of 'fine'. Closure is performed
  // by following 'property' roles.
  bool ItemInClosure(Handle property, Handle coarse, Handle fine);

  // Extract item types (P31) with closure over subclass of (P279).
  void ExtractItemTypes(Handle item, std::vector<Handle> *types);

  // Check if item is a direct or indirect instance of of a type.
  bool InstanceOf(Handle item, Handle type);

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

  // Symbols.
  Names names_;
  Name p_role_{names_, "role"};
  Name p_target_{names_, "target"};
  Name p_located_in_{names_, "P131"};
  Name p_location_{names_, "P276"};
  Name p_instance_of_{names_, "P31"};
  Name p_subclass_of_{names_, "P279"};
  Name p_of_{names_, "P642"};
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
  Name p_described_by_source_{names_, "P1343"};
  Name p_different_from_{names_, "P1889"};
  Name p_located_at_body_of_water_{names_, "P206"};
  Name p_located_on_street_{names_, "P669"};

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
  Facts(const FactCatalog *catalog)
      : catalog_(catalog), store_(catalog_->store_), list_(store_),
        path_(store_) {}

  // Whether closure will be performed on certain facts.
  bool closure() const { return closure_; }
  void set_closure(bool c) { closure_ = c; }

  // Whether numeric dates are extracted.
  bool numeric_dates() const { return numeric_dates_; }
  void set_numeric_dates(bool d) { numeric_dates_ = d; }

  // Extract facts for item.
  void Extract(Handle item);

  // Add fact expansion.
  void Expand(Handle property, Handle value);

  // Extract facts for a subset of properties for item.
  void ExtractFor(Handle item, const HandleSet &properties);

  // Extract simple fact with no backoff.
  void ExtractSimple(Handle value);

  // Skip extraction.
  void ExtractNothing(Handle value);

  // Extract simple property from item.
  void ExtractProperty(Handle item, const Name &property);

  // Extract qualified property from item.
  void ExtractQualifier(Handle item, const Name &qualifier);

  // Extract fact with backoff through transitive property relation.
  void ExtractClosure(Handle item, Handle relation);

  // Extract super-classes.
  void ExtractSuperclass(Handle item);

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

  // Get facts as an array of arrays.
  Handle AsArrays(Store *store) const;
  void AsArrays(Store *store, Handles *array) const;

  // Add value to current fact path.
  void push(Handle value) { path_.push_back(value); }
  void push(const Name &value) { path_.push_back(value.handle()); }

  // Remove last value from current fact path.
  void pop() { path_.pop_back(); }

  // Return the number of extracted facts.
  int size() const { return delimiters_.size(); }

  // Return interval for fact, i.e. fact(i) = list[begin(i):end(i)].
  int begin(int i) const { return i == 0 ? 0 : delimiters_[i - 1]; }
  int end(int i) const { return delimiters_[i]; }

  // Return length of fact chain.
  int length(int i) const { return end(i) - begin(i); }

  // Simple facts have length 2, i.e. a property and a value.
  bool simple(int i) const { return length(i) == 2; }

  // Return base property for fact, i.e. first value in fact path.
  Handle first(int i) const { return list_[begin(i)]; }

  // Return fact value, i.e. last value in fact path.
  Handle last(int i) const { return list_[end(i) - 1]; }

  // Return fingerprint for fact.
  uint64 fingerprint(int i) const {
    return store_->Fingerprint(&list_[begin(i)], &list_[end(i)]);
  }

  // Fact value list.
  const Handles &list() const { return list_; }

  // Fact delimiters.
  const std::vector<int> &delimiters() const { return delimiters_; }

  // Fact groups.
  const std::vector<int> &groups() const { return groups_; }

 private:
  // Catalog for facts.
  const FactCatalog *catalog_;

  // Store for fact properties and values.
  Store *store_;

  // Each extracted fact is a property path and a value, i.e. [P1,..,Pn,Q]. All
  // the extracted facts are concatenated together in a list, and the delimiters
  // marks the boundaries between the facts in the list, so the first fact is
  // stored in list[0:delimiters[0]], and the nth fact is stored in
  // list[delimiters[n-1]:delimiters[n]].
  Handles list_;
  std::vector<int> delimiters_;

  // List of fact group delimiters. A fact group is a set of facts extracted
  // from the same basic fact about the item. A group consists of the basic fact
  // as well as facts derived from this fact and qualified facts. The first
  // group contains facts between 0 and groups_[0]-1. The second group goes from
  // groups_[0] to groups_[1]-1 and so on.
  std::vector<int> groups_;

  // Current fact path [P1,...,Pn].
  Handles path_;

  // Whether closure expansion is enabled.
  bool closure_ = true;

  // Whether numeric dates are extracted.
  bool numeric_dates_ = false;
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

  // Type mapping for taxonomy.
  const HandleMap<int> &typemap() const { return typemap_; }

 private:
  // Fact catalog.
  const FactCatalog *catalog_;

  // Mapping from type to priority.
  HandleMap<int> typemap_;
};

}  // namespace nlp
}  // namespace sling

#endif  // SLING_NLP_KB_FACTS_H_
