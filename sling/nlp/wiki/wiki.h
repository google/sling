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

#ifndef SLING_NLP_WIKI_WIKI_H_
#define SLING_NLP_WIKI_WIKI_H_

#include <string>

#include "sling/base/types.h"
#include "sling/frame/object.h"
#include "sling/frame/store.h"

namespace sling {
namespace nlp {

// Wikipedia name spaces.
enum WikipediaNamespace {
  WIKIPEDIA_NAMESPACE_MAIN      = 0,
  WIKIPEDIA_NAMESPACE_USER      = 2,
  WIKIPEDIA_NAMESPACE_WIKIPEDIA = 4,
  WIKIPEDIA_NAMESPACE_FILE      = 6,
  WIKIPEDIA_NAMESPACE_MEDIAWIKI = 8,
  WIKIPEDIA_NAMESPACE_TEMPLATE  = 10,
  WIKIPEDIA_NAMESPACE_HELP      = 12,
  WIKIPEDIA_NAMESPACE_CATEGORY  = 14,
};

// Alias sources.
enum AliasSource {
  SRC_GENERIC                  = 0,
  SRC_WIKIDATA_LABEL           = 1,
  SRC_WIKIDATA_ALIAS           = 2,
  SRC_WIKIPEDIA_TITLE          = 3,
  SRC_WIKIPEDIA_REDIRECT       = 4,
  SRC_WIKIPEDIA_ANCHOR         = 5,
  SRC_WIKIPEDIA_DISAMBIGUATION = 6,
  SRC_WIKIDATA_FOREIGN         = 7,
  SRC_WIKIDATA_NATIVE          = 8,
  SRC_WIKIDATA_DEMONYM         = 9,
};

static const int kNumAliasSources = 10;

extern const char *kAliasSourceName[kNumAliasSources];

// Utility functions for Wikidata and Wikipedia.
class Wiki {
 public:
  // Split title into name and disambiguation.
  static void SplitTitle(const string &title,
                         string *name,
                         string *disambiguation);

  // Return id for Wikipedia page.
  static string Id(const string &lang, const string &title);
  static string Id(const string &lang,
                   const string &prefix,
                   const string &title);

  // Return URL for Wikipedia page.
  static string URL(const string &lang, const string &title);

  // Language priority order.
  static const char *language_priority[];
};

// Filter for auxiliary items. The auxiliary items in the knowledge base are
// items that are used infrequently and are stored in a separate knowledge
// base store.
class AuxFilter {
 public:
  // Initialize auxiliary item filter.
  void Init(Store *store);

  // Check if item is an auxiliary item.
  bool IsAux(const Frame &frame);

 private:
  // Auxiliary item types.
  HandleSet aux_types_;

  // Names.
  Names names_;
  Name n_wikipedia_{names_, "/w/item/wikipedia"};
  Name n_instanceof_{names_, "P31"};
};

}  // namespace nlp
}  // namespace sling

#endif  // SLING_NLP_WIKI_WIKI_H_

