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

#ifndef SLING_NLP_WIKI_WIKIPEDIA_MAP_H_
#define SLING_NLP_WIKI_WIKIPEDIA_MAP_H_

#include "sling/frame/object.h"
#include "sling/frame/store.h"
#include "sling/nlp/wiki/wiki.h"
#include "sling/string/text.h"

namespace sling {
namespace nlp {

// Mapping of Wikipedia ids to Wikidata ids. This also includes Wikipedia
// redirects, which will be resolved as part of the lookup.
class WikipediaMap {
 public:
  // Wikipedia page type.
  enum PageType {
    UNKNOWN = 0,
    ARTICLE = 1,
    DISAMBIGUATION = 2,
    CATEGORY = 3,
    LIST = 4,
    TEMPLATE = 5,
    INFOBOX = 6,
    REDIRECT = 7,
  };

  // Wikipedia page information.
  struct PageInfo {
    PageType type = UNKNOWN;  // page type
    Text title;               // title of original Wikipedia page
    Text source_id;           // original page Wikipedia id
    Text target_id;           // target page Wikipedia id
    Text target_title;        // title of target Wikipedia page
    Text qid;                 // Wikidata id of target page
  };

  // Initialize Wikipedia mapping.
  WikipediaMap();

  // Load Wikipedia-to-Wikidata mapping into mapping store.
  void LoadMapping(Text filename);

  // Load redirects into mapping store.
  void LoadRedirects(Text filename);

  // List of redirects.
  const Handles &redirects() const { return redirects_; }

  // Freeze mapping store.
  void Freeze() { store_.Freeze(); }

  // Mapping store.
  Store *store() { return &store_; }

  // Look up Wikipedia id and return Wikidata id. This also resolves redirects.
  // Returns the empty string id the id is not found or does not have the
  // expected type.
  Text Lookup(Text id, PageType type = UNKNOWN);

  // Return page information.
  bool GetPageInfo(Text id, PageInfo *info);
  bool GetPageInfo(Text lang, Text link, PageInfo *info);
  bool GetPageInfo(Text lang, Text prefix, Text link, PageInfo *info);

  // Get redirect information.
  void GetRedirectInfo(Handle redirect, PageInfo *info);

  // Look up Wikipedia link name and return Wikidata id for target.
  Text LookupLink(Text lang, Text link, PageType type = UNKNOWN);
  Text LookupLink(Text lang, Text prefix, Text link, PageType type = UNKNOWN);

 private:
  // Resolve redirects.
  Handle Resolve(Handle);

  // Frame store for mappings.
  Store::Options options_;
  Store store_{&options_};

  // List of redirects.
  Handles redirects_{&store_};

  // Page type mapping.
  HandleMap<PageType> typemap_;

  // Symbols.
  Handle n_qid_;
  Handle n_kind_;
  Handle n_redirect_;
  Handle n_redirect_title_;
  Handle n_redirect_link_;
};

}  // namespace nlp
}  // namespace sling

#endif  // SLING_NLP_WIKI_WIKIPEDIA_MAP_H_

