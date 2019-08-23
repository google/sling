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

#include "sling/nlp/wiki/wiki.h"

#include <string>

#include "sling/util/unicode.h"

namespace sling {
namespace nlp {

// Language priority order.
const char *Wiki::language_priority[] = {
  "en", "da", "sv", "no", "de", "fr", "es", "it", "nl", "pt", "pl", "fi",
  "ca", "eu", "la", "eo", "cs", "sh", "hu", "ro",
  "el", "ru", "uk", "sr", "bg",
  nullptr,
};

// Alias source names.
const char *kAliasSourceName[kNumAliasSources] = {
  "generic",
  "wikidata_label",
  "wikidata_alias",
  "wikipedia_title",
  "wikipedia_redirect",
  "wikipedia_anchor",
  "wikipedia_disambiguation",
  "wikidata_foreign",
  "wikidata_native",
  "wikidata_demonym",
  "wikipedia_link",
  "wikidata_name",
  "wikipedia_name",
  "wikipedia_nickname",
  "variation",
};

void Wiki::SplitTitle(const string &title,
                      string *name,
                      string *disambiguation) {
  // Find last phrase in parentheses.
  size_t open = -1;
  size_t close = -1;
  for (int i = 0; i < title.size(); ++i) {
    if (title[i] == '(') {
      open = i;
      close = -1;
    } else if (title[i] == ')') {
      close = i;
    }
  }

  // Split title into name and disambiguation parts.
  if (open > 1 && close == title.size() - 1) {
    int end = open - 1;
    while (end > 0 && title[end - 1] == ' ') end--;
    name->assign(title, 0, end);
    disambiguation->assign(title, open + 1, close - open - 1);
  } else {
    *name = title;
    disambiguation->clear();
  }
}

string Wiki::Id(const string &lang, const string &title) {
  string t;
  UTF8::ToTitleCase(title, &t);
  for (char &c : t) {
    if (c == ' ') c = '_';
  }
  return "/wp/" + lang + "/" + t;
}

string Wiki::Id(const string &lang, const string &prefix, const string &title) {
  string t;
  UTF8::ToTitleCase(title, &t);
  for (char &c : t) {
    if (c == ' ') c = '_';
  }
  return "/wp/" + lang + "/" + prefix + ":" + t;
}

string Wiki::URL(const string &lang, const string &title) {
  string t;
  UTF8::ToTitleCase(title, &t);
  for (char &c : t) {
    if (c == ' ') c = '_';
  }
  return "http://" + lang + ".wikipedia.org/wiki/" + t;
}


void WikimediaTypes::Init(Store *store) {
  CHECK(names_.Bind(store));
}

bool WikimediaTypes::IsCategory(Handle type) {
  return type == n_category_ ||
         type == n_disambiguation_category_ ||
         type == n_list_category_ ||
         type == n_template_category_ ||
         type == n_admin_category_ ||
         type == n_user_category_ ||
         type == n_user_language_category_ ||
         type == n_stub_category_ ||
         type == n_meta_category_ ||
         type == n_navbox_category_ ||
         type == n_infobox_templates_category_;
}

bool WikimediaTypes::IsDisambiguation(Handle type) {
  return type == n_disambiguation_;
}

bool WikimediaTypes::IsList(Handle type) {
  return type == n_list_;
}

bool WikimediaTypes::IsTemplate(Handle type) {
  return type == n_template_;
}

bool WikimediaTypes::IsInfobox(Handle type) {
  return type == n_infobox_;
}

bool WikimediaTypes::IsDuplicate(Handle type) {
  return type == n_permanent_duplicate_item_;
}

void AuxFilter::Init(Store *store) {
  const char *kAuxItemtypes[] = {
    "Q13442814",  // scientific article
    "Q17329259",  // encyclopedic article
    "Q17633526",  // Wikinews article
    "Q732577",    // publication
    "Q7187",      // gene
    "Q16521",     // taxon
    "Q8054",      // protein
    "Q11173",     // chemical compound
    nullptr,
  };
  for (const char **type = kAuxItemtypes; *type; ++type) {
    aux_types_.insert(store->Lookup(*type));
  }
  names_.Bind(store);
}

bool AuxFilter::IsAux(const Frame &frame) {
  // Never mark items in Wikipedia as aux.
  Frame wikipedia = frame.GetFrame(n_wikipedia_);
  if (wikipedia.valid() && wikipedia.size() > 0) return false;

  // Check item types.
  for (const Slot &slot : frame) {
    if (slot.name != n_instanceof_) continue;
    Handle type = frame.store()->Resolve(slot.value);
    if (aux_types_.count(type) > 0) return true;
  }

  return false;
}

}  // namespace nlp
}  // namespace sling

