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

#include <set>
#include <string>
#include <unordered_map>

#include "sling/base/logging.h"
#include "sling/base/types.h"
#include "sling/file/textmap.h"
#include "sling/frame/serialization.h"
#include "sling/nlp/document/phrase-tokenizer.h"
#include "sling/nlp/wiki/wiki.h"
#include "sling/task/frames.h"
#include "sling/task/task.h"
#include "sling/task/reducer.h"
#include "sling/util/unicode.h"

namespace sling {
namespace nlp {

// Compute Levenshtein edit distance between string in s of length m and string
// in t of length n. See https://en.wikipedia.org/wiki/Levenshtein_distance.
static int LevenshteinDistance(const int *s, int m, const int *t, int n) {
  // Skip common prefix since this does not affect edit distance.
  while (m > 0 && n > 0 && *s == *t) s++, t++, m--, n--;

  // Keep track of previous and current cost.
  std::vector<int> prev(n + 1);
  std::vector<int> curr(n + 1);

  // Initialize the previous distances, which is just the number of characters
  // to delete from t.
  for (int i = 0; i <= n; ++i) prev[i] = i;

  // Perform dynamic program to calculate the edit distance.
  for (int i = 0; i < m; ++i) {
    // Calculate current distances from the previous distances. First, edit
    // distance is delete i+1 characters from s to match empty t.
    curr[0] = i + 1;

    // Compute edit distance for remaining characters.
    for (int j = 0; j < n; ++j) {
      int deletion = prev[j + 1] + 1;
      int insertion = curr[j] + 1;
      int substitution = s[i] == t[j] ? prev[j] : prev[j] + 1;
      int cost = std::min(std::min(deletion, insertion), substitution);
      curr[j + 1] = cost;
    }

    // Swap current and previous for next iteration.
    prev.swap(curr);
  }

  return prev[n];
}

// Extract aliases for items.
class AliasExtractor : public task::FrameProcessor {
 public:
  void Startup(task::Task *task) override {
    string lang = task->Get("language", "en");
    language_ = commons_->Lookup("/lang/" + lang);
    skip_aux_ = task->Get("skip_aux", false);
    wikitypes_.Init(commons_);

    // Initialize filter.
    if (skip_aux_) filter_.Init(commons_);
    num_aux_items_ = task->GetCounter("num_aux_items");
  }

  void Process(Slice key, const Frame &frame) override {
    // Optionally skip aux items.
    if (skip_aux_ && filter_.IsAux(frame)) {
      num_aux_items_->Increment();
      return;
    }

    // Create frame with all aliases matching language.
    Store *store = frame.store();
    Builder a(store);
    for (const Slot &s : frame) {
      if (s.name == n_name_) {
        // Add item name as alias.
        AddAlias(&a, s.value, SRC_WIKIDATA_LABEL);
      } else if (s.name == n_alias_) {
        Frame alias(store, s.value);
        if (alias.GetHandle(n_lang_) == language_) {
          a.Add(n_alias_, alias);
        } else {
          // Add aliases in other languages as foreign alias.
          AddAlias(&a, alias.GetHandle(n_name_), SRC_WIKIDATA_FOREIGN,
                   alias.GetHandle(n_lang_), alias.GetInt(n_count_));
        }
      } else if (s.name == n_native_name_ || s.name == n_native_label_) {
        // Output native names/labels as native aliases.
        AddAlias(&a, store->Resolve(s.value), SRC_WIKIDATA_NATIVE);
      } else if (s.name == n_iso3166_country_code_2_ ||
                 s.name == n_iso3166_country_code_3_) {
        // Output country codes as alternative names.
        AddAlias(&a, store->Resolve(s.value), SRC_WIKIDATA_NAME);
      } else if (s.name == n_nickname_ || s.name == n_pseudonym_) {
        // Output nicknames and pseudonyms as alternative names.
        AddAlias(&a, store->Resolve(s.value), SRC_WIKIDATA_NAME);
      } else if (s.name == n_short_name_ ||
                 s.name == n_generic_name_ ||
                 s.name == n_birth_name_ ||
                 s.name == n_married_name_ ||
                 s.name == n_official_name_ ||
                 s.name == n_female_form_ ||
                 s.name == n_male_form_ ||
                 s.name == n_unit_symbol_ ||
                 s.name == n_demonym_) {
        // Output names as alternative or foreign names.
        Handle lang = Handle::nil();
        if (store->IsFrame(s.value)) {
          Frame f(store, s.value);
          lang = f.GetHandle(n_lang_);
        }
        if (lang.IsNil() || lang == language_) {
          AliasSource source = SRC_WIKIDATA_NAME;
          if (s.name == n_demonym_) source = SRC_WIKIDATA_DEMONYM;
          AddAlias(&a, store->Resolve(s.value), source);
        } else {
          AddAlias(&a, store->Resolve(s.value), SRC_WIKIDATA_FOREIGN);
        }
      } else if (s.name == n_instance_of_) {
        // Discard alias for non-entity items.
        Handle type = store->Resolve(s.value);
        if (wikitypes_.IsCategory(type) ||
            wikitypes_.IsDisambiguation(type) ||
            wikitypes_.IsInfobox(type) ||
            wikitypes_.IsTemplate(type) ||
            wikitypes_.IsDuplicate(type)) {
          return;
        }
      }
    }

    // Output aliases matching language.
    Frame aliases = a.Create();
    if (aliases.size() != 0) {
      Output(key, aliases);
    }
  }

  // Add alias.
  void AddAlias(Builder *aliases, Handle name, AliasSource source,
                Handle lang = Handle::nil(), int count = 0) {
    if (name.IsNil()) return;
    Builder alias(aliases->store());
    alias.Add(n_name_, name);
    if (!lang.IsNil()) alias.Add(n_lang_, lang);
    if (count > 0) alias.Add(n_count_, count);
    alias.Add(n_sources_, 1 << source);
    aliases->Add(n_alias_, alias.Create());
  }

 private:
  // Wiki page types.
  WikimediaTypes wikitypes_;

  // Language for aliases.
  Handle language_;

  // Skip auxiliary items.
  bool skip_aux_ = false;
  AuxFilter filter_;
  task::Counter *num_aux_items_;

  // Symbols.
  Name n_lang_{names_, "lang"};
  Name n_name_{names_, "name"};
  Name n_alias_{names_, "alias"};
  Name n_count_{names_, "count"};
  Name n_sources_{names_, "sources"};

  Name n_native_name_{names_, "P1559"};
  Name n_native_label_{names_, "P1705"};
  Name n_demonym_{names_, "P1549"};
  Name n_short_name_{names_, "P1813"};
  Name n_nickname_{names_, "P1449"};
  Name n_pseudonym_{names_, "P742"};
  Name n_generic_name_{names_, "P2561"};
  Name n_official_name_{names_, "P1448"};
  Name n_birth_name_{names_, "P1477"};
  Name n_married_name_{names_, "P2562"};
  Name n_female_form_{names_, "P2521"};
  Name n_male_form_{names_, "P3321"};
  Name n_iso3166_country_code_2_{names_, "P297"};
  Name n_iso3166_country_code_3_{names_, "P298"};
  Name n_unit_symbol_{names_, "P5061"};

  Name n_instance_of_{names_, "P31"};
};

REGISTER_TASK_PROCESSOR("alias-extractor", AliasExtractor);

class AliasReducer : public task::Reducer {
 public:
  // Group of aliases with same fingerprint.
  struct Alias {
    // Most common variant.
    string name;

    // Map of alias variants with same fingerprint with frequency counts.
    std::unordered_map<string, int> variants;

    // Unicode representation for normalized version of most common variant.
    ustring chars;

    // Frequency counts for case forms.
    int forms[NUM_CASE_FORMS] = {};

    // Bitmask with alias sources.
    int sources = 0;

    // Total number of occurrences for alias.
    int count = 0;

    // Whether the alias has been selected for output.
    bool selected = false;

    // Compute size of common prefix between this alias and another alias.
    int CommonPrefix(const Alias *other) const {
      int n = 0;
      while (n < chars.size() &&
             n < other->chars.size() &&
             chars[n] == other->chars[n]) {
        n++;
      }
      return n;
    }

    // Compute edit distance between this alias and another alias.
    int EditDistance(const Alias *other) {
      return LevenshteinDistance(chars.data(), chars.size(),
                                 other->chars.data(), other->chars.size());
    }
  };

  void Start(task::Task *task) override {
    Reducer::Start(task);

    // Load commons store.
    for (task::Binding *binding : task->GetInputs("commons")) {
      LoadStore(binding->resource()->name(), &commons_);
    }
    names_.Bind(&commons_);

    // Get parameters.
    string lang = task->Get("language", "en");
    language_ = commons_.Lookup("/lang/" + lang);
    task->Fetch("anchor_threshold", &anchor_threshold_);
    task->Fetch("majority_form_fraction", &majority_form_fraction_);
    task->Fetch("min_prefix", &min_prefix_);
    task->Fetch("max_edit_distance", &max_edit_distance_);
    CHECK_GE(majority_form_fraction_, 0.5);

    // Read alias corrections.
    Frame aliases(&commons_, "/w/aliases");
    if (aliases.valid()) {
      // Get corrections for language.
      Frame corrections = aliases.GetFrame(language_);
      if (corrections.valid()) {
        // Make map of alias corrections for each item.
        for (const Slot &s : corrections) {
          item_corrections_[s.name] = s.value;
        }
      }
    }
    commons_.Freeze();
  }

  void Reduce(const task::ReduceInput &input) override {
    Text qid = input.key();
    Store store(&commons_);
    std::unordered_map<uint64, Alias *> aliases;

    // Get alias corrections for item.
    std::set<uint64> forbidden;
    auto f = item_corrections_.find(store.Lookup(qid));
    if (f != item_corrections_.end()) {
      Frame correction_list(&store, f->second);
      for (const Slot &s : correction_list) {
        // Get alias and source.
        string name = String(&store, s.name).value();
        Handle source = s.value;

        // Compute fingerprint and case form.
        uint64 fp;
        CaseForm form;
        tokenizer_.FingerprintAndForm(name, &fp, &form);
        if (form == CASE_INVALID) continue;

        if (source == n_forbid_) {
          // Forbid alias for item.
          forbidden.insert(fp);
        } else {
          // Add new alias for item.
          Alias *a = aliases[fp];
          if (a == nullptr) {
            a = new Alias();
            aliases[fp] = a;
          }
          a->sources |= (1 << source.AsInt());
          a->count += 1;
          a->variants[name] += 1;
          a->forms[form] += 1;
        }
      }
    }

    // Collect all the aliases for the item.
    for (task::Message *message : input.messages()) {
      // Get next set of aliases for item.
      Frame batch = DecodeMessage(&store, message);

      // Get all aliases for item.
      for (const Slot &slot : batch) {
        if (slot.name != n_alias_) continue;
        Frame alias(&store, slot.value);
        string name = alias.GetString(n_name_);
        int count = alias.GetInt(n_count_, 1);
        int sources = alias.GetInt(n_sources_);

        // Check that alias is valid UTF-8.
        if (!UTF8::Valid(name)) {
          VLOG(9) << "Skipping invalid alias for " << qid << ": " << name;
          continue;
        }

        // Compute fingerprint and case form.
        uint64 fp;
        CaseForm form;
        tokenizer_.FingerprintAndForm(name, &fp, &form);
        if (form == CASE_INVALID) continue;

        // Check if alias has been forbidden.
        if (forbidden.count(fp) > 0) continue;

        // Update alias table.
        Alias *a = aliases[fp];
        if (a == nullptr) {
          a = new Alias();
          aliases[fp] = a;
        }
        a->sources |= sources;
        a->count += count;
        a->variants[name] += count;
        a->forms[form] += count;
      }
    }

    // Find most common variant for each alias.
    for (auto it : aliases) {
      Alias *alias = it.second;

      // Find most common variant.
      int max_count = -1;
      for (auto &variant : alias->variants) {
        if (variant.second > max_count) {
          max_count = variant.second;
          alias->name = variant.first;
        }
      }

      // Normalize name of the most common variant for alias and convert it to
      // Unicode code points for computing edit distance.
      string normalized;
      UTF8::Normalize(alias->name, tokenizer_.normalization(), &normalized);
      UTF8::DecodeString(normalized, &alias->chars);
    }

    // Select aliases based on sources.
    for (auto it : aliases) {
      Alias *alias = it.second;
      if (SelectAlias(alias)) alias->selected = true;
    }

    // Select aliases that are variations over already selected aliases.
    if (max_edit_distance_ > 0) {
      for (auto it : aliases) {
        Alias *alias = it.second;
        if (alias->selected) continue;
        if (alias->sources == WIKIDATA_FOREIGN) continue;

        // Check if alias is a variation over an already selected alias.
        bool variation = false;
        for (auto ot : aliases) {
          Alias *a = ot.second;
          if (!a->selected) continue;
          if (a->sources & VARIATION) continue;

          // A variation must have a common prefix with existing alias of a
          // minimum size and the edit distance must be below a threshold.
          if (alias->CommonPrefix(a) < min_prefix_) continue;
          if (alias->EditDistance(a) > max_edit_distance_) continue;

          // Variation found.
          variation = true;
          break;
        }

        // Select variation.
        if (variation) {
          alias->selected = true;
          alias->sources |= VARIATION;
        }
      }
    }

    // Build new set of selected aliases.
    Builder merged(&store);
    for (auto it : aliases) {
      Alias *alias = it.second;
      if (!alias->selected) continue;
      if (alias->name.empty()) continue;

      // Find majority form.
      int form = CASE_NONE;
      for (int f = 0; f < NUM_CASE_FORMS; ++f) {
        if (alias->forms[f] >= alias->count * majority_form_fraction_) {
          form = f;
          break;
        }
      }
      if (form == CASE_INVALID) continue;

      // Add alias to output.
      Builder a(&store);
      a.Add(n_name_, alias->name);
      a.Add(n_lang_, language_);
      a.Add(n_count_, alias->count);
      a.Add(n_sources_, alias->sources);
      if (form != CASE_NONE) a.Add(n_form_, form);
      merged.Add(n_alias_, a.Create());
    }

    // Output selected aliases.
    Output(input.shard(), task::CreateMessage(qid, merged.Create()));

    // Delete alias table.
    for (auto it : aliases) delete it.second;
  }

  // Check if alias should be selected.
  bool SelectAlias(Alias *alias) {
    // Keep aliases from "trusted" sources.
    if (alias->sources & (WIKIDATA_LABEL |
                          WIKIPEDIA_TITLE |
                          WIKIPEDIA_REDIRECT |
                          WIKIPEDIA_NAME |
                          WIKIDATA_ALIAS |
                          WIKIDATA_NAME)) {
      return true;
    }

    // Keep foreign, native and demonym, and nickname aliases supported by
    // Wikipedia aliases.
    if (alias->sources & (WIKIDATA_FOREIGN |
                          WIKIDATA_NATIVE |
                          WIKIDATA_DEMONYM |
                          WIKIPEDIA_NICKNAME)) {
      if (alias->sources & (WIKIPEDIA_ANCHOR |
                            WIKIPEDIA_LINK |
                            WIKIPEDIA_DISAMBIGUATION)) {
        return true;
      }
    }

    // Disambiguation links need to be backed by anchors.
    if (alias->sources & WIKIPEDIA_DISAMBIGUATION) {
      if (alias->sources & (WIKIPEDIA_ANCHOR | WIKIPEDIA_LINK)) return true;
    }

    // Pure anchors need high counts to be selected.
    if (alias->sources & (WIKIPEDIA_ANCHOR | WIKIPEDIA_LINK)) {
      if (alias->count >= anchor_threshold_) return true;
    }

    return false;
  }

 private:
  // Alias source masks.
  enum AliasSourceMask {
    GENERIC = 1 << SRC_GENERIC,
    WIKIDATA_LABEL = 1 << SRC_WIKIDATA_LABEL,
    WIKIDATA_ALIAS = 1 << SRC_WIKIDATA_ALIAS,
    WIKIPEDIA_TITLE  = 1 << SRC_WIKIPEDIA_TITLE,
    WIKIPEDIA_REDIRECT = 1 << SRC_WIKIPEDIA_REDIRECT,
    WIKIPEDIA_ANCHOR = 1 << SRC_WIKIPEDIA_ANCHOR,
    WIKIPEDIA_DISAMBIGUATION = 1 << SRC_WIKIPEDIA_DISAMBIGUATION,
    WIKIDATA_FOREIGN = 1 << SRC_WIKIDATA_FOREIGN,
    WIKIDATA_NATIVE = 1 << SRC_WIKIDATA_NATIVE,
    WIKIDATA_DEMONYM = 1 << SRC_WIKIDATA_DEMONYM,
    WIKIPEDIA_LINK = 1 << SRC_WIKIPEDIA_LINK,
    WIKIDATA_NAME = 1 << SRC_WIKIDATA_NAME,
    WIKIPEDIA_NAME = 1 << SRC_WIKIPEDIA_NAME,
    WIKIPEDIA_NICKNAME = 1 << SRC_WIKIPEDIA_NICKNAME,
    VARIATION = 1 << SRC_VARIATION,
  };

  // Commons store.
  Store commons_;

  // Symbols.
  Names names_;
  Name n_name_{names_, "name"};
  Name n_lang_{names_, "lang"};
  Name n_alias_{names_, "alias"};
  Name n_count_{names_, "count"};
  Name n_sources_{names_, "sources"};
  Name n_form_{names_, "form"};
  Name n_forbid_{names_, "forbid"};

  // Language.
  Handle language_;

  // Phrase tokenizer for computing phrase fingerprints.
  nlp::PhraseTokenizer tokenizer_;

  // Threshold for pure anchors.
  int anchor_threshold_ = 100;

  // Fraction of aliases that must have a certain case form for this form to
  // be considered the majority form.
  float majority_form_fraction_ = 0.75;

  // Minimum common prefix size for alias variations.
  int min_prefix_ = 2;

  // Maximum edit distance for alias variations (0=disabled).
  int max_edit_distance_ = 0;

  // Mapping from item id to corrections for item.
  HandleMap<Handle> item_corrections_;
};

REGISTER_TASK_PROCESSOR("alias-reducer", AliasReducer);

}  // namespace nlp
}  // namespace sling

