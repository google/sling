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
#include "sling/nlp/document/phrase-tokenizer.h"
#include "sling/nlp/wiki/wiki.h"
#include "sling/task/frames.h"
#include "sling/task/task.h"
#include "sling/task/reducer.h"
#include "sling/util/unicode.h"

namespace sling {
namespace nlp {

// Extract aliases from profiles.
class ProfileAliasExtractor : public task::FrameProcessor {
 public:
  void Startup(task::Task *task) override {
    string lang = task->Get("language", "en");
    language_ = commons_->Lookup("/lang/" + lang);
    skip_aux_ = task->Get("skip_aux", false);

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
      } else if (s.name == n_demonym_) {
        // Output demonyms as demonym aliases.
        AddAlias(&a, store->Resolve(s.value), SRC_WIKIDATA_DEMONYM);
      } else if (s.name == n_instance_of_) {
        // Discard categories, disambiguations, info boxes and templates.
        if (s.value == n_category_ ||
            s.value == n_disambiguation_ ||
            s.value == n_infobox_ ||
            s.value == n_template_ ||
            s.value == n_templates_category_) {
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
  // Symbols.
  Name n_lang_{names_, "lang"};
  Name n_name_{names_, "name"};
  Name n_alias_{names_, "alias"};
  Name n_count_{names_, "count"};
  Name n_sources_{names_, "sources"};

  Name n_native_name_{names_, "P1559"};
  Name n_native_label_{names_, "P1705"};
  Name n_demonym_{names_, "P1549"};

  Name n_instance_of_{names_, "P31"};
  Name n_category_{names_, "Q4167836"};
  Name n_disambiguation_{names_, "Q4167410"};
  Name n_template_{names_, "Q11266439"};
  Name n_infobox_{names_, "Q19887878"};
  Name n_templates_category_{names_, "Q23894233"};

  // Language for aliases.
  Handle language_;

  // Skip auxiliary items.
  bool skip_aux_ = false;
  AuxFilter filter_;
  task::Counter *num_aux_items_;
};

REGISTER_TASK_PROCESSOR("profile-alias-extractor", ProfileAliasExtractor);

class ProfileAliasReducer : public task::Reducer {
 public:
  struct Alias {
    std::unordered_map<string, int> variants;
    std::unordered_map<int, int> forms;
    int sources = 0;
    int count = 0;
  };

  void Start(task::Task *task) override {
    Reducer::Start(task);

    // Get parameters.
    string lang = task->Get("language", "en");
    language_ = commons_.Lookup("/lang/" + lang);
    names_.Bind(&commons_);
    commons_.Freeze();
    task->Fetch("anchor_threshold", &anchor_threshold_);
    task->Fetch("majority_form_fraction", &majority_form_fraction_);
    CHECK_GE(majority_form_fraction_, 0.5);

    // Read toxic aliases.
    TextMapInput aliases(task->GetInputFiles("toxic-aliases"));
    string alias;
    while (aliases.Read(nullptr, &alias, nullptr)) {
      uint64 fp = tokenizer_.Fingerprint(alias);
      toxic_aliases_.insert(fp);
    }
  }

  void Reduce(const task::ReduceInput &input) override {
    // Collect all the aliases for the item.
    Text qid = input.key();
    Store store(&commons_);
    std::unordered_map<uint64, Alias *> aliases;
    for (task::Message *message : input.messages()) {
      // Get next alias profile.
      Frame profile = DecodeMessage(&store, message);

      // Get all aliases from profile.
      for (const Slot &slot : profile) {
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

        // Update alias table.
        Alias *a = aliases[fp];
        if (a == nullptr) {
          a = new Alias;
          aliases[fp] = a;
        }
        a->sources |= sources;
        a->count += count;
        a->variants[name] += count;
        a->forms[form] += count;
      }
    }

    // Select aliases.
    Builder merged(&store);
    for (auto it : aliases) {
      bool toxic = toxic_aliases_.count(it.first) != 0;
      Alias *alias = it.second;
      if (!SelectAlias(alias, toxic)) continue;

      // Find most common variant.
      int max_count = -1;
      string name;
      for (auto &variant : alias->variants) {
        if (variant.second > max_count) {
          max_count = variant.second;
          name = variant.first;
        }
      }
      if (name.empty()) continue;

      // Find majority form.
      int form = CASE_NONE;
      for (auto &f : alias->forms) {
        if (f.second >= alias->count * majority_form_fraction_) {
          form = f.first;
          break;
        }
      }
      if (form == CASE_INVALID) continue;

      // Add alias to output.
      Builder a(&store);
      a.Add(n_name_, name);
      a.Add(n_lang_, language_);
      a.Add(n_count_, alias->count);
      a.Add(n_sources_, alias->sources);
      if (form != CASE_NONE) a.Add(n_form_, form);
      merged.Add(n_alias_, a.Create());
    }

    // Output alias profile.
    Output(input.shard(), task::CreateMessage(qid, merged.Create()));

    // Delete alias table.
    for (auto it : aliases) delete it.second;
  }

  // Check if alias should be selected.
  bool SelectAlias(Alias *alias, bool toxic) {
    // Keep aliases from trusted sources.
    if (alias->sources & (WIKIDATA_LABEL |
                          WIKIPEDIA_TITLE |
                          WIKIPEDIA_REDIRECT)) {
      return true;
    }

    // Only keep Wikidata alias if it is not toxic.
    if ((alias->sources & WIKIDATA_ALIAS) && !toxic) return true;

    // Keep foreign, native and demonym aliases supported by Wikipedia aliases.
    if (alias->sources & (WIKIDATA_FOREIGN |
                          WIKIDATA_NATIVE |
                          WIKIDATA_DEMONYM)) {
      if (alias->sources & (WIKIPEDIA_ANCHOR | WIKIPEDIA_DISAMBIGUATION)) {
        return true;
      }
    }

    // Disambiguation links need to be backed by anchors.
    if (alias->sources & WIKIPEDIA_DISAMBIGUATION) {
      if (alias->sources & WIKIPEDIA_ANCHOR) return true;
    }

    // Pure anchors need high counts to be selected.
    if (alias->sources & WIKIPEDIA_ANCHOR) {
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

  // Language.
  Handle language_;

  // Phrase tokenizer for computing phrase fingerprints.
  nlp::PhraseTokenizer tokenizer_;

  // Threshold for pure anchors.
  int anchor_threshold_ = 100;

  // Fraction of aliases that must have a certain case form for this form to
  // be considered the majority form.
  float majority_form_fraction_ = 0.75;

  // Fingerprint for toxic aliases.
  std::set<uint64> toxic_aliases_;
};

REGISTER_TASK_PROCESSOR("profile-alias-reducer", ProfileAliasReducer);

}  // namespace nlp
}  // namespace sling

