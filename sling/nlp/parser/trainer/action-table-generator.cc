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

#include "sling/nlp/parser/trainer/action-table-generator.h"

#include <algorithm>
#include <vector>

#include "sling/base/macros.h"
#include "sling/file/file.h"
#include "sling/nlp/parser/parser-state.h"
#include "sling/frame/object.h"
#include "sling/frame/serialization.h"
#include "sling/string/strcat.h"

namespace sling {
namespace nlp {

ActionTableGenerator::ActionTableGenerator(Store *global) {
  global_ = global;
  global_->Freeze();
  generator_.Init(global_);
}

void ActionTableGenerator::Add(const Document &document) {
  ActionTable table;
  GetUnknownSymbols(document);
  if (!per_sentence_) {
    Process(document, 0, document.num_tokens());
  } else {
    for (SentenceIterator s(&document); s.more(); s.next()) {
      Process(document, s.begin(), s.end() + 1);
    }
  }
}

void ActionTableGenerator::Save(const string &table_file,
                                const string &summary_file,
                                const string &unknown_symbols_file) const {
  CHECK(!table_file.empty());
  table_.Save(global_, coverage_percentile_, table_file);

  if (!summary_file.empty()) table_.OutputSummary(summary_file);
  if (!unknown_symbols_file.empty()) {
    OutputUnknownSymbols(unknown_symbols_file);
  }
}

void ActionTableGenerator::GetUnknownSymbols(const Document &document) {
  Store *store = document.store();
  Handle h_evokes = store->Lookup("/s/phrase/evokes");
  CHECK(!h_evokes.IsNil());
  for (int i = 0; i < document.num_spans(); ++i) {
    Span *span = document.span(i);
    for (const Slot &slot : span->mention()) {
      if (slot.name == h_evokes) {
        Frame f(store, slot.value);
        for (const Slot &slot2 : f) {
          Frame name = Object(store, slot2.name).AsFrame();
          Frame value = Object(store, slot2.value).AsFrame();
          if (name.valid() && name.IsProxy()) {
            unknown_[name.Id().str()]++;
          }
          if (slot2.name.IsIsA() && value.valid() && value.IsProxy()) {
            unknown_[value.Id().str()]++;
          }
        }
      }
    }
  }
}

void ActionTableGenerator::OutputUnknownSymbols(const string &filename) const {
  // Sort unknown symbols in descending order of count.
  std::vector<std::pair<int, string>> unknown;
  for (const auto &kv : unknown_) unknown.emplace_back(kv.second, kv.first);
  std::sort(unknown.rbegin(), unknown.rend());

  // Dump to a text file.
  string store;
  for (const auto &u : unknown) {
    StrAppend(&store, "; Count = ", u.first, "\n{= ", u.second, "}\n\n");
  }
  CHECK(File::WriteContents(filename, store));
}

void ActionTableGenerator::Process(
    const Document &document, int start, int end) {
  TransitionSequence gold_sequence;
  TransitionGenerator::Report report;
  generator_.Generate(
      document, start, end, &gold_sequence, &report);

  Store store(global_);
  ParserState state(&store, start, end);
  int actions = 0;
  int shift_action = 0;
  for (const ParserAction &action : gold_sequence.actions()) {
    table_.Add(action);
    if (!state.CanApply(action)) {
      LOG(FATAL) << "Can't apply gold action: "
          << action.ToString(document.store()) << " at state:\n "
          << state.DebugString();
    }
    state.Apply(action);
    if (action.type == ParserAction::SHIFT) {
      table_.set_max_actions_per_token(actions - shift_action);
      shift_action = actions;
    } else {
      actions++;
    }
  }
}

}  // namespace nlp
}  // namespace sling
