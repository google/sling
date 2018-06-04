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

#include "sling/nlp/parser/action-table.h"

#include "sling/base/logging.h"
#include "sling/base/types.h"
#include "sling/file/file.h"
#include "sling/frame/serialization.h"
#include "sling/string/numbers.h"
#include "sling/string/strcat.h"
#include "sling/string/text.h"

namespace sling {
namespace nlp {

void ActionTable::Add(const ParserAction &action) {
  // Add the action to the index if it is new.
  int index = Index(action);
  if (index == -1) {
    actions_.emplace_back(action);
    index = actions_.size() - 1;
    index_[action] = {index, 1};
    if (action.type == ParserAction::SHIFT) shift_index_ = index;
    if (action.type == ParserAction::STOP) stop_index_ = index;
  } else {
    index_[action].second++;
  }

  uint8 source = action.source;
  uint8 target = action.target;
  switch (action.type) {
    case ParserAction::EVOKE:
      span_length_.Add(action.length);
      break;
    case ParserAction::REFER: {
      span_length_.Add(action.length);
      break;
    }
    case ParserAction::ASSIGN:
      assign_source_.Add(source);
      overall_index_.Add(source);
      break;
    case ParserAction::CONNECT: {
      connect_source_.Add(source);
      connect_target_.Add(target);
      overall_index_.Add(source);
      overall_index_.Add(target);
      break;
    }
    case ParserAction::EMBED: {
      embed_target_.Add(target);
      overall_index_.Add(target);
      break;
    }
    case ParserAction::ELABORATE: {
      elaborate_source_.Add(source);
      overall_index_.Add(source);
      break;
    }
    case ParserAction::SHIFT:
    case ParserAction::STOP:
    default:
      break;
  }
}

void ActionTable::Allowed(
    const ParserState &state, std::vector<bool> *allowed) const {
  // See if SHIFT/STOP are allowed.
  if (state.done() || (state.current() == state.end())) {
    (*allowed)[StopIndex()] = true;
    return;
  }

  // If we are not at the end, we can always SHIFT.
  (*allowed)[ShiftIndex()] = true;

  // Allow all actions that the parser state permits,
  // except the ones beyond the index bounds.
  for (int i = 0; i < actions_.size(); ++i) {
    if (beyond_bounds_[i]) continue;
    const auto &action = actions_[i];
    if (state.CanApply(action)) {
      (*allowed)[i] = true;
    }
  }
}

// Returns the handle to 'symbol', which should already exist.
static Handle GetSymbol(const Store &store, Text symbol) {
  Handle h = store.LookupExisting(symbol);
  CHECK(!h.IsNil()) << symbol;
  return h;
}

// Looks up or creates handle for 'symbol'.
static Handle GetSymbol(Store *store, Text symbol) {
  Handle h = store->Lookup(symbol);
  CHECK(!h.IsNil()) << symbol;
  return h;
}

void ActionTable::Init(Store *store) {
  Frame top(store, "/table");
  CHECK(top.valid());

  // Get all the integer fields.
  max_refer_target_ = top.GetInt("/table/max_refer_target");
  max_embed_target_ = top.GetInt("/table/max_embed_target");
  max_elaborate_source_ = top.GetInt("/table/max_elaborate_source");
  max_connect_source_ = top.GetInt("/table/max_connect_source");
  max_connect_target_ = top.GetInt("/table/max_connect_target");
  max_assign_source_ = top.GetInt("/table/max_assign_source");
  max_span_length_ = top.GetInt("/table/max_span_length");
  max_actions_per_token_ = top.GetInt("/table/max_actions_per_token");
  frame_limit_ = top.GetInt("/table/frame_limit");

  // Compute the overall max index.
  if (max_index_ < max_refer_target_) max_index_ = max_refer_target_;
  if (max_index_ < max_embed_target_) max_index_ = max_embed_target_;
  if (max_index_ < max_elaborate_source_) max_index_ = max_elaborate_source_;
  if (max_index_ < max_connect_source_) max_index_ = max_connect_source_;
  if (max_index_ < max_connect_target_) max_index_ = max_connect_target_;
  if (max_index_ < max_assign_source_) max_index_ = max_assign_source_;

  // Read the action index.
  Array actions = top.Get("/table/actions").AsArray();
  CHECK(actions.valid());

  Handle action_type = GetSymbol(*store, "/table/action/type");
  Handle action_length = GetSymbol(*store, "/table/action/length");
  Handle action_source = GetSymbol(*store, "/table/action/source");
  Handle action_target = GetSymbol(*store, "/table/action/target");
  Handle action_role = GetSymbol(*store, "/table/action/role");
  Handle action_label = GetSymbol(*store, "/table/action/label");
  for (int i = 0; i < actions.length(); ++i) {
    ParserAction action;
    Frame item(store, actions.get(i));
    CHECK(item.valid());

    for (const Slot &slot : item) {
      if (slot.name == action_type) {
        action.type = static_cast<ParserAction::Type>(slot.value.AsInt());
      } else if (slot.name == action_length) {
        action.length = slot.value.AsInt();
      } else if (slot.name == action_source) {
        action.source = slot.value.AsInt();
      } else if (slot.name == action_target) {
        action.target = slot.value.AsInt();
      } else if (slot.name == action_role) {
        action.role = slot.value;
      } else if (slot.name == action_label) {
        action.label = slot.value;
      }
    }

    if (action.type == ParserAction::EVOKE ||
        action.type == ParserAction::REFER) {
      if (action.length == 0) action.length = 1;
    }

    actions_.emplace_back(action);
    index_[action] = {actions_.size() - 1, 0 /* raw count; unused */};

    // Get the indices of SHIFT and STOP actions.
    if (action.type == ParserAction::SHIFT) shift_index_ = actions_.size() - 1;
    if (action.type == ParserAction::STOP) stop_index_ = actions_.size() - 1;
  }

  beyond_bounds_.resize(actions_.size());
  for (int i = 0; i < actions_.size(); ++i) {
    const ParserAction &action = actions_[i];
    bool beyond_bounds = false;
    switch (action.type) {
      case ParserAction::EVOKE:
        beyond_bounds = action.length > max_span_length_;
        break;
      case ParserAction::REFER:
        beyond_bounds = action.length > max_span_length_;
        break;
      case ParserAction::CONNECT:
        beyond_bounds = (action.source > max_connect_source_) ||
            (action.target > max_connect_target_);
        break;
      case ParserAction::ASSIGN:
        beyond_bounds = action.source > max_assign_source_;
        break;
      case ParserAction::EMBED:
        beyond_bounds = action.target > max_embed_target_;
        break;
      case ParserAction::ELABORATE:
        beyond_bounds = action.source > max_elaborate_source_;
        break;
      default:
        break;
    }
    beyond_bounds_[i] = beyond_bounds;
  }
}

void ActionTable::Save(const Store *global,
                       int percentile,
                       const string &file) const {
  string s = Serialize(global, percentile);
  CHECK(File::WriteContents(file, s));
}

string ActionTable::Serialize(const Store *global, int percentile) const {
  Store store(global);
  Builder top(&store);
  top.AddId("/table");

  top.Add("/table/max_embed_target", embed_target_.PercentileBin(percentile));
  top.Add("/table/max_refer_target", refer_target_.PercentileBin(percentile));
  top.Add("/table/max_elaborate_source",
          elaborate_source_.PercentileBin(percentile));
  top.Add("/table/max_connect_source",
          connect_source_.PercentileBin(percentile));
  top.Add("/table/max_connect_target",
          connect_target_.PercentileBin(percentile));
  top.Add("/table/max_assign_source", assign_source_.PercentileBin(percentile));
  top.Add("/table/max_span_length", span_length_.PercentileBin(percentile));
  top.Add("/table/max_actions_per_token", max_actions_per_token_);
  top.Add("/table/frame_limit", frame_limit_);

  // Save the actions index.
  Handle action_type = GetSymbol(&store, "/table/action/type");
  Handle action_length = GetSymbol(&store, "/table/action/length");
  Handle action_source = GetSymbol(&store, "/table/action/source");
  Handle action_target = GetSymbol(&store, "/table/action/target");
  Handle action_role = GetSymbol(&store, "/table/action/role");
  Handle action_label = GetSymbol(&store, "/table/action/label");

  Array actions(&store, actions_.size());
  int index = 0;
  for (const ParserAction &action : actions_) {
    auto type = action.type;
    Builder b(&store);
    b.Add(action_type, static_cast<int>(type));

    if (type == ParserAction::REFER || type == ParserAction::EVOKE) {
      if (action.length > 1) {
        b.Add(action_length, static_cast<int>(action.length));
      }
    }
    if (type == ParserAction::ASSIGN ||
        type == ParserAction::ELABORATE ||
        type == ParserAction::CONNECT) {
      if (action.source != 0) {
        b.Add(action_source, static_cast<int>(action.source));
      }
    }
    if (type == ParserAction::EMBED ||
        type == ParserAction::REFER ||
        type == ParserAction::CONNECT) {
      if (action.target != 0) {
        b.Add(action_target, static_cast<int>(action.target));
      }
    }
    if (action.role.raw() != 0) b.Add(action_role, action.role);
    if (action.label.raw() != 0) b.Add(action_label, action.label);
    actions.set(index++, b.Create().handle());
  }
  top.Add("/table/actions", actions);

  // Add artificial links to symbols used in serialization. This is needed as
  // some action types might be unseen, so their corresponding symbols won't be
  // serialized. However we still want handles to them during Load().
  // For example, if we have only seen EVOKE, SHIFT, and STOP actions, then
  // the symbol /table/fp/refer for REFER won't be serialized unless the table
  // links to it.
  std::vector<Handle> symbols = {
    action_type, action_length, action_source, action_target,
    action_role, action_label
  };
  Array symbols_array(&store, symbols);
  top.Add("/table/symbols", symbols_array);

  StringEncoder encoder(&store);
  encoder.Encode(top.Create());
  return encoder.buffer();
}

void ActionTable::OutputSummary(const string &file) const {
  TableWriter writer;
  OutputSummary(&writer);
  writer.Write(file);
}

void ActionTable::OutputSummary(TableWriter *writer) const {
  writer->StartTable("Actions Summary");
  writer->SetColumns({"Action Type", "Unique Arg Combinations", "Raw Count"});

  // Action type -> unique, raw count.
  std::unordered_map<string, std::pair<int64, int64>> counts;
  static const string kOverall = "OVERALL";
  for (const ParserAction &action : actions_) {
    auto &c = counts[action.TypeName()];
    c.first++;
    counts[kOverall].first++;
    const auto &it = index_.find(action);
    c.second += it->second.second;
    counts[kOverall].second += it->second.second;
  }
  writer->AddNamedRow(kOverall);
  for (const auto &kv : counts) {
    if (kv.first != kOverall) writer->AddNamedRow(kv.first);
    writer->SetCell(kv.first, 0, kv.first);
    writer->SetCell(kv.first, 1, kv.second.first);
    writer->SetCell(kv.first, 2, kv.second.second);
  }

  // Display histograms.
  overall_index_.ToTable(writer);
  span_length_.ToTable(writer);
  refer_target_.ToTable(writer);
  embed_target_.ToTable(writer);
  elaborate_source_.ToTable(writer);
  connect_source_.ToTable(writer);
  connect_target_.ToTable(writer);
  assign_source_.ToTable(writer);

  // Other statistics.
  writer->StartTable("Other Action Statistics");
  writer->SetColumns({"Metric", "Value"});
  writer->AddRow("Maximum actions per token", max_actions_per_token_);
}

void ActionTable::Histogram::Add(int bin, int count) {
  if (counts_.size() < bin + 1) counts_.resize(bin + 1);
  counts_[bin] += count;
  total_ += count;
}

int ActionTable::Histogram::PercentileBin(int p) const {
  int64 cumulative =  0;
  int64 required = p * total_;
  for (int i = 0; i < counts_.size(); ++i) {
    cumulative += counts_[i] * 100;
    if (cumulative >= required) return i;
  }

  return counts_.size() - 1;
}

void ActionTable::Histogram::ToTable(TableWriter *writer) const {
  if (counts_.empty()) return;

  writer->StartTable(xaxis_);
  writer->SetColumns(
      {"Bin", "Count", "Cumulative Count", "Percentile (rounded down)"});

  int64 cumulative = 0;
  std::vector<int> special_percentiles = {99, 98, 95, 90};
  for (int i = 0; i < counts_.size(); ++i) {
    if (counts_[i] > 0) {
      cumulative += counts_[i];
      string row_name = StrCat(i);
      writer->AddNamedRow(row_name);
      writer->SetCell(row_name, 0, row_name);
      writer->SetCell(row_name, 1, counts_[i]);
      writer->SetCell(row_name, 2, cumulative);

      int percentile = (cumulative * 100) / total_;
      writer->SetCell(row_name, 3, percentile);

      // Display the special percentile rows in a special color.
      for (int j = 0; j < special_percentiles.size(); ++j) {
        if (special_percentiles[j] <= percentile) {
          writer->Annotate(row_name, 3, "--> ");
          special_percentiles.resize(j);
          break;
        }
      }
    }
  }
}

}  // namespace nlp
}  // namespace sling

