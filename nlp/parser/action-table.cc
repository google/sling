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

#include "nlp/parser/action-table.h"

#include "base/logging.h"
#include "base/types.h"
#include "file/file.h"
#include "frame/serialization.h"
#include "string/numbers.h"
#include "string/strcat.h"
#include "string/text.h"

namespace sling {
namespace nlp {

void ActionTable::Add(const ParserState &state,
                      const ParserAction &action,
                      uint64 fingerprint) {
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
      fingerprint_[fingerprint].evoke.emplace(index);
      span_length_.Add(action.length);
      break;
    case ParserAction::REFER: {
      ParserAction refer = action;
      refer.label = state.type(target);
      fingerprint_[fingerprint].refer.emplace(refer);
      span_length_.Add(action.length);
      break;
    }
    case ParserAction::ASSIGN:
      assign_source_.Add(source);
      overall_index_.Add(source);
      type_[state.type(source)].assign.emplace(action.role, action.label);
      break;
    case ParserAction::CONNECT: {
      connect_source_.Add(source);
      connect_target_.Add(target);
      overall_index_.Add(source);
      overall_index_.Add(target);
      Handle source_type = state.type(source);
      type_[source_type].connect[state.type(target)].insert(action.role);
      break;
    }
    case ParserAction::EMBED: {
      Handle target_type = state.type(target);
      embed_target_.Add(target);
      overall_index_.Add(target);
      type_[target_type].embed.emplace(action.label, action.role);
      break;
    }
    case ParserAction::ELABORATE: {
      elaborate_source_.Add(source);
      overall_index_.Add(source);
      Handle source_type = state.type(source);
      type_[source_type].elaborate.emplace(action.label, action.role);
      break;
    }
    case ParserAction::SHIFT:
    case ParserAction::STOP:
    default:
      break;
  }
}

void ActionTable::Allowed(
    const ParserState &state,
    const std::vector<uint64> &fingerprints,
    std::vector<bool> *allowed) const {
  // See if SHIFT/STOP are allowed.
  if (state.done() || (state.current() == state.end())) {
    (*allowed)[StopIndex()] = true;
    return;
  }

  // If we are not at the end, we can always SHIFT.
  (*allowed)[ShiftIndex()] = true;

  // If no additional checks are required then just allow all actions that the
  // parser state permits, except the ones beyond the index bounds.
  if (!action_checks()) {
    for (int i = 0; i < actions_.size(); ++i) {
      if (beyond_bounds_[i]) continue;
      const auto &action = actions_[i];
      if (state.CanApply(action)) {
        (*allowed)[i] = true;
      }
    }
    return;
  }

  // Output any EVOKE/REFER actions.
  ParserAction refer;
  refer.type = ParserAction::REFER;

  int attention_size = state.AttentionSize();
  int max_length = state.MaxEvokeLength(max_span_length_);

  for (int f = 0; f < max_length && f < fingerprints.size(); ++f) {
    uint64 fp = fingerprints[f];
    const auto &it = fingerprint_.find(fp);
    if (it == fingerprint_.end()) continue;

    // Allow seen EVOKE actions for the fingerprint.
    for (int action_index : it->second.evoke) {
      if (state.CanApply(Action(action_index))) (*allowed)[action_index] = true;
    }

    // For every REFER action for that fingerprint, do an additional type check.
    if (attention_size == 0) continue;  // no frames, so can't refer back
    for (const ParserAction &action : it->second.refer) {
      refer.length = action.length;
      for (int i = 0; i <= max_refer_target_ && i < attention_size; ++i) {
        if (state.type(i) != action.label) continue;
        refer.target = i;
        int index = Index(refer);
        if (index != -1 && state.CanApply(refer)) {
          (*allowed)[index] = true;
        }
      }
    }
  }

  // Go over the rest of the actions.
  for (int i = 0; i <= max_index_ && i < attention_size; ++i) {
    const auto &it = type_.find(state.type(i));
    if (it == type_.end()) continue;
    const TypeConstraint &type_constraints = it->second;

    // Output allowed ASSIGN actions.
    if (i <= max_assign_source_) {
      ParserAction action;
      action.type = ParserAction::ASSIGN;
      action.source = i;
      for (const auto &role_value : type_constraints.assign) {
        action.role = role_value.first;
        action.label = role_value.second;
        int index = Index(action);
        if (index != -1 && state.CanApply(action)) (*allowed)[index] = true;
      }
    }

    // Output allowed CONNECT actions.
    if (i <= max_connect_source_) {
      ParserAction action;
      action.type = ParserAction::CONNECT;
      action.source = i;
      for (int j = 0; j <= max_connect_target_ && j < attention_size; ++j) {
        const auto &it2 = type_constraints.connect.find(state.type(j));
        if (it2 != type_constraints.connect.end()) {
          action.target = j;
          for (Handle role : it2->second) {
            action.role = role;
            int index = Index(action);
            if (index != -1 && state.CanApply(action)) (*allowed)[index] = true;
          }
        }
      }
    }

    // Output allowed EMBED actions.
    if (i <= max_embed_target_) {
      ParserAction action;
      action.type = ParserAction::EMBED;
      action.target = i;
      for (const auto &source_type_role : type_constraints.embed) {
        action.label = source_type_role.first;
        action.role = source_type_role.second;
        int index = Index(action);
        if (index != -1 && state.CanApply(action)) (*allowed)[index] = true;
      }
    }

    // Output allowed ELABORATE actions.
    if (i <= max_elaborate_source_) {
      ParserAction action;
      action.type = ParserAction::ELABORATE;
      action.source = i;
      for (const auto &target_type_role : type_constraints.elaborate) {
        action.label = target_type_role.first;
        action.role = target_type_role.second;
        int index = Index(action);
        if (index != -1 && state.CanApply(action)) (*allowed)[index] = true;
      }
    }
  }
}

void ActionTable::Load(const Frame &frame,
                       Handle slot,
                       Handle subslot1,
                       Handle subslot2,
                       ActionTable::HandlePairSet *pairs) {
  Array array = frame.Get(slot).AsArray();
  if (!array.valid()) return;

  for (int i = 0; i < array.length(); ++i) {
    Frame f(frame.store(), array.get(i));
    CHECK(f.valid());
    Handle value1, value2;
    for (const Slot &s : f) {
      if (s.name == subslot1) value1 = s.value;
      if (s.name == subslot2) value2 = s.value;
    }
    CHECK(!value1.IsNil());
    CHECK(!value2.IsNil());
    pairs->emplace(value1, value2);
  }
}

void ActionTable::Save(const HandlePairSet &pairs,
                       Handle slot,
                       Handle subslot1,
                       Handle subslot2,
                       Builder *builder) const {
  if (pairs.empty()) return;

  Array array(builder->store(), pairs.size());
  int i = 0;
  for (const auto &p : pairs) {
    Builder b(builder->store());
    b.Add(subslot1, p.first);
    b.Add(subslot2, p.second);
    array.set(i++, b.Create().handle());
  }
  builder->Add(slot, array);
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

  // Read fingerprint-based constraints.
  Array fingerprint_constraints(store, top.GetHandle("/table/fp/constraints"));
  CHECK(fingerprint_constraints.valid());

  Handle fingerprint = GetSymbol(*store, "/table/fp");
  Handle fingerprint_evoke = GetSymbol(*store, "/table/fp/evoke");
  Handle fingerprint_refer = GetSymbol(*store, "/table/fp/refer");
  for (int i = 0; i < fingerprint_constraints.length(); ++i) {
    Frame constraint(store, fingerprint_constraints.get(i));
    CHECK(constraint.valid());

    // uint64 fingerprints are stored as strings.
    string s = constraint.GetString(fingerprint);
    CHECK(!s.empty());
    uint64 fp;
    CHECK(safe_strtou64(s, &fp)) << s;
    ActionTable::FingerprintConstraint &fp_constraint = fingerprint_[fp];

    Array evoke = constraint.Get(fingerprint_evoke).AsArray();
    if (evoke.valid()) {
      for (int j = 0; j < evoke.length(); ++j) {
        fp_constraint.evoke.emplace(evoke.get(j).AsInt());
      }
    }

    Array refer = constraint.Get(fingerprint_refer).AsArray();
    if (refer.valid()) {
      for (int j = 0; j < refer.length(); ++j) {
        ParserAction action;
        action.type = ParserAction::REFER;
        Frame refer_frame(store, refer.get(i));
        CHECK(refer_frame.valid());
        action.length = refer_frame.GetInt(action_length);
        if (action.length == 0) action.length = 1;
        action.label = refer_frame.GetHandle(action_label);
        fp_constraint.refer.emplace(action);
      }
    }
  }

  // Read type-based constraints.
  Handle type_role = GetSymbol(*store, "/table/type/role");
  Handle type_value = GetSymbol(*store, "/table/type/value");
  Handle type_source = GetSymbol(*store, "/table/type/source");
  Handle type_target = GetSymbol(*store, "/table/type/target");
  Handle type_assign = GetSymbol(*store, "/table/type/assign");
  Handle type_embed = GetSymbol(*store, "/table/type/embed");
  Handle type_elaborate = GetSymbol(*store, "/table/type/elaborate");
  Handle type_connect = GetSymbol(*store, "/table/type/connect");
  Array type_constraints(store, top.GetHandle("/table/type/constraints"));
  if (type_constraints.valid()) {
    for (int i = 0; i < type_constraints.length(); ++i) {
      Frame constraint(store, type_constraints.get(i));
      CHECK(constraint.valid());
      Handle type = constraint.GetHandle("/table/type");
      CHECK(!type.IsNil());

      ActionTable::TypeConstraint &tc = type_[type];
      Load(constraint, type_assign, type_role, type_value, &tc.assign);
      Load(constraint, type_embed, type_source, type_role, &tc.embed);
      Load(constraint, type_elaborate, type_target, type_role, &tc.elaborate);

      // Load CONNECT constraints.
      Array connect = constraint.Get(type_connect).AsArray();
      if (connect.valid()) {
        for (int i = 0; i < connect.length(); ++i) {
          Frame f(store, connect.get(i));
          CHECK(f.valid());
          Handle target = f.GetHandle(type_target);
          Array roles = f.Get(type_role).AsArray();
          CHECK(!target.IsNil());
          CHECK(roles.valid());
          for (int j = 0; j < roles.length(); ++j) {
            tc.connect[target].insert(roles.get(j));
          }
        }
      }
    }
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

  // Save fingerprint-based constraints.
  Handle fingerprint = GetSymbol(&store, "/table/fp");
  Handle fingerprint_evoke = GetSymbol(&store, "/table/fp/evoke");
  Handle fingerprint_refer = GetSymbol(&store, "/table/fp/refer");
  if (!fingerprint_.empty()) {
    Array fingerprint_constraints(&store, fingerprint_.size());
    index = 0;
    for (const auto &kv : fingerprint_) {
      Builder b(&store);
      uint64 fp = kv.first;
      b.Add(fingerprint, StrCat(fp));  // 64-bit numbers can't be saved as ints

      const FingerprintConstraint &constraint = kv.second;

      if (!constraint.evoke.empty()) {
        Array evoke(&store, constraint.evoke.size());
        int i = 0;
        for (int action_index : constraint.evoke) {
          evoke.set(i++, Handle::Integer(action_index));
        }
        b.Add(fingerprint_evoke, evoke);
      }

      if (!constraint.refer.empty()) {
        Array refer(&store, constraint.refer.size());
        int i = 0;
        for (const ParserAction &action : constraint.refer) {
          Builder b2(&store);
          if (action.length > 1) {
            b2.Add(action_length, static_cast<int>(action.length));
          }
          b2.Add(action_label, action.label);
          refer.set(i++, b2.Create().handle());
        }
        b.Add(fingerprint_refer, refer);
      }
      fingerprint_constraints.set(index++, b.Create().handle());
    }
    top.Add("/table/fp/constraints", fingerprint_constraints);
  }

  // Save type-based constraints.
  Handle type_role = GetSymbol(&store, "/table/type/role");
  Handle type_value = GetSymbol(&store, "/table/type/value");
  Handle type_source = GetSymbol(&store, "/table/type/source");
  Handle type_target = GetSymbol(&store, "/table/type/target");
  Handle type_assign = GetSymbol(&store, "/table/type/assign");
  Handle type_embed = GetSymbol(&store, "/table/type/embed");
  Handle type_elaborate = GetSymbol(&store, "/table/type/elaborate");
  Handle type_connect = GetSymbol(&store, "/table/type/connect");
  if (!type_.empty()) {
    Array type_constraints(&store, type_.size());
    index = 0;
    for (const auto &kv : type_) {
      Handle type = kv.first;
      Builder b(&store);
      b.Add("/table/type", type);

      const ActionTable::TypeConstraint &constraint = kv.second;
      Save(constraint.assign, type_assign, type_role, type_value, &b);
      Save(constraint.embed, type_embed, type_source, type_role, &b);
      Save(constraint.elaborate, type_elaborate, type_target, type_role, &b);

      if (constraint.connect.size() > 0) {
        Array array(&store, constraint.connect.size());
        int i = 0;
        for (const auto &connect_kv : constraint.connect) {
          Builder b2(&store);
          b2.Add(type_target, connect_kv.first);

          Array roles(&store, connect_kv.second.size());
          int j = 0;
          for (Handle role : connect_kv.second) roles.set(j++, role);
          b2.Add(type_role, roles);
          array.set(i++, b2.Create().handle());
        }
        b.Add(type_connect, array);
      }
      type_constraints.set(index++, b.Create().handle());
    }
    top.Add("/table/type/constraints", type_constraints);
  }

  // Add artificial links to symbols used in serialization. This is needed as
  // some action types might be unseen, so their corresponding symbols won't be
  // serialized. However we still want handles to them during Load().
  // For example, if we have only seen EVOKE, SHIFT, and STOP actions, then
  // the symbol /table/fp/refer for REFER won't be serialized unless the table
  // links to it.
  std::vector<Handle> symbols = {
    action_type, action_length, action_source, action_target, action_role,
    action_label, fingerprint, fingerprint_evoke, fingerprint_refer,
    type_role, type_value, type_source, type_target, type_assign, type_embed,
    type_elaborate, type_connect
  };
  Array symbols_array(&store, symbols);
  top.Add("/table/symbols", symbols_array);

  StringEncoder encoder(&store);  // TODO(grahul): move to FileEncoder
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

  // Arguments summary.
  writer->StartTable("Action Arguments Summary");
  writer->SetColumns({"Metric", "Value"});
  writer->AddRow("Num unique phrase fp", static_cast<int>(fingerprint_.size()));
  writer->AddRow("Fingerprint map load factor (as %)",
                 fingerprint_.load_factor() * 100);
  int max_evoke_size = 0;
  HandleSet evoke_types;
  for (const auto &kv : fingerprint_) {
    if (kv.second.evoke.size() > max_evoke_size) {
      max_evoke_size = kv.second.evoke.size();
    }
    for (int index : kv.second.evoke) {
      evoke_types.insert(Action(index).label);
    }
  }
  writer->AddRow("Maximum EVOKE actions for a fingerprint", max_evoke_size);
  writer->AddRow("Total types involved in EVOKE actions",
                 static_cast<int64>(evoke_types.size()));
  writer->AddRow("Num types used as keys in constraints map",
                 static_cast<int64>(type_.size()));
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

  int cumulative = 0;
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

