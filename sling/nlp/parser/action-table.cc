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
#include "sling/string/text.h"

namespace sling {
namespace nlp {

void ActionTable::Add(const ParserAction &action) {
  // Add the action to the table if it is new.
  int index = Index(action);
  if (index == -1) {
    int index = actions_.size();
    actions_.emplace_back(action);
    mapping_[action] = index;
  }
}

void ActionTable::Init(Store *store) {
  Frame top(store, "/table");
  CHECK(top.valid());

  // Get all the integer fields.
  max_actions_per_token_ = top.GetInt("/table/max_actions_per_token", 5);
  frame_limit_ = top.GetInt("/table/frame_limit", 5);

  // Read the action index.
  Array actions = top.Get("/table/actions").AsArray();
  CHECK(actions.valid());

  Handle n_type = store->Lookup("/table/action/type");
  Handle n_length = store->Lookup("/table/action/length");
  Handle n_source = store->Lookup("/table/action/source");
  Handle n_target = store->Lookup("/table/action/target");
  Handle n_role = store->Lookup("/table/action/role");
  Handle n_label = store->Lookup("/table/action/label");
  Handle n_delegate = store->Lookup("/table/action/delegate");
  for (int i = 0; i < actions.length(); ++i) {
    ParserAction action;
    Frame item(store, actions.get(i));
    CHECK(item.valid());

    for (const Slot &slot : item) {
      if (slot.name == n_type) {
        action.type = static_cast<ParserAction::Type>(slot.value.AsInt());
      } else if (slot.name == n_length) {
        action.length = slot.value.AsInt();
      } else if (slot.name == n_source) {
        action.source = slot.value.AsInt();
      } else if (slot.name == n_target) {
        action.target = slot.value.AsInt();
      } else if (slot.name == n_role) {
        action.role = slot.value;
      } else if (slot.name == n_label) {
        action.label = slot.value;
      } else if (slot.name == n_delegate) {
        action.delegate = slot.value.AsInt();
      }
    }

    Add(action);
  }
}

void ActionTable::Save(const Store *global, const string &file) const {
  string s = Serialize(global);
  CHECK(File::WriteContents(file, s));
}

string ActionTable::Serialize(const Store *global) const {
  // Build frame with action table.
  Store store(global);
  Builder table(&store);
  table.AddId("/table");
  Write(&table);

  StringEncoder encoder(&store);
  encoder.Encode(table.Create());
  return encoder.buffer();
}

void ActionTable::Write(Builder *frame) const {
  // Save the action table.
  Store *store = frame->store();
  Handle n_type = store->Lookup("/table/action/type");
  Handle n_length = store->Lookup("/table/action/length");
  Handle n_source = store->Lookup("/table/action/source");
  Handle n_target = store->Lookup("/table/action/target");
  Handle n_role = store->Lookup("/table/action/role");
  Handle n_label = store->Lookup("/table/action/label");
  Handle n_delegate = store->Lookup("/table/action/delegate");

  Array actions(store, actions_.size());
  int index = 0;
  for (const ParserAction &action : actions_) {
    auto type = action.type;
    Builder b(store);
    b.Add(n_type, static_cast<int>(type));

    if (type == ParserAction::REFER || type == ParserAction::EVOKE) {
      if (action.length > 0) {
        b.Add(n_length, static_cast<int>(action.length));
      }
    }
    if (type == ParserAction::ASSIGN ||
        type == ParserAction::ELABORATE ||
        type == ParserAction::CONNECT) {
      if (action.source != 0) {
        b.Add(n_source, static_cast<int>(action.source));
      }
    }
    if (type == ParserAction::EMBED ||
        type == ParserAction::REFER ||
        type == ParserAction::CONNECT) {
      if (action.target != 0) {
        b.Add(n_target, static_cast<int>(action.target));
      }
    }
    if (type == ParserAction::CASCADE) {
      b.Add(n_delegate, static_cast<int>(action.delegate));
    }
    if (!action.role.IsNil()) b.Add(n_role, action.role);
    if (!action.label.IsNil()) b.Add(n_label, action.label);
    actions.set(index++, b.Create().handle());
  }
  frame->Add("/table/actions", actions);
}

}  // namespace nlp
}  // namespace sling

