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
  max_actions_per_token_ = top.GetInt("/table/max_actions_per_token");
  frame_limit_ = top.GetInt("/table/frame_limit");

  // Read the action index.
  Array actions = top.Get("/table/actions").AsArray();
  CHECK(actions.valid());

  Handle action_type = store->LookupExisting("/table/action/type");
  Handle action_length = store->LookupExisting("/table/action/length");
  Handle action_source = store->LookupExisting("/table/action/source");
  Handle action_target = store->LookupExisting("/table/action/target");
  Handle action_role = store->LookupExisting("/table/action/role");
  Handle action_label = store->LookupExisting("/table/action/label");
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

    Add(action);
  }
}

void ActionTable::Save(const Store *global, const string &file) const {
  string s = Serialize(global);
  CHECK(File::WriteContents(file, s));
}

string ActionTable::Serialize(const Store *global) const {
  Store store(global);
  Builder top(&store);
  top.AddId("/table");

  // Save the action table.
  Handle action_type = store.Lookup("/table/action/type");
  Handle action_length = store.Lookup("/table/action/length");
  Handle action_source = store.Lookup("/table/action/source");
  Handle action_target = store.Lookup("/table/action/target");
  Handle action_role = store.Lookup("/table/action/role");
  Handle action_label = store.Lookup("/table/action/label");

  Array actions(&store, actions_.size());
  int index = 0;
  for (const ParserAction &action : actions_) {
    auto type = action.type;
    Builder b(&store);
    b.Add(action_type, static_cast<int>(type));

    if (type == ParserAction::REFER || type == ParserAction::EVOKE) {
      if (action.length > 0) {
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
    if (!action.role.IsNil()) b.Add(action_role, action.role);
    if (!action.label.IsNil()) b.Add(action_label, action.label);
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

}  // namespace nlp
}  // namespace sling

