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
#include "sling/frame/serialization.h"

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

void ActionTable::Read(const Frame &frame) {
  Array actions = frame.Get("actions").AsArray();
  CHECK(actions.valid());

  Store *store = frame.store();
  Handle n_type = store->Lookup("type");
  Handle n_length = store->Lookup("length");
  Handle n_source = store->Lookup("source");
  Handle n_target = store->Lookup("target");
  Handle n_role = store->Lookup("role");
  Handle n_label = store->Lookup("label");
  Handle n_delegate = store->Lookup("delegate");
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

void ActionTable::Write(Builder *frame) const {
  // Save the action table.
  Store *store = frame->store();
  Handle n_type = store->Lookup("type");
  Handle n_length = store->Lookup("length");
  Handle n_source = store->Lookup("source");
  Handle n_target = store->Lookup("target");
  Handle n_role = store->Lookup("role");
  Handle n_label = store->Lookup("label");
  Handle n_delegate = store->Lookup("delegate");

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
    if (type == ParserAction::ASSIGN || type == ParserAction::CONNECT) {
      if (action.source != 0) {
        b.Add(n_source, static_cast<int>(action.source));
      }
    }
    if (type == ParserAction::REFER || type == ParserAction::CONNECT) {
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
  frame->Add("actions", actions);
}

}  // namespace nlp
}  // namespace sling

