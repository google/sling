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

#include "nlp/parser/roles.h"

namespace sling {
namespace nlp {

void RoleSet::Init(const ActionTable &actions) {
  for (int i = 0; i < actions.NumActions(); ++i) {
    const auto &action = actions.Action(i);
    if (action.type == ParserAction::CONNECT ||
        action.type == ParserAction::ASSIGN ||
        action.type == ParserAction::EMBED ||
        action.type == ParserAction::ELABORATE) {
      if (roles_.find(action.role) == roles_.end()) {
        int index = roles_.size();
        roles_[action.role] = index;
      }
    }
  }
}

void RoleGraph::Compute(const ParserState &state,
                        int limit,
                        const RoleSet &roles) {
  limit_ = limit;
  num_roles_ = roles.size();
  int k = limit_;
  if (k > state.AttentionSize()) k = state.AttentionSize();
  for (int source = 0; source < k; ++source) {
    Handle handle = state.frame(state.Attention(source));
    const FrameDatum *frame = state.store()->GetFrame(handle);
    for (const Slot *slot = frame->begin(); slot < frame->end(); ++slot) {
      int target = -1;
      if (slot->value.IsIndex()) {
        target = state.AttentionIndex(slot->value.AsIndex(), k);
        if (target == -1) continue;
      }

      int role = roles.Lookup(slot->name);
      if (role == -1) continue;

      edges_.emplace_back(source, role, target);
    }
  }
}

}  // namespace nlp
}  // namespace sling

