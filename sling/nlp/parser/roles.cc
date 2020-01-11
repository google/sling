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

#include "sling/nlp/parser/roles.h"

namespace sling {
namespace nlp {

void RoleSet::Add(Handle role) {
  if (!role.IsNil() && roles_.find(role) == roles_.end()) {
    int index = roles_.size();
    roles_[role] = index;
  }
}

void RoleSet::Add(const std::vector<ParserAction> &actions) {
  for (const ParserAction &action : actions) Add(action.role);
}

void RoleSet::GetList(std::vector<Handle> *list) const {
  list->resize(roles_.size());
  for (auto &it : roles_) {
    (*list)[it.second] = it.first;
  }
}

void RoleGraph::Compute(const ParserState &state,
                        int limit,
                        const RoleSet &roles) {
  limit_ = limit;
  num_roles_ = roles.size();
  int k = limit_;
  edges_.clear();
  if (k > state.AttentionSize()) k = state.AttentionSize();
  for (int source = 0; source < k; ++source) {
    Handle handle = state.Attention(source).frame;
    const FrameDatum *frame = state.store()->GetFrame(handle);
    for (const Slot *slot = frame->begin(); slot < frame->end(); ++slot) {
      int target = -1;
      if (slot->value.IsLocalRef()) {
        target = state.AttentionIndex(slot->value, k);
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

