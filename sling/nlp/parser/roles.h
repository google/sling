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

#ifndef SLING_NLP_PARSER_ROLES_H_
#define SLING_NLP_PARSER_ROLES_H_

#include <functional>
#include <vector>

#include "sling/frame/object.h"
#include "sling/nlp/parser/parser-action.h"
#include "sling/nlp/parser/parser-state.h"

namespace sling {
namespace nlp {

// A mapping of roles to role ids extracted from the action set.
class RoleSet {
 public:
  // Add role to role set.
  void Add(Handle role);

  // Add roles from action list.
  void Add(const std::vector<ParserAction> &actions);

  // Look up role id for role. Return -1 if role is unknown.
  int Lookup(Handle role) const {
    const auto &it = roles_.find(role);
    if (it == roles_.end()) return -1;
    return it->second;
  }

  // Return the number of roles in the role set.
  int size() const { return roles_.size(); }

  // Get list of roles.
  void GetList(std::vector<Handle> *list) const;

 private:
  // Mapping from role handle to role id.
  HandleMap<int> roles_;
};

// A role graph represents the roles edges between the top frames in the
// attention buffer of a parser state. This is used for extracting role features
// for the frame semantic parser.
class RoleGraph {
 public:
  // Feature emitter function.
  typedef std::function<void(int feature)> Emit;

  // Compute role graph from parser state.
  void Compute(const ParserState &state, int limit, const RoleSet &roles);

  // Emit (source, role) features.
  void out(Emit emit) const {
    for (const Edge &e : edges_) {
      emit(e.source * num_roles_ + e.role);
    }
  }

  // Emit (role, target) features.
  void in(Emit emit) const {
    for (const Edge &e : edges_) {
      if (e.target != -1) {
        emit(e.target * num_roles_ + e.role);
      }
    }
  }

  // Emit (source, target) features.
  void unlabeled(Emit emit) const {
    for (const Edge &e : edges_) {
      if (e.target != -1) {
        emit(e.source + e.target * limit_);
      }
    }
  }

  // Emit (source, role, target) features.
  void labeled(Emit emit) const {
    for (const Edge &e : edges_) {
      if (e.target != -1) {
        emit(e.source * limit_ * num_roles_ + e.target * num_roles_ + e.role);
      }
    }
  }

 private:
  // Edge in role graph.
  struct Edge {
    Edge(int s, int r, int t) : source(s), role(r), target(t) {}
    int source;
    int role;
    int target;
  };

  // The maximum number of frames to use from the attention buffer.
  int limit_ = 0;

  // Number of roles in role set.
  int num_roles_ = 0;

  // Edges in the role graph.
  std::vector<Edge> edges_;
};

}  // namespace nlp
}  // namespace sling

#endif  // SLING_NLP_PARSER_ROLES_H_

