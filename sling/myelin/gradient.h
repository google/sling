// Copyright 2018 Google Inc.
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

#ifndef SLING_MYELIN_GRADIENT_H_
#define SLING_MYELIN_GRADIENT_H_

#include <vector>
#include <unordered_map>

#include "sling/myelin/builder.h"
#include "sling/myelin/flow.h"

namespace sling {
namespace myelin {

// Gradients for function.
class Gradients : public FlowBuilder {
 public:
  // Initialize adjoints for gradient derivation.
  Gradients(Flow *flow, Flow::Function *primal,
            std::vector<Flow::Variable *> &vars);

  // Get adjoint for primal variable.
  Flow::Variable *d(Flow::Variable *x) {
    return x->constant() ? Const(0.0f) : adjoints_[x];
  }

  // Get primal variable reference.
  Flow::Variable *v(Flow::Variable *x) {
    return x->constant() ? x : GetReference(x);
  }

  // Add term to adjoint for variable.
  void add(Flow::Variable *x, Flow::Variable *term) {
    auto f = adjoints_.find(x);
    if (f == adjoints_.end()) return;
    auto *dx = f->second;
    terms_[dx] = (terms_[dx] == nullptr) ? term : Add(terms_[dx], term);
  }

  // Finalize gradient function.
  Flow::Function *Finalize();

 private:
  // Get reference to primal variable.
  Flow::Variable *GetReference(Flow::Variable *x);

  // Mapping from primal variables to adjoint.
  std::unordered_map<Flow::Variable *, Flow::Variable *> adjoints_;

  // Terms for adjoint.
  std::unordered_map<Flow::Variable *, Flow::Variable *> terms_;

  // Instance variable pointing to primal variables.
  Flow::Variable *instance_;

  // References to primal variables (lazily initialized).
  std::unordered_map<Flow::Variable *, Flow::Variable *> refs_;
};

// Build gradient for function.
Flow::Function *Gradient(Flow *flow,
                         Flow::Function *func,
                         const Transformations &library);

}  // namespace myelin
}  // namespace sling

#endif  // SLING_MYELIN_BUILDER_H_

