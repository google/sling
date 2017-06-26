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

#include "myelin/kernel/generic.h"

#include <vector>

#include "myelin/compute.h"

namespace sling {
namespace myelin {

// array.cc
void RegisterArrayKernels(Library *library);

// generic-math.cc
void RegisterGenericMath(Library *library);

// generic-matmul.cc
void RegisterGenericMatMul(Library *library);

// generic-operators.cc
void RegisterGenericOperators(Library *library);

// Rename operations with aliases.
class RenameTransformer : public Transformer {
 public:
  bool Transform(Flow *flow) override {
    // Rename BiasAdd to Add.
    int renames = 0;
    for (Flow::Operation *op : flow->ops()) {
      if (op->type == "BiasAdd") {
        op->type = "Add";
        renames++;
      }
    }

    return renames > 0;
  }
};

// Remove identity ops.
class IdentityTransformer : public Transformer {
 public:
  bool Transform(Flow *flow) override {
    // Eliminate no-ops.
    std::vector<Flow::Operation *> noops;
    for (Flow::Operation *op : flow->ops()) {
      if (op->type == "Identity" ||
          op->type == "Const" ||


          op->type == "Variable" ||
          op->type == "VariableV2" ||
          op->type == "Placeholder" ||
          op->type == "Enter") {
        noops.push_back(op);
      }
    }

    // Remove no-ops from the flow and eliminate the intermediate variables.
    for (Flow::Operation *op : noops) {
      flow->Eliminate(op);
    }

    return !noops.empty();
  }
};

// Combine ops.
class CombineTransformer : public Transformer {
 public:
  bool Transform(Flow *flow) override {
    int combines = 0;
    while (Combine(flow, "MatMul", "Add", "MatMulAdd") ||
           Combine(flow, "MatMul", "Relu", "MatMulRelu") ||
           Combine(flow, "MatMulAdd", "Relu", "MatMulAddRelu")) {
      combines++;
    }
    return combines > 0;
  }

  // Try to find combinations and replace them with a combined op.
  bool Combine(Flow *flow, const string &first, const string &second,
               const string &combined) {
    // Find operations that can be combined.
    for (Flow::Operation *op : flow->ops()) {
      if (op->type != first) continue;
      if (op->outputs.size() != 1) continue;
      Flow::Variable *var = op->outputs[0];
      if (var->consumers.size() != 1) continue;
      if (var->consumers[0]->type != second) continue;
      if (var->consumers[0]->task != op->task) continue;

      flow->Fuse(op, var->consumers[0], combined);
      return true;
    }
    return false;
  }
};

// Type inference for standard ops.
class StandardTyper : public Typer {
 public:
  bool InferTypes(Flow::Operation *op) override {
    // Infer shape for matrix multiplication.
    if (op->type == "MatMul" ||
        op->type == "MatMulAdd" ||
        op->type == "MatMulRelu" ||
        op->type == "MatMulAddRelu") {
      if (op->indegree() >= 2 && op->outdegree() == 1) {
        Flow::Variable *a = op->inputs[0];
        Flow::Variable *b = op->inputs[1];
        Flow::Variable *c = op->outputs[0];

        // Matrix multipled by matrix.
        if (a->rank() == 2 && b->rank() == 2 && a->dim(1) == b->dim(0)) {
          c->shape.assign(a->dim(0), b->dim(1));
          return true;
        }
      }
    }

    // Infer shape for element-wise operation.
    if (op->type == "Add" ||
        op->type == "BiasAdd" ||
        op->type == "Mul" ||
        op->type == "Sub" ||
        op->type == "Tanh" ||
        op->type == "Sigmoid" ||
        op->type == "Relu") {
      if (op->indegree() > 0 && op->outdegree() > 0) {
        // Determine output rank.
        Shape shape;
        int rank = 0;
        for (Flow::Variable *in : op->inputs) {
          if (in->rank() > rank) rank = in->rank();
        }
        shape.fill(rank, 1);

        // Determine output shape based on broadcast semantics.
        for (Flow::Variable *in : op->inputs) {
          int depth = rank - in->rank();
          for (int d = 0; d < in->rank(); ++d) {
            if (shape.dim(d + depth) < in->dim(d)) {
              shape.set(d + depth, in->dim(d));
            }
          }
        }

        // Set shape for outputs.
        for (Flow::Variable *out : op->outputs) {
          out->shape = shape;
        }
        return true;
      }
    }

    // Infer shape for concat operation.
    if (op->type == "ConcatV2") {
      int n = op->GetAttr("N", 0);
      if (n > 0 && op->indegree() == n + 1 && op->outdegree() == 1) {
        Flow::Variable *axis = op->inputs[n];
        Flow::Variable *result = op->outputs[0];
        if (axis->type == DT_INT32 &&
            axis->rank() == 0 &&
            axis->data != nullptr) {
          int a = *reinterpret_cast<int *>(axis->data);
          Shape concat = op->inputs[0]->shape;
          bool compatible = true;
          for (int i = 1; i < n; ++i) {
            Flow::Variable *input = op->inputs[i];
            if (input->shape.rank() == concat.rank()) {
              for (int d = 0; d < concat.rank(); ++d) {
                if (d == a) {
                  concat.set(d, concat.dim(d) + input->shape.dim(d));
                } else if (concat.dim(d) != input->shape.dim(d)) {
                  compatible = false;
                }
              }
            } else {
              compatible = false;
            }
          }
          if (compatible) {
            result->shape = concat;
            return true;
          }
        }
      }
    }

    // Infer shape for reshaping operation.
    if (op->type == "Reshape") {
      if (op->indegree() == 2 && op->outdegree() == 1) {
        Flow::Variable *shape = op->inputs[1];
        Flow::Variable *result = op->outputs[0];
        if (shape->type == DT_INT32 &&
            shape->rank() == 1 &&
            shape->data != nullptr) {
          // The output shape is constant.
          int *dims = reinterpret_cast<int *>(shape->data);
          result->shape.clear();
          for (int d = 0; d < shape->dim(0); ++d) {
            result->shape.add(dims[d] == -1 ? 1 : dims[d]);
          }
          return true;
        }
      }
    }

    return false;
  }
};

// Register generic library.
void RegisterGenericLibrary(Library *library) {
  // Register transformations.
  library->RegisterTransformer(new RenameTransformer());
  library->RegisterTransformer(new IdentityTransformer());
  library->RegisterTransformer(new CombineTransformer());

  // Register type inference.
  library->RegisterTyper(new StandardTyper());

  // Register kernels.
  RegisterArrayKernels(library);
  RegisterGenericMath(library);
  RegisterGenericMatMul(library);
  RegisterGenericOperators(library);
}

}  // namespace myelin
}  // namespace sling

