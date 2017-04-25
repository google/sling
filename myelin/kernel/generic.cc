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

#include "myelin/compute.h"

namespace sling {
namespace myelin {

// generic-math.cc
void RegisterGenericMath(Library *library);

// generic-matmul.cc
void RegisterGenericMatMul(Library *library);

// generic-operators.cc
void RegisterGenericOperators(Library *library);

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
      if (op->indegree() > 0 && op->outdegree() == 1) {
        // Determine output rank.
        Flow::Variable *out = op->outputs[0];
        int rank = 0;
        for (Flow::Variable *in : op->inputs) {
          if (in->rank() > rank) rank = in->rank();
        }
        out->shape.fill(rank, 1);

        // Determine output shape based on broadcast semantics.
        for (Flow::Variable *in : op->inputs) {
          int depth = rank - in->rank();
          for (int d = 0; d < in->rank(); ++d) {
            if (out->dim(d + depth) < in->dim(d)) {
              out->shape.set(d + depth, in->dim(d));
            }
          }
        }
        return true;
      }
    }

    // Infer shape for lookup operation.
    if (op->type == "Lookup") {
      if (op->indegree() == 2 && op->outdegree() == 1) {
        Flow::Variable *embeddings = op->inputs[1];
        Flow::Variable *result = op->outputs[0];
        if (embeddings->rank() == 2) {
          result->shape.assign(1, embeddings->dim(1));
          return true;
        }
      }
    }

    // Infer shape for collect operation.
    if (op->type == "Collect") {
      if (op->indegree() == 2 && op->outdegree() == 1) {
        Flow::Variable *features = op->inputs[0];
        Flow::Variable *embeddings = op->inputs[1];
        Flow::Variable *result = op->outputs[0];
        if (features->rank() == 2 && embeddings->rank() == 2) {
          // Add extra element for OOV indicator.
          result->shape.assign(features->dim(1), embeddings->dim(1) + 1);
          return true;
        }
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

// Register generic transformations.
void RegisterGenericTransformations(Library *library) {
  // Register identity ops.
  library->RegisterIdentityOp("Identity");
  library->RegisterIdentityOp("Const");
  library->RegisterIdentityOp("Variable");
  library->RegisterIdentityOp("VariableV2");
  library->RegisterIdentityOp("Placeholder");
  library->RegisterIdentityOp("Enter");
  library->RegisterIdentityOp("FeatureVector");

  // Register combined ops.
  library->RegisterCombinedOp("MatMul", "Add", "MatMulAdd");
  library->RegisterCombinedOp("MatMul", "BiasAdd", "MatMulAdd");
  library->RegisterCombinedOp("MatMul", "Relu", "MatMulRelu");
  library->RegisterCombinedOp("MatMulAdd", "Relu", "MatMulAddRelu");

  // Register type inference.
  library->RegisterTyper(new StandardTyper());
}

// Register generic kernels.
void RegisterGenericKernels(Library *library) {
  RegisterGenericMath(library);
  RegisterGenericMatMul(library);
  RegisterGenericOperators(library);
}

}  // namespace myelin
}  // namespace sling

