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

#include "sling/myelin/kernel/generic.h"

#include <algorithm>
#include <vector>

#include "sling/myelin/compute.h"

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
      } else if (op->type == "Reshape") {
        // Eliminate reshaping if input and output shapes are equal.
        if (op->indegree() == 2 && op->outdegree() == 1) {
          Flow::Variable *in = op->inputs[0];
          Flow::Variable *out = op->outputs[0];
          if (in->shape.defined() &&
              out->shape.defined() &&
              in->shape == out->shape &&
              in->type == out->type) {
            Flow::Variable *shape = op->inputs[1];
            op->RemoveInput(shape);
            noops.push_back(op);
          }
        }
      } else if (op->type == "Concat" || op->type == "ConcatV2") {
        // Eliminate concatenations with only one input.
        int n = op->GetAttr("N", 0);
        if (n == 1) {
          Flow::Variable *axis = op->inputs[n];
          op->RemoveInput(axis);
          noops.push_back(op);
        }
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
      if (var->out) continue;
      if (!var->shape.defined()) continue;
      if (op->indegree() >= 1) {
        // Only combine for vector inputs.
        Flow::Variable *input = op->inputs[0];
        if (input->rank() == 2 && input->dim(0) > 1) continue;
      }

      flow->Fuse(op, var->consumers[0], combined);
      return true;
    }
    return false;
  }
};

// Flattens nested concatenations, if possible.  E.g.,
// tf.concat([a, tf.concat([b, c], 1), d], 1) = tf.concat([a, b, c, d], 1)
class FlattenConcatTransformer : public Transformer {
 public:
  bool Transform(Flow *flow) override {
    bool transformed = false;
    while (TryFlattenOnce(flow)) transformed = true;
    return transformed;
  }

 private:
  // Returns true if the operation is a concatenation.
  static bool IsConcat(const Flow::Operation &operation) {
    if (operation.type != "ConcatV2") return false;
    if (!operation.HasAttr("N")) return false;
    const int num_to_concat = operation.GetAttr("N", -1);
    if (num_to_concat <= 0) return false;
    if (operation.indegree() != num_to_concat + 1) return false;
    if (operation.outdegree() != 1) return false;
    return true;
  }

  // Flattens one nested concatenation and returns true, if possible.
  static bool TryFlattenOnce(Flow *flow) {
    // Search for a parent and child concat, where both have the same axis and
    // the result of the child concat is only used by the parent concat.
    for (Flow::Operation *child : flow->ops()) {
      if (!IsConcat(*child)) continue;

      // The child should have only one consumer, the parent.
      Flow::Variable *child_result = child->outputs[0];
      if (child_result->consumers.size() != 1) continue;
      Flow::Operation *parent = child_result->consumers[0];
      if (!IsConcat(*parent)) continue;

      // The axes (i.e., final inputs) should match.
      int parent_axis = 0, child_axis = 0;
      if (!parent->inputs.back()->GetData(&parent_axis)) continue;
      if (!child->inputs.back()->GetData(&child_axis)) continue;
      if (parent_axis != child_axis) continue;

      // The child axis will be pruned, so it should have no other dependencies.
      if (child->inputs.back()->consumers.size() != 1) continue;
      if (child->inputs.back()->producer != nullptr) continue;

      Flatten(flow, parent, child);
      return true;
    }

    return false;
  }

  // Flattens the child concatenation into the parent concatenation by replacing
  // the child with the inputs it concatenates.
  static void Flatten(Flow *flow, Flow::Operation *parent,
                      Flow::Operation *child) {
    VLOG(9) << "Flattening " << child->type << " (" << child->name << ") into "
            << parent->type << " (" << parent->name << ")";

    // Find the index of the child among the parent's inputs.  This is where the
    // child's inputs should be inserted.
    Flow::Variable *child_result = child->outputs[0];
    const int child_index =
        std::find(parent->inputs.begin(), parent->inputs.end(), child_result) -
        parent->inputs.begin();
    CHECK_LT(child_index, parent->inputs.size())
        << "parent=" << parent->name << " child=" << child->name;

    // Discard the child's axis; it is redundant with the parent's axis.
    Flow::Variable *child_axis = child->inputs.back();
    child->RemoveInput(child_axis);
    flow->DeleteVariable(child_axis);

    // Discard the child's result; it will be replaced with the child's inputs.
    child->RemoveOutput(child_result);
    parent->RemoveInput(child_result);
    flow->DeleteVariable(child_result);

    // Move the child's inputs to the parent.
    while (!child->inputs.empty()) {
      Flow::Variable *input = child->inputs.back();  // iterate back to front
      child->MoveInput(input, parent);

      // MoveInput() appends to the parent's input list, so pop and reinsert it
      // at the proper location.  Since we iterate the child's inputs backwards,
      // it suffices to repeatedly insert at the same index.
      CHECK_EQ(input, parent->inputs.back());
      parent->inputs.pop_back();
      parent->inputs.insert(parent->inputs.begin() + child_index, input);
    }

    flow->DeleteOperation(child);
    parent->SetAttr("N", static_cast<int>(parent->inputs.size() - 1));
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

        if (c->type == DT_INVALID) c->type = a->type;

        // Matrix multiplied by matrix.
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
        op->type == "Relu" ||
        op->type == "Calculate") {
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

        if (op->outputs[0]->type == DT_INVALID) {
          op->outputs[0]->type = op->inputs[0]->type;
        }
        return true;
      }
    }

    // Infer shape for concat operation.
    if (op->type == "ConcatV2") {
      int n = op->GetAttr("N", 0);
      if (n > op->indegree()) return false;
      int axis;
      if (op->indegree() != n + 1 || !op->inputs[n]->GetData(&axis)) axis = 0;

      if (n > 0 && op->outdegree() == 1) {
        Flow::Variable *result = op->outputs[0];
        Shape concat = op->inputs[0]->shape;
        bool compatible = true;
        for (int i = 1; i < n; ++i) {
          Flow::Variable *input = op->inputs[i];
          if (input->shape.rank() == concat.rank()) {
            for (int d = 0; d < concat.rank(); ++d) {
              if (d == axis) {
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

    // Infer shape for reshaping operation.
    if (op->type == "Reshape") {
      if (op->indegree() == 2 && op->outdegree() == 1) {
        Flow::Variable *shape = op->inputs[1];
        Flow::Variable *result = op->outputs[0];
        std::vector<int> dims;
        if (shape->GetData(&dims)) {
          result->shape.clear();
          for (int dim : dims) {
            // Unspecified dimensions (-1) typically correspond to to the input
            // sequence length.  Since Myelin cells process one input element at
            // a time, -1 can be replaced with 1.
            result->shape.add(dim == -1 ? 1 : dim);
          }
          if (op->outputs[0]->type == DT_INVALID) {
            op->outputs[0]->type = op->inputs[0]->type;
          }
          return true;
        }
      }
    }

    // Infer shape for gather operation.
    if (op->type == "Gather") {
      // For the 2-arg form tf.gather(params, indices):
      //   result.type = params.dtype.
      //   result.shape = indices.shape + params.shape[1:].
      // https://www.tensorflow.org/api_docs/python/tf/gather
      // Note that there is also a 3-arg form tf.gather(params, indices, axis).
      if (op->indegree() == 2 && op->outdegree() == 1) {
        Flow::Variable *params = op->inputs[0];
        Flow::Variable *indices = op->inputs[1];
        Flow::Variable *result = op->outputs[0];
        result->type = params->type;
        result->shape = indices->shape;
        for (int i = 1; i < params->shape.rank(); ++i) {
          result->shape.add(params->shape.dim(i));
        }
        return true;
      }
    }

    // Infer shape for argmax operation.
    if (op->type == "ArgMax") {
      if (op->indegree() == 1 && op->outdegree() == 1) {
        Flow::Variable *y = op->outputs[0];
        if (y->type == DT_INVALID) y->type = DT_INT32;
        y->shape.redim(0);
        return true;
      }
    }

    // Infer shape for tf.fill(dims, value).
    if (op->type == "Fill") {
      std::vector<int> dims;
      if (op->indegree() == 2 && op->outdegree() == 1 &&
          op->inputs[0]->GetData(&dims)) {
        op->outputs[0]->shape = Shape(dims);
        return true;
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
  library->RegisterTransformer(new FlattenConcatTransformer());

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
