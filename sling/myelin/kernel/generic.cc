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
#include "sling/myelin/macro-assembler.h"

#define __ masm->

namespace sling {
namespace myelin {

using namespace jit;

// array.cc
void RegisterArrayKernels(Library *library);

// generic-math.cc
void RegisterGenericMath(Library *library);

// generic-matmul.cc
void RegisterGenericMatMul(Library *library);

// generic-operators.cc
void RegisterGenericOperators(Library *library);

// Reference op for accessing parameters in other cells of the network. Looks up
// tensor 'var' in instance and outputs a reference to the tensor.
class Reference : public Kernel {
 public:
  string Name() override { return "Reference"; }
  string Operation() override { return "Reference"; }

  bool Supports(Step *step) override {
    // Check inputs and outputs.
    if (step->indegree() != 1 || step->outdegree() != 1) return false;

    // Lookup variable.
    Tensor *var = GetReference(step);
    if (var == nullptr) {
      LOG(WARNING) << "Missing/unknown reference variable for " << step->name();
      return false;
    }

    // Check types.
    Tensor *instance = step->input(0);
    Tensor *ref = step->output(0);
    if (instance->type() != DT_RESOURCE || !instance->ref()) return false;
    if (ref->type() != var->type() || !ref->ref()) return false;

    return true;
  }

  void Adjust(Step *step) override {
    // Propagate alignment constraints from reference to variable.
    Tensor *var = GetReference(step);
    CHECK(var != nullptr);
    step->output(0)->Link(var);

    // Propagate corresponding sparsity tensors.
    if (var->sparse()) {
      Tensor *sparse_ref = step->output(0)->MakeSparse(true);
      sparse_ref->Link(var->sparse());
    }
  }

  void Generate(Step *step, MacroAssembler *masm) override {
    // Get inputs and outputs.
    Tensor *instance = step->input(0);
    Tensor *ref = step->output(0);
    Tensor *var = GetReference(step);
    CHECK(instance->IsLocal());
    CHECK(ref->IsLocal());
    CHECK(var != nullptr);

    // Output reference to variable in other instance.
    Register addr = masm->rr().alloc();
    if (var->IsGlobal()) {
      __ load_extern(addr, var->data(), var->name());
    } else {
      __ movq(addr, Operand(masm->instance(), instance->offset()));
      if (var->ref()) {
        __ movq(addr, Operand(addr, var->offset()));
      } else if (var->offset() != 0) {
        __ addq(addr, Immediate(var->offset()));
      }
    }
    __ movq(Operand(masm->instance(), ref->offset()), addr);

    // Output reference to sparsity vector.
    if (ref->sparse()) {
      CHECK(ref->sparse()->IsLocal());
      CHECK(var->sparse()->IsLocal());
      __ movq(addr, Operand(masm->instance(), instance->offset()));
      if (var->sparse()->ref()) {
        __ movq(addr, Operand(addr, var->sparse()->offset()));
      } else if (var->sparse()->offset() != 0) {
        __ addq(addr, Immediate(var->sparse()->offset()));
      }
      __ movq(Operand(masm->instance(), ref->sparse()->offset()), addr);
    }
  }

  int64 Complexity(const Step *step) override {
    return 0;
  }

  // Get referenced tensor.
  static Tensor *GetReference(Step *step) {
    const string &varname = step->GetAttr("var");
    if (varname.empty()) return nullptr;
    return step->cell()->network()->GetParameter(varname);
  }
};

// Rename operations with aliases.
class RenameTransformer : public Transformer {
 public:
  string Name() override { return "RenameTransformer"; }

  bool Transform(Flow *flow) override {
    // Rename BiasAdd to Add.
    int renames = 0;
    for (Flow::Operation *op : flow->ops()) {
      if (op->type == "BiasAdd") {
        op->type = "Add";
        renames++;
      }
      if (op->type == "ConcatV2") {
        op->type = "Concat";
        renames++;
      }
      if (op->type == "GatherV2") {
        op->type = "Gather";
        renames++;
      }
    }

    return renames > 0;
  }
};

// Remove identity ops.
class IdentityTransformer : public Transformer {
 public:
  string Name() override { return "IdentityTransformer"; }

  bool Transform(Flow *flow) override {
    // Eliminate no-ops.
    std::vector<Flow::Operation *> noops;
    for (Flow::Operation *op : flow->ops()) {
      if (op->type == "Const" ||
          op->type == "Variable" ||
          op->type == "VariableV2" ||
          op->type == "Placeholder" ||
          op->type == "Enter") {
        noops.push_back(op);
      } else if (op->type == "Identity") {
        // Eliminate identity if there is no implicit broadcasting involved.
        if (op->indegree() == 1 && op->outdegree() == 1) {
          Flow::Variable *in = op->inputs[0];
          Flow::Variable *out = op->outputs[0];
          if (!out->shape.missing() && in->shape != out->shape) continue;
          if (in->type != out->type) continue;

          // Assignment of global constant to output needs to be materialized.
          if (out->out() && in->global()) continue;

          // Assignment of local to global needs to be materialized.
          if (in->local() && out->global()) continue;

          noops.push_back(op);
        }
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
      } else if (op->type == "Concat") {
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

// Flattens nested concatenations, if possible.  E.g.,
// tf.concat([a, tf.concat([b, c], 1), d], 1) = tf.concat([a, b, c, d], 1)
class FlattenConcatTransformer : public Transformer {
 public:
  string Name() override { return "FlattenConcatTransformer"; }

  bool Transform(Flow *flow) override {
    bool transformed = false;
    while (TryFlattenOnce(flow)) transformed = true;
    return transformed;
  }

 private:
  // Returns true if the operation is a concatenation.
  static bool IsConcat(const Flow::Operation &operation) {
    if (operation.type != "Concat") return false;
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
      if (child_result->usages() != 1) continue;
      Flow::Operation *parent = child_result->consumers[0];
      if (!IsConcat(*parent)) continue;

      // The axes (i.e., final inputs) should match.
      int parent_axis = 0, child_axis = 0;
      if (!parent->inputs.back()->GetData(&parent_axis)) continue;
      if (!child->inputs.back()->GetData(&child_axis)) continue;
      if (parent_axis != child_axis) continue;

      // The child axis will be pruned, so it should have no other dependencies.
      if (child->inputs.back()->usages() != 1) continue;
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

// Normalizes "Gather" operations. Removes the "axis" input when it is zero.
class GatherTransformer : public Transformer {
 public:
  string Name() override { return "GatherTransformer"; }

  bool Transform(Flow *flow) override {
    // Remove the "axis" input when it is zero.
    bool transformed = false;
    for (Flow::Operation *op : flow->ops()) {
      if (op->type != "Gather") continue;  // types were normalized above

      // When present, the axis is the third argument.
      if (op->indegree() != 3) continue;
      Flow::Variable *axis = op->inputs.back();

      // The axis must be constant and zero.
      int32 axis32 = -1;
      if (!axis->GetData(&axis32)) continue;
      if (axis32 != 0) continue;

      // The axis will be pruned, so it should have no other dependencies.
      if (axis->usages() != 1) continue;
      if (axis->producer != nullptr) continue;

      op->RemoveInput(axis);
      flow->DeleteVariable(axis);
      transformed = true;
    }

    return transformed;
  }
};

// Type inference for standard ops.
class StandardTyper : public Typer {
 public:
  string Name() override { return "StandardTyper"; }

  bool InferTypes(Flow *flow, Flow::Operation *op) override {
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
        if (a->rank() == 2 && b->rank() == 2) {
          bool ta = op->GetAttr("transpose_a", false);
          bool tb = op->GetAttr("transpose_b", false);
          bool tc = op->GetAttr("transpose_c", false);
          int a_rows =  ta ? a->dim(1) : a->dim(0);
          int a_cols =  ta ? a->dim(0) : a->dim(1);
          int b_rows =  tb ? b->dim(1) : b->dim(0);
          int b_cols =  tb ? b->dim(0) : b->dim(1);
          if (a_cols == b_rows) {
            if (tc) {
              c->shape.assign(b_cols, a_rows);
            } else {
              c->shape.assign(a_rows, b_cols);
            }
            return true;
          }
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
    if (op->type == "Concat") {
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

    // Infer shape for identity operation.
    if (op->type == "Identity") {
      if (op->indegree() == 1 && op->outdegree() == 1) {
        Flow::Variable *input = op->inputs[0];
        Flow::Variable *output = op->outputs[0];
        if (output->shape.missing()) {
          output->shape = input->shape;
        }
        if (output->type == DT_INVALID) {
          output->type = input->type;
        }
        return true;
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

    // Infer shape for pooling gather operation.
    if (op->type == "GatherAvg" ||
        op->type == "GatherSum" ||
        op->type == "GatherMax") {
      if (op->indegree() == 2 && op->outdegree() == 1) {
        Flow::Variable *params = op->inputs[0];
        Flow::Variable *result = op->outputs[0];
        result->type = params->type;
        result->shape = params->shape;
        result->shape.set(0, 1);
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

    // Infer shape for reference operation.
    if (op->type == "Reference") {
      if (op->indegree() == 1 && op->outdegree() == 1) {
        Flow::Variable *ref = op->outputs[0];
        Flow::Variable *external = flow->Var(op->GetAttr("var"));
        ref->type = external->type;
        ref->shape = external->shape;
        ref->set_ref();
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

// Register generic transforms.
void RegisterGenericTransforms(Library *library) {
  library->RegisterTransformer(new RenameTransformer());
  library->RegisterTransformer(new IdentityTransformer());
  library->RegisterTransformer(new FlattenConcatTransformer());
  library->RegisterTransformer(new GatherTransformer());

  // Register type inference.
  library->RegisterTyper(new StandardTyper());
}

// Register generic library.
void RegisterGenericLibrary(Library *library) {
  library->Register(new Reference());
  RegisterArrayKernels(library);
  RegisterGenericMath(library);
  RegisterGenericMatMul(library);
  RegisterGenericOperators(library);
}

}  // namespace myelin
}  // namespace sling
