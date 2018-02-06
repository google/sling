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

#include "sling/myelin/kernel/arithmetic.h"

#include <map>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

#include "sling/base/logging.h"
#include "sling/base/types.h"
#include "sling/myelin/compute.h"
#include "sling/myelin/express.h"
#include "sling/myelin/macro-assembler.h"
#include "sling/myelin/generator/elementwise.h"
#include "sling/myelin/generator/expression.h"

#define __ masm->

namespace sling {
namespace myelin {

using namespace jit;

// Mapping from flow variables to expression variables.
typedef std::map<Flow::Variable *, Express::Var *> VarMap;

// Convert operation type to expression op.
static Express::OpType OpType(const string &op) {
  // Operations that can be fused into Calculate operations.
  static std::unordered_map<string, Express::OpType> ops {
    {"Add", Express::ADD},
    {"Sub", Express::SUB},
    {"Mul", Express::MUL},
    {"Div", Express::DIV},
    {"RealDiv", Express::DIV},
    {"Minimum", Express::MIN},
    {"Maximum", Express::MAX},

    {"Log", Express::LOG},
    {"Exp", Express::EXP},
    {"Sigmoid", Express::SIGMOID},
    {"Tanh", Express::TANH},

    {"Negate", Express::NEG},
    {"Abs", Express::ABS},
    {"Relu", Express::RELU},
    {"Softsign", Express::SOFTSIGN},
    {"Softplus", Express::SOFTPLUS},
    {"LogSigmoid", Express::LOGSIGMOID},
    {"Reciprocal", Express::RECIPROCAL},
    {"Square", Express::SQUARE},
  };

  auto f = ops.find(op);
  return f == ops.end() ? Express::INVALID : f->second;
}

// Check if operation is a candidate for Calculate ops.
static bool IsCalculateOp(Flow::Operation *op) {
  return op->type == "Calculate" || OpType(op->type) != Express::INVALID;
}

// Initialize expression for flow operation.
static void InitExpression(Flow::Operation *op, Express *expr, bool expand) {
  if (op->type == "Calculate") {
    // Build expression from expression recipe attribute on op.
    const string &recipe = op->GetAttr("expr");
    if (!recipe.empty()) expr->Parse(recipe, expand);
  } else {
    // Add op with inputs and output.
    CHECK_EQ(op->outdegree(), 1);
    std::vector<Express::Var *> args(op->indegree());
    for (int i = 0; i < op->indegree(); ++i) {
      args[i] = expr->Variable(Express::INPUT, i);
    }
    Express::Op *func = expr->Function(OpType(op->type), args, expand);
    func->Assign(expr->Variable(Express::OUTPUT, 0));
    expr->CompactTempVars();
  }

  // Mark constant inputs.
  for (int i = 0; i < op->indegree(); ++i) {
    auto *input = op->inputs[i];
    if (input->constant() && input->elements() == 1) {
      int const_id = -1;
      if (input->type == DT_FLOAT) {
        float value = *reinterpret_cast<const float *>(input->data);
        if (value == 0.0) {
          const_id = Express::ZERO;
        } else if (value == 1.0) {
          const_id = Express::ONE;
        } else if (value == 0.5) {
          const_id = Express::HALF;
        } else if (value == 2.0) {
          const_id = Express::TWO;
        } else if (value == -1.0) {
          const_id = Express::N1;
        }
      }
      auto *var = expr->Variable(Express::INPUT, i);
      if (const_id != -1) {
        var->type = Express::NUMBER;
        var->id = const_id;
      } else {
        var->type = Express::CONST;
      }
    }
  }
}

// Initialize expression for step.
void InitExpression(const Step *step, Express *expr, bool expand) {
  if (step->type() == "Calculate") {
    // Build expression from expression recipe attribute on op.
    const string &recipe = step->GetAttr("expr");
    if (!recipe.empty()) expr->Parse(recipe, expand);
  } else {
    // Add op with inputs and output.
    CHECK_EQ(step->outdegree(), 1);
    std::vector<Express::Var *> args(step->indegree());
    for (int i = 0; i < step->indegree(); ++i) {
      args[i] = expr->Variable(Express::INPUT, i);
    }
    Express::Op *func = expr->Function(OpType(step->type()), args, expand);
    func->Assign(expr->Variable(Express::OUTPUT, 0));
    expr->CompactTempVars();
  }

  // Mark constant inputs.
  for (int i = 0; i < step->indegree(); ++i) {
    if (step->input(i)->IsConstant() && step->input(i)->elements() == 1) {
      expr->Variable(Express::INPUT, i)->type = Express::CONST;
    }
  }
}

// Expression code generator for element-wise operations.
struct Expression {
  // Initialize expression.
  Expression(const Step *step, MacroAssembler *masm, int spare_regs = 0)
      : index(step, masm) {
    // Determine output type and shape from the first output.
    output = step->output(0);
    Type type = output->type();

    // Compute the maximum common size between inputs and outputs.
    DCHECK_GE(step->outdegree(), 1);
    int elements = output->elements();
    for (auto *input : step->inputs()) {
      if (input->IsConstant() && input->elements() == 1) continue;
      int common = output->shape().CommonSize(input->shape());
      if (common < elements) elements = common;
    }

    // Compile expression to be computed.
    InitExpression(step, &expr, true);

    // Select expression generator.
    generator = ExpressionGenerator::Select(expr, type, elements);
    CHECK(generator != nullptr);

    // Initialize expression and index generators.
    generator->Initalize(expr, type, spare_regs, &index);
  }

  ~Expression() { delete generator; }

  // Allocate registers.
  bool AllocateRegisters() {
    return index.AllocateRegisters();
  }

  // Generate code for expression loop.
  void Generate(MacroAssembler *masm) {
    generator->GenerateInit(masm);
    index.BeginLoop();
    generator->GenerateBody(masm);
    index.EndLoop();
  }

  // Compute complexity.
  int64 Complexity() {
    return output->shape().elements() * expr.Complexity();
  }

  // Representative output from expression.
  Tensor *output;

  // Expression to be compiled.
  Express expr;

  // Index generator for element-wise operation.
  ElementwiseIndexGenerator index;

  // Code generator for expression.
  ExpressionGenerator *generator;
};

// Convert division with constant c to multiplication with constant 1/c to
// take advantage of mul being much faster than div.
class DivToMulTransformer : public Transformer {
 public:
  bool Transform(Flow *flow) override {
    int updates = 0;
    for (Flow::Operation *op : flow->ops()) {
      // Look for Div(x, c) where c is a non-shared scalar float constant.
      if (op->type != "Div" && op->type != "RealDiv") continue;
      if (op->indegree() != 2) continue;
      Flow::Variable *second = op->inputs[1];
      if (second->type != DT_FLOAT || second->elements() != 1) continue;
      if (!second->constant() || second->consumers.size() != 1) continue;

      // Change Div(x,c) to Mul(x,1/c).
      CHECK_EQ(second->size, sizeof(float));
      op->type = "Mul";
      float multiplier = 1.0 / *reinterpret_cast<const float *>(second->data);
      char *buffer = flow->AllocateMemory(sizeof(float));
      *reinterpret_cast<float *>(buffer) = multiplier;
      second->data = buffer;
      updates++;
    }
    return updates > 0;
  }
};

// Combine arithmetic operators into expressions that can be computed by a
// Calculate kernel.
class ExpressionTransformer : public Transformer {
 public:
  bool Transform(Flow *flow) override {
    // Make list of ops that can potentially be included in Calculate ops.
    std::vector<Flow::Operation *> candidates;
    for (Flow::Operation *op : flow->ops()) {
      if (IsCalculateOp(op) && !op->GetAttr("strict", false)) {
        candidates.push_back(op);
      }
    }

    // Find candidate pairs to merge into combined Calculate ops.
    bool again = true;
    int num_combines = 0;
    while (again) {
      again = false;
      for (int i = 0; i < candidates.size(); ++i) {
        Flow::Operation *op = candidates[i];
        if (op == nullptr) continue;

        // Check if producer of one of the inputs is also a candidate.
        for (auto *input : op->inputs) {
          if (input->producer == nullptr) continue;
          if (!IsCalculateOp(input->producer)) continue;
          if (input->producer->GetAttr("strict", false)) continue;

          // Try to combine op with producer.
          if (Combine(flow, input->producer, op)) {
            // Remove op from candidate list and try again.
            candidates[i] = nullptr;
            num_combines++;
            again = true;
            break;
          }
        }
      }
    }
    VLOG(3) << num_combines << " of " << candidates.size() << " ops combined";

    return num_combines > 0;
  }

  bool Combine(Flow *flow, Flow::Operation *first, Flow::Operation *second) {
    // Check that ops have the same types and output shapes.
    if (first->indegree() < 1 || first->outdegree() < 1) return false;
    if (second->indegree() < 1 || second->outdegree() < 1) return false;
    Type type = first->outputs[0]->type;
    const Shape &shape = first->outputs[0]->shape;
    for (auto *input : first->inputs) {
      if (input->type != type) return false;
      if (!input->shape.defined()) return false;
      if (!input->shape.IsCompatible(shape)) return false;
    }
    for (auto *input : second->inputs) {
      if (input->type != type) return false;
      if (!input->shape.defined()) return false;
      if (!input->shape.IsCompatible(shape)) return false;
    }
    for (auto *output : first->outputs) {
      if (output->type != type) return false;
      if (!output->shape.defined()) return false;
      if (output->shape != shape) return false;
    }
    for (auto *output : second->outputs) {
      if (output->type != type) return false;
      if (!output->shape.defined()) return false;
      if (output->shape != shape) return false;
    }

    // Check for indirect dependencies between ops.
    for (auto *v : second->inputs) {
      if (v->producer != first && v->DependsOn(first)) return false;
    }

    // Compute fused expression.
    string fused_recipe = FuseExpressions(first, second);

    // Fuse the two ops and set expression recipe for the fused Calculate op.
    Flow::Operation *fused = flow->Fuse(first, second, "Calculate", true);
    fused->SetAttr("expr", fused_recipe);

    return true;
  }

  string FuseExpressions(Flow::Operation *first, Flow::Operation *second) {
    // Build first expression.
    Express expr1;
    InitExpression(first, &expr1, false);
    VarMap vars1;
    MapVars(first, &expr1, &vars1);

    // Build second expression.
    Express expr2;
    InitExpression(second, &expr2, false);
    VarMap vars2;
    MapVars(second, &expr2, &vars2);

    // Build expression variable mapping for mapping variables in the second
    // expression to variables in the first expression.
    Express::Map mapping;
    int next_input = first->inputs.size();
    int next_output = first->outputs.size();
    for (Flow::Variable *v : second->inputs) {
      if (first->IsInput(v)) {
        // Map input from second op to input from first op.
        mapping[vars2[v]] = vars1[v];
      } else if (first->IsOutput(v)) {
        if (v->consumers.size() == 1 && !v->out) {
          // Second op is the only consumer of the output from the first op,
          // so the input can be turned into a temporary variable.
          int id = vars1[v]->id;
          vars1[v]->type = Express::TEMP;
          vars1[v]->id = -1;

          // Adjust numbering of output variables from the first op.
          next_output--;
          for (auto *o : expr1.vars()) {
            if (o->type == Express::OUTPUT && o->id > id) {
              o->id--;
            }
          }
        }

        // Map input from second op to output from first op.
        mapping[vars2[v]] = vars1[v];
      } else {
        // Map input from second op to a new input in the merged expression.
        mapping[vars2[v]] = expr1.Variable(InputType(v), next_input++);
      }
    }
    for (Flow::Variable *v : second->outputs) {
      // Map output from second op to a new output in the merged expression.
      mapping[vars2[v]] = expr1.Variable(Express::OUTPUT, next_output++);
    }
    expr1.CompactTempVars();
    expr2.CompactTempVars();

    // Merge second expression into the first one.
    expr1.Merge(&expr2, mapping);

    // Return merged recipe.
    return expr1.AsRecipe();
  }

  // Build mapping from flow variables to expression variables.
  static void MapVars(Flow::Operation *op, Express *expr, VarMap *varmap) {
    // Map input variables.
    for (int i = 0; i < op->indegree(); ++i) {
      (*varmap)[op->inputs[i]] = expr->Variable(InputType(op->inputs[i]), i);
    }

    // Map output variables.
    for (int i = 0; i < op->outdegree(); ++i) {
      (*varmap)[op->outputs[i]] = expr->Variable(Express::OUTPUT, i);
    }
  }

  // Determine input variable type.
  static Express::VarType InputType(Flow::Variable *var) {
    if (var->constant() && var->elements() == 1) {
      return Express::CONST;
    } else {
      return Express::INPUT;
    }
  }
};

// Kernel for computing arithmetic expressions.
class Calculate : public Kernel {
 public:
  Calculate(const string &name, const string &operation, int arity = -1)
      : name_(name), operation_(operation), arity_(arity) {}

  string Name() override { return name_; }
  string Operation() override { return operation_; }

  bool Supports(Step *step) override {
    // Check that operation is compatible.
    if (step->type() != operation_) return false;
    if (arity_ != -1 && step->indegree() != arity_) return false;

    // Check that inputs and outputs have the compatible types and shapes.
    if (step->indegree() < 1 || step->outdegree() < 1) return false;
    Type type = step->output(0)->type();
    const Shape &shape = step->output(0)->shape();
    for (auto *input : step->inputs()) {
      if (input->type() != type) return false;
      if (!input->Compatible(step->output(0))) return false;
    }
    for (auto *output : step->outputs()) {
      if (output->type() != type) return false;
      if (output->shape() != shape) return false;
    }

    // Strict math not supported.
    if (step->GetAttr("strict", false)) return false;

    // Dense encoding required.
    for (auto *input : step->inputs()) input->RequireDense();
    for (auto *output : step->outputs()) output->RequireDense();

    return true;
  }

  void Adjust(Step *step) override {
    Expression expression(step, nullptr);
    step->set_variant(expression.generator->Name());

    // Set alignment.
    int alignment = expression.generator->VectorSize();
    for (auto *input : step->inputs()) {
      input->SetMiniumAlignment(alignment);
      input->RequireDense();
      input->RequireStandardOrder();
    }
    for (auto *output : step->outputs()) {
      output->SetMiniumAlignment(alignment);
      output->RequireDense();
      output->RequireStandardOrder();
    }

    // Enable sharing of inputs and outputs.
    for (int i = 0; i < step->indegree(); ++i) {
      for (int j = 0; j < step->outdegree(); ++j) {
        if (step->input(i)->shape() == step->output(j)->shape()) {
          if (step->AllowInPlace(i, j)) break;
        }
      }
    }
  }

  void Generate(Step *step, MacroAssembler *masm) override {
    // Check how many spare register we have for hoisting constant out of the
    // loop body. This is only done for floating-point operations to avoid
    // register pressure on the regular x64 integer registers which are also
    // used for the loop indexing.
    int spare_regs = 0;
    Type type = step->output(0)->type();
    if (type == DT_FLOAT || type == DT_DOUBLE) {
      // Perform dry-run to estimate the number of SIMD registers needed.
      MacroAssembler dryrun_masm(nullptr, 0, masm->options());
      Expression dryrun_expr(step, &dryrun_masm, 0);
      CHECK(dryrun_expr.AllocateRegisters()) << "Register overflow";

      // Count the number of spare SIMD registers.
      if (!dryrun_expr.index.single()) {
        while (dryrun_masm.mm().try_alloc() != -1) spare_regs++;
      }
    }

    // Generate code for element-wise expression evaluation.
    Expression expression(step, masm, spare_regs);
    CHECK(expression.AllocateRegisters()) << "Register overflow";
    expression.Generate(masm);
  }

  int64 Complexity(const Step *step) override {
    Expression expression(step, nullptr);
    return expression.Complexity();
  }

 private:
  string name_;       // kernel name
  string operation_;  // kernel operation
  int arity_;         // number of inputs
};

// Register arithmetic library.
void RegisterArithmeticLibrary(Library *library) {
  library->Register(new Calculate("AddExpr", "Add", 2));
  library->Register(new Calculate("SubExpr", "Sub", 2));
  library->Register(new Calculate("MulExpr", "Mul", 2));
  library->Register(new Calculate("DivExpr", "Div", 2));
  library->Register(new Calculate("MaxExpr", "Maximum", 2));
  library->Register(new Calculate("MinExpr", "Minimum", 2));

  library->Register(new Calculate("LogExpr", "Log", 1));
  library->Register(new Calculate("ExpExpr", "Exp", 1));
  library->Register(new Calculate("SigmoidExpr", "Sigmoid", 1));
  library->Register(new Calculate("TanhExpr", "Tanh", 1));
  library->Register(new Calculate("Calculate", "Calculate"));

  library->Register(new Calculate("NegateExpr", "Negate", 1));
  library->Register(new Calculate("AbsExpr", "Abs", 1));
  library->Register(new Calculate("ReluExpr", "Relu", 1));
  library->Register(new Calculate("SoftsignExpr", "Softsign", 1));
  library->Register(new Calculate("SoftplusExpr", "Softplus", 1));
  library->Register(new Calculate("LogSigmoidExpr", "LogSigmoid", 1));
  library->Register(new Calculate("ReciprocalExpr", "Reciprocal", 1));
  library->Register(new Calculate("SquareExpr", "Square", 1));
}

// Register arithmetic transforms.
void RegisterArithmeticTransforms(Library *library) {
  library->RegisterTransformer(new ExpressionTransformer());
  library->RegisterTransformer(new DivToMulTransformer());
}

}  // namespace myelin
}  // namespace sling

