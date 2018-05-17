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

#include <math.h>
#include <map>
#include <set>
#include <string>
#include <unordered_map>
#include <utility>
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

    {"Neg", Express::NEG},
    {"Abs", Express::ABS},
    {"Relu", Express::RELU},
    {"ReluGrad", Express::RELUGRAD},
    {"Softsign", Express::SOFTSIGN},
    {"Softplus", Express::SOFTPLUS},
    {"LogSigmoid", Express::LOGSIGMOID},
    {"Reciprocal", Express::RECIPROCAL},
    {"Square", Express::SQUARE},
    {"Sqrt", Express::SQRT},
  };

  auto f = ops.find(op);
  return f == ops.end() ? Express::INVALID : f->second;
}

// Check if operation is a candidate for Calculate ops.
static bool IsCalculateOp(Flow::Operation *op) {
  return op->type == "Calculate" || OpType(op->type) != Express::INVALID;
}

// Check if operation is an assignment op.
static bool IsAssignmentOp(Flow::Operation *op) {
  return op->type == "Assign";
}

// Initialize expression for flow operation.
static void InitExpression(Flow::Operation *op, Express *expr, bool expand) {
  if (op->type == "Calculate") {
    // Build expression from expression recipe attribute on op.
    const string &recipe = op->GetAttr("expr");
    if (!recipe.empty()) expr->Parse(recipe, expand);
  } else if (op->type == "Assign") {
    const string &recipe = op->GetAttr("expr");
    expr->Parse(recipe.empty() ? "@0=Id(%1)" : recipe, expand);
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

  // Mark constant and scalar inputs.
  for (int i = 0; i < op->indegree(); ++i) {
    auto *input = op->inputs[i];
    if (input->elements() == 1) {
      int const_id = -1;
      if (input->constant()) {
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
      }
      auto *var = expr->Variable(Express::INPUT, i);
      if (const_id != -1) {
        var->type = Express::NUMBER;
        var->id = const_id;
      } else if (input->constant()) {
        var->type = Express::CONST;
      } else {
        var->single = true;
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
  } else if (step->type() == "Assign") {
    const string &recipe = step->GetAttr("expr");
    expr->Parse(recipe.empty() ? "@0=Id(%1)" : recipe, expand);
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

  // Mark scalar and constant inputs.
  for (int i = 0; i < step->indegree(); ++i) {
    if (step->input(i)->elements() == 1) {
      Express::Var *var = expr->Variable(Express::INPUT, i);
      if (step->input(i)->constant()) {
        var->type = Express::CONST;
      } else {
        var->single = true;
      }
    }
  }
}

// Expression code generator for element-wise operations.
struct Expression {
  // Initialize expression.
  Expression(const Step *step, MacroAssembler *masm, int spare_regs = 0)
      : index(step, masm) {
    // Determine output type and shape from the first output (or first input
    // for assignment op).
    assign = step->type() == "Assign";
    prototype = assign ? step->input(0) : step->output(0);
    Type type = prototype->type();

    // Compute the maximum common size between inputs and outputs. Scalars are
    // not used for computing the maximum size since these can be broadcast to
    // the vector size.
    int elements = prototype->elements();
    for (auto *input : step->inputs()) {
      if (input->elements() == 1) continue;
      int common = prototype->shape().CommonSize(input->shape());
      if (common < elements) elements = common;
    }

    // Compile expression to be computed.
    InitExpression(step, &expr, true);

    // Clear single flag for scalar ops since broadcasting and hoisting is not
    // needed in this case.
    if (elements == 1) {
      for (auto *v : expr.vars()) v->single = false;
    }

    // Select expression generator.
    generator = ExpressionGenerator::Select(expr, type, elements);
    CHECK(generator != nullptr);

    // Initialize expression and index generators.
    generator->Initialize(expr, type, spare_regs, &index);
  }

  ~Expression() { delete generator; }

  // Allocate registers.
  bool AllocateRegisters() {
    return index.AllocateRegisters();
  }

  // Generate code for expression loop.
  void Generate(MacroAssembler *masm) {
    index.GenerateInit();
    generator->GenerateInit(masm);
    index.GenerateLoopBegin();
    generator->GenerateBody(masm);
    index.GenerateLoopEnd();
  }

  // Compute complexity.
  int64 Complexity() {
    return prototype->shape().elements() * expr.Complexity();
  }

  // Compute how many spare register we have for hoisting constant out of the
  // loop body. This is only done for floating-point operations to avoid
  // register pressure on the regular x64 integer registers which are also
  // used for the loop indexing.
  static int SpareRegs(const Step *step, const Options &options) {
    int spare_regs = 0;
    bool assign = step->type() == "Assign";
    Type type = assign ? step->input(0)->type() : step->output(0)->type();
    if (type == DT_FLOAT || type == DT_DOUBLE) {
      // Perform dry-run to estimate the number of SIMD registers needed.
      MacroAssembler masm(nullptr, 0, options);
      Expression expr(step, &masm, 0);
      CHECK(expr.AllocateRegisters()) << "Register overflow";

      // Count the number of spare SIMD registers.
      bool extended = expr.index.extended_regs();
      while (masm.mm().try_alloc(extended) != -1) spare_regs++;
    }
    return spare_regs;
  }

  // Representative output from expression.
  Tensor *prototype;

  // Expression to be compiled.
  Express expr;

  // Index generator for element-wise operation.
  ElementwiseIndexGenerator index;

  // Code generator for expression.
  ExpressionGenerator *generator;

  // Assignment expression.
  bool assign;
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
      if (!second->constant() || second->usages() != 1) continue;

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

// Convert addition where last term is negated to subtraction.
class AddNegToSubTransformer : public Transformer {
 public:
  bool Transform(Flow *flow) override {
    int updates = 0;
    for (Flow::Operation *op : flow->Find("Neg|1:Add")) {
      Flow::Operation *add = op;
      Flow::Operation *neg = add->inputs[1]->producer;
      if (neg->outputs[0]->usages() == 1) {
        flow->Eliminate(neg);
        add->type = "Sub";
        updates++;
      }
    }
    return updates > 0;
  }
};

// Combine arithmetic operators into expressions that can be computed by a
// Calculate kernel.
class ExpressionTransformer : public Transformer {
 public:
  bool Transform(Flow *flow) override {
    // Make list of ops that can potentially be included in Calculate or
    // Assign op merging.
    std::vector<Flow::Operation *> candidates;
    for (Flow::Operation *op : flow->ops()) {
      if (IsCalculateOp(op) || IsAssignmentOp(op)) {
        if (!op->GetAttr("strict", false)) {
          candidates.push_back(op);
        }
      }
    }

    // Merge calculate ops into assignment.
    int num_combines = 0;
    bool again = true;
    while (again) {
      again = false;
      for (int i = 0; i < candidates.size(); ++i) {
        Flow::Operation *op = candidates[i];
        if (op == nullptr) continue;
        if (!IsAssignmentOp(op)) continue;

        // Check if producer of one of the inputs is a calculate op.
        for (auto *input : op->inputs) {
          Flow::Operation *producer = input->producer;
          if (producer == nullptr) continue;
          if (!IsCalculateOp(producer)) continue;
          if (producer->GetAttr("strict", false)) continue;

          // Assignment must be the sole consumer of all the outputs from the
          // producer.
          bool contained = true;
          for (auto *v : producer->outputs) {
            if (v->usages() != 1 ||v->consumers[0] != op || v->out()) {
              contained = false;
              break;
            }
          }
          if (!contained) continue;

          // Try to combine op with producer.
          if (Combine(flow, producer, op)) {
            // Remove op from candidate list and try again.
            candidates[i] = nullptr;
            num_combines++;
            again = true;
            break;
          }
        }
      }
    }

    // Merge calculate ops.
    again = true;
    while (again) {
      again = false;
      // Merge calculate ops.
      for (int i = 0; i < candidates.size(); ++i) {
        Flow::Operation *op = candidates[i];
        if (op == nullptr) continue;
        if (!IsCalculateOp(op)) continue;

        // Check if producer of one of the inputs is also a candidate.
        for (auto *input : op->inputs) {
          Flow::Operation *producer = input->producer;
          if (producer == nullptr) continue;
          if (!IsCalculateOp(producer)) continue;
          if (producer->GetAttr("strict", false)) continue;

          // Try to combine op with producer.
          if (Combine(flow, producer, op)) {
            // Remove op from candidate list and try again.
            candidates[i] = nullptr;
            num_combines++;
            again = true;
            break;
          }
        }
      }
    }

    return num_combines > 0;
  }

  bool Combine(Flow *flow, Flow::Operation *first, Flow::Operation *second) {
    // Check that ops have the same types and output shapes.
    bool assign = IsAssignmentOp(second);
    if (first->indegree() < 1) return false;
    if (first->outdegree() < 1) return false;
    if (second->indegree() < 1) return false;
    if (!assign && second->outdegree() < 1) return false;
    Flow::Variable *prototype = assign ? first->inputs[0] : first->outputs[0];
    Type type = prototype->type;
    const Shape &shape = prototype->shape;
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
    Flow::Variable *target = assign ? second->inputs[0] : nullptr;
    Flow::Operation *fused = flow->Fuse(first, second,
                                        assign ? "Assign" : "Calculate",
                                        true);

    // Make sure that the assignment target is still the first input to the
    // combined op.
    if (assign && fused->inputs[0] != target) {
      // Get the input index of the target variable.
      int target_index = fused->InputIndex(target);
      CHECK(target_index != -1);

      // Swap target variable with first input.
      Express expr;
      expr.Parse(fused_recipe, false);
      auto *vt = expr.Variable(Express::INPUT, target_index);
      auto *v0 = expr.Variable(Express::INPUT, 0);
      vt->id = 0;
      v0->id = target_index;
      fused_recipe = expr.AsRecipe();
      std::swap(fused->inputs[0], fused->inputs[target_index]);
    }

    // Set fused expression for combined op.
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
    bool assign = IsAssignmentOp(second);
    Express expr2;
    InitExpression(second, &expr2, false);
    VarMap vars2;
    MapVars(second, &expr2, &vars2);

    // Build expression variable mapping for mapping variables in the second
    // expression to variables in the first expression.
    Express::Map mapping;
    int next_input = first->inputs.size();
    int next_output = first->outputs.size();
    if (assign && second->outdegree() == 0) {
      // Add implicit output for assignment target.
      Express::Var *v2 = expr2.Variable(Express::OUTPUT, 0);
      Express::Var *v1 = expr1.Variable(Express::OUTPUT, next_output++);
      mapping[v2] = v1;
    }
    for (Flow::Variable *v : second->inputs) {
      if (first->IsInput(v)) {
        // Map input from second op to input from first op.
        mapping[vars2[v]] = vars1[v];
      } else if (first->IsOutput(v)) {
        if (v->usages() == 1 && !v->out()) {
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

// Eliminate unused inputs to calculate ops. These are usually constants that
// have been replaced with system constants.
class RemoveUnusedInputs : public Transformer {
 public:
  bool Transform(Flow *flow) override {
    int num_eliminates = 0;
    for (Flow::Operation *op : flow->ops()) {
      bool calculate = op->type == "Calculate";
      bool assign = op->type == "Assign";
      if (calculate || assign) {
        Express expr;
        InitExpression(op, &expr, false);
        for (int i = 0; i < op->inputs.size(); ++i) {
          if (expr.Lookup(Express::INPUT, i) == nullptr &&
              expr.Lookup(Express::CONST, i) == nullptr) {
            if (assign && i == 0) continue;
            expr.EliminateInput(i);
            op->RemoveInput(op->inputs[i]);
            op->SetAttr("expr", expr.AsRecipe());
            num_eliminates++;
            break;
          }
        }
      }
    }

    return num_eliminates > 0;
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

    // Check that inputs and outputs have compatible types and shapes.
    bool assign = step->type() == "Assign";
    if (step->indegree() < 1) return false;
    if (!assign && step->outdegree() < 1) return false;
    Tensor *prototype = assign ? step->input(0) : step->output(0);
    Type type = prototype->type();
    const Shape &shape = prototype->shape();
    for (auto *input : step->inputs()) {
      if (input->type() != type) return false;
      if (!input->Compatible(prototype)) return false;
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

    if (step->type() == "Assign") {
      // Link output reference to assignment target.
      if (step->outdegree() == 1) {
        step->input(0)->Link(step->output(0));
      }
    } else {
      // Enable sharing of inputs and outputs.
      expression.expr.ComputeLiveRanges();
      for (int i = 0; i < step->indegree(); ++i) {
        Tensor *input = step->input(i);
        Express::Var *ivar = expression.expr.Lookup(Express::INPUT, i);
        if (ivar == nullptr) continue;

        for (int j = 0; j < step->outdegree(); ++j) {
          Tensor *output = step->output(j);
          Express::Var *ovar = expression.expr.Lookup(Express::OUTPUT, j);
          if (ovar == nullptr) continue;

          // The input and output can be shared if they have the same format and
          // their live ranges do not overlap.
          if (input->shape() == output->shape() && !ivar->overlaps(ovar)) {
            if (step->AllowInPlace(i, j)) {
              break;
            }
          }
        }
      }
    }
  }

  void Generate(Step *step, MacroAssembler *masm) override {
    // Generate code for element-wise expression evaluation.
    int spare_regs = Expression::SpareRegs(step, masm->options());
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

// Kernel for computing softmax or log-softmax.
class Softmax : public Kernel {
 public:
  Softmax(bool log) : log_(log) {}

  string Name() override { return log_ ? "LogSoftmax" : "Softmax"; }
  string Operation() override { return log_ ? "LogSoftmax" : "Softmax"; }

  bool Supports(Step *step) override {
    // Requires SSE or AVX support.
    if (!CPU::Enabled(AVX) && !CPU::Enabled(SSE)) return false;

    // Check inputs and outputs.
    if (step->indegree() != 1 || step->outdegree() != 1) return false;
    Tensor *x = step->input(0);
    Tensor *y = step->output(0);

    // Check type.
    if (x->type() != DT_FLOAT) return false;
    if (y->type() != DT_FLOAT) return false;

    // Input and output must have same shape.
    if (!x->HasSameShape(y)) return false;

    return true;
  }

  void Adjust(Step *step) override {
    Tensor *x = step->input(0);
    Tensor *y = step->output(0);
    int align = 16;
    if (CPU::Enabled(AVX)) align = 32;
    if (CPU::Enabled(AVX512F)) align = 64;
    x->SetMiniumAlignment(align);
    x->RequireDense();
    x->RequireStandardOrder();
    y->SetMiniumAlignment(align);
    y->RequireDense();
    y->RequireStandardOrder();
    step->AllowInPlace(0, 0);
  }

  void Generate(Step *step, MacroAssembler *masm) override {
    // Get input and output.
    Tensor *x = step->input(0);
    Tensor *y = step->output(0);
    int n = y->elements();

    // Compile expression for preprocessing.
    Express preexpr;
    preexpr.Parse("!0=Id(%0)", true);

    // Compile expression for computing y=exp(x-max(x)) storing the result in an
    // intermediate register.
    Express expr;
    expr.Parse("!1=Exp(Sub(%0,!0));@0=Id(!1)", true);

    // Compile expression for post-processing.
    Express postexpr;
    postexpr.Parse(log_ ? "@0=Log(Mul(%0,!0))" : "@0=Mul(%0,!0)", true);

    // Determine vector size for main block computation.
    int vecsize = 1;
    if (masm->Enabled(AVX512F) && n >= 16) {
      vecsize = 16;
    } else if (masm->Enabled(AVX) && n >= 8) {
      vecsize = 8;
    } else if (masm->Enabled(SSE) && n >= 4) {
      vecsize = 4;
    }
    int m = (n / vecsize) * vecsize;
    int r = vecsize == 1 ? n : n % vecsize;

    // Allocate registers.
    Register input = masm->rr().alloc();
    Register output = masm->rr().alloc();
    Register offset = masm->rr().alloc();
    ZMMRegister sum = masm->mm().allocz(false);
    ZMMRegister elem = masm->mm().allocz(false);
    ZMMRegister max = masm->mm().allocz(false);
    SIMDRegisters basemm = masm->mm();

    // Load tensor locations.
    __ LoadTensorAddress(input, x);
    if (y->SharedWith(x)) {
      output = input;
    } else {
      __ LoadTensorAddress(output, y);
    }

    // For numerical stability we compute softmax(x)=softmax(x-max(x)), so first
    // we find the maximum input element. Initialize max to -inf.
    float neginf = -INFINITY;
    if (masm->Enabled(AVX512F)) {
      __ vmovaps(max, masm->GetConstant(neginf, 16)->address());
    } else if (masm->Enabled(AVX)) {
      __ vmovaps(max.ymm(), masm->GetConstant(neginf, 8)->address());
    } else {
      __ movaps(max.xmm(), masm->GetConstant(neginf, 4)->address());
    }

    // Find max element for main block.
    __ xorq(offset, offset);
    if (vecsize > 1) {
      UnaryExpression expression(preexpr, masm, input, jit::no_reg, offset, m);

      // Loop over all main elements.
      expression.generator->GenerateInit(masm);
      Label l;
      __ bind(&l);

      // Find max element for next block.
      expression.generator->GenerateBody(masm);
      if (masm->Enabled(AVX)) {
        __ vmaxps(max.ymm(), max.ymm(), expression.ymm(0));
      } else {
        __ maxps(max.xmm(), expression.xmm(0));
      }

      if (m > vecsize || r > 0) {
        __ addq(offset, Immediate(vecsize * sizeof(float)));
      }
      if (m > vecsize) {
        __ cmpq(offset, Immediate(m * sizeof(float)));
        __ j(less, &l);
      }

      // Reduce max element vector.
      if (masm->Enabled(AVX)) {
        if (vecsize > 8) {
          __ vshuff32x4(elem, max, max, 0x0E);
          __ vmaxps(max, max, elem);
        }
        if (vecsize > 4) {
          __ vperm2f128(elem.ymm(), max.ymm(), max.ymm(), 1);
          __ vmaxps(max.ymm(), max.ymm(), elem.ymm());
        }
        __ vpermilps(elem.ymm(), max.ymm(), 0x0E);
        __ vmaxps(max.ymm(), max.ymm(), elem.ymm());
        __ vpermilps(elem.ymm(), max.ymm(), 0x01);
        __ vmaxps(max.ymm(), max.ymm(), elem.ymm());
      } else {
        __ shufps(elem.xmm(), max.xmm(), 0x0E);
        __ maxps(max.xmm(), elem.xmm());
        __ shufps(elem.xmm(), max.xmm(), 0x01);
        __ maxps(max.xmm(), elem.xmm());
      }

      masm->mm() = basemm;
    }

    // Find max element for residual block.
    if (r > 0) {
      UnaryExpression expression(preexpr, masm, input, jit::no_reg, offset, 1);

      // Loop over all residual elements.
      expression.generator->GenerateInit(masm);
      Label l;
      __ bind(&l);
      expression.generator->GenerateBody(masm);
      if (masm->Enabled(AVX)) {
        __ vmaxss(max.ymm(), max.ymm(), expression.ymm(0));
      } else {
        __ maxss(max.xmm(), expression.xmm(0));
      }

      if (r > 1) {
        __ addq(offset, Immediate(sizeof(float)));
        __ cmpq(offset, Immediate(n * sizeof(float)));
        __ j(less, &l);
      }

      masm->mm() = basemm;
    }

    // Clear sum register.
    if (masm->Enabled(AVX512F)) {
      __ vxorps(sum, sum, sum);
    } else if (masm->Enabled(AVX)) {
      __ vxorps(sum.ymm(), sum.ymm(), sum.ymm());
    } else {
      __ xorps(sum.xmm(), sum.xmm());
    }

    // Compute exp(x) for main block.
    __ xorq(offset, offset);
    if (vecsize > 1) {
      UnaryExpression expression(expr, masm, input, output, offset, m);

      // Broadcast max value over vector.
      ZMMRegister i0 = expression.zmm(0);
      if (masm->Enabled(AVX512F)) {
        __ vbroadcastss(i0, max);
      } else if (masm->Enabled(AVX2)) {
        __ vbroadcastss(i0.ymm(), max.ymm());
      } else if (masm->Enabled(AVX)) {
        __ vshufps(i0.ymm(), max.ymm(), max.ymm(), 0);
        __ vperm2f128(i0.ymm(), i0.ymm(), i0.ymm(), 0);
      } else {
        __ movaps(i0.xmm(), max.xmm());
        __ shufps(i0.xmm(), i0.xmm(), 0);
      }

      // Loop over all main elements.
      expression.generator->GenerateInit(masm);
      Label l;
      __ bind(&l);

      // Compute exp(x) for the next block.
      expression.generator->GenerateBody(masm);

      // Sum up results.
      if (masm->Enabled(AVX512F)) {
        __ vaddps(sum, sum, expression.zmm(1));
      } else if (masm->Enabled(AVX)) {
        __ vaddps(sum.ymm(), sum.ymm(), expression.ymm(1));
      } else {
        __ addps(sum.xmm(), expression.xmm(1));
      }

      if (m > vecsize || r > 0) {
        __ addq(offset, Immediate(vecsize * sizeof(float)));
      }
      if (m > vecsize) {
        __ cmpq(offset, Immediate(m * sizeof(float)));
        __ j(less, &l);
      }

      // Reduce sum.
      if (masm->Enabled(AVX)) {
        if (vecsize > 8) {
          __ vshuff32x4(elem, sum, sum, 0x0E);
          __ vaddps(sum, sum, elem);
        }
        if (vecsize > 4) {
          __ vperm2f128(elem.ymm(), sum.ymm(), sum.ymm(), 1);
          __ vhaddps(sum.ymm(), sum.ymm(), elem.ymm());
        }
        __ vhaddps(sum.ymm(), sum.ymm(), sum.ymm());
        __ vhaddps(sum.ymm(), sum.ymm(), sum.ymm());
      } else {
        __ haddps(sum.xmm(), sum.xmm());
        __ haddps(sum.xmm(), sum.xmm());
      }

      masm->mm() = basemm;
    }

    // Compute exp(x) for residual block.
    if (r > 0) {
      UnaryExpression expression(expr, masm, input, output, offset, 1);

      // Loop over all residual elements.
      if (masm->Enabled(AVX)) {
        __ vmovaps(expression.ymm(0), max.ymm());
      } else {
        __ movaps(expression.xmm(0), max.xmm());
      }
      expression.generator->GenerateInit(masm);
      Label l;
      __ bind(&l);

      // Compute exp(x) for the next element in residual block.
      expression.generator->GenerateBody(masm);

      // Sum up results.
      if (masm->Enabled(AVX)) {
        __ vaddss(sum.ymm(), sum.ymm(), expression.ymm(1));
      } else {
        __ addss(sum.xmm(), expression.xmm(1));
      }

      if (r > 1) {
        __ addq(offset, Immediate(sizeof(float)));
        __ cmpq(offset, Immediate(n * sizeof(float)));
        __ j(less, &l);
      }

      masm->mm() = basemm;
    }

    // Compute 1/sum for normalization. Multiplication is faster than division.
    if (masm->Enabled(AVX512F)) {
      __ vrcpss(sum.xmm(), sum.xmm(), sum.xmm());
      __ vbroadcastss(sum, sum);
    } else if (masm->Enabled(AVX)) {
      __ vrcpss(sum.xmm(), sum.xmm(), sum.xmm());
      if (masm->Enabled(AVX2)) {
        __ vbroadcastss(sum.ymm(), sum.ymm());
      } else {
        __ vshufps(sum.ymm(), sum.ymm(), sum.ymm(), 0);
        __ vperm2f128(sum.ymm(), sum.ymm(), sum.ymm(), 0);
      }
    } else {
      __ rcpss(sum.xmm(), sum.xmm());
      __ shufps(sum.xmm(), sum.xmm(), 0);
    }

    // Normalize output for main block.
    __ xorq(offset, offset);
    if (vecsize > 1) {
      UnaryExpression expression(postexpr, masm, output, output, offset, m);

      // Load scaling factor.
      if (masm->Enabled(AVX512F)) {
        __ vmovaps(expression.zmm(0), sum);
      } else if (masm->Enabled(AVX)) {
        __ vmovaps(expression.ymm(0), sum.ymm());
      } else {
        __ movaps(expression.xmm(0), sum.xmm());
      }
      expression.generator->GenerateInit(masm);

      // Loop over main elements.
      Label l;
      __ bind(&l);

      // Compute normalization for the next block.
      expression.generator->GenerateBody(masm);

      if (m > vecsize || r > 0) {
        __ addq(offset, Immediate(vecsize * sizeof(float)));
      }
      if (m > vecsize) {
        __ cmpq(offset, Immediate(m * sizeof(float)));
        __ j(less, &l);
      }

      masm->mm() = basemm;
    }

    // Normalize output for residual block.
    if (r > 0) {
      UnaryExpression expression(postexpr, masm, output, output, offset, 1);

      // Load scaling factor.
      if (masm->Enabled(AVX)) {
        __ vmovaps(expression.ymm(0), sum.ymm());
      } else {
        __ movaps(expression.xmm(0), sum.xmm());
      }
      expression.generator->GenerateInit(masm);

      // Loop over all residual elements.
      expression.generator->GenerateInit(masm);
      Label l;
      __ bind(&l);

      // Compute normalization for the next residual element.
      expression.generator->GenerateBody(masm);

      if (r > 1) {
        __ addq(offset, Immediate(sizeof(float)));
        __ cmpq(offset, Immediate(n * sizeof(float)));
        __ j(less, &l);
      }
      masm->mm() = basemm;
    }
  }

  int64 Complexity(const Step *step) override {
    int ops = 30;
    if (log_) ops += 42;
    return step->input(0)->elements() * ops + 10;
  }

 private:
  // Index generator for unary function.
  class UnaryIndexGenerator : public IndexGenerator {
   public:
    UnaryIndexGenerator(MacroAssembler *masm,
                        Register input,
                        Register output,
                        Register offset)
        : IndexGenerator(masm),
          input_(input), output_(output), offset_(offset) {}

    void Initialize(size_t vecsize) override {
      vecsize_ = vecsize;
    }

    jit::Operand addr(Express::Var *var) override {
      if (var->type == Express::NUMBER) {
        float number = Express::NumericFlt32(var->id);
        int repeat = vecsize_ / sizeof(float);
        return masm_->GetConstant(number, repeat)->address();
      } else if (var->type == Express::INPUT) {
        return Operand(input_, offset_);
      } else if (var->type == Express::OUTPUT) {
        return Operand(output_, offset_);
      } else {
        LOG(INFO) << var->AsString();
        UNSUPPORTED;
        return Operand(no_reg);
      }
    }

    const void *data(Express::Var *var) override {
      UNSUPPORTED;
      return nullptr;
    }

    int vecsize() const { return vecsize_; }

   private:
    Register input_;    // input base register
    Register output_;   // output base register
    Register offset_;   // displacement register
    int vecsize_;       // vector size
  };

  // Unary Expression.
  struct UnaryExpression {
    UnaryExpression(const Express &expr,
                    MacroAssembler *masm,
                    Register input,
                    Register output,
                    Register offset,
                    int size)
        : index(masm, input, output, offset) {
      generator = ExpressionGenerator::Select(expr, DT_FLOAT, size);
      CHECK(generator != nullptr);
      generator->Initialize(expr, DT_FLOAT, 0, &index);
      CHECK(index.AllocateRegisters()) << "Register overflow";
    }

    ~UnaryExpression() { delete generator; }

    // Get register for register-based variable.
    int reg(int id) { return generator->RegisterNumber(Express::REGISTER, id); }
    YMMRegister ymm(int id) { return index.ymm(reg(id)); }
    XMMRegister xmm(int id) { return index.xmm(reg(id)); }
    ZMMRegister zmm(int id) { return index.zmm(reg(id)); }

    UnaryIndexGenerator index;
    ExpressionGenerator *generator;
  };

  bool log_; // log(softmax(x)) vs. softmax(x)
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
  library->Register(new Calculate("Assign", "Assign"));

  library->Register(new Calculate("NegExpr", "Neg", 1));
  library->Register(new Calculate("AbsExpr", "Abs", 1));
  library->Register(new Calculate("ReluExpr", "Relu", 1));
  library->Register(new Calculate("ReluGradExpr", "ReluGrad", 2));
  library->Register(new Calculate("SoftsignExpr", "Softsign", 1));
  library->Register(new Calculate("SoftplusExpr", "Softplus", 1));
  library->Register(new Calculate("LogSigmoidExpr", "LogSigmoid", 1));
  library->Register(new Calculate("ReciprocalExpr", "Reciprocal", 1));
  library->Register(new Calculate("SquareExpr", "Square", 1));
  library->Register(new Calculate("SqrtExpr", "Sqrt", 1));

  library->Register(new Softmax(false));
  library->Register(new Softmax(true));
}

// Register arithmetic transforms.
void RegisterArithmeticTransforms(Library *library) {
  library->RegisterTransformer(new ExpressionTransformer());
  library->RegisterTransformer(new RemoveUnusedInputs());
  library->RegisterTransformer(new DivToMulTransformer());
  library->RegisterTransformer(new AddNegToSubTransformer());
}

}  // namespace myelin
}  // namespace sling

