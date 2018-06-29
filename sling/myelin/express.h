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

#ifndef SLING_MYELIN_EXPRESS_H_
#define SLING_MYELIN_EXPRESS_H_

#include <map>
#include <string>
#include <vector>

#include "sling/base/logging.h"
#include "sling/base/types.h"

namespace sling {
namespace myelin {

// Express is an intermediate representation (IR) of lists of expressions. An
// expression is a computation of outputs from inputs using a fixed set of
// functions. Express uses single static assignment (SSA) form to represent the
// computations as a sequence of operations on variables. The following kinds of
// variables are supported:
//
//   %n: memory-based input variable
//   #n: constant variable
//   !n: register-based variable
//   @n: memory-based output variable
//   $n: temporary register variable
//   _n: number
//
// An Express recipe is a text format for representing computations over
// inputs variables to produce the output variables. A recipe has the following
// grammar:
//
//   <recipe> := <assignment> | <assignment> ';' <recipe>
//   <assignment> := <variable> '=' <expression>
//   <expression> := <variable> | <operation>
//   <operation> := <name> '(' <arg list> ')'
//   <arg list> := <arg> | <arg> ',' <arg list>
//   <arg> := <variable> | <expression>
//   <variable> := <input variable> | <constant> | <register>
//                 <output variable> | <temp variable> | <number>
//   <input variable> := '%' <integer>
//   <constant> := '#' <integer>
//   <register> := '!' <integer>
//   <output variable> := '@' <integer>
//   <temp variable> := '$' <integer>
//   <number> := '_' <integer>
//
class Express {
 public:
  struct Var;
  struct Op;

  // Variable type.
  enum VarType {INPUT, REGISTER, CONST, OUTPUT, TEMP, NUMBER};

  // Operation type.
  enum OpType {
    MOV,         // identity operation, r=a
    ADD,         // addition, r=a+b
    SUB,         // subtraction, r=a-b
    MUL,         // multiplication, r=a*b
    DIV,         // division, r=a/b
    MINIMUM,     // minimum, r=max(a,b)
    MAXIMUM,     // maximum, r=min(a,b)

    NEG,         // negative, r=-x
    ABS,         // absolute value, r=|x|=max(x,neg(x))
    RELU,        // rectified linear unit, r=max(0,a)
    RELUGRAD,    // rectified linear unit gradient, r=(x>0)*y
    SOFTSIGN,    // softsign, r=x/(|x|+1)
    SOFTPLUS,    // softplus, r=log(exp(x)+1)
    LOGSIGMOID,  // log sigmoid, r=log(1/(1+exp(-x)))=-softplus(-x))
    RECIPROCAL,  // reciprocal value, r=1/x
    SQUARE,      // square, r=x*x
    SQRT,        // square root, r=x^(1/2)

    LOG,         // natural logarithm, r=log(a)
    EXP,         // natural exponential function, r=exp(a)
    SIGMOID,     // sigmoid function, r=1/(1+exp(-a))
    TANH,        // hyperbolic tangent, r=tanh(a)
    LOG2,        // base-2 logarithm, r=log2(a)
    EXP2,        // base-2 exponential function, r=2^a

    MULADD132,   // fused multiply/add, r=a*c+b
    MULADD213,   // fused multiply/add, r=b*a+c
    MULADD231,   // fused multiply/add, r=b*c+a
    MULSUB132,   // fused multiply/sub, r=a*c-b
    MULSUB213,   // fused multiply/sub, r=b*a-c
    MULSUB231,   // fused multiply/sub, r=b*c-a

    CMPEQOQ,     // compare equal (ordered, non-signaling)
    CMPLTOQ,     // compare less than (ordered, non-signaling)
    CMPGTOQ,     // compare greater than (ordered, non-signaling)
    CMPNGEUQ,    // compare not greater or equal (unordered, non-signaling)

    SHR23,       // shift right 23 bits
    SHL23,       // shift left 23 bits
    AND,         // logical and
    OR,          // logical or
    ANDNOT,      // logical and not
    FLOOR,       // floor function
    CVTFLTINT,   // float to integer conversion
    CVTINTFLT,   // integer to float conversion
    SUBINT,      // integer subtraction

    SUM,         // sum reduction
    PRODUCT,     // product reduction
    MIN,         // min reduction
    MAX,         // max reduction

    INVALID,     // invalid operation
  };

  // System-defined numeric constants.
  enum ConstantNumber {
    ZERO, ONE, HALF, TWO, N1, P9, N9, P127, LN2, NLN2, LOG2E,
    PINF, NINF, MIN_NORM_POS, INV_MANT_MASK, MAX_MANT,
    CEPHES_SQRTHF,
    CEPHES_LOG_P0, CEPHES_LOG_P1, CEPHES_LOG_P2, CEPHES_LOG_P3, CEPHES_LOG_P4,
    CEPHES_LOG_P5, CEPHES_LOG_P6, CEPHES_LOG_P7, CEPHES_LOG_P8,
    CEPHES_LOG_Q1, CEPHES_LOG_Q2,
    EXP_HI, EXP_LO,
    CEPHES_LOG2EF, CEPHES_EXP_P0, CEPHES_EXP_P1, CEPHES_EXP_P2, CEPHES_EXP_P3,
    CEPHES_EXP_P4, CEPHES_EXP_P5,
    ALPHA_1, ALPHA_3, ALPHA_5, ALPHA_7, ALPHA_9, ALPHA_11, ALPHA_13,
    BETA_0, BETA_2, BETA_4, BETA_6,
    NUM_CONSTANTS,
  };

  // Variable mapping.
  typedef std::map<Var *, Var *> Map;

  // Variable in expression.
  struct Var {
    Var(const Var &other) = default;
    Var(VarType type, int id) : type(type), id(id), producer(nullptr) {}
    string AsString() const;
    void GetRecipe(string *recipe) const;

    // Return the number of usages of variable.
    int usages() const { return consumers.size(); }

    // An inlined variable is a temporary variable that is only needed in a
    // single context.
    bool inlined() const { return type == TEMP && usages() == 1; }

    // Redirect all consumers of variable to another variable.
    void Redirect(Var *other);

    // Temporary variables and register-based inputs are in registers.
    bool IsRegister() const { return type == TEMP || type == REGISTER; }

    // Check if live range for variable overlaps with another variable.
    bool overlaps(Var *other) const {
      return first->index < other->last->index &&
             other->first->index < last->index;
    }

    VarType type;                 // variable type
    int id;                       // variable id (-1 for unassigned temps)
    Op *producer;                 // operation producing value for variable
    std::vector<Op *> consumers;  // consumers of variable
    bool single = false;          // single-element memory variable

    // Live range for variable.
    Op *first = nullptr;          // first usage of variable
    Op *last = nullptr;           // last usage of variable
    int reg = -1;                 // register number for variable
  };

  // Operation in expression.
  struct Op {
    Op(const Op &other) = default;
    Op(OpType type) : type(type) {}
    string AsString() const;
    void GetRecipe(string *recipe) const;
    bool EqualTo(Op *other) const;
    string AsInstruction() const;

    // Assign result of operation to variable.
    void Assign(Var *var, bool reassign = false);

    // Add argument.
    void AddArgument(Var *arg);

    // Remove all arguments.
    void ClearArguments();

    // Return number of arguments.
    int arity() const { return args.size(); }

    // Check if operation is commutative.
    bool commutative() const {
      return type == ADD || type == MUL ||
             type == MINIMUM || type == MAXIMUM ||
             type == AND || type == OR;
    }

    // Check if operation is a reduction.
    bool reduction() const {
      return type == SUM || type == PRODUCT || type == MIN || type == MAX;
    }

    // Check if operation is a no-op.
    bool nop() const { return type == MOV && dst != -1 && src == dst; }

    // Operation is computing result = type(args...).
    OpType type;                  // operation type
    Var *result = nullptr;        // variable where result is stored
    std::vector<Var *> args;      // operation arguments

    // Register assignment for operands.
    int dst = -1;                 // register for first operand
    int src = -1;                 // register for second operand
    int src2 = -1;                // register for third operand
    int acc = -1;                 // register for accumulation
    bool first_is_dest = false;   // first argument is also destination
    int index = -1;               // operation index
  };

  // Target platform.
  enum Target {INTEL, NVIDIA};

  // Instruction model with instruction forms supported by target architecture
  // for rewriting expression operations.
  struct Model {
    // Move instruction formats.
    bool mov_reg_reg = false;       // dst = src
    bool mov_reg_imm = false;       // dst = imm
    bool mov_reg_mem = false;       // dst = [mem]
    bool mov_mem_reg = false;       // [mem] = src
    bool mov_mem_imm = false;       // [mem] = imm

    // Two-operand instruction formats.
    bool op_reg_reg = false;        // dst = op(dst, src)
    bool op_reg_imm = false;        // dst = op(dst, imm)
    bool op_reg_mem = false;        // dst = op(dst, [mem])
    bool op_mem_reg = false;        // [mem] = op([mem], src)
    bool op_mem_imm = false;        // [mem] = op([mem], imm)

    // Three-operand instruction formats.
    bool op_reg_reg_reg = false;    // dst = op(src1, src2)
    bool op_reg_reg_imm = false;    // dst = op(src, imm)
    bool op_reg_reg_mem = false;    // dst = op(src, [mem])
    bool op_mem_reg_reg = false;    // [mem] = op(src, src2)

    // Unary function instruction formats.
    bool func_reg_reg = false;      // dst = op(src)
    bool func_reg_imm = false;      // dst = op(imm)
    bool func_reg_mem = false;      // dst = op([mem])
    bool func_mem_reg = false;      // [mem] = op(src)
    bool func_mem_imm = false;      // [mem] = op(imm)

    // Fused multiply instruction formats.
    bool fm_reg_reg_reg = false;    // dst = op(dst, src1, src2)
    bool fm_reg_reg_imm = false;    // dst = op(dst, src, imm)
    bool fm_reg_reg_mem = false;    // dst = op(dst, src, [mem])
  };

  Express(Target target = INTEL) : target_(target) {}
  ~Express();

  // Parse an expression recipe and add it to the expression. Intrinsic
  // functions can be expanded into basic operations.
  void Parse(const string &recipe, bool expand = false);

  // Return recipe for expression.
  void GetRecipe(string *recipe) const;
  string AsRecipe() const {
    string str;
    GetRecipe(&str);
    return str;
  }

  // Add new operation to expression.
  Op *Operation(OpType type);
  Op *OperationBefore(Op *pos, OpType type);
  Op *OperationAfter(Op *pos, OpType type);

  // Add function with optional intrinsics expansion. The result variable is not
  // set for the returned op.
  Op *Function(OpType type, std::vector<Var *> &args, bool expand = false);

  // Find variable in expression or add a new variable if it does not exist.
  Var *Variable(VarType type, int id);

  // Look up variable in expression. Return null if variable does not exist.
  Var *Lookup(VarType type, int id) const;

  // Add new temp variable to expression.
  Var *Temp() { return Variable(TEMP, -1); }

  // Add new number variable.
  Var *Number(ConstantNumber number);

  // Count the number of variables of a certain type.
  int NumVars(VarType type) const;

  // Count the number of ops of a certain type.
  int NumOps(OpType type) const;

  // Check if expression has node of a certain type.
  bool Has(OpType type) const { return NumOps(type) > 0; }

  // Compact temporary variable ids and return the number of temporary variable.
  int CompactTempVars();

  // Eliminate common subexpressions.
  void EliminateCommonSubexpressions();

  // Cache constants and move the loads outside the body of the code. Each
  // cached constant takes up an additional register, so the number of cached
  // constants is limited to the number of spare registers. Loop-invariant ops
  // are also moved out of the body, notably loads of single element memory
  // input variables.
  void Hoist(int limit);

  // Cache inputs and results used in multiple ops in temporary variables.
  void CacheResults();

  // Compute live range for each variable.
  void ComputeLiveRanges();

  // Return maximum number of active temp variables.
  int MaxActiveTemps() const;

  // Copy operations and variables from another expression.
  void Copy(const Express &other);

  // Merge variable and operations from another expression into this
  // expression. The variables are mapped through the mapping which maps
  // variables in the other expression to variables in this expression.
  void Merge(Express *other, const Map &varmap);

  // Fuse operations. All occurrences of outer(inner(a,b),c) are changed to
  // left(a,b,c) and all occurrences of outer(a,inner(b,c)) to right(a,b,c).
  void Fuse(OpType outer, OpType inner, OpType left, OpType right);
  void FuseMulAdd() { Fuse(ADD, MUL, MULADD213, MULADD231); }
  void FuseMulSub() { Fuse(SUB, MUL, MULSUB213, INVALID); }

  // Optimize expression by performing the following transformations:
  //   - eliminate common subexpressions
  //   - fuse multiply and add/subtract
  //   - cache inputs and results used in multiple ops in temporary variables
  //   - hoist loop-invariant instructions outside the loop
  void Optimize(bool fma, int spare_regs);

  // Rewrite expression to match instruction formats supported by target
  // architecture. The expression is assumed to be in static single assignment
  // form. The expression is rewritten by adding additional temporary variables
  // to the rewritten expression so only the supported instruction forms are
  // needed for computing the expression.
  bool Rewrite(const Model &model, Express *rewritten) const;

  // Allocate registers for operands. Return the number of registers used.
  int AllocateRegisters();

  // Generate instructions according to the instruction model and allocate
  // registers for operands.
  bool Generate(const Model &model, Express *rewritten) const;

  // Returns the number of register used by expression.
  int NumRegs() const;

  // Computes the complexity of the expression. This counts the number of
  // operations needed to compute the expression. This does not include move
  // operations.
  int Complexity() const;

  // Delete unused input variable from expression by renumbering all following
  // input variables.
  void EliminateInput(int id);

  // Variables.
  const std::vector<Var *> vars() const { return vars_; }

  // Operations.
  const std::vector<Op *> ops() const { return ops_; }

  // First operation in the body. All instructions before are loop invariant.
  int body() const { return body_; }

  // Expression building.
  Var *Do(OpType type, Var *x) {
    Op *op = Operation(type);
    op->AddArgument(x);
    op->Assign(Temp());
    return op->result;
  }
  Var *Do(OpType type, Var *x, Var *y) {
    Op *op = Operation(type);
    op->AddArgument(x);
    op->AddArgument(y);
    op->Assign(Temp());
    return op->result;
  }
  Var *Do(OpType type, Var *x, Var *y, Var *z) {
    Op *op = Operation(type);
    op->AddArgument(x);
    op->AddArgument(y);
    op->AddArgument(z);
    op->Assign(Temp());
    return op->result;
  }

  // Build expressions for simple operations.
  Var *Add(Var *x, Var *y) { return Do(ADD, x, y); }
  Var *Sub(Var *x, Var *y) { return Do(SUB, x, y); }
  Var *Mul(Var *x, Var *y) { return Do(MUL, x, y); }
  Var *Div(Var *x, Var *y) { return Do(DIV, x, y); }
  Var *Minimum(Var *x, Var *y) { return Do(MINIMUM, x, y); }
  Var *Maximum(Var *x, Var *y) { return Do(MAXIMUM, x, y); }
  Var *CmpGt(Var *x, Var *y) { return Do(CMPGTOQ, x, y); }
  Var *And(Var *x, Var *y) { return Do(AND, x, y); }
  Var *Zero() { return Number(ZERO); }
  Var *One() { return Number(ONE); }

  // Build expressions for intrinsic functions.
  Var *Log(Var *x);
  Var *Exp(Var *x);
  Var *Tanh(Var *x);

  // Build expressions for composite functions.
  Var *MulAdd(Var *x, Var *y, Var *z) { return Add(Mul(x, y), z); }
  Var *Neg(Var *x) { return target_ == NVIDIA ? Do(NEG, x) : Sub(Zero(), x); }
  Var *Abs(Var *x) {
    return target_ == NVIDIA ? Do(ABS, x) : Maximum(x, Neg(x));
  }
  Var *Relu(Var *x) { return Maximum(x, Zero()); }
  Var *ReluGrad(Var *x, Var *y) { return Mul(And(CmpGt(x, Zero()), One()), y); }
  Var *Softsign(Var *x) { return Div(x, Add(Abs(x), One())); }
  Var *Softplus(Var *x) { return Log(Add(Exp(x), One())); }
  Var *LogSigmoid(Var *x) { return Neg(Softplus(Neg(x))); }
  Var *Reciprocal(Var *x) {
    return target_ == NVIDIA ? Do(RECIPROCAL, x) : Div(One(), x);
  }
  Var *Square(Var *x) { return Mul(x, x); }
  Var *Sqrt(Var *x) { return Do(SQRT, x); }
  Var *Sigmoid(Var *x) { return Reciprocal(Add(One(), Exp(Neg(x)))); }

  // Look up op type for op name. Return INVALID for unknown op name.
  static OpType Lookup(const string &opname);

  // Return op name for op type.
  static const string &OpName(OpType type);

  // Return value for system-defined numeric constant.
  static float NumericFlt32(int number) { return constants[number].flt; }
  static double NumericFlt64(int number) { return constants[number].dbl; }

  // Return system constant number for identity value for op.
  static int IdentityValue(OpType type);

 private:
  // Try to eliminate identical operations from expression. Return true if any
  // operations were removed.
  bool TryToEliminateOps();

  // Try to fuse operation with the producer of the first argument.
  bool TryFuseFirst(Op *op, OpType type, OpType combined);

  // Try to fuse operation with the producer of the second argument.
  bool TryFuseSecond(Op *op, OpType type, OpType combined);

  // Remove variable.
  void RemoveVar(Var *var);

  // Remove operation.
  void RemoveOp(Op *op);

  // Variables in expression.
  std::vector<Var *> vars_;

  // Operations in expression.
  std::vector<Op *> ops_;

  // First operation in the body. All instructions before are loop invariant. If
  // body is 0 (the default), there are no loop invariant instructions.
  int body_ = 0;

  // Target platform.
  Target target_;

  // System-defined numeric constants.
  struct Constant { float flt; double dbl; };
  static Constant constants[NUM_CONSTANTS];
};

}  // namespace myelin
}  // namespace sling

#endif  // SLING_MYELIN_EXPRESS_H_

