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
    // Binary operators.
    MOV,         // identity operation, r=a
    ADD,         // addition, r=a+b
    SUB,         // subtraction, r=a-b
    MUL,         // multiplication, r=a*b
    DIV,         // division, r=a/b
    MINIMUM,     // minimum, r=max(a,b)
    MAXIMUM,     // maximum, r=min(a,b)

    // Functions.
    NEG,         // negative, r=-x
    ABS,         // absolute value, r=|x|=max(x,neg(x))
    SIGN,        // sign of value, r=x<0?-x:x
    RELU,        // rectified linear unit, r=max(0,a)
    SOFTSIGN,    // softsign, r=x/(|x|+1)
    SOFTPLUS,    // softplus, r=log(exp(x)+1)
    LOGSIGMOID,  // log sigmoid, r=log(1/(1+exp(-x)))=-softplus(-x))
    RECIPROCAL,  // reciprocal value, r=1/x
    SQUARE,      // square, r=x*x
    SQRT,        // square root, r=x^(1/2)
    RSQRT,       // reciprocal square root, r=1/sqrt(x)=x^(-1/2)

    LOG,         // natural logarithm, r=log(x)
    EXP,         // natural exponential function, r=exp(x)
    SIGMOID,     // sigmoid function, r=1/(1+exp(-x))
    ERF,         // error function, r=erf(x)
    LOG2,        // base-2 logarithm, r=log2(x)
    EXP2,        // base-2 exponential function, r=2^x
    POW,         // power function, r=x^y

    // Trigonometric functions.
    SIN,         // sine function, r=sin(x)
    COS,         // cosine function, r=cos(x)
    TAN,         // tangent function, r=tan(x)
    COT,         // cotangent, r=cot(x)=1/tan(x)=cos(x)/sin(x)
    SEC,         // secant, r=sec(x)=1/cos(x)
    CSC,         // cosecant, r=csc(x)=1/sin(x)

    // Inverse trigonometric functions.
    ASIN,        // inverse sine function, r=asin(x)
    ACOS,        // inverse cosine function, r=acos(x)
    ATAN,        // inverse tangent function, r=atan(x)
    ACOT,        // inverse cotangent, r=acot(x)
    ASEC,        // inverse secant, r=asec(x)
    ACSC,        // inverse cosecant, r=acsc(x)

    // Hyperbolic functions.
    SINH,        // hyperbolic sine function, r=sinh(x)
    COSH,        // hyperbolic cosine function, r=cosh(x)
    TANH,        // hyperbolic tangent, r=tanh(x)
    COTH,        // hyperbolic cotangent, r=coth(x)
    SECH,        // hyperbolic secant, r=sech(x)
    CSCH,        // hyperbolic cosecant, r=csch(x)

    // Inverse hyperbolic functions.
    ASINH,       // inverse hyperbolic sine function, r=asinh(x)
    ACOSH,       // inverse hyperbolic cosine function, r=acosh(x)
    ATANH,       // inverse hyperbolic tangent, r=atanh(x)
    ACOTH,       // inverse hyperbolic cotangent, r=acoth(x)
    ASECH,       // inverse hyperbolic secant, r=asech(x)
    ACSCH,       // inverse hyperbolic cosecant, r=acsch(x)

    // Fused multiply.
    MULADD132,   // fused multiply/add, r=a*c+b
    MULADD213,   // fused multiply/add, r=b*a+c
    MULADD231,   // fused multiply/add, r=b*c+a
    MULSUB132,   // fused multiply/sub, r=a*c-b
    MULSUB213,   // fused multiply/sub, r=b*a-c
    MULSUB231,   // fused multiply/sub, r=b*c-a

    // Comparison.
    CMPEQOQ,     // compare equal (ordered, non-signaling)
    CMPNEUQ,     // compare not equal (unordered, non-signaling)
    CMPLTOQ,     // compare less than (ordered, non-signaling)
    CMPLEOQ,     // compare less than or equal (ordered, non-signaling)
    CMPGTOQ,     // compare greater than (ordered, non-signaling)
    CMPGEOQ,     // compare greater than or equal (ordered, non-signaling)

    // Logical operators.
    AND,         // logical and, p=a&b
    OR,          // logical or, p=a|b
    XOR,         // logical exclusive or, p=a^b
    ANDNOT,      // logical and not, p=a&!b
    NOT,         // logical not, p=!a

    // Conditionals operators.
    COND,        // conditional expression, r=p?a:b
    SELECT,      // conditional selection, r=p?a:0

    // Rounding operations.
    FLOOR,       // floor function, round down towards -inf
    CEIL,        // ceil function, round up towards +inf
    ROUND,       // round to nearest
    TRUNC,       // round towards zero

    // Integer operations.
    BITAND,      // bitwise and
    BITOR,       // bitwise or
    BITXOR,      // bitwise xor
    BITANDNOT,   // bitwise and not
    BITEQ,       // bitwise equality test
    CVTFLTINT,   // float to integer conversion
    CVTINTFLT,   // integer to float conversion
    CVTEXPINT,   // convert float exponent to integer
    CVTINTEXP,   // convert integer to float exponent
    QUADSIGN,    // shift bit 2 to sign bit
    ADDINT,      // integer addition
    SUBINT,      // integer subtraction

    // Reductions.
    SUM,         // sum reduction
    PRODUCT,     // product reduction
    MIN,         // min reduction
    MAX,         // max reduction
    ALL,         // and reduction
    ANY,         // or reduction
    COUNT,       // predicate reduction

    INVALID,     // invalid operation
    NUM_OPTYPES
  };

  // System-defined numeric constants.
  enum ConstantNumber {
    ZERO, ONE, TWO, HALF, N1, P9, N9, LN2, NLN2, LOG2E,
    PI, PIO2, PIO4, FOPI, TANPIO8, TAN3PIO8,
    PINF, NINF, QNAN, MIN_NORM_POS, INV_MANT_MASK, MAX_MANT,
    SIGN_MASK, INV_SIGN_MASK, I1, I2, I4, INV_I1,
    CEPHES_SQRTHF,
    CEPHES_LOG_P0, CEPHES_LOG_P1, CEPHES_LOG_P2, CEPHES_LOG_P3, CEPHES_LOG_P4,
    CEPHES_LOG_P5, CEPHES_LOG_P6, CEPHES_LOG_P7, CEPHES_LOG_P8,
    CEPHES_LOG_Q1, CEPHES_LOG_Q2,
    EXP_HI, EXP_LO, EXP_BIAS,
    CEPHES_LOG2EF, CEPHES_EXP_P0, CEPHES_EXP_P1, CEPHES_EXP_P2, CEPHES_EXP_P3,
    CEPHES_EXP_P4, CEPHES_EXP_P5,
    ALPHA_1, ALPHA_3, ALPHA_5, ALPHA_7, ALPHA_9, ALPHA_11, ALPHA_13,
    BETA_0, BETA_2, BETA_4, BETA_6,
    ERF_A1, ERF_A2, ERF_A3, ERF_A4, ERF_A5, ERF_P,
    CEPHES_MINUS_DP1, CEPHES_MINUS_DP2, CEPHES_MINUS_DP3,
    SINCOF_P0, SINCOF_P1, SINCOF_P2,
    COSCOF_P0, COSCOF_P1, COSCOF_P2,
    ATAN_P0, ATAN_P1, ATAN_P2, ATAN_P3,
    NUM_CONSTANTS,
  };

  // Variable mapping.
  typedef std::map<Var *, Var *> Map;

  // Variable in expression.
  struct Var {
    Var(const Var &other) = default;
    Var(VarType type, int id) : type(type), id(id), producer(nullptr) {}
    char TypeCode() const;
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
    bool unhoistable = false;     // variable is not loop invariant
    bool predicate = false;       // predicate variable

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
             type == BITAND || type == BITOR || type == BITXOR ||
             type == CMPEQOQ || type == CMPNEUQ || type == BITEQ ||
             type == AND || type == OR || type == XOR;
    }

    // Check if operation is a fused multiply.
    bool fma() const {
      return type >= MULADD132 && type <= MULSUB231;
    }

    // Check if operation is a comparison.
    bool compare() const {
      return (type >= CMPEQOQ && type <= CMPGEOQ) || type == BITEQ;
    }

    // Check if operation is a logic operation.
    bool logic() const {
      return type >= AND && type <= NOT;
    }

    // Check if operation is a conditional operation.
    bool conditional() const {
      return type >= COND && type <= SELECT;
    }

    // Check if operation is a reduction.
    bool reduction() const {
      return type >= SUM && type <= COUNT;
    }

    // Check if operation is a predicate reduction.
    bool preduction() const {
      return type == ALL || type == ANY;
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
    int mask = -1;                // register for masking
    int acc = -1;                 // register for accumulation
    bool first_is_dest = false;   // first argument is also destination
    int index = -1;               // operation index
  };

  // Instruction model with instruction forms supported by target architecture
  // for translating and rewriting expressions.
  struct Model {
    // Instruction model name.
    const char *name = "Generic";

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

    // Conditional instruction formats.
    bool cond_reg_reg_reg = false;  // dst = op(dst, pred, src, src2)
    bool cond_reg_reg_mem = false;  // dst = op(dst, pred, src, [mem])
    bool cond_reg_mem_reg = false;  // dst = op(dst, pred, [mem], src2)
    bool cond_reg_mem_mem = false;  // dst = op(dst, pred, [mem], [mem])

    // Separate registers for predicates.
    bool predicate_regs = false;

    // Only use register operands for logic ops.
    bool logic_in_regs = false;

    // Supported instructions.
    bool instr[NUM_OPTYPES] = {false};

    // Set supported instructions.
    void instruction_set(std::initializer_list<OpType> ops) {
      for (OpType op : ops) instr[op] = true;
    }
  };

  Express() : model_(nullptr) {}
  Express(const Model *model) : model_(model) {}
  ~Express();

  // Parse an expression recipe and add it to the expression. Intrinsic
  // functions can be expanded into basic operations.
  void Parse(const string &recipe);

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
  Op *Function(OpType type, std::vector<Var *> &args);

  // Expand intrinsic operation or return null if no expansion is available.
  Var *Expand(OpType type, std::vector<Var *> &args);

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

  // Check if expression has any nodes of a set of type.
  bool Has(std::initializer_list<OpType> ops) const;

  // Compact temporary variable ids and return the number of temporary variable.
  int CompactTempVars();

  // Eliminate redundant move instructions.
  void EliminateRedundantMoves();

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

  // Translate expression replacing instructions that are not supported by
  // instruction model with more basic instructions.
  void Translate(Express *translated) const;

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
  // Registers for predicates can be allocated in a separate pool.
  int AllocateRegisters(bool predicate_regs = false);

  // Generate instructions according to the instruction model and allocate
  // registers for operands.
  bool Generate(const Model &model, Express *rewritten) const;

  // Returns the number of register used by expression.
  int NumRegs() const;

  // Return vector with flag for each register indicating if the register is
  // a predicate register.
  void GetRegisterTypes(std::vector<bool> *regs) const;

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
  Var *Zero() { return Number(ZERO); }
  Var *One() { return Number(ONE); }
  Var *Two() { return Number(TWO); }
  Var *Ones() { return Number(QNAN); }

  Var *CmpEq(Var *x, Var *y) { return Do(CMPEQOQ, x, y); }
  Var *CmpNe(Var *x, Var *y) { return Do(CMPNEUQ, x, y); }
  Var *CmpLt(Var *x, Var *y) { return Do(CMPLTOQ, x, y); }
  Var *CmpLe(Var *x, Var *y) { return Do(CMPLEOQ, x, y); }
  Var *CmpGt(Var *x, Var *y) { return Do(CMPGTOQ, x, y); }
  Var *CmpGe(Var *x, Var *y) { return Do(CMPGEOQ, x, y); }

  Var *And(Var *x, Var *y) { return Do(AND, x, y); }
  Var *Or(Var *x, Var *y) { return Do(OR, x, y); }
  Var *Xor(Var *x, Var *y) { return Do(XOR, x, y); }
  Var *Not(Var *x) {
    return Supports(NOT) ? Do(NOT, x) : Xor(Ones(), x);
  }
  Var *AndNot(Var *x, Var *y) {
    return Supports(ANDNOT) ? Do(ANDNOT, x) : And(Not(x), y);
  }

  Var *Cond(Var *p, Var *x, Var *y) { return Do(COND, p, x, y); }
  Var *Select(Var *p, Var *x) {
    return Supports(SELECT) ? Do(SELECT, p, x) : Cond(p, x, Zero());
  }

  // Build expressions for intrinsic functions.
  Var *Log(Var *x);
  Var *Exp(Var *x);
  Var *Erf(Var *x);

  // Build expressions for trigonometric functions.
  Var *Trig(OpType type, Var *x);
  Var *Sin(Var *x) { return Trig(SIN, x); }
  Var *Cos(Var *x) { return Trig(COS, x); }
  Var *Tan(Var *x) { return Trig(TAN, x); }
  Var *Cot(Var *x) { return Trig(COT, x); }
  Var *Sec(Var *x) { return Trig(SEC, x); }
  Var *Csc(Var *x) { return Trig(CSC, x); }

  // Build expressions for inverse trigonometric functions.
  Var *Asin(Var *x);
  Var *Acos(Var *x);
  Var *Atan(Var *x);
  Var *Acot(Var *x) { return Sub(Number(PIO2), Atan(x)); }
  Var *Asec(Var *x) { return Acos(Reciprocal(x)); }
  Var *Acsc(Var *x) { return Asin(Reciprocal(x)); }

  // Build expressions for hyperbolic functions.
  Var *HyperTrig(OpType type, Var *x);
  Var *Sinh(Var *x) { return HyperTrig(SINH, x); }
  Var *Cosh(Var *x) { return HyperTrig(COSH, x); }
  Var *Tanh(Var *x);
  Var *Coth(Var *x) { return HyperTrig(COTH, x); }
  Var *Sech(Var *x) { return HyperTrig(SECH, x); }
  Var *Csch(Var *x) { return HyperTrig(CSCH, x); }

  // Build expressions for inverse hyperbolic functions.
  Var *Asinh(Var *x) {
    return Log(Add(x, Sqrt(Add(Square(x), One()))));
  }
  Var *Acosh(Var *x) {
    return Log(Add(x, Sqrt(Sub(Square(x), One()))));
  }
  Var *Atanh(Var *x) {
    return Mul(Log(Div(Add(One(), x), Sub(One(), x))), Number(HALF));
  }
  Var *Acoth(Var *x) {
    return Mul(Log(Div(Add(x, One()), Sub(x, One()))), Number(HALF));
  }
  Var *Asech(Var *x) {
    return Log(Add(Reciprocal(x), Sqrt(Sub(Square(Reciprocal(x)), One()))));
  }
  Var *Acsch(Var *x) {
    return Log(Add(Reciprocal(x), Sqrt(Add(Square(Reciprocal(x)), One()))));
  }

  // Build expressions for composite functions.
  Var *MulAdd(Var *x, Var *y, Var *z) { return Add(Mul(x, y), z); }
  Var *Neg(Var *x) {
    return Supports(NEG) ? Do(NEG, x) : Sub(Zero(), x);
  }
  Var *Abs(Var *x) {
    return Supports(ABS) ? Do(ABS, x) : Maximum(x, Neg(x));
  }
  Var *Sign(Var *x) {
    if (Supports(SIGN)) return Do(SIGN, x);
    return Cond(CmpLt(x, Zero()), Number(N1),
                                  Select(CmpGt(x, Zero()), One()));
  }
  Var *Relu(Var *x) {
    return Supports(RELU) ? Do(RELU, x) : Maximum(x, Zero());
  }
  Var *Softsign(Var *x) { return Div(x, Add(Abs(x), One())); }
  Var *Softplus(Var *x) { return Log(Add(Exp(x), One())); }
  Var *LogSigmoid(Var *x) { return Neg(Softplus(Neg(x))); }
  Var *Reciprocal(Var *x) {
    return Supports(RECIPROCAL) ? Do(RECIPROCAL, x) : Div(One(), x);
  }
  Var *Square(Var *x) {
    return Supports(SQUARE) ? Do(SQUARE, x) : Mul(x, x);
  }
  Var *Sqrt(Var *x) { return Do(SQRT, x); }
  Var *Rsqrt(Var *x) {
    return Supports(RSQRT) ? Do(RSQRT, x) : Reciprocal(Sqrt(x));
  }
  Var *Pow(Var *x, Var *y) {
    if (Supports(POW)) return Do(POW, x, y);
    if (Supports(EXP2) && Supports(LOG2)) return Do(EXP2, Mul(y, Do(LOG2, x)));
    return Exp(Mul(y, Log(x)));
  }
  Var *Sigmoid(Var *x) { return Reciprocal(Add(One(), Exp(Neg(x)))); }

  Var *Sum(Var *x) { return Do(SUM, x); }
  Var *Count(Var *p) {
    return Supports(COUNT) ? Do(COUNT, p) : Sum(Select(p, One()));
  }

  // Commute operation so op(a,b) = commute(op)(b,a).
  static OpType Commute(OpType type);

  // Look up op type for op name. Return INVALID for unknown op name.
  static OpType Lookup(const string &opname);

  // Return op name for op type.
  static const string &OpName(OpType type);

  // Return value for system-defined numeric constant.
  static float NumericFlt32(int number) { return constants[number].flt; }
  static double NumericFlt64(int number) { return constants[number].dbl; }

  // Return system constant number for neutral element for op.
  static int NeutralValue(OpType type);

  // Set approximate math flag.
  bool approx() const { return approx_; }
  void set_approx(bool approx) { approx_ = approx; }

  // Check if expression is sparse computation compatible, i.e. it evaluates to
  // zero when the non-single inputs are zero.
  bool SparseCompatible() const;

  // Check if expression sparse assignment compatible, i.e. it evaluates to
  // zero plus the first input when the remaining non-single inputs are zero.
  bool SparseAssignCompatible() const;

 private:
  // Check if variable always evaluates to zero when non-single inputs are zero.
  bool ZeroFixpoint(Var *x) const;

  // Check if variable always evaluates to zero.
  bool AlwaysZero(Var *x) const;

  // Check if variable always evaluates to one.
  bool AlwaysOne(Var *x) const;

  // Check if y always evaluates to x.
  bool Identical(Var *y, Var *x) const;

  // Check if y evaluates to x+z where z does not depend on x.
  bool Additive(Var *y, Var *x) const;

  // Check if y depends on x.
  bool DependsOn(Var *y, Var *x) const;

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

  // Check if operation is supported by instruction model.
  bool Supports(OpType type) const {
    if (!approx_ && (type == RECIPROCAL || type == RSQRT)) return false;
    return model_ == nullptr || model_->instr[type];
  }

  // Variables in expression.
  std::vector<Var *> vars_;

  // Operations in expression.
  std::vector<Op *> ops_;

  // First operation in the body. All instructions before are loop invariant. If
  // body is 0 (the default), there are no loop invariant instructions.
  int body_ = 0;

  // Target instruction model.
  const Model *model_ = nullptr;

  // Allow approximate math functions.
  bool approx_ = false;

  // System-defined numeric constants.
  struct Constant { float flt; double dbl; };
  static Constant constants[NUM_CONSTANTS];
};

}  // namespace myelin
}  // namespace sling

#endif  // SLING_MYELIN_EXPRESS_H_

