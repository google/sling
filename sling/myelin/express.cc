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

#include "sling/myelin/express.h"

#include <algorithm>
#include <map>
#include <set>
#include <string>
#include <vector>

namespace sling {
namespace myelin {

namespace {

// Mapping from operation name to operation type.
static std::map<string, Express::OpType> optypes = {
  {"Id", Express::MOV},
  {"Add", Express::ADD},
  {"Sub", Express::SUB},
  {"Mul", Express::MUL},
  {"Div", Express::DIV},
  {"Minimum", Express::MINIMUM},
  {"Maximum", Express::MAXIMUM},

  {"Neg", Express::NEG},
  {"Abs", Express::ABS},
  {"Sign", Express::SIGN},
  {"Relu", Express::RELU},
  {"Softsign", Express::SOFTSIGN},
  {"Softplus", Express::SOFTPLUS},
  {"LogSigmoid", Express::LOGSIGMOID},
  {"Reciprocal", Express::RECIPROCAL},
  {"Square", Express::SQUARE},
  {"Sqrt", Express::SQRT},
  {"Rsqrt", Express::RSQRT},
  {"Log", Express::LOG},
  {"Exp", Express::EXP},
  {"Sigmoid", Express::SIGMOID},
  {"Erf", Express::ERF},
  {"Log2", Express::LOG2},
  {"Exp2", Express::EXP2},
  {"Pow", Express::POW},

  {"Sin", Express::SIN},
  {"Cos", Express::COS},
  {"Tan", Express::TAN},
  {"Cot", Express::COT},
  {"Sec", Express::SEC},
  {"Csc", Express::CSC},

  {"Asin", Express::ASIN},
  {"Acos", Express::ACOS},
  {"Atan", Express::ATAN},
  {"Acot", Express::ACOT},
  {"Asec", Express::ASEC},
  {"Acsc", Express::ACSC},

  {"Sinh", Express::SINH},
  {"Cosh", Express::COSH},
  {"Tanh", Express::TANH},
  {"Coth", Express::COTH},
  {"Sech", Express::SECH},
  {"Csch", Express::CSCH},

  {"Asinh", Express::ASINH},
  {"Acosh", Express::ACOSH},
  {"Atanh", Express::ATANH},
  {"Acoth", Express::ACOTH},
  {"Asech", Express::ASECH},
  {"Acsch", Express::ACSCH},

  {"MulAdd132", Express::MULADD132},
  {"MulAdd213", Express::MULADD213},
  {"MulAdd231", Express::MULADD231},
  {"MulSub132", Express::MULSUB132},
  {"MulSub213", Express::MULSUB213},
  {"MulSub231", Express::MULSUB231},

  {"CmpEq", Express::CMPEQOQ},
  {"CmpNe", Express::CMPNEUQ},
  {"CmpLt", Express::CMPLTOQ},
  {"CmpLe", Express::CMPLEOQ},
  {"CmpGt", Express::CMPGTOQ},
  {"CmpGe", Express::CMPGEOQ},

  {"And", Express::AND},
  {"Or", Express::OR},
  {"Xor", Express::XOR},
  {"AndNot", Express::ANDNOT},
  {"Not", Express::NOT},

  {"Cond", Express::COND},
  {"Select", Express::SELECT},

  {"Floor", Express::FLOOR},
  {"Ceil", Express::CEIL},
  {"Round", Express::ROUND},
  {"Trunc", Express::TRUNC},

  {"BitAnd", Express::BITAND},
  {"BitOr", Express::BITOR},
  {"BitXor", Express::BITXOR},
  {"BitAndNot", Express::BITANDNOT},
  {"BitEq", Express::BITEQ},
  {"CvtFltInt", Express::CVTFLTINT},
  {"CvtIntFlt", Express::CVTINTFLT},
  {"CvtExpInt", Express::CVTEXPINT},
  {"CvtIntExp", Express::CVTINTEXP},
  {"QuadSign", Express::QUADSIGN},
  {"AddInt", Express::ADDINT},
  {"SubInt", Express::SUBINT},

  {"Sum", Express::SUM},
  {"Product", Express::PRODUCT},
  {"Min", Express::MIN},
  {"Max", Express::MAX},
  {"All", Express::ALL},
  {"Any", Express::ANY},
  {"Count", Express::COUNT},
};

static const string opname[] = {
  "Id",
  "Add", "Sub", "Mul", "Div",
  "Minimum", "Maximum",
  "Neg", "Abs", "Sign", "Relu", "Softsign", "Softplus", "LogSigmoid",
  "Reciprocal", "Square", "Sqrt", "Rsqrt",
  "Log", "Exp", "Sigmoid", "Erf", "Log2", "Exp2", "Pow",
  "Sin", "Cos", "Tan", "Cot", "Sec", "Csc",
  "Asin", "Acos", "Atan", "Acot", "Asec", "Acsc",
  "Sinh", "Cosh", "Tanh", "Coth", "Sech", "Csch",
  "Asinh", "Acosh", "Atanh", "Acoth", "Asech", "Acsch",
  "MulAdd132", "MulAdd213", "MulAdd231",
  "MulSub132", "MulSub213", "MulSub231",
  "CmpEq", "CmpNe", "CmpLt", "CmpLe", "CmpGt", "CmpGe",
  "And", "Or", "Xor", "AndNot", "Not", "Cond", "Select",
  "Floor", "Ceil", "Round", "Trunc",
  "BitAnd", "BitOr", "BitXor", "BitAndNot", "BitEq",
  "CvtFltInt", "CvtIntFlt", "CvtExpInt", "CvtIntExp", "QuadSign",
  "AddInt", "SubInt",
  "Sum", "Product", "Min", "Max", "All", "Any", "Count",
  "???",
};

// Variable mapping.
class VariableMap {
 public:
  VariableMap(Express *expr) : expr_(expr) {}

  Express::Var *operator[](Express::Var *var) {
    Express::Var *&m = mapping_[var];
    if (m == nullptr) {
      auto &vars = expr_->vars();
      if (std::find(vars.begin(), vars.end(), var) != vars.end()) {
        // Existing variable in expression.
        return var;
      }

      // Copy variable and update mapping.
      m = expr_->Variable(var->type, var->id);
      m->predicate = var->predicate;
      m->single = var->single;
      m->unhoistable = var->unhoistable;
    }
    return m;
  }

  void rename(Express::Var *from, Express::Var *to) {
    mapping_[from] = to;
    to->type = from->type;
    to->id = from->id;
    to->predicate = from->predicate;
    to->single = from->single;
    to->unhoistable = from->unhoistable;
  }

 private:
  Express *expr_;
  std::map<Express::Var *, Express::Var *> mapping_;
};

// Register allocator.
class RegisterAllocator {
 public:
  RegisterAllocator(bool typed) : typed_(typed) {}

  // Allocate register for variable.
  int Allocate(Express::Var *var) {
    // Check if a register has already been allocated.
    int regno = -1;
    for (int r = 0; r < reg_.size(); ++r) {
      if (reg_[r] == var) return r;
      if (regno == -1 && reg_[r] == nullptr) {
        if (!typed_ || var->predicate == predicate_[r]) regno = r;
      }
    }

    if (regno == -1) {
      // Allocate new register.
      regno = reg_.size();
      reg_.push_back(var);
      predicate_.push_back(var->predicate);
    } else {
      // Assign unused register to variable.
      reg_[regno] = var;
    }

    var->reg = regno;
    return regno;
  }

  // Allocate specific register for variable. Return -1 if register cannot
  // be allocated to variable.
  int Allocate(Express::Var *var, int regno) {
    // Check if register is already allocated.
    CHECK(regno < reg_.size());
    if (reg_[regno] != nullptr) return -1;

    // Check that register types match.
    if (typed_ && var->predicate != predicate_[regno]) return -1;

    // Allocate register for variable.
    reg_[regno] = var;
    var->reg = regno;
    return regno;
  }

  // Allocate spare register not assigned to variable.
  int AllocateExtra(bool predicate) {
    static Express::Var extra(Express::REGISTER, -1);
    int regno = reg_.size();
    reg_.push_back(&extra);
    predicate_.push_back(predicate);
    return regno;
  }

  // Get register allocated for variable. Return -1 if no register is allocated.
  int Get(Express::Var *var) const {
    return var->reg;
  }

  // Free register used by variable.
  void Free(Express::Var *var) {
    CHECK(Allocated(var));
    reg_[var->reg] = nullptr;
  }

  // Check if register is allocated for variable.
  bool Allocated(Express::Var *var) const {
    return var->reg != -1 && reg_[var->reg] == var;
  }

  // Return the maximum number of register allocated.
  int max() const { return reg_.size(); }

 private:
  // In typed allocation mode, predicates and non-predicate variables are not
  // allowed to share registers.
  bool typed_;

  // Variable currently assigned to register or null if the register is not
  // currently allocated.
  std::vector<Express::Var *> reg_;

  // Variable type for register (predicate vs. non-predicate).
  std::vector<bool> predicate_;
};

template <class Dest, class Source>
inline Dest bit_cast(const Source &source) {
  static_assert(sizeof(Dest) == sizeof(Source), "size error");
  Dest dest;
  memcpy(&dest, &source, sizeof(dest));
  return dest;
}

// Recipe parser for converting a string to an expression.
class RecipeParser {
 public:
  // Initialize parser.
  RecipeParser(const string &recipe, Express *expr) {
    recipe_ = ptr_ = recipe.data();
    end_ = ptr_ + recipe.size();
    expr_ = expr;
  }

  // Parse recipe.
  void Parse() {
    // Parse list of assignment expressions.
    ParseAssignment();
    while (is(';')) {
      next();
      ParseAssignment();
    }

    // Check that all the input has been consumed.
    if (more()) Error("Syntax error in expression");

    // Assign ids to intermediate variables.
    expr_->CompactTempVars();
  }

  // Parse assignment expression.
  void ParseAssignment() {
    // Parse assignment variable.
    Express::Var *var = ParseVariable();
    if (var->type == Express::INPUT) {
      Error("Cannot assign to input variable");
    }

    // Consume '='.
    if (current() != '=') Error("Expected '=' in expression");
    next();

    // Parse expression.
    Express::Op *expr = ParseExpression();

    // Assign expression to variable.
    expr->Assign(var);
  }

  // Parse expression.
  Express::Op *ParseExpression() {
    // Parse operation name.
    if (!isletter()) Error("Operation name expected in expression");
    const char *start = ptr_;
    while (isletter() || isdigit()) ptr_++;
    string opname(start, ptr_ - start);

    // Parse argument list.
    if (current() != '(') Error("Expected '(' in expression");
    next();
    std::vector<Express::Var *> args;
    args.push_back(ParseArgument());
    while (current() == ',') {
      next();
      args.push_back(ParseArgument());
    }
    if (current() != ')') Error("Expected ')' in expression");
    next();

    // Create operation.
    Express::OpType optype = Express::Lookup(opname);
    CHECK(optype != Express::INVALID) << opname;
    return expr_->Function(optype, args);
  }

  // Parse argument.
  Express::Var *ParseArgument() {
    if (isvar()) {
      // Return variable as argument.
      return ParseVariable();
    } else {
      // Parse expression and assign to intermediate variable. The intermediate
      // variable is assigned a unique negative id which will later be fixed up.
      Express::Op *expr = ParseExpression();
      Express::Var *var = expr_->Temp();
      expr->Assign(var);
      return var;
    }
  }

  // Parse variable name.
  Express::Var *ParseVariable() {
    // Parse variable type.
    Express::VarType type;
    if (is('%')) {
      type = Express::INPUT;
    } else if (is('!')) {
      type = Express::REGISTER;
    } else if (is('#')) {
      type = Express::CONST;
    } else if (is('@')) {
      type = Express::OUTPUT;
    } else if (is('$')) {
      type = Express::TEMP;
    } else if (is('_')) {
      type = Express::NUMBER;
    } else {
      Error("Unknown variable type in expression");
    }
    next();

    // Parse single qualifier (only used for testing).
    bool single = false;
    if (is('\'')) {
      single = true;
      next();
    }

    // Parse variable id.
    int id = 0;
    int digits = 0;
    while (current() >= '0' && current() <= '9') {
      id = id * 10 + (current() - '0');
      next();
      digits++;
    }
    if (digits == 0) Error("Variable id expected in expression");

    // Return variable.
    Express::Var *var = expr_->Variable(type, id);
    var->single = single;
    return var;
  }

  // Output error.
  void Error(const char *msg) {
    string prefix = string(recipe_, ptr_ - recipe_);
    string suffix = string(ptr_, end_ - ptr_);
    LOG(FATAL) << msg << ": " << prefix << "➤" << suffix;
  }

  // Current input character.
  char current() { return *ptr_; }

  // Consume next input character.
  void next() { ptr_++; }

  // Check if the next input matches.
  bool is(char ch) const { return more() && *ptr_ == ch; }
  bool isdigit() const { return more() && *ptr_ >= '0' && *ptr_ <= '9'; }
  bool isupper() const { return more() && *ptr_ >= 'A' && *ptr_ <= 'Z'; }
  bool islower() const { return more() && *ptr_ >= 'a' && *ptr_ <= 'z'; }
  bool isletter() const { return isupper() || islower(); }
  bool isvar() const {
    return is('%') || is('!') || is('@') || is('$') || is('#') || is('_');
  }

  // Check if the whole expression has been parsed.
  bool more() const { return ptr_ < end_; }
  bool done() const { return ptr_ == end_; }

 private:
  const char *recipe_;         // expression recipe
  const char *ptr_;            // current position for parser
  const char *end_;            // end of parsed recipe
  Express *expr_;              // target expression
};

}  // namespace

#define FLT_FROM_INT(x) bit_cast<float>(x)
#define DBL_FROM_INT(x) bit_cast<double>(x)

// System-defined numeric constants.
#define FLTCONST(x) {x##f, x}
#define DBLCONST(a, b) {a##f, b}
#define INTCONST(a, b) {FLT_FROM_INT(a), DBL_FROM_INT(b)}
Express::Constant Express::constants[Express::NUM_CONSTANTS] = {
  FLTCONST(0.0),    // ZERO
  FLTCONST(1.0),    // ONE
  FLTCONST(2.0),    // TWO
  FLTCONST(0.5),    // HALF
  FLTCONST(-1.0),   // N1
  FLTCONST(9.0),    // P9
  FLTCONST(-9.0),   // N9

  FLTCONST(0.6931471805599453),      // LN2
  FLTCONST(-0.6931471805599453),     // NLN2
  FLTCONST(1.442695021629333),       // LOG2E
  FLTCONST(3.14159265358979323846),  // PI
  FLTCONST(1.5707963267948966192),   // PIO2 (pi/2)
  FLTCONST(0.7853981633974483096),   // PIO4 (pi/4)
  FLTCONST(1.27323954473516),        // FOPI (4/pi)
  FLTCONST(0.4142135623730950),      // TANPIO8 (tan pi/8)
  FLTCONST(2.414213562373095),       // TAN3PIO8 (tan 3pi/8)

  INTCONST(0x7F800000, 0x7FF0000000000000LL),      // PINF
  INTCONST(0xFF800000, 0xFFF0000000000000LL),      // NINF
  INTCONST(0xFFFFFFFF, 0xFFFFFFFFFFFFFFFFLL),      // QNAN

  INTCONST(0x00800000, 0x0020000000000000LL),      // MIN_NORM_POS
  INTCONST(~0x7F800000, ~0x7FF0000000000000LL),    // INV_MANT_MASK
  INTCONST(0x7F, 0x3FFLL),                         // MAX_MANT

  INTCONST(0x80000000, 0x8000000000000000LL),      // SIGN_MASK
  INTCONST(~0x80000000, ~0x8000000000000000LL),    // INV_SIGN_MASK
  INTCONST(1, 1LL),                                // I1
  INTCONST(2, 2LL),                                // I2
  INTCONST(4, 4LL),                                // I4
  INTCONST(~1, ~1LL),                              // INV_I1

  // Polynomial coefficients for natural logarithm.
  FLTCONST(0.707106781186547524),  // CEPHES_SQRTHF
  FLTCONST(7.0376836292E-2),       // CEPHES_LOG_P0
  FLTCONST(-1.1514610310E-1),      // CEPHES_LOG_P1
  FLTCONST(1.1676998740E-1),       // CEPHES_LOG_P2
  FLTCONST(-1.2420140846E-1),      // CEPHES_LOG_P3
  FLTCONST(+1.4249322787E-1),      // CEPHES_LOG_P4
  FLTCONST(-1.6668057665E-1),      // CEPHES_LOG_P5
  FLTCONST(+2.0000714765E-1),      // CEPHES_LOG_P6
  FLTCONST(-2.4999993993E-1),      // CEPHES_LOG_P7
  FLTCONST(+3.3333331174E-1),      // CEPHES_LOG_P8
  FLTCONST(-2.12194440E-4),        // CEPHES_LOG_Q1
  FLTCONST(0.693359375),           // CEPHES_LOG_Q2

  // Clamping interval for exponential function.
  DBLCONST(88.3762626647950, 709.437),      // EXP_HI
  DBLCONST(-88.3762626647949, -709.436),    // EXP_LO
  DBLCONST(127.0, 1023.0),                  // EXP_BIAS

  // Polynomial coefficients for exponential function.
  FLTCONST(1.44269504088896341),   // CEPHES_LOG2EF
  FLTCONST(1.9875691500E-4),       // CEPHES_EXP_P0
  FLTCONST(1.3981999507E-3),       // CEPHES_EXP_P1
  FLTCONST(8.3334519073E-3),       // CEPHES_EXP_P2
  FLTCONST(4.1665795894E-2),       // CEPHES_EXP_P3
  FLTCONST(1.6666665459E-1),       // CEPHES_EXP_P4
  FLTCONST(5.0000001201E-1),       // CEPHES_EXP_P5

  // Monomial coefficients of the numerator polynomial for tanh (odd).
  FLTCONST(-2.76076847742355e-16),  // ALPHA_1
  FLTCONST(2.00018790482477e-13),   // ALPHA_3
  FLTCONST(-8.60467152213735e-11),  // ALPHA_5
  FLTCONST(5.12229709037114e-08),   // ALPHA_7
  FLTCONST(1.48572235717979e-05),   // ALPHA_9
  FLTCONST(6.37261928875436e-04),   // ALPHA_11
  FLTCONST(4.89352455891786e-03),   // ALPHA_13

  // Monomial coefficients of the denominator polynomial for tanh (even).
  FLTCONST(1.19825839466702e-06),  // BETA_0
  FLTCONST(1.18534705686654e-04),  // BETA_2
  FLTCONST(2.26843463243900e-03),  // BETA_4
  FLTCONST(4.89352518554385e-03),  // BETA_6

  // Polynomial coefficients for error function.
  FLTCONST(0.254829592),           // ERF_A1
  FLTCONST(-0.284496736),          // ERF_A2
  FLTCONST(1.421413741),           // ERF_A3
  FLTCONST(-1.453152027),          // ERF_A4
  FLTCONST(1.061405429),           // ERF_A5
  FLTCONST(0.3275911),             // ERF_P

  // Constants for sine and cosine functions.
  FLTCONST(-0.78515625),                // CEPHES_MINUS_DP1
  FLTCONST(-2.4187564849853515625e-4),  // CEPHES_MINUS_DP2
  FLTCONST(-3.77489497744594108e-8),    // CEPHES_MINUS_DP3
  FLTCONST(-1.9515295891e-4),           // SINCOF_P0
  FLTCONST(8.3321608736e-3),            // SINCOF_P1
  FLTCONST(-1.6666654611e-1),           // SINCOF_P2
  FLTCONST(2.443315711809948e-5),       // COSCOF_P0
  FLTCONST(-1.388731625493765e-3),      // COSCOF_P1
  FLTCONST(4.166664568298827e-2),       // COSCOF_P2

  // Constants for arctan.
  FLTCONST(8.05374449538e-2),           // ATAN_P0
  FLTCONST(-1.38776856032e-1),          // ATAN_P1
  FLTCONST(1.99777106478e-1),           // ATAN_P2
  FLTCONST(-3.33329491539e-1),          // ATAN_P3
};

int Express::NeutralValue(OpType type) {
  switch (type) {
    case SUM: return ZERO;
    case PRODUCT: return ONE;
    case MIN: return PINF;
    case MAX: return NINF;
    case ALL: return QNAN;
    case ANY: return ZERO;
    default: return ZERO;
  }
}

Express::OpType Express::Commute(OpType type) {
  switch (type) {
    case CMPLTOQ: return CMPGEOQ;
    case CMPLEOQ: return CMPGTOQ;
    case CMPGTOQ: return CMPLEOQ;
    case CMPGEOQ: return CMPLTOQ;
    default: return type;
  }
}

Express::OpType Express::Lookup(const string &opname) {
  auto f = optypes.find(opname);
  return f == optypes.end() ? INVALID : f->second;
}

const string &Express::OpName(OpType type) {
  return opname[type];
}

void Express::Parse(const string &recipe) {
  RecipeParser parser(recipe, this);
  parser.Parse();
}

Express::~Express() {
  for (auto *v : vars_) delete v;
  for (auto *o : ops_) delete o;
}

void Express::GetRecipe(string *recipe) const {
  bool first = true;
  for (Op *op : ops_) {
    if (!op->result->inlined()) {
      if (!first) recipe->push_back(';');
      first = false;
      op->result->GetRecipe(recipe);
      recipe->push_back('=');
      op->GetRecipe(recipe);
    }
  }
}

Express::Var *Express::Lookup(VarType type, int id) const {
  for (Var *v : vars_) {
    if (v->type == type && v->id == id) return v;
  }
  return nullptr;
}

Express::Var *Express::Variable(VarType type, int id) {
  // Look for existing variable.
  if (id != -1) {
    for (Var *v : vars_) {
      if (v->type == type && v->id == id) return v;
    }
  }

  // Add new variable.
  Var *v = new Var(type, id);
  vars_.push_back(v);
  return v;
}

Express::Op *Express::Operation(OpType type) {
  Op *op = new Op(type);
  ops_.push_back(op);
  return op;
}

Express::Op *Express::OperationBefore(Op *pos, OpType type) {
  Op *op = new Op(type);
  auto f = std::find(ops_.begin(), ops_.end(), pos);
  CHECK(f != ops_.end());
  ops_.insert(f, op);
  return op;
}

Express::Op *Express::OperationAfter(Op *pos, OpType type) {
  Op *op = new Op(type);
  auto f = std::find(ops_.begin(), ops_.end(), pos);
  CHECK(f != ops_.end());
  ops_.insert(f + 1, op);
  return op;
}

Express::Op *Express::Function(OpType type, std::vector<Var *> &args) {
  // Create new op with arguments.
  Express::Op *op = Operation(type);
  for (Var *arg : args) op->AddArgument(arg);
  return op;
}

Express::Var *Express::Expand(OpType type, std::vector<Var *> &args) {
  Express::Var *result = nullptr;
  if (args.size() == 1) {
    switch (type) {
      case Express::NEG: result = Neg(args[0]); break;
      case Express::NOT: result = Not(args[0]); break;
      case Express::ABS: result = Abs(args[0]); break;
      case Express::SIGN: result = Sign(args[0]); break;
      case Express::RELU: result = Relu(args[0]); break;
      case Express::SOFTSIGN: result = Softsign(args[0]); break;
      case Express::SOFTPLUS: result = Softplus(args[0]); break;
      case Express::LOGSIGMOID: result = LogSigmoid(args[0]); break;
      case Express::RECIPROCAL: result = Reciprocal(args[0]); break;
      case Express::SQUARE: result = Square(args[0]); break;
      case Express::RSQRT: result = Rsqrt(args[0]); break;
      case Express::LOG: result = Log(args[0]); break;
      case Express::EXP: result = Exp(args[0]); break;
      case Express::SIGMOID: result = Sigmoid(args[0]); break;
      case Express::ERF: result = Erf(args[0]); break;
      case Express::SIN: result = Sin(args[0]); break;
      case Express::COS: result = Cos(args[0]); break;
      case Express::TAN: result = Tan(args[0]); break;
      case Express::COT: result = Cot(args[0]); break;
      case Express::SEC: result = Sec(args[0]); break;
      case Express::CSC: result = Csc(args[0]); break;
      case Express::ASIN: result = Asin(args[0]); break;
      case Express::ACOS: result = Acos(args[0]); break;
      case Express::ATAN: result = Atan(args[0]); break;
      case Express::ACOT: result = Acot(args[0]); break;
      case Express::ASEC: result = Asec(args[0]); break;
      case Express::ACSC: result = Acsc(args[0]); break;
      case Express::SINH: result = Sinh(args[0]); break;
      case Express::COSH: result = Cosh(args[0]); break;
      case Express::TANH: result = Tanh(args[0]); break;
      case Express::COTH: result = Coth(args[0]); break;
      case Express::SECH: result = Sech(args[0]); break;
      case Express::CSCH: result = Csch(args[0]); break;
      case Express::ASINH: result = Asinh(args[0]); break;
      case Express::ACOSH: result = Acosh(args[0]); break;
      case Express::ATANH: result = Atanh(args[0]); break;
      case Express::ACOTH: result = Acoth(args[0]); break;
      case Express::ASECH: result = Asech(args[0]); break;
      case Express::ACSCH: result = Acsch(args[0]); break;
      case Express::COUNT: result = Count(args[0]); break;
      default: ;
    }
  } else if (args.size() == 2) {
    switch (type) {
      case Express::ANDNOT: result = AndNot(args[0], args[1]); break;
      case Express::SELECT: result = Select(args[0], args[1]); break;
      case Express::POW: result = Pow(args[0], args[1]); break;
      default: ;
    }
  }

  return result;
}

Express::Var *Express::Number(ConstantNumber number) {
  // Add number variable.
  Var *v = new Var(NUMBER, number);
  vars_.push_back(v);
  return v;
}

void Express::RemoveVar(Var *var) {
  // Check that variable is unused.
  DCHECK(var->producer == nullptr);
  DCHECK(var->consumers.empty());

  // Delete variable.
  auto f = std::find(vars_.begin(), vars_.end(), var);
  DCHECK(f != vars_.end());
  vars_.erase(f);
  delete var;
}

void Express::RemoveOp(Op *op) {
  // Remove operation as producer of result.
  if (op->result != nullptr) {
    DCHECK(op == op->result->producer);
    op->result->producer = nullptr;
  }

  // Remove operation as consumer of argument variables.
  op->ClearArguments();

  // Delete operation.
  auto f = std::find(ops_.begin(), ops_.end(), op);
  DCHECK(f != ops_.end());
  ops_.erase(f);
  delete op;
}

int Express::NumVars(VarType type) const {
  int n = 0;
  for (Var *v : vars_) {
    if (v->type == type) n++;
  }
  return n;
}

int Express::NumOps(OpType type) const {
  int n = 0;
  for (Op *op : ops_) {
    if (op->type == type) n++;
  }
  return n;
}

bool Express::Has(std::initializer_list<OpType> ops) const {
  // Build instruction set.
  bool instr[NUM_OPTYPES] = {false};
  for (OpType op : ops) instr[op] = true;

  // Check if expression contains any instructions in the set.
  for (Op *op : ops_) {
    if (instr[op->type]) return true;
  }
  return false;
}

int Express::Complexity() const {
  int n = 0;
  for (Op *op : ops_) {
    switch (op->type) {
      case MOV:
        break;
      case MULADD132:
      case MULADD213:
      case MULADD231:
      case MULSUB132:
      case MULSUB213:
      case MULSUB231:
      case ANDNOT:
        n += 2;
        break;
      default:
        n += 1;
    }
  }
  return n;
}

bool Express::SparseCompatible() const {
  for (Var *v : vars_) {
    if (v->type == OUTPUT && !ZeroFixpoint(v)) return false;
  }
  return true;
}

bool Express::SparseAssignCompatible() const {
  Var *in = Lookup(INPUT, 0);
  Var *out = Lookup(OUTPUT, 0);
  if (in == nullptr || out == nullptr) return false;
  return Additive(out, in) && SparseCompatible();
}

bool Express::ZeroFixpoint(Var *x) const {
  Op *op = x->producer;
  if (op == nullptr) {
    if (x->type == NUMBER) return x->id == ZERO;
    return !x->single;
  }
  switch (op->type) {
    case MOV:
      return ZeroFixpoint(op->args[0]);
    case ADD:
    case SUB:
    case MINIMUM:
    case MAXIMUM:
    case AND:
    case BITAND:
      return ZeroFixpoint(op->args[0]) && ZeroFixpoint(op->args[1]);
    case MUL:
    case OR:
    case BITOR:
      return ZeroFixpoint(op->args[0]) || ZeroFixpoint(op->args[1]);
    case DIV:
    case POW:
      return ZeroFixpoint(op->args[0]);
    case NEG: case ABS: case SIGN: case RELU: case SOFTSIGN:
    case SQUARE: case SQRT: case SIGMOID: case ERF:
    case SIN: case TAN:
    case ASIN: case ATAN:
    case SINH: case TANH:
    case ASINH: case ATANH:
    case FLOOR: case CEIL: case ROUND: case TRUNC:
    case SUM:
      return ZeroFixpoint(op->args[0]);
    default:
      return false;
  }
}

bool Express::AlwaysZero(Var *x) const {
  if (x->type == NUMBER && x->id == ZERO) return true;
  Op *op = x->producer;
  if (op == nullptr) return false;
  switch (op->type) {
    case MOV:
      return AlwaysZero(op->args[0]);
    case ADD:
      return AlwaysZero(op->args[0]) && AlwaysZero(op->args[1]);
    case SUB:
      return (AlwaysZero(op->args[0]) && AlwaysZero(op->args[1])) ||
             (AlwaysOne(op->args[0]) && AlwaysOne(op->args[1]));
    case MUL:
      return AlwaysZero(op->args[0]) || AlwaysZero(op->args[1]);
    case DIV:
      return AlwaysZero(op->args[0]) && !AlwaysZero(op->args[1]);
    default:
      return op->arity() == 1 && AlwaysZero(op->args[0]) && ZeroFixpoint(x);
  }
}

bool Express::AlwaysOne(Var *x) const {
  if (x->type == NUMBER && x->id == ONE) return true;
  Op *op = x->producer;
  if (op == nullptr) return false;
  switch (op->type) {
    case MOV:
      return AlwaysOne(op->args[0]);
    case ADD:
      return (AlwaysOne(op->args[0]) && AlwaysZero(op->args[1])) ||
             (AlwaysZero(op->args[0]) && AlwaysOne(op->args[1]));
    case SUB:
      return AlwaysOne(op->args[0]) && AlwaysZero(op->args[1]);
    case MUL:
    case DIV:
      return AlwaysOne(op->args[0]) && AlwaysOne(op->args[1]);
    default:
      return false;
  }
}

bool Express::Identical(Var *y, Var *x) const {
  if (x == y) return true;
  Op *op = y->producer;
  if (op == nullptr) return false;
  switch (op->type) {
    case MOV:
      return Identical(op->args[0], x);
    case ADD:
      return (Identical(op->args[0], x) && AlwaysZero(op->args[1])) ||
             (Identical(op->args[1], x) && AlwaysZero(op->args[0]));
    case MUL:
      return (Identical(op->args[0], x) && AlwaysOne(op->args[1])) ||
             (Identical(op->args[1], x) && AlwaysOne(op->args[0]));
    case SUB:
      return Identical(op->args[0], x) && AlwaysZero(op->args[1]);
    case DIV:
      return Identical(op->args[0], x) && AlwaysOne(op->args[1]);
    default:
      return false;
  }
}

bool Express::Additive(Var *y, Var *x) const {
  if (x == y) return true;
  Op *op = y->producer;
  if (op == nullptr) return false;
  switch (op->type) {
    case MOV:
      return Additive(op->args[0], x);
    case ADD:
      return (Identical(op->args[0], x) && !DependsOn(op->args[1], x)) ||
             (Identical(op->args[1], x) && !DependsOn(op->args[0], x));
    case SUB:
      return Identical(op->args[0], x) && !DependsOn(op->args[1], x);
    case MUL:
      return (Identical(op->args[0], x) && AlwaysOne(op->args[1])) ||
             (Identical(op->args[1], x) && AlwaysOne(op->args[0]));
    default:
      return false;
  }
}

bool Express::DependsOn(Var *y, Var *x) const {
  if (x == y) return true;
  if (y->producer != nullptr) {
    for (Var *arg : y->producer->args) {
      if (DependsOn(arg, x)) return true;
    }
  }
  return false;
}

void Express::EliminateInput(int id) {
  for (Var *v : vars_) {
    if ((v->type == INPUT || v->type == CONST) && v->id > id) v->id--;
  }
}

int Express::CompactTempVars() {
  int n = 0;
  for (Var *v : vars_) {
    if (v->type == REGISTER) n = v->id + 1;
  }
  for (Var *v : vars_) {
    if (v->type == TEMP) v->id = n++;
  }
  return n;
}

void Express::EliminateRedundantMoves() {
  int eliminated = 0;
  for (int i = 0; i < ops_.size(); ++i) {
    Op *op = ops_[i];
    if (op->type != MOV) continue;
    Var *src = op->args[0];
    Var *dst = op->result;
    if (src->usages() != 1) continue;
    if (dst->type != TEMP) continue;

    dst->Redirect(src);
    RemoveOp(op);
    RemoveVar(dst);
    eliminated++;
    i--;
  }
  if (eliminated > 0) CompactTempVars();
}

void Express::EliminateCommonSubexpressions() {
  // Coalesce system constant variables.
  std::map<int, Var *> sysvars;
  std::vector<Var *> unused;
  for (Var *v : vars_) {
    if (v->type == NUMBER) {
      auto f = sysvars.find(v->id);
      if (f == sysvars.end()) {
        sysvars[v->id] = v;
      } else {
        v->Redirect(f->second);
        unused.push_back(v);
      }
    }
  }
  for (Var *v : unused) RemoveVar(v);

  // Keep trying to eliminate ops until no more can be removed.
  int iterations = 0;
  while (TryToEliminateOps()) iterations++;

  // Compact temporary variables if some operations were eliminated.
  if (iterations > 0) CompactTempVars();
}

bool Express::TryToEliminateOps() {
  // Find matching ops.
  for (int i = 0; i < ops_.size(); ++i) {
    Op *op1 = ops_[i];
    for (int j = i + 1; j < ops_.size(); ++j) {
      Op *op2 = ops_[j];
      if (op1->EqualTo(op2)) {
        Var *v1 = op1->result;
        Var *v2 = op2->result;
        if (v1->type == TEMP) {
          // Eliminate ith operation.
          std::swap(ops_[i], ops_[j]);
          v1->Redirect(v2);
          RemoveOp(op1);
          RemoveVar(v1);
          return true;
        } else if (v2->type == TEMP) {
          // Eliminate jth operation.
          v2->Redirect(v1);
          RemoveOp(op2);
          RemoveVar(v2);
          return true;
        } else {
          // Two output variables. Change second op to move op.
          v2->Redirect(v1);
          op2->type = MOV;
          op2->ClearArguments();
          op2->AddArgument(v1);
          return true;
        }
      }
    }
  }
  return false;
}

void Express::Hoist(int limit) {
  // Collect all existing hoisted variables.
  std::set<Var *> hoisted;
  for (int i = 0; i < body_; ++i) {
    hoisted.insert(ops_[i]->result);
  }

  // Hoist const loads outside the body until limit reached.
  int new_temps = 0;
  for (int r = 0; r < limit; ++r) {
    // Find constant or number variable with the most usages.
    Var *candidate = nullptr;
    for (Var *v : vars_) {
      if (v->unhoistable) continue;
      if (v->type == CONST || v->type == NUMBER) {
        if (hoisted.count(v) == 0) {
          if (candidate == nullptr || v->usages() > candidate->usages()) {
            candidate = v;
          }
        }
      }
    }

    // Stop if no candidate for hoisting was found.
    if (candidate == nullptr || candidate->usages() == 0) break;

    // Allocate temp for constant and update all usages.
    Var *temp = Temp();
    candidate->consumers.swap(temp->consumers);
    for (Op *o : ops_) {
      for (int i = 0; i < o->args.size(); ++i) {
        if (o->args[i] == candidate) {
          o->args[i] = temp;
        }
      }
    }

    // Assign constant to temp variable.
    Op *assign = OperationBefore(ops_[body_], MOV);
    assign->Assign(temp);
    assign->AddArgument(candidate);
    body_++;
    hoisted.insert(candidate);
    hoisted.insert(temp);
    new_temps++;
  }
  if (new_temps > 0) CompactTempVars();

  // Single element inputs and constants are also considered hoisted since
  // these are by definition loop invariant.
  for (Var *var : vars_) {
    if (var->type == NUMBER || var->type == CONST || var->single) {
      hoisted.insert(var);
    }
  }

  // Hoist loop-invariant operations.
  bool again = true;
  while (again) {
    again = false;
    for (int i = body_; i < ops_.size(); ++i) {
      Op *op = ops_[i];

      // Check if all arguments are cached.
      bool invariant = !op->result->unhoistable;
      for (Var *arg : op->args) {
        if (hoisted.count(arg) == 0 || arg->unhoistable) {
          invariant = false;
          break;
        }
      }

      // Never hoist reduction ops.
      if (op->reduction()) continue;

      // Move instruction out of the body if it is loop-invariant.
      if (invariant) {
        for (int j = i; j > body_; --j) ops_[j] = ops_[j - 1];
        ops_[body_++] = op;
        hoisted.insert(op->result);
        again = true;
        break;
      }
    }
  }
}

void Express::CacheResults() {
  int cached_vars = 0;
  for (int n = 0; n < vars_.size(); ++n) {
    Var *var = vars_[n];
    if (var->type == OUTPUT && var->usages() > 0) {
      // Make temp variable and update all usages to use this instead.
      Op *op = var->producer;
      CHECK(op != nullptr);
      var->producer = nullptr;
      Var *temp = Temp();
      op->result = temp;
      var->consumers.swap(temp->consumers);
      for (Op *o : ops_) {
        for (int i = 0; i < o->args.size(); ++i) {
          if (o->args[i] == var) o->args[i] = temp;
        }
      }

      // Assign temp variable to output.
      Op *assign = OperationAfter(op, MOV);
      assign->Assign(var);
      assign->AddArgument(temp);
      cached_vars++;
    } else if (var->type == INPUT && var->usages() > 0) {
      // Single-element input variables must be cached but other input variables
      // are only cached if there are two or more usages.
      if (var->single || var->usages() > 1) {
        // Make temp variable and update all usages to use this instead.
        Var *temp = Temp();
        var->consumers.swap(temp->consumers);
        Op *first = nullptr;
        for (Op *o : ops_) {
          for (int i = 0; i < o->args.size(); ++i) {
            if (o->args[i] == var) {
              o->args[i] = temp;
              if (first == nullptr) first = o;
            }
          }
        }
        CHECK(first != nullptr);

        // Assign temp variable to input.
        Op *assign = OperationBefore(first, MOV);
        assign->Assign(temp);
        assign->AddArgument(var);
        cached_vars++;
      }
    }
  }
  if (cached_vars > 0) CompactTempVars();
}

void Express::ComputeLiveRanges() {
  // All variables assigned before the start of the body need to have their live
  // range extended to the end.
  if (ops_.empty()) return;
  Op *begin = ops_.front();
  Op *end = ops_.back();
  for (int i = 0; i < body_; ++i) {
    ops_[i]->result->last = end;
  }

  // Inputs must be kept alive from the beginning and outputs must be kept
  // alive until the end.
  for (Var *var : vars_) {
    if (var->type == INPUT) var->first = begin;
    if (var->type == OUTPUT) var->last = end;
  }

  // Compute live ranges for the remaining variables.
  int index = 0;
  for (Op *op : ops_) {
    op->index = index++;
    if (op->result->first == nullptr) op->result->first = op;
    if (op->result->last != end) op->result->last = op;
    for (Var *arg : op->args) {
      if (arg->first == nullptr) arg->first = op;
      if (arg->last != end) arg->last = op;
    }
  }
}

int Express::MaxActiveTemps() const {
  int active = 0;
  int max_active = 0;
  for (Op *op : ops_) {
    if (op->result->first == op && op->result->type == TEMP) active++;
    if (active > max_active) max_active = active;
    for (Var *arg : op->args) {
      if (arg->last == op && arg->type == TEMP) active--;
    }
  }
  return max_active;
}

void Express::Copy(const Express &other) {
  // Expression must be empty.
  CHECK(vars_.empty());
  CHECK(ops_.empty());
  vars_.reserve(other.vars().size());
  ops_.reserve(other.ops().size());
  body_ = other.body_;

  // Copy variables.
  std::map<Var *, Var *> varmap;
  for (Var *var : other.vars()) {
    Var *v = new Var(*var);
    vars_.push_back(v);
    varmap[var] = v;
  }

  // Copy operations.
  std::map<Op *, Op *> opmap;
  for (Op *op : other.ops()) {
    Op *o = new Op(*op);
    ops_.push_back(o);
    opmap[op] = o;
  }

  // Map pointers.
  for (Var *var : vars_) {
    var->producer = opmap[var->producer];
    for (Op *&op : var->consumers) op = opmap[op];
    var->first = opmap[var->first];
    var->last = opmap[var->last];
  }
  for (Op *op : ops_) {
    op->result = varmap[op->result];
    for (Var *&var : op->args) var = varmap[var];
  }
}

void Express::Merge(Express *other, const Map &varmap) {
  // Move variables that are not mapped.
  bool temps_moved = false;
  for (Var *var : other->vars_) {
    auto f = varmap.find(var);
    if (f == varmap.end()) {
      vars_.push_back(var);
      if (var->type == TEMP) temps_moved = true;
    } else {
      delete var;
    }
  }

  // Move operations and map arguments.
  for (Op *op : other->ops_) {
    ops_.push_back(op);
    auto f = varmap.find(op->result);
    if (f != varmap.end()) {
      op->result = f->second;
      f->second->producer = op;
    }
    for (int i = 0; i < op->args.size(); ++i) {
      auto f = varmap.find(op->args[i]);
      if (f != varmap.end()) {
        op->args[i] = f->second;
        f->second->consumers.push_back(op);
      }
    }
  }

  // Clear the vars and ops in the other expression.
  other->vars_.clear();
  other->ops_.clear();

  // Rename temporary variables if needed.
  if (temps_moved) CompactTempVars();
}

void Express::Translate(Express *translated) const {
  VariableMap varmap(translated);
  std::vector<Var *> args;
  for (Op *op : ops_) {
    // Translate arguments.
    args.clear();
    for (Var *arg : op->args) args.push_back(varmap[arg]);

    // Translate operation.
    Var *result = nullptr;
    if (!translated->Supports(op->type)) {
      // Try to expand unsupported operation.
      result = translated->Expand(op->type, args);
      if (result != nullptr) {
        varmap.rename(op->result, result);
      } else {
        LOG(WARNING) << "Cannot translate unsupported op " << OpName(op->type)
                     << " in " << AsRecipe();
      }
    }
    if (result == nullptr) {
      // Copy operation verbatim.
      Op *copy = translated->Operation(op->type);
      for (Var *arg : args) copy->AddArgument(arg);
      copy->Assign(varmap[op->result]);
    }
  }
  translated->CompactTempVars();
}

void Express::Fuse(OpType outer, OpType inner, OpType left, OpType right) {
  bool again = true;
  while (again) {
    again = false;
    for (Op *op : ops_) {
      if (op->type != outer) continue;
      if (op->arity() != 2) continue;
      if (TryFuseFirst(op, inner, left)) {
        again = true;
      } else if (TryFuseSecond(op, inner, right)) {
        again = true;
      }
      if (again) break;
    }
  }
}

bool Express::TryFuseFirst(Op *op, OpType type, OpType combined) {
  // Check if combined op is supported.
  if (combined == INVALID) return false;

  // Check if intermediate variable is only use as an intermediate.
  Var *intermediate = op->args[0];
  if (!intermediate->inlined()) return false;

  // Check that the producer of the intermediate variable is the correct type.
  Op *sub = intermediate->producer;
  if (sub == nullptr || sub->type != type) return false;
  if (sub->arity() != 2) return false;

  // Combine ops.
  Var *a = sub->args[0];
  Var *b = sub->args[1];
  Var *c = op->args[1];

  op->type = combined;
  op->ClearArguments();
  op->AddArgument(a);
  op->AddArgument(b);
  op->AddArgument(c);

  RemoveOp(sub);
  RemoveVar(intermediate);

  return true;
}

bool Express::TryFuseSecond(Op *op, OpType type, OpType combined) {
  // Check if combined op is supported.
  if (combined == INVALID) return false;

  // Check if intermediate variable is only use as an intermediate.
  Var *intermediate = op->args[1];
  if (!intermediate->inlined()) return false;

  // Check that the producer of the intermediate variable is the correct type.
  Op *sub = intermediate->producer;
  if (sub == nullptr || sub->type != type) return false;
  if (sub->arity() != 2) return false;

  // Combine ops.
  Var *a = op->args[0];
  Var *b = sub->args[0];
  Var *c = sub->args[1];

  op->type = combined;
  op->ClearArguments();
  op->AddArgument(a);
  op->AddArgument(b);
  op->AddArgument(c);

  RemoveOp(sub);
  RemoveVar(intermediate);

  return true;
}

void Express::Optimize(bool fma, int spare_regs) {
  // Eliminate common subexpressions.
  EliminateCommonSubexpressions();

  // Optionally fuse multiply and add/sub.
  if (fma) {
    FuseMulAdd();
    if (Supports(MULSUB132)) FuseMulSub();
  }

  // Cache inputs and results used in multiple ops in temporary variables.
  CacheResults();

  // Hoist loop-invariant ops out of the body. The spare registers are used for
  // pre-loading constants.
  Hoist(spare_regs);
}

bool Express::Rewrite(const Model &model, Express *rewritten) const {
  // Target expression must be empty.
  CHECK_EQ(rewritten->vars().size(), 0);

  // Mapping from original variables to variables in rewritten expression.
  VariableMap varmap(rewritten);

  // Translate all ops to conform to target model.
  bool success = true;
  for (Op *op : ops_) {
    // Get operation type, result, and arguments.
    OpType type = op->type;
    Var *result = op->result;
    std::vector<Var *> args = op->args;
    Var *source = nullptr;
    Var *source2 = nullptr;
    Var *source3 = nullptr;
    Var *destination = nullptr;
    bool first_is_dest = false;

    // Keep track of the beginning of the body.
    if (op == ops_[body_]) rewritten->body_ = rewritten->ops_.size();

    // Rewrite operation.
    if (op->arity() == 1) {
      if (type == MOV) {
        // Move operations.
        switch (result->type) {
          case TEMP:
          case REGISTER:
            // Move value into register.
            switch (args[0]->type) {
              case INPUT:
              case OUTPUT:
                if (!model.mov_reg_mem) success = false;
                break;
              case TEMP:
              case REGISTER:
                if (!model.mov_reg_reg) success = false;
                break;
              case CONST:
              case NUMBER:
                if (!model.mov_reg_imm) success = false;
                break;
            }
            break;

          case OUTPUT:
            // Move value into output variable.
            switch (args[0]->type) {
              case INPUT:
                // Add temp variable for input.
                source = rewritten->Temp();
                break;
              case OUTPUT:
                // Add temp variable for output.
                destination = rewritten->Temp();
                break;
              case TEMP:
              case REGISTER:
                if (!model.mov_mem_reg) success = false;
                break;
              case CONST:
              case NUMBER:
                if (!model.mov_mem_imm) {
                  // Add temp variable for constant.
                  destination = rewritten->Temp();
                }
                break;
            }
            break;

          case INPUT:
          case CONST:
          case NUMBER:
            // Assignment to inputs and constants not allowed.
            success = false;
            break;
        }
      } else {
        // Unary operator. Reductions are output to accumulator.
        switch (op->reduction() ? REGISTER : result->type) {
          case TEMP:
          case REGISTER:
            switch (args[0]->type) {
              case INPUT:
              case OUTPUT:
                if (!model.func_reg_mem || args[0]->single ||
                    (model.logic_in_regs && op->logic())) {
                  // Add temp variable for input.
                  source = rewritten->Temp();
                  if (!model.func_reg_reg) success = false;
                }
                break;
              case TEMP:
              case REGISTER:
                if (!model.func_reg_reg) success = false;
                break;
              case CONST:
              case NUMBER:
                if (!model.func_reg_imm) {
                  // Add temp variable for input.
                  source = rewritten->Temp();
                  if (!model.func_reg_reg) success = false;
                }
                break;
            }
            break;

          case OUTPUT:
            switch (args[0]->type) {
              case INPUT:
              case OUTPUT:
                if (model.logic_in_regs && op->logic()) {
                  // Add temp variables for input and output.
                  destination = rewritten->Temp();
                  source = rewritten->Temp();
                  if (!model.func_reg_reg) success = false;
                } else if (model.func_reg_mem) {
                  // Add temp variable for output.
                  destination = rewritten->Temp();
                } else if (model.func_mem_reg) {
                  // Add temp variable for input.
                  source = rewritten->Temp();
                } else {
                  // Add temp variables for input and output.
                  destination = rewritten->Temp();
                  source = rewritten->Temp();
                  if (!model.func_reg_reg) success = false;
                }
                break;
              case TEMP:
              case REGISTER:
                if (!model.func_mem_reg ||
                    (model.logic_in_regs && op->logic())) {
                  // Add temp variable for output.
                  destination = rewritten->Temp();
                  if (!model.func_reg_reg) success = false;
                }
                break;
              case CONST:
              case NUMBER:
                if (!model.func_mem_imm) {
                  // Add temp variable for output.
                  destination = rewritten->Temp();
                  if (!model.func_reg_imm) {
                    // Add temp variable for input.
                    source = rewritten->Temp();
                    if (!model.func_reg_reg) success = false;
                  }
                }
                break;
            }
            break;

          case INPUT:
          case CONST:
          case NUMBER:
            // Assignment to inputs and constants not allowed.
            success = false;
        }
      }
    } else if (op->arity() == 2 && type != SELECT) {
      // Binary operator.
      switch (result->type) {
        case TEMP:
        case REGISTER:
        case OUTPUT:
          if (model.op_reg_reg_reg) {
            // Three-operand instruction. Try to put the memory operand last if
            // operation is commutative.
            if (model.op_reg_reg_mem &&
                (op->commutative() || op->compare()) &&
                !args[0]->IsRegister() && args[1]->IsRegister()) {
              type = Commute(type);
              std::swap(args[0], args[1]);
            }

            // Put first argument into a register.
            if (!args[0]->IsRegister()) {
              source = rewritten->Temp();
            }

            // Put second argument into a register if memory operands are not
            // supported.
            if (model.logic_in_regs && op->logic()) {
              source2 = rewritten->Temp();
            } else if (args[1]->type == CONST || args[1]->type == NUMBER) {
              if (!model.op_reg_reg_imm) {
                source2 = rewritten->Temp();
              }
            } else if (!args[1]->IsRegister()) {
              if (!model.op_reg_reg_mem || args[1]->single) {
                source2 = rewritten->Temp();
              }
            }

            // Put destination into a register if memory destinations are not
            // supported or if second argument is not in a register.
            if (model.logic_in_regs && op->logic()) {
              destination = rewritten->Temp();
            } else {
              bool arg1_in_reg = args[1]->IsRegister() || source2 != nullptr;
              if (result->type == OUTPUT &&
                  (!arg1_in_reg || !model.op_mem_reg_reg)) {
                destination = rewritten->Temp();
              }
            }
          } else if (model.op_reg_reg) {
            // Two-operand instruction.
            Var *dest = result;
            first_is_dest = true;

            // Try to put the memory operand last if operation is commutative.
            if (model.op_reg_mem &&
                (op->commutative() || op->compare()) &&
                !args[0]->IsRegister() && args[1]->IsRegister()) {
              type = Commute(type);
              std::swap(args[0], args[1]);
            }

            // Put result and first argument in the same location.
            if (result != args[0] || !model.op_mem_reg) {
              // Put result in temp register if result is an output.
              if (result->type == OUTPUT) {
                dest = destination = rewritten->Temp();
              }

              // Move first argument to destination.
              Op *mov = rewritten->Operation(MOV);
              mov->Assign(varmap[dest], true);
              mov->AddArgument(varmap[args[0]]);
              switch (args[0]->type) {
                case INPUT:
                case OUTPUT:
                  if (!model.mov_reg_mem) success = false;
                  break;
                case TEMP:
                case REGISTER:
                  if (!model.mov_reg_reg) success = false;
                  break;
                case CONST:
                case NUMBER:
                  if (!model.mov_reg_imm) success = false;
                  break;
              }
              args[0] = dest;
            }

            // Make second argument available for instruction.
            switch (args[1]->type) {
              case INPUT:
              case OUTPUT:
                // Put second operand into register if memory operands are not
                // supported.
                if (dest->type != TEMP ||
                    !model.op_reg_mem ||
                    args[1]->single ||
                    (model.logic_in_regs && op->logic())) {
                  source2 = rewritten->Temp();
                }
                break;
              case TEMP:
              case REGISTER:
                break;
              case CONST:
              case NUMBER:
                // Put second operand into register if immediate operands are
                // not supported.
                if (dest->type == TEMP) {
                  if (!model.op_reg_imm) {
                    source2 = rewritten->Temp();
                  }
                } else {
                  if (!model.op_mem_imm) {
                    source2 = rewritten->Temp();
                  }
                }
                break;
            }
          } else {
            success = false;
          }
          break;

        case INPUT:
        case CONST:
        case NUMBER:
          // Assignment to inputs and constants not allowed.
          success = false;
          break;
      }
    } else if (op->arity() == 2 && type == SELECT) {
      // Put predicate into a register.
      if (!args[0]->IsRegister()) {
        source = rewritten->Temp();
      }

      // Put source into a register if memory operands are not available.
      if (args[1]->type == CONST || args[1]->type == NUMBER) {
        if (!model.func_reg_imm) {
          source2 = rewritten->Temp();
        }
      } else if (!args[1]->IsRegister()) {
        if (!model.func_reg_mem || args[1]->single) {
          source2 = rewritten->Temp();
        }
      }

      // Put destination into a register.
      if (!result->IsRegister()) {
        destination = rewritten->Temp();
      }
    } else if (op->arity() == 3) {
      // Ternary operator.
      Var *dest = result;

      if (op->fma()) {
        // Fused multiply.
        first_is_dest = true;

        // Try to put memory operand last for FMA.
        if (model.fm_reg_reg_mem) {
          if (!args[1]->IsRegister() && args[2]->IsRegister()) {
            // Swap second and third argument.
            std::swap(args[1], args[2]);
            switch (type) {
              case MULADD132: type = MULADD213; break;
              case MULADD213: type = MULADD132; break;
              case MULADD231: break;
              case MULSUB132: type = MULSUB213; break;
              case MULSUB213: type = MULSUB132; break;
              case MULSUB231: break;
              default: success = false;
            }
          } else if (!args[0]->IsRegister() && args[2]->IsRegister()) {
            // Swap first and third argument.
            std::swap(args[0], args[2]);
            switch (type) {
              case MULADD132: break;
              case MULADD213: type = MULADD231; break;
              case MULADD231: type = MULADD213; break;
              case MULSUB132: break;
              case MULSUB213: type = MULSUB231; break;
              case MULSUB231: type = MULSUB213; break;
              default: success = false;
            }
          }
        }

        // Put result and first argument in the same location.
        if (result != args[0]) {
          // Put result in temp register if result is an output.
          if (result->type == OUTPUT) {
            dest = destination = rewritten->Temp();
          }

          // Move first argument to destination.
          Op *mov = rewritten->Operation(MOV);
          mov->Assign(varmap[dest], true);
          mov->AddArgument(varmap[args[0]]);
          switch (args[0]->type) {
            case INPUT:
            case OUTPUT:
              if (!model.mov_reg_mem) success = false;
              break;
            case TEMP:
            case REGISTER:
              if (!model.mov_reg_reg) success = false;
              break;
            case CONST:
            case NUMBER:
              if (!model.mov_reg_imm) success = false;
              break;
          }
          args[0] = dest;
        }

        // Make sure second operand is in register.
        if (!args[1]->IsRegister()) {
          source2 = rewritten->Temp();
        }

        // Make third argument available for instruction.
        if (args[2]->type == CONST || args[2]->type == NUMBER) {
          if (!model.fm_reg_reg_imm) {
            source3 = rewritten->Temp();
          }
        } else if (!args[2]->IsRegister()) {
          if (!model.fm_reg_reg_mem || args[2]->single) {
            source3 = rewritten->Temp();
          }
        }
      } else if (op->type == COND) {
        // Put predicate into a register.
        if (!args[0]->IsRegister()) {
          source = rewritten->Temp();
        }

        // Put second operand into a register if memory operands are not
        // available.
        if (args[1]->IsRegister()) {
          if (!model.cond_reg_reg_reg && !model.cond_reg_reg_mem) return false;
        } else {
          if (!(model.cond_reg_mem_reg || model.cond_reg_mem_mem) ||
              args[1]->single) {
            source2 = rewritten->Temp();
          }
        }

        // Put third operand into a register if memory operands are not
        // available.
        if (args[2]->IsRegister()) {
          if (!model.cond_reg_reg_reg && !model.cond_reg_mem_reg) return false;
        } else {
          if (!(model.cond_reg_reg_mem || model.cond_reg_mem_mem) ||
              args[2]->single) {
            source3 = rewritten->Temp();
          }
        }

        // Put destination into a register.
        if (!result->IsRegister()) {
          destination = rewritten->Temp();
        }
      }
    } else {
      LOG(WARNING) << "Unsupported op: " << op->AsString();
      success = false;
    }

    // Assign first argument to source.
    if (source != nullptr) {
      if (!model.mov_reg_mem) success = false;
      Op *mov = rewritten->Operation(MOV);
      mov->Assign(source);
      mov->AddArgument(varmap[args[0]]);
      args[0] = source;
    }

    // Assign second argument to source2.
    if (source2 != nullptr) {
      if (!model.mov_reg_mem) success = false;
      Op *mov = rewritten->Operation(MOV);
      mov->Assign(source2);
      mov->AddArgument(varmap[args[1]]);
      args[1] = source2;
    }

    // Assign third argument to source3.
    if (source3 != nullptr) {
      if (!model.mov_reg_mem) success = false;
      Op *mov = rewritten->Operation(MOV);
      mov->Assign(source3);
      mov->AddArgument(varmap[args[2]]);
      args[2] = source3;
    }

    // Translate operation.
    Op *instr = rewritten->Operation(type);
    instr->first_is_dest = first_is_dest;
    if (result != nullptr) {
      if (destination != nullptr) {
        // Use destination as temporary for result.
        if (!model.mov_mem_reg) success = false;
        instr->Assign(destination, true);
        Op *mov = rewritten->Operation(MOV);
        mov->Assign(varmap[result], true);
        mov->AddArgument(destination);
      } else {
        // Assign directly to result.
        instr->Assign(varmap[result], true);
      }
    }
    for (Var *arg : args) {
      instr->AddArgument(varmap[arg]);
    }
  }

  rewritten->CompactTempVars();
  return success;
}

int Express::AllocateRegisters(bool predicate_regs) {
  RegisterAllocator regs(predicate_regs);

  // Allocate registers for register-based variables.
  for (Var *var : vars_) {
    if (var->type == REGISTER) regs.Allocate(var);
  }

  // Allocate/get/free destination and source registers for ops.
  std::vector<Var *> free_vars;
  std::vector<int> free_regs;
  for (Op *op : ops_) {
    // Get registers for source operands.
    free_vars.clear();
    for (int i = 0; i < op->arity(); ++i) {
      // Register should already be allocated for source arguments.
      Var *arg = op->args[i];
      if (!arg->IsRegister()) continue;
      int reg = regs.Get(arg);
      CHECK(reg != -1) << i;

      // Assign register.
      if (op->conditional()) {
        switch (i) {
          case 0: op->mask = reg; break;
          case 1: op->src = reg; break;
          case 2: op->src2 = reg; break;
          default: LOG(FATAL) << "Too many arguments";
        }
      } else if (op->first_is_dest) {
        switch (i) {
          case 0: op->dst = reg; break;
          case 1: op->src = reg; break;
          case 2: op->src2 = reg; break;
          default: LOG(FATAL) << "Too many arguments";
        }
      } else {
        switch (i) {
          case 0: op->src = reg; break;
          case 1: op->src2 = reg; break;
          default: LOG(FATAL) << "Too many arguments";
        }
      }

      // Free register if this op is the last usage.
      if (arg->type == TEMP && arg->last == op) {
        free_vars.push_back(arg);
      }
    }

    // Free source registers after last usage.
    free_regs.clear();
    for (Var *v : free_vars) {
      if (regs.Allocated(v)) {
        free_regs.push_back(v->reg);
        regs.Free(v);
      }
    }

    // Get register for result.
    if (op->result->type == TEMP) {
      if (op->result->first == op) {
        // Allocate register for result. Steal source register if available.
        for (int r : free_regs) {
          if (op->dst != -1) break;
          op->dst = regs.Allocate(op->result, r);
        }
        if (op->dst == -1) {
          // Allocate new register.
          op->dst = regs.Allocate(op->result);
        }
      } else {
        op->dst = regs.Get(op->result);
      }
      CHECK(op->dst != -1);
    } else if (op->result->type == REGISTER) {
      op->dst = op->result->reg;
      CHECK(op->dst != -1);
    }

    // Get register for accumulation.
    if (op->reduction()) {
      op->acc = regs.AllocateExtra(op->preduction());
      CHECK(op->acc != -1);
    }
  }

  return regs.max();
}

bool Express::Generate(const Model &model, Express *rewritten) const {
  // Rewrite expression using instruction model.
  if (!Rewrite(model, rewritten)) return false;

  // Compute live ranges for all variables.
  rewritten->ComputeLiveRanges();

  // Allocate registers for temporary variables.
  rewritten->AllocateRegisters(model.predicate_regs);

  return true;
}

int Express::NumRegs() const {
  int num_regs = 0;
  for (auto *var : vars_) {
    if (var->reg != -1 && var->reg + 1 > num_regs) num_regs = var->reg + 1;
  }
  for (auto *op : ops_) {
    if (op->acc != -1 && op->acc + 1 > num_regs) num_regs = op->acc + 1;
  }
  return num_regs;
}

void Express::GetRegisterTypes(std::vector<bool> *regs) const {
  regs->clear();
  for (auto *var : vars_) {
    if (var->reg == -1) continue;
    if (var->reg >= regs->size()) regs->resize(var->reg + 1);
    if (var->predicate) (*regs)[var->reg] = true;
  }
  for (auto *op : ops_) {
    if (op->acc == -1) continue;
    if (op->acc >= regs->size()) regs->resize(op->acc + 1);
    if (op->preduction()) (*regs)[op->acc] = true;
  }
}

// Natural logarithm.
// See also: http://software-lisc.fbk.eu/avx_mathfun/avx_mathfun.h
Express::Var *Express::Log(Var *x) {
  if (Supports(LOG2)) {
    // Compute natural logarithm from base-2 logarithm.
    return Mul(Do(LOG2, x), Number(LN2));
  } else {
    // Logarithm of negative input is NaN.
    Var *valid = CmpGe(x, Zero());

    // Truncate input values to the minimum positive normal.
    x = Maximum(x, Number(MIN_NORM_POS));

    // Part 1: x = frexpf(x, e).
    Var *emm0 = Do(CVTEXPINT, x);
    emm0 = Do(SUBINT, emm0, Number(MAX_MANT));
    Var *e = Add(Do(CVTINTFLT, emm0), One());

    // Keep only the fractional part.
    x = Do(BITAND, x, Number(INV_MANT_MASK));
    x = Do(BITOR, x, Number(HALF));

    // Part 2: Shift the inputs from the range [0.5,1) to [sqrt(1/2),sqrt(2)]
    // and shift by -1. The values are then centered around 0, which improves
    // the stability of the polynomial evaluation.
    //   if (x < SQRTHF) {
    //     e -= 1;
    //     x = x + x - 1.0;
    //   } else {
    //     x = x - 1.0;
    //   }
    Var *mask = CmpLt(x, Number(CEPHES_SQRTHF));
    Var *tmp = Select(mask, x);
    x = Sub(x, One());
    e = Sub(e, Select(mask, One()));
    x = Add(x, tmp);
    Var *z = Square(x);

    // Part 3: Compute the polynomial approximation.
    Var *y = Number(CEPHES_LOG_P0);
    y = MulAdd(y, x, Number(CEPHES_LOG_P1));
    y = MulAdd(y, x, Number(CEPHES_LOG_P2));
    y = MulAdd(y, x, Number(CEPHES_LOG_P3));
    y = MulAdd(y, x, Number(CEPHES_LOG_P4));
    y = MulAdd(y, x, Number(CEPHES_LOG_P5));
    y = MulAdd(y, x, Number(CEPHES_LOG_P6));
    y = MulAdd(y, x, Number(CEPHES_LOG_P7));
    y = MulAdd(y, x, Number(CEPHES_LOG_P8));
    y = Mul(y, x);
    y = Mul(y, z);

    tmp = Mul(e, Number(CEPHES_LOG_Q1));
    y = Add(y, tmp);
    tmp = Mul(z, Number(HALF));
    y = Sub(y, tmp);
    tmp = Mul(e, Number(CEPHES_LOG_Q2));
    x = Add(x, y);
    x = Add(x, tmp);

    x = Cond(valid, x, Number(QNAN));  // negative arg will be NaN

    return x;
  }
}

// Exponential function.
// Works by writing x = m*log(2) + r, where m = floor(x/log(2)+1/2) and r is
// the remainder. The result is then exp(x) = 2^m*exp(r), where exp(r) is in the
// range [-1,1).
// See also: https://git.io/vHyVR
Express::Var *Express::Exp(Var *x) {
  if (Supports(EXP2)) {
    // Compute e^x = 2^(x * log2(e)).
    return Do(EXP2, Mul(x, Number(LOG2E)));
  } else {
    // Clamp x.
    Var *arg = x;
    x = Maximum(Minimum(x, Number(EXP_HI)), Number(EXP_LO));

    // Express exp(x) as exp(m*ln(2) + r), start by extracting
    // m = floor(x/ln(2) + 0.5).
    Var *m = Do(FLOOR, MulAdd(x, Number(CEPHES_LOG2EF), Number(HALF)));

    // Compute r = x - m*ln(2).
    Var *r = MulAdd(m, Number(NLN2), x);

    // Compute r^2.
    Var *r2 = Square(r);

    // Compute polynomial.
    Var *y = Number(CEPHES_EXP_P0);
    y = MulAdd(y, r, Number(CEPHES_EXP_P1));
    y = MulAdd(y, r, Number(CEPHES_EXP_P2));
    y = MulAdd(y, r, Number(CEPHES_EXP_P3));
    y = MulAdd(y, r, Number(CEPHES_EXP_P4));
    y = MulAdd(y, r, Number(CEPHES_EXP_P5));
    y = MulAdd(y, r2, r);
    y = Add(y, One());

    // Compute emm0 = 2^m.
    Var *emm0 = Do(CVTINTEXP, Do(CVTFLTINT, Add(m, Number(EXP_BIAS))));

    // Return 2^m * exp(r).
    return Maximum(Mul(y, emm0), arg);
  }
}

// Trigonometric functions.
// See also: http://software-lisc.fbk.eu/avx_mathfun/avx_mathfun.h
Express::Var *Express::Trig(OpType type, Var *x) {
  // Compute directly using sin and cos if supported.
  if (Supports(SIN) && Supports(COS)) {
    switch (type) {
      case SIN: return Do(SIN, x);
      case COS: return Do(COS, x);
      case TAN: return Div(Do(SIN, x), Do(COS, x));
      case COT: return Div(Do(COS, x), Do(SIN, x));
      case SEC: return Reciprocal(Do(COS, x));
      case CSC: return Reciprocal(Do(SIN, x));
      default: LOG(FATAL) << "Unsupported trigonometric function";
    }
  }

  // Check if sine and/or cosine are needed.
  bool sin_needed = (type != COS && type != SEC);
  bool cos_needed = (type != SIN && type != CSC);

  // Extract the sign bit for sine.
  Var *sign = nullptr;
  if (sin_needed) {
    sign = Do(BITAND, x, Number(SIGN_MASK));
  }

  // Take the absolute value.
  x = Do(BITAND, x, Number(INV_SIGN_MASK));

  // Scale by 4/pi.
  Var *y = Mul(x, Number(FOPI));

  // Get the integer part of y.
  Var *j = Do(CVTFLTINT, y);

  // Compute j=(j+1) & (~1).
  j = Do(ADDINT, j, Number(I1));
  j = Do(BITAND, j, Number(INV_I1));
  y = Do(CVTINTFLT, j);

  // Get the sign swap masks.
  Var *signswap_sin = nullptr;
  Var *signswap_cos = nullptr;
  Var *four = Number(I4);
  if (sin_needed) {
    signswap_sin = Do(BITXOR, sign, Do(QUADSIGN, Do(BITAND, j, four)));
  }
  if (cos_needed) {
    signswap_cos = Do(QUADSIGN, Do(BITANDNOT, Do(SUBINT, j, Number(I2)), four));
  }

  // Get the polynomial selection mask. There is one polynomial for 0<=x<=pi/4
  // and another one for pi/4<x<=pi/2. Both branches will be computed.
  Var *polymask = Do(BITEQ, Do(BITAND, j, Number(I2)), Zero());

  // Extended precision modular arithmetic.
  // x = ((x - y * DP1) - y * DP2) - y * DP3
  x = Add(x, Mul(y, Number(CEPHES_MINUS_DP1)));
  x = Add(x, Mul(y, Number(CEPHES_MINUS_DP2)));
  x = Add(x, Mul(y, Number(CEPHES_MINUS_DP3)));

  // Evaluate the first polynomial (0<=x<=pi/4).
  Var *z = Mul(x, x);
  Var *p1 = Number(COSCOF_P0);
  p1 = Mul(p1, z);
  p1 = Add(p1, Number(COSCOF_P1));
  p1 = Mul(p1, z);
  p1 = Add(p1, Number(COSCOF_P2));
  p1 = Mul(p1, z);
  p1 = Mul(p1, z);
  p1 = Sub(p1, Mul(z, Number(HALF)));
  p1 = Add(p1, One());

  // Evaluate the second polynomial (pi/4<=x<=pi/2).
  Var *p2 = Number(SINCOF_P0);
  p2 = Mul(p2, z);
  p2 = Add(p2, Number(SINCOF_P1));
  p2 = Mul(p2, z);
  p2 = Add(p2, Number(SINCOF_P2));
  p2 = Mul(p2, z);
  p2 = Mul(p2, x);
  p2 = Add(p2, x);

  // Compute sine and cosine.
  Var *sin = nullptr;
  Var *cos = nullptr;
  if (sin_needed) {
    sin = Do(BITXOR, Cond(polymask, p2, p1), signswap_sin);
  }
  if (cos_needed) {
    Var *y1 = Sub(p1, Select(Not(polymask), p1));
    Var *y2 = Sub(p2, Select(polymask, p2));
    cos = Do(BITXOR, Add(y1, y2), signswap_cos);
  }

  // Return result.
  switch (type) {
    case SIN: return sin;
    case COS: return cos;
    case TAN: return Div(sin, cos);
    case COT: return Div(cos, sin);
    case SEC: return Reciprocal(cos);
    case CSC: return Reciprocal(sin);
    default: LOG(FATAL) << "Unsupported trigonometric function";
  }
}

Express::Var *Express::Asin(Var *x) {
  return Atan(Mul(x, Rsqrt(Mul(Add(One(), x), Sub(One(), x)))));
}

Express::Var *Express::Acos(Var *x) {
  return Add(Atan(Div(Sqrt(Mul(Add(One(), x), Sub(One(), x))), x)),
             Select(CmpLt(x, Zero()), Number(PI)));
}

Express::Var *Express::Atan(Var *x) {
  // Extract the sign bit.
  Var *sign = Do(BITAND, x, Number(SIGN_MASK));

  // Take the absolute value.
  x = Do(BITAND, x, Number(INV_SIGN_MASK));

  // Range reduction.
  Var *over_tan3pio8 = CmpGt(x, Number(TAN3PIO8));
  Var *over_tanpio8 = CmpGt(x, Number(TANPIO8));
  x = Cond(over_tan3pio8, Div(Number(N1), x),
           Cond(over_tanpio8, Div(Sub(x, One()), Add(x, One())), x));
  Var *y = Cond(over_tan3pio8, Number(PIO2),
                Cond(over_tanpio8, Number(PIO4), Zero()));

  // Compute polynomial.
  Var *z = Square(x);
  Var *p = Number(ATAN_P0);
  p = MulAdd(p, z, Number(ATAN_P1));
  p = MulAdd(p, z, Number(ATAN_P2));
  p = MulAdd(p, z, Number(ATAN_P3));
  y = Add(y, Add(Mul(Mul(p, z), x), x));

  // Adjust sign.
  y = Do(BITXOR, y, sign);
  return y;
}

Express::Var *Express::HyperTrig(OpType type, Var *x) {
  // Compute exp(x), exp(-x), and exp(2x).
  Var *pexp = nullptr;
  Var *nexp = nullptr;
  Var *texp = nullptr;
  if (type == TANH || type == COTH) {
    texp = Exp(Mul(x, Two()));
  } else {
    pexp = Exp(x);
    nexp = Exp(Neg(x));
  }

  switch (type) {
    case SINH: return Div(Sub(pexp, nexp), Two());
    case COSH: return Div(Add(pexp, nexp), Two());
    case TANH: return Div(Sub(texp, One()), Add(texp, One()));
    case COTH: return Div(Add(texp, One()), Sub(texp, One()));
    case SECH: return Div(Two(), Add(pexp, nexp));
    case CSCH: return Div(Two(), Sub(pexp, nexp));
    default: LOG(FATAL) << "Unsupported hyperbolic function";
  }
}

// Hyperbolic tangent function.
// Compute 13/6-degree rational interpolant which is accurate up to a couple of
// ulp in the range [-9, 9], outside of which the fl(tanh(x)) = +/-1.
// See: https://git.io/vHyiz
Express::Var *Express::Tanh(Var *x) {
  if (Supports(EXP) || Supports(EXP2)) {
    // Compute tanh(x) = 2*sigmoid(2*x) - 1.
    return Sub(Mul(Sigmoid(Mul(x, Two())), Two()), One());
  } else {
    // Clamp the inputs to the range [-9, 9] since anything outside this range
    // is +/-1.0.
    x = Maximum(Minimum(x, Number(P9)), Number(N9));

    // Since the polynomials are odd/even, we need x^2.
    Var *x2 = Square(x);

    // Evaluate the numerator polynomial p.
    Var *p = Number(ALPHA_1);
    p = MulAdd(x2, p, Number(ALPHA_3));
    p = MulAdd(x2, p, Number(ALPHA_5));
    p = MulAdd(x2, p, Number(ALPHA_7));
    p = MulAdd(x2, p, Number(ALPHA_9));
    p = MulAdd(x2, p, Number(ALPHA_11));
    p = MulAdd(x2, p, Number(ALPHA_13));
    p = Mul(x, p);

    // Evaluate the denominator polynomial q.
    Var *q = Number(BETA_0);
    q = MulAdd(x2, q, Number(BETA_2));
    q = MulAdd(x2, q, Number(BETA_4));
    q = MulAdd(x2, q, Number(BETA_6));

    // Divide the numerator by the denominator.
    return Div(p, q);
  }
}

// Gauss error function.
// See: Abramowitz & Stegun: Handbook of Mathematical Functions, formula 7.1.26.
Express::Var *Express::Erf(Var *x) {
  // Get sign and absolute value.
  Var *sign = Cond(CmpLt(x, Zero()), Number(N1), One());
  x = Abs(x);

  // A&S formula 7.1.26.
  Var *t = Reciprocal(Add(One(), Mul(Number(ERF_P), x)));
  Var *p = MulAdd(Number(ERF_A5), t, Number(ERF_A4));
  p = MulAdd(p, t, Number(ERF_A3));
  p = MulAdd(p, t, Number(ERF_A2));
  p = MulAdd(p, t, Number(ERF_A1));
  Var *y = Sub(One(), Mul(Mul(p, t), Exp(Neg(Square(x)))));

  return Mul(sign, y);
}

void Express::Var::Redirect(Var *other) {
  // Update all consumers to use the other variable.
  for (Op *consumer : consumers) {
    for (int i = 0; i < consumer->args.size(); ++i) {
      if (consumer->args[i] == this) consumer->args[i] = other;
    }
    other->consumers.push_back(consumer);
  }
  consumers.clear();
}

char Express::Var::TypeCode() const {
  switch (type) {
    case INPUT:    return '%';
    case REGISTER: return '!';
    case CONST:    return '#';
    case OUTPUT:   return '@';
    case TEMP:     return '$';
    case NUMBER:   return '_';
  }
  return '?';
}

string Express::Var::AsString() const {
  return TypeCode() + std::to_string(id);
}

void Express::Var::GetRecipe(string *recipe) const {
  recipe->push_back(TypeCode());
  recipe->append(std::to_string(id));
}

string Express::Op::AsString() const {
  string str;
  str.append(OpName(type));
  str.push_back('(');
  bool first = true;
  for (auto *arg : args) {
    if (!first) str.push_back(',');
    str.append(arg->AsString());
    first = false;
  }
  str.push_back(')');
  return str;
}

string Express::Op::AsInstruction() const {
  // Opcode.
  string str;
  if (type == MOV) {
    str.append("Mov ");
  } else {
    str.append(OpName(type));
    str.push_back(' ');
  }

  // Destination operand.
  if (dst != -1) {
    str.push_back('r');
    str.append(std::to_string(dst));
  } else {
    result->GetRecipe(&str);
  }

  int first = first_is_dest || conditional() ? 1 : 0;
  int second = first + 1;

  // Mask.
  if (mask != -1) {
    str.append("[r");
    str.append(std::to_string(mask));
    str.append("]");
  }

  // Source operand.
  if (src != -1) {
    str.push_back(',');
    str.push_back('r');
    str.append(std::to_string(src));
  } else if (arity() > first) {
    str.push_back(',');
    args[first]->GetRecipe(&str);
  }

  // Second source operand.
  if (src2 != -1) {
    str.push_back(',');
    str.push_back('r');
    str.append(std::to_string(src2));
  } else if (arity() > second) {
    str.push_back(',');
    args[second]->GetRecipe(&str);
  }

  // Accumulator.
  if (acc != -1) {
    str.append(" {r");
    str.append(std::to_string(acc));
    str.append("}");
  }

  return str;
}

void Express::Op::GetRecipe(string *recipe) const {
  recipe->append(OpName(type));
  recipe->push_back('(');
  bool first = true;
  for (auto *arg : args) {
    if (!first) recipe->push_back(',');
    first = false;
    if (arg->inlined()) {
      CHECK(arg->producer != nullptr);
      arg->producer->GetRecipe(recipe);
    } else {
      arg->GetRecipe(recipe);
    }
  }
  recipe->push_back(')');
}

void Express::Op::Assign(Var *var, bool reassign) {
  // Remove any previous assignment.
  CHECK(reassign || var->producer == nullptr);
  if (result != nullptr) result->producer = nullptr;

  // Set new assignment.
  result = var;
  var->producer = this;

  // Set predicate flag for result.
  if (logic() || compare() || preduction()) var->predicate = true;
}

void Express::Op::AddArgument(Var *arg) {
  // Add operation as consumer of variable.
  arg->consumers.push_back(this);

  // Add argument to operation.
  args.push_back(arg);

  // Set predicate flag for argument.
  if (logic() || type == COUNT ||
      (conditional() && args.size() == 1) ||
      (type == MOV && arg->predicate)) {
    arg->predicate = true;
  }
}

void Express::Op::ClearArguments() {
  // Remove operation as consumer of argument variables.
  for (Var *arg : args) {
    auto f = std::find(arg->consumers.begin(), arg->consumers.end(), this);
    DCHECK(f != arg->consumers.end());
    arg->consumers.erase(f);
  }
  args.clear();
}

bool Express::Op::EqualTo(Op *other) const {
  if (type != other->type) return false;
  if (arity() != other->arity()) return false;
  if (commutative() && args.size() == 2) {
    if (args[0] == other->args[1] &&
        args[1] == other->args[0]) {
      return true;
    }
  }
  for (int i = 0; i < args.size(); ++i) {
    if (args[i] != other->args[i]) return false;
  }
  return true;
}

}  // namespace myelin
}  // namespace sling

