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

#include <string>

#include "myelin/compute.h"
#include "myelin/macro-assembler.h"

#define __ masm->

namespace sling {
namespace myelin {

using namespace jit;

enum BinOp {ADD, SUB, MUL};

// Compute element-wise float binary operator.
class GenericFltBinaryOperator : public Kernel {
 public:
  GenericFltBinaryOperator(BinOp op) : op_(op) {}

  bool Supports(Step *step) override {
    // Requires CPU with SSE support.
    if (!CPU::Enabled(SSE)) return false;

    // Check inputs and outputs.
    if (step->inputs().size() != 2) return false;
    if (step->outputs().size() != 1) return false;
    Tensor *a = step->input(0);
    Tensor *b = step->input(1);
    Tensor *c = step->output(0);

    // Check type.
    if (a->type() != DT_FLOAT) return false;
    if (b->type() != DT_FLOAT) return false;
    if (c->type() != DT_FLOAT) return false;

    // Input and output must have same number of elements.
    if (a->shape().elements() != c->shape().elements()) return false;
    if (b->shape().elements() != c->shape().elements()) return false;

    return true;
  }

  void Adjust(Step *step) override {
    step->input(0)->CompatibleAlign(step->output(0));
    step->input(1)->CompatibleAlign(step->output(0));
    step->AllowInPlace(0, 0);
  }

  void Generate(Step *step, MacroAssembler *masm) override {
    Registers &rr = masm->rr();
    SIMDRegisters &mm = masm->mm();
    Label l;

    Tensor *a = step->input(0);
    Tensor *b = step->input(1);
    Tensor *c = step->output(0);

    Register ofs = rr.alloc();
    Register input1 = rr.alloc();
    Register input2 = rr.alloc();
    Register output = rr.alloc();
    XMMRegister elem = mm.allocx();

    // Load tensor locations.
    __ LoadTensorAddress(input1, a);
    __ LoadTensorAddress(input2, b);
    if (a->SharedWith(c)) {
      output = input1;
    } else {
      __ LoadTensorAddress(output, c);
    }
    __ xorq(ofs, ofs);

    // Loop over elements in input tensors.
    __ LoopStart(&l);

    // Compute c = f(a, b).
    __ movss(elem, Operand(input1, ofs));
    switch (op_) {
      case ADD:
        __ addss(elem, Operand(input2, ofs));
        break;

      case SUB:
        __ subss(elem, Operand(input2, ofs));
        break;

      case MUL:
        __ mulss(elem, Operand(input2, ofs));
        break;
    }
    __ movss(Operand(output, ofs), elem);

    // Next element.
    __ addq(ofs, Immediate(sizeof(float)));
    __ cmpq(ofs, Immediate(a->size()));
    __ j(less, &l);
  }

  int64 Complexity(const Step *step) override {
    return step->input(0)->elements();
  }

 private:
  BinOp op_;
};

// Element-wise float add.
class GenericFltAdd : public GenericFltBinaryOperator {
 public:
  GenericFltAdd() : GenericFltBinaryOperator(ADD) {}
  string Name() override { return "GenFltAdd"; }
  string Operation() override { return "Add"; }
};

// Element-wise float subtract.
class GenericFltSub : public GenericFltBinaryOperator {
 public:
  GenericFltSub() : GenericFltBinaryOperator(SUB) {}
  string Name() override { return "GenFltSub"; }
  string Operation() override { return "Sub"; }
};

// Element-wise float multiply.
class GenericFltMul : public GenericFltBinaryOperator {
 public:
  GenericFltMul() : GenericFltBinaryOperator(MUL) {}
  string Name() override { return "GenFltMul"; }
  string Operation() override { return "Mul"; }
};

// Compute element-wise integer binary operator.
class GenericIntBinaryOperator : public Kernel {
 public:
  GenericIntBinaryOperator(BinOp op) : op_(op) {}

  static bool IsIntType(Type t) {
    return t == DT_INT8 || t == DT_INT16 || t == DT_INT32 || t == DT_INT64;
  }

  bool Supports(Step *step) override {
    // Requires CPU with SSE support.
    if (!CPU::Enabled(SSE)) return false;

    // Check inputs and outputs.
    if (step->inputs().size() != 2) return false;
    if (step->outputs().size() != 1) return false;
    Tensor *a = step->input(0);
    Tensor *b = step->input(1);
    Tensor *c = step->output(0);

    // Check type.
    if (!IsIntType(a->type())) return false;
    if (!IsIntType(b->type())) return false;
    if (!IsIntType(c->type())) return false;

    // Input and output must have same number of elements.
    if (a->shape().elements() != c->shape().elements()) return false;
    if (b->shape().elements() != c->shape().elements()) return false;

    return true;
  }

  void Adjust(Step *step) override {
    step->input(0)->CompatibleAlign(step->output(0));
    step->input(1)->CompatibleAlign(step->output(0));
  }

  void Generate(Step *step, MacroAssembler *masm) override {
    Registers &rr = masm->rr();
    Label l;

    Tensor *a = step->input(0);
    Tensor *b = step->input(1);
    Tensor *c = step->output(0);

    Register x = rr.alloc();
    Register y = rr.alloc();
    Register index = rr.alloc();
    Register input1 = rr.alloc();
    Register input2 = rr.alloc();
    Register output = rr.alloc();

    // Load tensor locations.
    __ LoadTensorAddress(input1, a);
    __ LoadTensorAddress(input2, b);
    __ LoadTensorAddress(output, c);
    __ xorq(index, index);

    // Loop over elements in input tensors.
    __ LoopStart(&l);

    // Compute c = f(a, b).
    __ LoadInteger(x, input1, index, a->type());
    __ LoadInteger(y, input2, index, b->type());
    switch (op_) {
      case ADD:
        __ addq(x, y);
        break;

      case SUB:
        __ subq(x, y);
        break;

      case MUL:
        __ imulq(x, y);
        break;
    }
    __ StoreInteger(output, index, x, c->type());

    // Next element.
    __ incq(index);
    __ cmpq(index, Immediate(a->shape().elements()));
    __ j(less, &l);
  }

  int64 Complexity(const Step *step) override {
    return step->input(0)->elements();
  }

 private:
  BinOp op_;
};

// Element-wise integer add.
class GenericIntAdd : public GenericIntBinaryOperator {
 public:
  GenericIntAdd() : GenericIntBinaryOperator(ADD) {}
  string Name() override { return "GenIntAdd"; }
  string Operation() override { return "Add"; }
};

// Element-wise integer subtract.
class GenericIntSub : public GenericIntBinaryOperator {
 public:
  GenericIntSub() : GenericIntBinaryOperator(SUB) {}
  string Name() override { return "GenIntSub"; }
  string Operation() override { return "Sub"; }
};

// Element-wise interger multiply.
class GenericIntMul : public GenericIntBinaryOperator {
 public:
  GenericIntMul() : GenericIntBinaryOperator(MUL) {}
  string Name() override { return "GenIntMul"; }
  string Operation() override { return "Mul"; }
};

// Compute y = constant - x.
class GenericFltConstSub : public Kernel {
 public:
  string Name() override { return "GenFltConstSub"; }
  string Operation() override { return "Sub"; }

  bool Supports(Step *step) override {
    // Requires CPU with SSE support.
    if (!CPU::Enabled(SSE)) return false;

    // Check inputs and outputs.
    if (step->indegree() != 2) return false;
    if (step->outdegree() != 1) return false;
    Tensor *c = step->input(0);
    Tensor *x = step->input(1);
    Tensor *y = step->output(0);

    // Check types.
    if (c->type() != DT_FLOAT) return false;
    if (x->type() != DT_FLOAT) return false;
    if (y->type() != DT_FLOAT) return false;

    // Check shapes.
    if (c->shape().elements() != 1 || !c->IsConstant()) return false;
    if (x->shape().elements() != y->shape().elements()) return false;

    return true;
  }

  void Adjust(Step *step) override {
    step->input(1)->CompatibleAlign(step->output(0));
    step->AllowInPlace(1, 0);
  }

  void Generate(Step *step, MacroAssembler *masm) override {
    Registers &rr = masm->rr();
    SIMDRegisters &mm = masm->mm();
    Label l;

    Tensor *c = step->input(0);
    Tensor *x = step->input(1);
    Tensor *y = step->output(0);

    Register ofs = rr.alloc();
    Register input = rr.alloc();
    Register output = rr.alloc();
    Register constant = rr.alloc();
    XMMRegister elem = mm.allocx();

    // Load tensor locations.
    __ LoadTensorAddress(input, x);
    if (x->SharedWith(y)) {
      output = input;
    } else {
      __ LoadTensorAddress(output, y);
    }

    // Load constant.
    __ movl(constant, Immediate(c->value<int32_t>()));

    // Loop over elements in input tensors.
    __ xorq(ofs, ofs);
    __ LoopStart(&l);

    // Compute y = constant - x.
    __ movq(elem, constant);
    __ subss(elem, Operand(input, ofs));
    __ movss(Operand(output, ofs), elem);

    // Next element.
    __ addq(ofs, Immediate(sizeof(float)));
    __ cmpq(ofs, Immediate(x->size()));
    __ j(less, &l);
  }

  int64 Complexity(const Step *step) override {
    return step->input(0)->elements();
  }
};

// Compute z = x0 * x1 + x2 * x3.
class GenericFltMulTwoAdd : public Kernel {
 public:
  string Name() override { return "GenFltMulTwoAdd"; }
  string Operation() override { return "MulTwoAdd"; }

  bool Supports(Step *step) override {
    // Check inputs and outputs.
    if (step->inputs().size() != 4) return false;
    if (step->outputs().size() != 1) return false;
    Tensor *x[4];
    for (int i = 0; i < 4; ++i) x[i] = step->input(i);
    Tensor *y = step->output(0);

    // Check type.
    for (int i = 0; i < 4; ++i) {
      if (x[i]->type() != DT_FLOAT) return false;
    }
    if (y->type() != DT_FLOAT) return false;

    // Input and output must have same shape.
    for (int i = 0; i < 4; ++i) {
      if (!x[i]->HasSameShape(y)) return false;
    }

    return true;
  }

  void Adjust(Step *step) override {
    for (int i = 0; i < 4; ++i) {
      step->input(i)->SameAlign(step->output(0));
    }
  }

  void Generate(Step *step, MacroAssembler *masm) override {
    Registers &rr = masm->rr();
    SIMDRegisters &mm = masm->mm();
    Label l;

    Tensor *x[4];
    for (int i = 0; i < 4; ++i) x[i] = step->input(i);
    Tensor *y = step->output(0);

    Register ofs = rr.alloc();
    Register input[4];
    for (int i = 0; i < 4; ++i) input[i] = rr.alloc();
    Register output = rr.alloc();
    XMMRegister sum = mm.allocx();
    XMMRegister term = mm.allocx();

    // Load tensor locations.
    for (int i = 0; i < 4; ++i) {
      __ LoadTensorAddress(input[i], x[i]);
    }
    __ LoadTensorAddress(output, y);
    __ xorq(ofs, ofs);

    // Loop over elements in input tensors.
    __ LoopStart(&l);

    // Compute y = x0 * x1 + x2 * x3.
    __ movss(sum, Operand(input[0], ofs));
    __ mulss(sum, Operand(input[1], ofs));
    __ movss(term, Operand(input[2], ofs));
    __ mulss(term, Operand(input[3], ofs));
    __ addss(sum, term);
    __ movss(Operand(output, ofs), sum);

    // Next element.
    __ addq(ofs, Immediate(sizeof(float)));
    __ cmpq(ofs, Immediate(y->size()));
    __ j(less, &l);
  }

  int64 Complexity(const Step *step) override {
    return step->input(0)->elements() * 3;
  }
};

void RegisterGenericOperators(Library *library) {
  // Computes  : y = c - x element-wise
  // Input     : c: float32[1] const
  //             x: float32[d1,...,dn]
  // Output    : y: float32[d1,...,dn]
  library->Register(new GenericFltConstSub());

  // Computes  : y = x0 * x1 + x2 * x3 element-wise
  // Input     : x0: float32[d1,...,dn]
  //             x1: float32[d1,...,dn]
  //             x2: float32[d1,...,dn]
  //             x3: float32[d1,...,dn]
  // Output    : y: float32[d1,...,dn]
  library->Register(new GenericFltMulTwoAdd());

  // Computes  : c = a + b element-wise
  // Input     : a: float32[d1,...,dn]
  //             b: float32[d1,...,dn]
  // Output    : c: float32[d1,...,dn]
  library->Register(new GenericFltAdd());

  // Computes  : c = a - b element-wise
  // Input     : a: float32[d1,...,dn]
  //             b: float32[d1,...,dn]
  // Output    : c: float32[d1,...,dn]
  library->Register(new GenericFltSub());

  // Computes  : c = a * b element-wise
  // Input     : a: float32[d1,...,dn]
  //             b: float32[d1,...,dn]
  // Output    : c: float32[d1,...,dn]
  library->Register(new GenericFltMul());

  // Computes  : c = a + b element-wise
  // Input     : a: int8/16/32/64[d1,...,dn]
  //             b: int8/16/32/64[d1,...,dn]
  // Output    : c: int8/16/32/64[d1,...,dn]
  library->Register(new GenericIntAdd());

  // Computes  : c = a - b element-wise
  // Input     : a: int8/16/32/64[d1,...,dn]
  //             b: int8/16/32/64[d1,...,dn]
  // Output    : c: int8/16/32/64[d1,...,dn]
  library->Register(new GenericIntSub());

  // Computes  : c = a * b element-wise
  // Input     : a: int8/16/32/64[d1,...,dn]
  //             b: int8/16/32/64[d1,...,dn]
  // Output    : c: int8/16/32/64[d1,...,dn]
  library->Register(new GenericIntMul());
}

}  // namespace myelin
}  // namespace sling

