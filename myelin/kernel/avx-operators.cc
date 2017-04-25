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

#include "myelin/kernel/avx.h"

#include <string>

#include "myelin/compute.h"
#include "myelin/macro-assembler.h"

#define __ masm->

namespace sling {
namespace myelin {

using namespace jit;

enum BinOp {ADD, SUB, MUL};

// Compute element-wise float binary operator using AVX.
class AVXFltBinaryOperator : public Kernel {
 public:
  AVXFltBinaryOperator(BinOp op) : op_(op) {}

  bool Supports(Step *step) override {
    // Requires CPU with AVX support.
    if (!CPU::Enabled(AVX)) return false;

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
    Tensor *a = step->input(0);
    Tensor *b = step->input(1);
    Tensor *c = step->output(0);

    a->AlignLast(8);
    a->SetMiniumAlignment(8 * sizeof(float));
    b->AlignLast(8);
    b->SetMiniumAlignment(8 * sizeof(float));
    c->AlignLast(8);
    c->SetMiniumAlignment(8 * sizeof(float));
    a->CompatibleAlign(c);
    b->CompatibleAlign(c);
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
    YMMRegister elem = mm.allocy();

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
    __ vmovaps(elem, Operand(input1, ofs));
    switch (op_) {
      case ADD:
        __ vaddps(elem, elem, Operand(input2, ofs));
        break;

      case SUB:
        __ vsubps(elem, elem, Operand(input2, ofs));
        break;

      case MUL:
        __ vmulps(elem, elem, Operand(input2, ofs));
        break;
    }
    __ vmovaps(Operand(output, ofs), elem);

    // Next element.
    __ addq(ofs, Immediate(8 * sizeof(float)));
    __ cmpq(ofs, Immediate(a->size()));
    __ j(less, &l);
  }

 private:
  BinOp op_;
};

// Element-wise float add using AVX.
class AVXFltAdd : public AVXFltBinaryOperator {
 public:
  AVXFltAdd() : AVXFltBinaryOperator(ADD) {}
  string Name() override { return "AVXFltAdd"; }
  string Operation() override { return "Add"; }
};

// Element-wise float subtract using AVX.
class AVXFltSub : public AVXFltBinaryOperator {
 public:
  AVXFltSub() : AVXFltBinaryOperator(SUB) {}
  string Name() override { return "AVXFltSub"; }
  string Operation() override { return "Sub"; }
};

// Element-wise float multiply using AVX.
class AVXFltMul : public AVXFltBinaryOperator {
 public:
  AVXFltMul() : AVXFltBinaryOperator(MUL) {}
  string Name() override { return "AVXFltMul"; }
  string Operation() override { return "Mul"; }
};

// Compute element-wise integer binary operator using AVX.
class AVXIntBinaryOperator : public Kernel {
 public:
  AVXIntBinaryOperator(BinOp op) : op_(op) {}

  bool Supports(Step *step) override {
    // Requires CPU with AVX-2 support.
    if (!CPU::Enabled(AVX2)) return false;

    // Check inputs and outputs.
    if (step->inputs().size() != 2) return false;
    if (step->outputs().size() != 1) return false;
    Tensor *a = step->input(0);
    Tensor *b = step->input(1);
    Tensor *c = step->output(0);

    // Check type.
    if (a->type() != DT_INT8 &&
        a->type() != DT_INT16 &&
        a->type() != DT_INT32 &&
        a->type() != DT_INT64) {
      return false;
    }
    if (b->type() != a->type()) return false;
    if (c->type() != a->type()) return false;

    // Input and output must have same number of elements.
    if (a->shape().elements() != c->shape().elements()) return false;
    if (b->shape().elements() != c->shape().elements()) return false;

    // Only add and subtract supported.
    if (op_ != ADD && op_ != SUB) return false;

    return true;
  }

  void Adjust(Step *step) override {
    Tensor *a = step->input(0);
    Tensor *b = step->input(1);
    Tensor *c = step->output(0);

    int regsize = 32;
    int align = regsize / a->element_size();
    a->AlignLast(align);
    a->SetMiniumAlignment(regsize);
    b->AlignLast(align);
    b->SetMiniumAlignment(regsize);
    c->AlignLast(align);
    c->SetMiniumAlignment(regsize);
    a->CompatibleAlign(c);
    b->CompatibleAlign(c);
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
    YMMRegister elem = mm.allocy();

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
    __ vmovdqa(elem, Operand(input1, ofs));
    switch (op_) {
      case ADD:
        if (b->type() == DT_INT8) {
          __ vpaddb(elem, elem, Operand(input2, ofs));
        } else if (b->type() == DT_INT16) {
          __ vpaddw(elem, elem, Operand(input2, ofs));
        } else if (b->type() == DT_INT32) {
          __ vpaddd(elem, elem, Operand(input2, ofs));
        } else {
          __ vpaddq(elem, elem, Operand(input2, ofs));
        }
        break;

      case SUB:
        if (b->type() == DT_INT8) {
          __ vpsubb(elem, elem, Operand(input2, ofs));
        } else if (b->type() == DT_INT16) {
          __ vpsubw(elem, elem, Operand(input2, ofs));
        } else if (b->type() == DT_INT32) {
          __ vpsubd(elem, elem, Operand(input2, ofs));
        } else {
          __ vpsubq(elem, elem, Operand(input2, ofs));
        }
        break;

      case MUL:
        break;
    }
    __ vmovdqa(Operand(output, ofs), elem);

    // Next element.
    __ addq(ofs, Immediate(32));
    __ cmpq(ofs, Immediate(a->size()));
    __ j(less, &l);
  }

 private:
  BinOp op_;
};

// Element-wise integer add using AVX.
class AVXIntAdd : public AVXIntBinaryOperator {
 public:
  AVXIntAdd() : AVXIntBinaryOperator(ADD) {}
  string Name() override { return "AVXIntAdd"; }
  string Operation() override { return "Add"; }
};

// Element-wise integer subtract using AVX.
class AVXIntSub : public AVXIntBinaryOperator {
 public:
  AVXIntSub() : AVXIntBinaryOperator(SUB) {}
  string Name() override { return "AVXIntSub"; }
  string Operation() override { return "Sub"; }
};

// Compute y = constant - x using AVX.
class AVXFltConstSub : public Kernel {
 public:
  string Name() override { return "AVXFltConstSub"; }
  string Operation() override { return "Sub"; }

  bool Supports(Step *step) override {
    // Requires CPU with AVX support.
    if (!CPU::Enabled(AVX)) return false;

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
    Tensor *x = step->input(1);
    Tensor *y = step->output(0);

    x->AlignLast(8);
    x->SetMiniumAlignment(8 * sizeof(float));
    y->AlignLast(8);
    y->SetMiniumAlignment(8 * sizeof(float));
    x->CompatibleAlign(y);
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
    Register imm = rr.alloc();
    YMMRegister constant = mm.allocy();
    YMMRegister elem = mm.allocy();

    // Load tensor locations.
    __ LoadTensorAddress(input, x);
    if (x->SharedWith(y)) {
      output = input;
    } else {
      __ LoadTensorAddress(output, y);
    }

    // Load constant.
    __ movl(imm, Immediate(c->value<int32_t>()));
    __ movq(constant.xmm(), imm);
      if (masm->Enabled(AVX2)) {
      __ vbroadcastss(constant, constant);
    } else {
      __ shufps(constant.xmm(), constant.xmm(), 0);
      __ vinsertf128(constant, constant, constant.xmm(), 1);
    }

    // Loop over elements in input tensors.
    __ xorq(ofs, ofs);
    __ LoopStart(&l);

    // Compute y = constant - x.
    __ vsubps(elem, constant, Operand(input, ofs));
    __ vmovaps(Operand(output, ofs), elem);

    // Next element.
    __ addq(ofs, Immediate(8 * sizeof(float)));
    __ cmpq(ofs, Immediate(x->size()));
    __ j(less, &l);

    // TODO: set padding elements to zero.
  }
};

// Compute z = x0 * x1 + x2 * x3 using AVX.
class AVXFltMulTwoAdd : public Kernel {
 public:
  string Name() override { return "AVXFltMulTwoAdd"; }
  string Operation() override { return "MulTwoAdd"; }

  bool Supports(Step *step) override {
    // Requires CPU with AVX support.
    if (!CPU::Enabled(AVX)) return false;

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
    step->output(0)->AlignLast(8);
    step->output(0)->SetMiniumAlignment(8 * sizeof(float));
    for (int i = 0; i < 4; ++i) {
      step->input(i)->AlignLast(8);
      step->input(i)->SetMiniumAlignment(8 * sizeof(float));
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
    YMMRegister term1 = mm.allocy();
    YMMRegister term2 = mm.allocy();
    YMMRegister sum = mm.allocy();

    // Load tensor locations.
    for (int i = 0; i < 4; ++i) {
      __ LoadTensorAddress(input[i], x[i]);
    }
    __ LoadTensorAddress(output, y);
    __ xorq(ofs, ofs);

    // Loop over elements in input tensors.
    __ LoopStart(&l);

    // Compute y = x0 * x1 + x2 * x3.
    __ vmovaps(term1, Operand(input[0], ofs));
    __ vmulps(term1, term1, Operand(input[1], ofs));
    __ vmovaps(term2, Operand(input[2], ofs));
    __ vmulps(term2, term2, Operand(input[3], ofs));
    __ vaddps(sum, term1, term2);
    __ vmovaps(Operand(output, ofs), sum);

    // Next element.
    __ addq(ofs, Immediate(8 * sizeof(float)));
    __ cmpq(ofs, Immediate(y->size()));
    __ j(less, &l);
  }
};

void RegisterAVXOperators(Library *library) {
  // Computes  : y = c - x element-wise
  // Input     : c: float32[1] const
  //             x: float32[d1,...,dn]
  // Output    : y: float32[d1,...,dn]
  // Requires  : AVX
  library->Register(new AVXFltConstSub());

  // Computes  : y = x0 * x1 + x2 * x3 element-wise
  // Input     : x0: float32[d1,...,dn]
  //             x1: float32[d1,...,dn]
  //             x2: float32[d1,...,dn]
  //             x3: float32[d1,...,dn]
  // Output    : y: float32[d1,...,dn]
  // Requires  : AVX
  library->Register(new AVXFltMulTwoAdd());

  // Computes  : c = a + b element-wise
  // Input     : a: float32[d1,...,dn]
  //             b: float32[d1,...,dn]
  // Output    : c: float32[d1,...,dn]
  // Requires  : AVX
  library->Register(new AVXFltAdd());

  // Computes  : c = a - b element-wise
  // Input     : a: float32[d1,...,dn]
  //             b: float32[d1,...,dn]
  // Output    : c: float32[d1,...,dn]
  // Requires  : AVX
  library->Register(new AVXFltSub());

  // Computes  : c = a * b element-wise
  // Input     : a: float32[d1,...,dn]
  //             b: float32[d1,...,dn]
  // Output    : c: float32[d1,...,dn]
  // Requires  : AVX
  library->Register(new AVXFltMul());

  // Computes  : c = a + b element-wise
  // Input     : a: int8/16/32/64[d1,...,dn]
  //             b: int8/16/32/64[d1,...,dn]
  // Output    : c: int8/16/32/64[d1,...,dn]
  // Requires  : AVX
  library->Register(new AVXIntAdd());

  // Computes  : c = a - b element-wise
  // Input     : a: int8/16/32/64[d1,...,dn]
  //             b: int8/16/32/64[d1,...,dn]
  // Output    : c: int8/16/32/64[d1,...,dn]
  // Requires  : AVX
  library->Register(new AVXIntSub());
}

}  // namespace myelin
}  // namespace sling

