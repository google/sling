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

#include "sling/myelin/kernel/avx.h"

#include <string>

#include "sling/myelin/compute.h"
#include "sling/myelin/macro-assembler.h"

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

    a->MinAlignLast(8);
    a->SetMiniumAlignment(8 * sizeof(float));
    b->MinAlignLast(8);
    b->SetMiniumAlignment(8 * sizeof(float));
    c->MinAlignLast(8);
    c->SetMiniumAlignment(8 * sizeof(float));
    a->CompatibleAlign(c);
    b->CompatibleAlign(c);
    if (!step->AllowInPlace(0, 0)) step->AllowInPlace(1, 0);
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
    } else if (b->SharedWith(c)) {
      output = input2;
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

  int64 Complexity(const Step *step) override {
    return step->input(0)->elements();
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
    a->MinAlignLast(align);
    a->SetMiniumAlignment(regsize);
    b->MinAlignLast(align);
    b->SetMiniumAlignment(regsize);
    c->MinAlignLast(align);
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

  int64 Complexity(const Step *step) override {
    return step->input(0)->elements();
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

void RegisterAVXOperators(Library *library) {
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

