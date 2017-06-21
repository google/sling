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

#include <stddef.h>
#include <string>

#include "myelin/compute.h"
#include "myelin/macro-assembler.h"

#define __ masm->

namespace sling {
namespace myelin {

using namespace jit;

// Constants.
namespace {

struct TanhConstants {
  FloatVec8 plus_9;
  FloatVec8 minus_9;
  FloatVec8 alpha[7];
  FloatVec8 beta[4];
};

TanhConstants tanh_const = {
  // Input range.
  CONST8(9.0f),
  CONST8(-9.0f),

  // The monomial coefficients of the numerator polynomial (odd).
  {
    CONST8(-2.76076847742355e-16f),
    CONST8(2.00018790482477e-13f),
    CONST8(-8.60467152213735e-11f),
    CONST8(5.12229709037114e-08f),
    CONST8(1.48572235717979e-05f),
    CONST8(6.37261928875436e-04f),
    CONST8(4.89352455891786e-03f),
  },

  // The monomial coefficients of the denominator polynomial (even).
  {
    CONST8(1.19825839466702e-06f),
    CONST8(1.18534705686654e-04f),
    CONST8(2.26843463243900e-03f),
    CONST8(4.89352518554385e-03f),
  },
};

struct ExpConstants {
  FloatVec8 one;
  FloatVec8 half;

  FloatVec8 exphi;
  FloatVec8 explo;

  FloatVec8 p127;
  FloatVec8 nln2;

  FloatVec8 cephes_log2ef;
  FloatVec8 unused;

  FloatVec8 cephes_exp[6];
  FloatVec8 cephes_exp_c1;
  FloatVec8 cephes_exp_c2;
} __attribute__ ((aligned (64)));

ExpConstants exp_const = {
  CONST8(1.0f),
  CONST8(0.5f),

  CONST8(88.3762626647950f),
  CONST8(-88.3762626647949f),

  CONST8(127.0f),
  CONST8(-0.6931471805599453f),

  CONST8(1.44269504088896341f),
  CONST8(0),

  {
    CONST8(1.9875691500e-4f),
    CONST8(1.3981999507e-3f),
    CONST8(8.3334519073e-3f),
    CONST8(4.1665795894e-2f),
    CONST8(1.6666665459e-1f),
    CONST8(5.0000001201e-1f),
  },

  CONST8(0.693359375f),
  CONST8(-2.12194440e-4f),
};

}  // namespace

// Compute element-wise hyperbolic tangent for a tensor using AVX.
// This implementation is derived from the Eigen library.
class AVXFltTanh : public Kernel {
 public:
  string Name() override { return "AVXFltTanh"; }
  string Operation() override { return "Tanh"; }

  bool Supports(Step *step) override {
    // Requires CPU with AVX support.
    if (!CPU::Enabled(AVX)) return false;

    // Check inputs and outputs.
    if (step->inputs().size() != 1) return false;
    if (step->outputs().size() != 1) return false;
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
    x->MinAlignLast(8);
    x->SetMiniumAlignment(8 * sizeof(float));
    y->MinAlignLast(8);
    y->SetMiniumAlignment(8 * sizeof(float));
    x->SameAlign(y);
    step->AllowInPlace(0, 0);
  }

  void Generate(Step *step, MacroAssembler *masm) override {
    Registers &rr = masm->rr();
    SIMDRegisters &mm = masm->mm();
    Label l;

    // Allocate registers.
    Register input = rr.alloc();
    Register output = rr.alloc();
    Register consts = rr.alloc();
    Register ofs = rr.alloc();

    YMMRegister x = mm.allocy();
    YMMRegister x2 = mm.allocy();
    YMMRegister p = mm.allocy();
    YMMRegister q = mm.allocy();

    // Load tensor and constant locations.
    __ LoadTensorAddress(input, step->input(0));
    if (step->output(0)->SharedWith(step->input(0))) {
      output = input;
    } else {
      __ LoadTensorAddress(output, step->output(0));
    }
    __ movp(consts, static_cast<void *>(&tanh_const));
    __ xorq(ofs, ofs);

    // Loop over elements in tensor, eight floats at a time.
    __ LoopStart(&l);

    // Load input.
    __ vmovaps(x, Operand(input, ofs));

    // Clamp the inputs to the range [-9, 9] since anything outside this range
    // is +/-1.0 in single-precision.
    __ vminps(x, x, Operand(consts, offsetof(TanhConstants, plus_9)));
    __ vmaxps(x, x, Operand(consts, offsetof(TanhConstants, minus_9)));

    // Compute x^2.
    __ vmulps(x2, x, x);

    // Compute the numerator polynomial.
    // p = alpha_0
    __ vmovaps(p, Operand(consts, offsetof(TanhConstants, alpha)));
    for (int i = 1; i < 7; ++i) {
      // p = p * x^2 + alpha_i
      int disp = offsetof(TanhConstants, alpha) + i * sizeof(float) * 8;
      if (masm->Enabled(FMA3)) {
        __ vfmadd213ps(p, x2, Operand(consts, disp));
      } else {
        __ vmulps(p, p, x2);
        __ vaddps(p, p, Operand(consts, disp));
      }
    }
    // p = p * x
    __ vmulps(p, p, x);

    // Compute the denominator polynomial.
    // q = beta_0
    __ vmovaps(q, Operand(consts, offsetof(TanhConstants, beta)));
    for (int i = 1; i < 4; ++i) {
      // p = p * x^2 + alpha_i
      int disp = offsetof(TanhConstants, beta) + i * sizeof(float) * 8;
      if (masm->Enabled(FMA3)) {
        __ vfmadd213ps(q, x2, Operand(consts, disp));
      } else {
        __ vmulps(q, q, x2);
        __ vaddps(q, q, Operand(consts, disp));
      }
    }

    // Divide the numerator by the denominator.
    __ vdivps(x, p, q);

    // Save result in output.
    __ vmovaps(Operand(output, ofs), x);

    // Next batch.
    __ addq(ofs, Immediate(8 * sizeof(float)));
    __ cmpq(ofs, Immediate(step->input(0)->size()));
    __ j(less, &l);
  }

  int64 Complexity(const Step *step) override {
    return step->input(0)->elements() * (5 + 6 * 3 + 3 * 3);
  }
};

// Compute element-wise exponential function for a tensor using AVX.
// This implementation is derived from the Eigen library.
// Works by writing x = m*log(2) + r, where m = floor(x/log(2)+1/2)  and  r is
// the remainder. The result is then exp(x) = 2^m*exp(r), where exp(r) is in the
// range [-1,1).
class AVXFltExpBase : public Kernel {
 public:
  AVXFltExpBase(bool sigmoid) : sigmoid_(sigmoid) {}

  bool Supports(Step *step) override {
    // Requires CPU with AVX support.
    if (!CPU::Enabled(AVX)) return false;

    // Check inputs and outputs.
    if (step->inputs().size() != 1) return false;
    if (step->outputs().size() != 1) return false;
    Tensor *x = step->input(0);
    Tensor *y = step->output(0);

    // Check type.
    if (x->type() != DT_FLOAT) return false;
    if (y->type() != DT_FLOAT) return false;

    // Input and output must have same shape.
    if (x->rank() != y->rank()) return false;
    for (int d = 0; d < x->rank(); ++d) {
      if (x->dim(d) != y->dim(d)) return false;
    }

    return true;
  }

  void Adjust(Step *step) override {
    Tensor *x = step->input(0);
    Tensor *y = step->output(0);
    x->MinAlignLast(8);
    x->SetMiniumAlignment(8 * sizeof(float));
    y->MinAlignLast(8);
    y->SetMiniumAlignment(8 * sizeof(float));
    step->AllowInPlace(0, 0);
  }

  void Generate(Step *step, MacroAssembler *masm) override {
    Registers &rr = masm->rr();
    SIMDRegisters &mm = masm->mm();
    Label l;

    // Allocate registers.
    Register input = rr.alloc();
    Register output = rr.alloc();
    Register consts = rr.alloc();
    Register ofs = rr.alloc();

    YMMRegister x = mm.allocy();
    YMMRegister m = mm.allocy();
    YMMRegister r = mm.allocy();
    YMMRegister r2 = mm.allocy();
    YMMRegister y = mm.allocy();
    YMMRegister emm0 = mm.allocy();
    XMMRegister hi = mm.allocx();

    YMMRegister zero = sigmoid_ ? mm.allocy() : no_ymm_reg;
    YMMRegister one = mm.allocy();
    YMMRegister half = mm.allocy();
    YMMRegister exphi = mm.allocy();
    YMMRegister explo = mm.allocy();
    YMMRegister p127 = mm.allocy();
    YMMRegister nln2 = mm.allocy();
    YMMRegister log2ef = mm.allocy();

    // Load tensor and constant locations.
    __ LoadTensorAddress(input, step->input(0));
    if (step->output(0)->SharedWith(step->input(0))) {
      output = input;
    } else {
      __ LoadTensorAddress(output, step->output(0));
    }
    __ movp(consts, static_cast<void *>(&exp_const));
    __ xorq(ofs, ofs);

    // Initialize constants.
    if (sigmoid_) {
      __ vxorps(zero, zero, zero);
    }
    __ vmovaps(one, Operand(consts, offsetof(ExpConstants, one)));
    __ vmovaps(half, Operand(consts, offsetof(ExpConstants, half)));
    __ vmovaps(exphi, Operand(consts, offsetof(ExpConstants, exphi)));
    __ vmovaps(explo, Operand(consts, offsetof(ExpConstants, explo)));
    __ vmovaps(p127, Operand(consts, offsetof(ExpConstants, p127)));
    __ vmovaps(nln2, Operand(consts, offsetof(ExpConstants, nln2)));
    __ vmovaps(log2ef, Operand(consts, offsetof(ExpConstants, cephes_log2ef)));

    // Loop over elements in tensor, eight floats at a time.
    __ LoopStart(&l);

    // Load input.
    __ vmovaps(x, Operand(input, ofs));

    // Negate x for sigmoid.
    if (sigmoid_) {
      __ vsubps(x, zero, x);
    }

    // Clamp x.
    __ vminps(x, x, exphi);
    __ vmaxps(x, x, explo);

    // Express exp(x) as exp(m*ln(2) + r), start by extracting
    // m = floor(x/ln(2) + 0.5).
    if (masm->Enabled(FMA3)) {
      __ vmovaps(m, log2ef);
      __ vfmadd213ps(m, x, half);
    } else {
      __ vmulps(m, x, log2ef);
      __ vaddps(m, m, half);
    }
    __ vroundps(m, m, kRoundDown);

    // Get r = x - m*ln(2). If no FMA instructions are available, m*ln(2) is
    // subtracted out in two parts, m*C1+m*C2 = m*ln(2), to avoid accumulating
    // truncation errors.
    if (masm->Enabled(FMA3)) {
      __ vmovaps(r, nln2);
      __ vfmadd213ps(r, m, x);
    } else {
      __ vmulps(r, m, Operand(consts, offsetof(ExpConstants, cephes_exp_c1)));
      __ vsubps(r, x, r);
      __ vmulps(r2, m, Operand(consts, offsetof(ExpConstants, cephes_exp_c2)));
      __ vsubps(r, r, r2);
    }

    // Compute r^2.
    __ vmulps(r2, r, r);

    // Compute polynomial.
    __ vmovaps(y, Operand(consts, offsetof(ExpConstants, cephes_exp)));
    for (int i = 1; i < 6; ++i) {
      // y = y * r + cephes_exp_i
      int disp = offsetof(ExpConstants, cephes_exp) + i * sizeof(float) * 8;
      if (masm->Enabled(FMA3)) {
        __ vfmadd213ps(y, r, Operand(consts, disp));
      } else {
        __ vmulps(y, y, r);
        __ vaddps(y, y, Operand(consts, disp));
      }
    }

    // y = y * r2 + r
    if (masm->Enabled(FMA3)) {
      __ vfmadd213ps(y, r2, r);
    } else {
      __ vmulps(y, y, r2);
      __ vaddps(y, y, r);
    }

    // y = y + 1.0
    __ vaddps(y, y, one);

    // emm0 = 2^m.
    __ vaddps(emm0, m, p127);
    __ vcvttps2dq(emm0, emm0);

    // emm0 = emm0 << 23.
    if (masm->Enabled(AVX2)) {
      __ vpslld(emm0, emm0, 23);
    } else {
      __ vextractf128(hi, emm0, 1);
      __ vpslld(hi, hi, 23);
      __ vpslld(emm0.xmm(), emm0.xmm(), 23);
      __ vinsertf128(emm0, emm0, hi, 1);
    }

    // y = 2^m * exp(r).
    __ vmulps(y, y, emm0);

    // Compute sigmoid(x) = 1 / (1 + exp(-x)).
    if (sigmoid_) {
      __ vaddps(y, one, y);
      __ vdivps(y, one, y);
    }

    // Save result in output.
    __ vmovaps(Operand(output, ofs), y);

    // Next batch.
    __ addq(ofs, Immediate(8 * sizeof(float)));
    __ cmpq(ofs, Immediate(step->input(0)->size()));
    __ j(less, &l);

    // TODO: set padding elements to zero because sigmoid(0)=0.5 and exp(0)=1.
  }

  int64 Complexity(const Step *step) override {
    return step->input(0)->elements() * (16 + 5 * 2 + (sigmoid_ ? 3 : 0));
  }

 private:
  bool sigmoid_;
};

class AVXFltExp : public AVXFltExpBase {
 public:
  AVXFltExp() : AVXFltExpBase(false) {}

  string Name() override { return "AVXFltExp"; }
  string Operation() override { return "Exp"; }
};

class AVXFltSigmoid : public AVXFltExpBase {
 public:
  AVXFltSigmoid() : AVXFltExpBase(true) {}

  string Name() override { return "AVXFltSigmoid"; }
  string Operation() override { return "Sigmoid"; }
};

void RegisterAVXMath(Library *library) {
  // Computes  : y = tanh(x) element-wise
  // Input     : x: float32[d1,...,dn]
  // Output    : y: float32[d1,...,dn]
  // Requires  : AVX
  // Supports  : FMA3
  library->Register(new AVXFltTanh());

  // Computes  : y = exp(x) element-wise
  // Input:    : x: float32[d1,...,dn]
  // Output:   : y: float32[d1,...,dn]
  // Requires  : AVX
  // Supports  : FMA3
  library->Register(new AVXFltExp());

  // Computes  : y = sigmoid(x) = 1 / (1 + exp(-x)) element-wise
  // Input     : x: float32[d1,...,dn]
  // Output    : y: float32[d1,...,dn]
  // Requires  : AVX
  // Supports  : FMA3
  library->Register(new AVXFltSigmoid());
}

}  // namespace myelin
}  // namespace sling

