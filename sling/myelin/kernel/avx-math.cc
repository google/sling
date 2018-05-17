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

#include <math.h>
#include <stddef.h>
#include <string>

#include "sling/myelin/compute.h"
#include "sling/myelin/macro-assembler.h"

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

    // Strict math not supported.
    if (step->GetAttr("strict", false)) return false;

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
    __ load_extern(consts, static_cast<void *>(&tanh_const), "tanh_const");
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

    // Strict math not supported.
    if (step->GetAttr("strict", false)) return false;

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
    __ load_extern(consts, static_cast<void *>(&exp_const), "exp_const");
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
    __ vroundps(m, m, round_down);

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

// Compute argmax of input using AVX.
class AVXFltArgMax : public Kernel {
 public:
  string Name() override { return "AVXFltArgMax"; }
  string Operation() override { return "ArgMax"; }

  bool Supports(Step *step) override {
    // Requires CPU with AVX2 support.
    if (!CPU::Enabled(AVX2)) return false;

    // Check inputs and outputs.
    if (step->inputs().size() != 1) return false;
    if (step->outputs().size() != 1) return false;
    Tensor *x = step->input(0);
    Tensor *y = step->output(0);

    // Check type.
    if (x->type() != DT_FLOAT) return false;
    if (y->type() != DT_INT32 && y->type() != DT_INT64) return false;
    if (y->elements() != 1) return false;

    return true;
  }

  void Generate(Step *step, MacroAssembler *masm) override {
    // Get input and output.
    Tensor *x = step->input(0);
    Tensor *y = step->output(0);
    int main_elements = (x->elements() / 8) * 8;

    // Assign registers.
    Register input = masm->rr().alloc();
    Register output = masm->rr().alloc();
    Register ofs = masm->rr().alloc();
    Register best = masm->rr().alloc();
    YMMRegister eight = masm->mm().allocy();
    YMMRegister index = masm->mm().allocy();
    YMMRegister value = masm->mm().allocy();
    YMMRegister mask = masm->mm().allocy();
    XMMRegister maxval = masm->mm().allocx();
    YMMRegister maxval0 = masm->mm().allocy();
    YMMRegister best0 = masm->mm().allocy();

    // Load tensor locations.
    __ LoadTensorAddress(input, x);
    __ LoadTensorAddress(output, y);

    if (main_elements > 0) {
      // Initialize variables.
      static int idx_init[8] = {0, 1, 2, 3, 4, 5, 6, 7};
      auto *neginf = masm->GetConstant<float>(-INFINITY, 8);
      auto *negone = masm->GetConstant<int>(-1, 8);
      auto *plus8 = masm->GetConstant<int>(8, 8);
      auto *indices = masm->GetData(idx_init, sizeof(idx_init));
      __ vmovaps(index, Operand(indices->address()));
      __ vmovaps(eight, Operand(plus8->address()));
      __ vmovaps(maxval0, Operand(neginf->address()));
      __ vmovaps(best0, Operand(negone->address()));
      __ xorq(ofs, ofs);

      // Find argmax for main elements, eight elements at a time.
      const static int CMP_LE = 2;
      Label loop1;
      __ LoopStart(&loop1);
      __ vmovaps(value, Operand(input, ofs));
      __ vcmpps(mask, maxval0, value, CMP_LE);
      __ vblendvps(maxval0, maxval0, value, mask);
      __ vblendvps(best0, best0, index, mask);
      __ vpaddd(index, index, eight);  // requires avx2
      __ addq(ofs, Immediate(8 * sizeof(float)));
      __ cmpq(ofs, Immediate(main_elements * sizeof(float)));
      __ j(less, &loop1);

      // Reduce from 8 to 4.
      YMMRegister maxval1 = masm->mm().allocy();
      YMMRegister best1 = masm->mm().allocy();
      __ vperm2f128(maxval1, maxval0, maxval0, 1);
      __ vperm2f128(best1, best0, best0, 1);
      __ vcmpps(mask, maxval1, maxval0, CMP_LE);
      __ vblendvps(maxval1, maxval1, maxval0, mask);
      __ vblendvps(best1, best1, best0, mask);

      // Reduce from 4 to 2.
      YMMRegister maxval2 = masm->mm().allocy();
      YMMRegister best2 = masm->mm().allocy();
      __ vpermilps(maxval2, maxval1, 0x0E);
      __ vpermilps(best2, best1, 0x0E);
      __ vcmpps(mask, maxval2, maxval1, CMP_LE);
      __ vblendvps(maxval2, maxval2, maxval1, mask);
      __ vblendvps(best2, best2, best1, mask);

      // Reduce from 2 to 1.
      YMMRegister maxval3 = masm->mm().allocy();
      YMMRegister best3 = masm->mm().allocy();
      __ vpermilps(maxval3, maxval2, 0x01);
      __ vpermilps(best3, best2, 0x01);
      __ vcmpps(mask, maxval3, maxval2, CMP_LE);
      __ vblendvps(maxval3, maxval3, maxval2, mask);
      __ vblendvps(best3, best3, best2, mask);

      __ vmovss(maxval, maxval, maxval3.xmm());
      __ movq(best, best3.xmm());
    } else {
      auto *neginf = masm->GetConstant<float>(-INFINITY);
      __ movq(best, Immediate(-1));
      __ vmovss(maxval, Operand(neginf->address()));
    }

   // Reduce residual elements.
   if (main_elements < x->elements()) {
      Register idx = masm->rr().alloc();
      __ movq(idx, Immediate(main_elements));
      Label loop2;
      __ LoopStart(&loop2);
      __ vmovss(value.xmm(), Operand(input, idx, times_4));
      Label l2;
      __ vucomiss(value.xmm(), maxval);
      __ j(below_equal, &l2);
      __ vmovss(maxval, maxval, value.xmm());
      __ movq(best, idx);
      __ bind(&l2);
      __ incq(idx);
      __ cmpq(idx, Immediate(x->elements()));
      __ j(less, &loop2);
    }

    // Save output.
    if (y->type() == DT_INT32) {
      __ movl(Operand(output), best);
    } else {
      __ movq(Operand(output), best);
    }
  }

  int64 Complexity(const Step *step) override {
    return step->input(0)->elements();
  }
};

// Compute L2 norm of input, norm = sqrt(sum(x^2)).
class Norm : public Kernel {
 public:
  string Name() override { return "Norm"; }
  string Operation() override { return "Norm"; }

  bool Supports(Step *step) override {
    // Requires CPU with AVX or SSE support.
    if (!CPU::Enabled(AVX) || !CPU::Enabled(SSE)) return false;

    // Check inputs and outputs.
    if (step->inputs().size() != 1) return false;
    if (step->outputs().size() != 1) return false;
    Tensor *x = step->input(0);
    Tensor *norm = step->output(0);

    // Check type.
    if (x->type() != DT_FLOAT) return false;
    if (norm->type() != DT_FLOAT) return false;
    if (norm->elements() != 1) return false;

    return true;
  }

  void Adjust(Step *step) override {
    Tensor *x = step->input(0);
    int align = 16;
    if (CPU::Enabled(AVX)) align = 32;
    if (CPU::Enabled(AVX512F)) align = 64;
    x->SetMiniumAlignment(align);
  }

  void Generate(Step *step, MacroAssembler *masm) override {
    // Get input and output.
    Tensor *x = step->input(0);
    Tensor *norm = step->output(0);

    // Determine vector size for main block computation.
    bool avx512 = masm->Enabled(AVX512F);
    int n = 1;
    for (int d = 0; d < x->rank(); ++d) n *= x->aligned(d);
    int vecsize = 1;
    if (avx512 && n > 1) {
      vecsize = 16;
    } else if (masm->Enabled(AVX) && n >= 8) {
      vecsize = 8;
    } else if (masm->Enabled(SSE) && n >= 4) {
      vecsize = 4;
    }
    int m = (n / vecsize) * vecsize;
    int r = vecsize == 1 ? n : n % vecsize;

    // Assign registers.
    Register input = masm->rr().alloc();
    Register output = masm->rr().alloc();
    Register offset = masm->rr().alloc();
    ZMMRegister sum = masm->mm().allocz(false);
    ZMMRegister elem = masm->mm().allocz(false);
    ZMMRegister l2 = masm->mm().allocz(false);
    OpmaskRegister mask = masm->kk().alloc();

    // Load tensor locations.
    __ LoadTensorAddress(input, x);
    __ LoadTensorAddress(output, norm);
    if (avx512) {
      __ vxorps(sum, sum, sum);
    } else {
      __ vxorps(sum.ymm(), sum.ymm(), sum.ymm());
    }

    // Compute sum of squares for main elements.
    __ xorq(offset, offset);
    if (vecsize > 1) {
      if (m > 0) {
        Label l;
        __ bind(&l);
        if (avx512) {
          __ vmovaps(elem, Operand(input, offset));
          __ vfmadd231ps(sum, elem, elem);
        } else if (masm->Enabled(AVX)) {
          if (vecsize == 8) {
            __ vmovaps(elem.ymm(), Operand(input, offset));
            if (masm->Enabled(FMA3)) {
              __ vfmadd231ps(sum.ymm(), elem.ymm(), elem.ymm());
            } else {
              __ vmulps(elem.ymm(), elem.ymm(), elem.ymm());
              __ vaddps(sum.ymm(), sum.ymm(), elem.ymm());
            }
          } else {
            __ vmovaps(elem.xmm(), Operand(input, offset));
            if (masm->Enabled(FMA3)) {
              __ vfmadd231ps(sum.xmm(), elem.xmm(), elem.xmm());
            } else {
              __ vmulps(elem.xmm(), elem.xmm(), elem.xmm());
              __ vaddps(sum.xmm(), sum.xmm(), elem.xmm());
            }
          }
        } else {
          __ vmovaps(elem.xmm(), Operand(input, offset));
          __ mulps(elem.xmm(), elem.xmm());
          __ addps(sum.xmm(), elem.xmm());
        }
        if (m > vecsize || r > 0) {
          __ addq(offset, Immediate(vecsize * sizeof(float)));
        }
        if (m > vecsize) {
          __ cmpq(offset, Immediate(m * sizeof(float)));
          __ j(less, &l);
        }
      }

      // Compute residual for AVX512 mode using masking.
      if (avx512 && r > 0) {
        __ LoadMask(r, mask);
        __ vmovaps(elem, Operand(input, offset), Mask(mask, zeroing));
        __ vfmadd231ps(sum, elem, elem);
      }

      // Reduce sum.
      if (masm->Enabled(AVX)) {
        if (vecsize == 16) {
          __ vshuff32x4(elem, sum, sum, 0x0E);
          __ vaddps(sum, sum, elem);
        }
        if (vecsize == 8) {
          __ vperm2f128(elem.ymm(), sum.ymm(), sum.ymm(), 1);
          __ vhaddps(sum.ymm(), sum.ymm(), elem.ymm());
        }
        __ vhaddps(sum.ymm(), sum.ymm(), sum.ymm());
        __ vhaddps(sum.ymm(), sum.ymm(), sum.ymm());
      } else {
        __ haddps(sum.xmm(), sum.xmm());
        __ haddps(sum.xmm(), sum.xmm());
      }
    }

    // Compute sum of squares for residual elements.
    if (!avx512 && r > 0) {
      Label l;
      __ bind(&l);
      if (masm->Enabled(AVX)) {
        __ vmovss(elem.xmm(), Operand(input, offset));
        if (masm->Enabled(FMA3)) {
          __ vfmadd231ss(sum.xmm(), elem.xmm(), elem.xmm());
        } else {
          __ vmulss(elem.xmm(), elem.xmm(), elem.xmm());
          __ vaddss(sum.xmm(), sum.xmm(), elem.xmm());
        }
      } else {
        __ movss(elem.xmm(), Operand(input, offset));
        __ mulss(elem.xmm(), elem.xmm());
        __ addss(sum.xmm(), elem.xmm());
      }
      if (r > 1) {
        __ addq(offset, Immediate(sizeof(float)));
        __ cmpq(offset, Immediate(n * sizeof(float)));
        __ j(less, &l);
      }
    }

    // Take square root of sum.
    if (masm->Enabled(AVX)) {
      __ vsqrtss(l2.xmm(), sum.xmm(), sum.xmm());
      __ vmovss(Operand(output), l2.xmm());
    } else {
      __ sqrtss(l2.xmm(), sum.xmm());
      __ movss(Operand(output), l2.xmm());
    }
  }

  int64 Complexity(const Step *step) override {
    return step->input(0)->elements() * 3 + 5;
  }
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

  // Computes  : y = argmax(x)
  // Input     : x: float32[d1,...,dn]
  // Output    : y: int32/int64
  // Requires  : AVX
  library->Register(new AVXFltArgMax());

  // Computes  : y = |x|
  // Input     : x: float32[d1,...,dn]
  // Output    : y: float32
  // Requires  : AVX or SSE
  library->Register(new Norm());
}

}  // namespace myelin
}  // namespace sling

