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
#include <string>

#include "sling/myelin/compute.h"
#include "sling/myelin/macro-assembler.h"

#define __ masm->

namespace sling {
namespace myelin {

using namespace jit;

// Compute argmax (or argmin) of input using AVX.
class AVXFltArgMax : public Kernel {
 public:
  AVXFltArgMax(bool minimum) : minimum_(minimum) {}

  string Name() override { return minimum_ ? "AVXFltArgMin" : "AVXFltArgMax"; }
  string Operation() override { return minimum_ ? "ArgMin" : "ArgMax"; }

  bool Supports(Step *step) override {
    // Requires CPU with AVX2 support.
    if (!CPU::Enabled(AVX2)) return false;

    // Check inputs and outputs.
    if (step->indegree() != 1) return false;
    if (step->outdegree() != 1) return false;
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

    // Get extremum value.
    float limit = minimum_ ? INFINITY : -INFINITY;
    auto *inf = masm->GetConstant<float>(limit, 8);

    if (main_elements > 0) {
      // Initialize variables.
      static int idx_init[8] = {0, 1, 2, 3, 4, 5, 6, 7};
      auto *indices = masm->GetData(idx_init, sizeof(idx_init));

      // Find argmax/argmin for main elements, eight elements at a time.
      const static int CMP_LE = 2;
      const static int CMP_GE = 13;
      int compare = minimum_ ? CMP_GE : CMP_LE;
      if (main_elements > 8) {
        auto *plus8 = masm->GetConstant<int>(8, 8);
        auto *none = masm->GetConstant<int>(-1, 8);
        __ vmovaps(index, Operand(indices->address()));
        __ vmovaps(eight, Operand(plus8->address()));
        __ vmovaps(maxval0, Operand(inf->address()));
        __ vmovaps(best0, Operand(none->address()));
        __ xorq(ofs, ofs);
        Label loop1;
        __ LoopStart(&loop1);
        __ vmovaps(value, Operand(input, ofs));
        __ vcmpps(mask, maxval0, value, compare);
        __ vblendvps(maxval0, maxval0, value, mask);
        __ vblendvps(best0, best0, index, mask);
        __ vpaddd(index, index, eight);  // requires avx2
        __ addq(ofs, Immediate(8 * sizeof(float)));
        __ cmpq(ofs, Immediate(main_elements * sizeof(float)));
        __ j(less, &loop1);
      } else {
        __ vmovaps(maxval0, Operand(input));
        __ vmovaps(best0, Operand(indices->address()));
      }

      // Reduce from 8 to 4.
      YMMRegister maxval1 = masm->mm().allocy();
      YMMRegister best1 = masm->mm().allocy();
      __ vperm2f128(maxval1, maxval0, maxval0, 1);
      __ vperm2f128(best1, best0, best0, 1);
      __ vcmpps(mask, maxval1, maxval0, compare);
      __ vblendvps(maxval1, maxval1, maxval0, mask);
      __ vblendvps(best1, best1, best0, mask);

      // Reduce from 4 to 2.
      YMMRegister maxval2 = masm->mm().allocy();
      YMMRegister best2 = masm->mm().allocy();
      __ vpermilps(maxval2, maxval1, 0x0E);
      __ vpermilps(best2, best1, 0x0E);
      __ vcmpps(mask, maxval2, maxval1, compare);
      __ vblendvps(maxval2, maxval2, maxval1, mask);
      __ vblendvps(best2, best2, best1, mask);

      // Reduce from 2 to 1.
      YMMRegister maxval3 = masm->mm().allocy();
      YMMRegister best3 = masm->mm().allocy();
      __ vpermilps(maxval3, maxval2, 0x01);
      __ vpermilps(best3, best2, 0x01);
      __ vcmpps(mask, maxval3, maxval2, compare);
      __ vblendvps(maxval3, maxval3, maxval2, mask);
      __ vblendvps(best3, best3, best2, mask);

      __ vmovss(maxval, maxval, maxval3.xmm());
      __ movq(best, best3.xmm());
    } else {
      __ movq(best, Immediate(-1));
      __ vmovss(maxval, Operand(inf->address()));
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
      __ j(minimum_ ? above_equal : below_equal, &l2);
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

 private:
  bool minimum_;  // compute argmin instead of argmax
};

void RegisterAVXMath(Library *library) {
  // Computes  : y = argmax(x)
  // Input     : x: float32[d1,...,dn]
  // Output    : y: int32/int64
  // Requires  : AVX
  library->Register(new AVXFltArgMax(false));

  // Computes  : y = argmin(x)
  // Input     : x: float32[d1,...,dn]
  // Output    : y: int32/int64
  // Requires  : AVX
  library->Register(new AVXFltArgMax(true));
}

}  // namespace myelin
}  // namespace sling

