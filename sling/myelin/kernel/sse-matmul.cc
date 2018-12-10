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

#include "sling/myelin/kernel/sse.h"

#include <string>

#include "sling/myelin/compute.h"
#include "sling/myelin/macro-assembler.h"

#define __ masm->

namespace sling {
namespace myelin {

using namespace jit;

// Vector-matrix multiplication for CPUs with SSE.
class SSEFltVecMatMulBase : public Kernel {
 public:
  // Maximum number of loop unrolls.
  static const int kMaxUnrolls = 4;

  // Maximum number of adder registers.
  static const int kMaxAdders = 2;

  SSEFltVecMatMulBase(bool bias, bool relu)
      : bias_(bias), relu_(relu) {}

  bool Supports(Step *step) override {
    // Requires CPU with SSE3 support.
    if (!CPU::Enabled(SSE3)) return false;

    // Two or three float 2D tensor inputs and one 2D tensor output.
    if (step->inputs().size() != (bias_ ? 3 : 2)) return false;
    if (step->outputs().size() != 1) return false;
    Tensor *x = step->input(0);
    Tensor *W = step->input(1);
    Tensor *y = step->output(0);
    if (x->rank() != 2 || x->type() != DT_FLOAT) return false;
    if (W->rank() != 2 || W->type() != DT_FLOAT) return false;
    if (y->rank() != 2 || y->type() != DT_FLOAT) return false;

    // Transpose not supported.
    if (step->GetAttr("transpose_a", false)) return false;
    if (step->GetAttr("transpose_b", false)) return false;
    if (step->GetAttr("transpose_c", false)) return false;

    // Check shape. First input must be a row vector.
    if (x->dim(0) != 1 || x->dim(1) != W->dim(0)) return false;
    if (y->dim(0) != x->dim(0) || y->dim(1) != W->dim(1)) return false;

    // The matrix must support column-major order.
    if (!W->SupportsOrder(COLUMN_MAJOR)) return false;

    // Check bias vector.
    if (bias_) {
      Tensor *b = step->input(2);
      if (b->type() != DT_FLOAT) return false;
      if (b->rank() == 1) {
        if (b->dim(0) != y->dim(1)) return false;
      } else if (b->rank() == 2) {
        if (b->dim(0) != 1 || b->dim(1) != y->dim(1)) return false;
      } else {
        return false;
      }
    }

    // Horizontal summation is not strict math compatible.
    if (step->GetAttr("strict", false)) return false;

    return true;
  }

  void Adjust(Step *step) override {
    Tensor *x = step->input(0);
    Tensor *W = step->input(1);

    int alignment = 4 * sizeof(float);

    x->MinAlign({1, 4});
    x->SetMiniumAlignment(alignment);

    W->MinAlign({4, 1});
    W->SetMiniumAlignment(alignment);
    W->RequireOrder(COLUMN_MAJOR);
  }

  void Generate(Step *step, MacroAssembler *masm) override {
    Registers &rr = masm->rr();
    SIMDRegisters &mm = masm->mm();
    Label l1, l2;

    // Get input and output tensors.
    Tensor *x = step->input(0);
    Tensor *W = step->input(1);
    Tensor *b = bias_ ? step->input(2) : nullptr;
    Tensor *y = step->output(0);

    // Get matrix dimensions.
    int rows = W->dim(0);
    int cols = W->dim(1);
    int row_size = W->stride(1);

    // Compute the number of unrolls and adders.
    int unrolls = 1;
    for (int i = 2; i <= kMaxUnrolls; ++i) {
      if (W->aligned(0) % (i * 4) == 0) unrolls = i;
    }
    int adders = unrolls;
    if (adders > kMaxAdders) adders = kMaxAdders;

    step->set_variant("U" + std::to_string(unrolls) +
                      "A" + std::to_string(adders));

    // Allocate general registers.
    Register row = rr.alloc();
    Register col = rr.alloc();
    Register matrix = rr.alloc();
    Register input = rr.alloc();
    Register output = rr.alloc();
    Register vector = bias_ ? rr.alloc() : no_reg;

    // Allocate SIMD registers.
    std::vector<XMMRegister> elem;
    for (int i = 0; i < unrolls; ++i) {
      elem.push_back(mm.allocx());
    }
    std::vector<XMMRegister> sum;
    for (int i = 0; i < adders; ++i) {
      sum.push_back(mm.allocx());
    }
    XMMRegister zero = relu_ ? mm.allocx() : no_xmm_reg;

    // Load tensor locations.
    __ LoadTensorAddress(input, x);
    __ LoadTensorAddress(matrix, W);
    if (bias_) {
      __ LoadTensorAddress(vector, b);
    }
    __ LoadTensorAddress(output, y);
    __ xorq(col, col);
    if (relu_) {
      __ xorps(zero, zero);
    }

    // Outer loop over columns.
    __ LoopStart(&l1);
    __ xorq(row, row);
    if (bias_) {
      __ movss(sum[0], Operand(vector, col, times_4));
    }
    for (int i = (bias_ ? 1 : 0); i < adders; ++i) {
       __ xorps(sum[i], sum[i]);
    }

    // Inner loop over rows.
    __ LoopStart(&l2);

    for (int i = 0; i < unrolls; ++i) {
      // Load x[row:row+4].
      int disp = 4 * i * sizeof(float);
      __ movaps(elem[i], Operand(input, row, times_4, disp));
    }

    for (int i = 0; i < unrolls; ++i) {
      int disp = 4 * i * sizeof(float);
      // Multiply x[row:row+4] with W[row:row+4,col].
      __ mulps(elem[i], Operand(matrix, row, times_4, disp));

      // Sum dot product in parallel.
      __ addps(sum[i % adders], elem[i]);
    }

    // Move to next row.
    __ addq(row, Immediate(4 * unrolls));
    __ cmpq(row, Immediate(rows));

    __ j(less, &l2);

    // Sum adders in sum[0].
    for (int i = 1; i < adders; ++i) {
      __ addps(sum[0], sum[i]);
    }

    // Add elements in sum[0] horizontally.
    __ haddps(sum[0], sum[0]);
    __ haddps(sum[0], sum[0]);

    // Compute relu.
    if (relu_) {
      __ maxps(sum[0], zero);
    }

    // Save to y[col].
    __ movss(Operand(output, col, times_4), sum[0]);
    __ addq(col, Immediate(1));

    // Move to next column.
    __ addq(matrix, Immediate(row_size));
    __ cmpq(col, Immediate(cols));
    __ j(less, &l1);
  }

  int64 Complexity(const Step *step) override {
    int64 ops = step->input(1)->elements() * 2;
    if (bias_) ops += step->input(2)->elements();
    if (relu_) ops += step->output(0)->elements();
    return ops;
  }

 protected:
  bool bias_;    // add bias vector to result, y=Wx+b
  bool relu_;    // apply rectified linear unit, y=max(0,Wx+b)
};

class SSEFltVecMatMul : public SSEFltVecMatMulBase {
 public:
  SSEFltVecMatMul() : SSEFltVecMatMulBase(false, false) {}

  string Name() override { return "SSEFltVecMatMul"; }
  string Operation() override { return "MatMul"; }
};

class SSEFltVecMatMulAdd : public SSEFltVecMatMulBase {
 public:
  SSEFltVecMatMulAdd() : SSEFltVecMatMulBase(true, false) {}

  string Name() override { return "SSEFltVecMatMulAdd"; }
  string Operation() override { return "MatMulAdd"; }
};

class SSEFltVecMatMulRelu : public SSEFltVecMatMulBase {
 public:
  SSEFltVecMatMulRelu() : SSEFltVecMatMulBase(false, true) {}

  string Name() override { return "SSEFltVecMatMulRelu"; }
  string Operation() override { return "MatMulRelu"; }
};

class SSEFltVecMatMulAddRelu : public SSEFltVecMatMulBase {
 public:
  SSEFltVecMatMulAddRelu() : SSEFltVecMatMulBase(true, true) {}

  string Name() override { return "SSEFltVecMatMulAddRelu"; }
  string Operation() override { return "MatMulAddRelu"; }
};

void RegisterSSEMatMul(Library *library) {
  // Computes  : y = x * W
  // Input     : x: float32[1,n]
  //             W: float32[n,m] column-major
  // Output    : y: float32[1,m]
  // Requires  : SSE3
  library->Register(new SSEFltVecMatMul());

  // Computes  : y = x * W + b
  // Input     : x: float32[1,n]
  //             W: float32[n,m] column-major
  //             b: float32[1,n]
  // Output    : y: float32[1,m]
  // Requires  : SSE3
  library->Register(new SSEFltVecMatMulAdd());

  // Computes  : y = max(0, x * W)
  // Input     : x: float32[1,n]
  //             W: float32[n,m] column-major
  // Output    : y: float32[1,m]
  // Requires  : SSE3
  library->Register(new SSEFltVecMatMulRelu());

  // Computes  : y = max(0, x * W + b)
  // Input     : x: float32[1,n]
  //             W: float32[n,m] column-major
  //             b: float32[1,n]
  // Output    : y: float32[1,m]
  // Requires  : SSE3
  library->Register(new SSEFltVecMatMulAddRelu());
}

}  // namespace myelin
}  // namespace sling

