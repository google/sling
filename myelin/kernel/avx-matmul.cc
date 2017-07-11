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

// Base class for vector-matrix multiplication for CPUs with AVX.
class AVXVecMatMulBase : public Kernel {
 public:
  AVXVecMatMulBase(bool bias, bool relu, Order order, Type itype, Type otype)
      : bias_(bias), relu_(relu), order_(order), itype_(itype), otype_(otype) {}

  bool Supports(Step *step) override {
    // Requires CPU with AVX support.
    if (!CPU::Enabled(AVX)) return false;

    // Two or three 2D tensor inputs and one 2D tensor output.
    if (step->inputs().size() != (bias_ ? 3 : 2)) return false;
    if (step->outputs().size() != 1) return false;
    Tensor *x = step->input(0);
    Tensor *W = step->input(1);
    Tensor *y = step->output(0);
    if (x->rank() != 2 || x->type() != itype_) return false;
    if (W->rank() != 2 || W->type() != itype_) return false;
    if (y->rank() != 2 || y->type() != otype_) return false;

    // Check shape. First input must be a row vector.
    if (x->dim(0) != 1 || x->dim(1) != W->dim(0)) return false;
    if (y->dim(0) != x->dim(0) || y->dim(1) != W->dim(1)) return false;

    // The matrix must support required order.
    if (!W->SupportsOrder(order_)) return false;

    // Transpose not supported.
    if (step->GetAttr("transpose_a", false)) return false;
    if (step->GetAttr("transpose_b", false)) return false;

    // Check bias vector.
    if (bias_) {
      Tensor *b = step->input(2);
      if (b->type() != otype_) return false;
      if (b->rank() == 1) {
        if (b->dim(0) != y->dim(1)) return false;
      } else if (b->rank() == 2) {
        if (b->dim(0) != 1 || b->dim(1) != y->dim(1)) return false;
      } else {
        return false;
      }
    }

    return true;
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
  Order order_;  // required order for matrix
  Type itype_;   // input type
  Type otype_;   // output type
};

// Vertical float vector-matrix multiplication for CPUs with AVX.
class AVXFltVecMatMulVBase : public AVXVecMatMulBase {
 public:
  // Maximum number of loop unrolls.
  static const int kMaxUnrolls = 8;

  AVXFltVecMatMulVBase(bool bias, bool relu)
      : AVXVecMatMulBase(bias, relu, ROW_MAJOR, DT_FLOAT, DT_FLOAT) {}

  void Adjust(Step *step) override {
    // Get input and output tensors.
    Tensor *x = step->input(0);
    Tensor *W = step->input(1);
    Tensor *b = bias_ ? step->input(2) : nullptr;
    Tensor *y = step->output(0);

    // Align to one ymm register (256 bits, 32 bytes).
    int byte_alignment = 256 / 8;
    x->SetMiniumAlignment(byte_alignment);
    W->SetMiniumAlignment(byte_alignment);
    y->SetMiniumAlignment(byte_alignment);
    if (bias_) b->SetMiniumAlignment(byte_alignment);

    // Rows must be aligned to ymm boundaries to support aligned loads.
    W->MinAlign({8, 1});
    W->SetRequiredOrder(ROW_MAJOR);
  }

  void Generate(Step *step, MacroAssembler *masm) override {
    Registers &rr = masm->rr();
    SIMDRegisters &mm = masm->mm();
    Label l1, l2, l3;

    // Get input and output tensors.
    Tensor *x = step->input(0);
    Tensor *W = step->input(1);
    Tensor *b = bias_ ? step->input(2) : nullptr;
    Tensor *y = step->output(0);

    // FMA is not strict math compatible.
    bool fma = masm->Enabled(FMA3);
    bool strict = step->GetAttr("strict", false);
    if (strict) {
      fma = false;
      step->set_variant("strict");
    }

    // Get matrix dimensions.
    int rows = W->dim(0);
    int cols = W->dim(1);
    int main_cols = (cols  / 8) * 8;
    int remaining_cols = cols - main_cols;

    // Compute the number of unrolls.
    int unrolls = 0;
    for (int i = 1; i <= kMaxUnrolls; ++i) {
      int batch_size = i * 8;
      if (main_cols >= batch_size && main_cols % batch_size == 0) unrolls = i;
    }
    if (step->variant().empty()) {
      string variant = "U" + std::to_string(unrolls);
      if (remaining_cols > 0) variant += "R" + std::to_string(remaining_cols);
      step->set_variant(variant);
    }

    // Allocate general registers.
    Register rowofs = rr.alloc();
    Register colofs = rr.alloc();
    Register m = rr.alloc();
    Register matrix = rr.alloc();
    Register input = rr.alloc();
    Register output = rr.alloc();
    Register vector = bias_ ? rr.alloc() : no_reg;

    // Allocate SIMD registers.
    std::vector<YMMRegister> sum;
    for (int i = 0; i < std::max(unrolls, 4); ++i) {
      sum.push_back(mm.allocy());
    }
    std::vector<YMMRegister> acc;
    for (int i = 0; i < 4; ++i) {
      acc.push_back(mm.allocy());
    }
    YMMRegister elem = mm.allocy();
    YMMRegister zero = relu_ ? mm.allocy() : no_ymm_reg;

    // Load tensor locations.
    __ LoadTensorAddress(input, x);
    __ LoadTensorAddress(matrix, W);
    if (bias_) {
      __ LoadTensorAddress(vector, b);
    }
    __ LoadTensorAddress(output, y);

    // Initialize SIMD register to zero for relu.
    if (relu_) {
      __ vxorps(zero, zero, zero);
    }

    // Compute main columns.
    if (unrolls > 0) {
      // Outer loop over matrix column blocks.
      __ xorq(colofs, colofs);
      __ LoopStart(&l1);

      // Initialize block with bias or zero.
      for (int i = 0; i < unrolls; ++i) {
        if (bias_ && !strict) {
          __ vmovaps(sum[i], Operand(vector, colofs, times_1, i * 32));
        } else {
          __ vxorps(sum[i], sum[i], sum[i]);
        }
      }
      __ movq(m, matrix);
      __ xorq(rowofs, rowofs);

      // Inner loop over rows.
      __ LoopStart(&l2);

      // Load x[row].
      __ vbroadcastss(elem, Operand(input, rowofs));

      // Multiply x[row] with W[row,col:col+n] and add to sum.
      for (int i = 0; i < unrolls; ++i) {
        if (fma) {
          __ vfmadd231ps(sum[i], elem, Operand(m, i * 32));
        } else {
          __ vmulps(acc[i % 4], elem, Operand(m, i * 32));
          __ vaddps(sum[i], sum[i], acc[i % 4]);
        }
      }

      // Next row.
      if (rows > 1) {
        __ addq(m, Immediate(W->stride(0)));
        __ addq(rowofs, Immediate(sizeof(float)));
        __ cmpq(rowofs, Immediate(rows * sizeof(float)));
        __ j(less, &l2);
      }

      // Save to y[col:col+n].
      for (int i = 0; i < unrolls; ++i) {
        // Add bias last in strict mode.
        if (bias_ && strict) {
          __ vaddps(sum[i], sum[i], Operand(vector, colofs, times_1, i * 32));
        }

        // Compute relu.
        if (relu_) {
          __ vmaxps(sum[i], sum[i], zero);
        }
        __ vmovaps(Operand(output, colofs, times_1, i * 32), sum[i]);
      }

      // Next matrix column block.
      if (main_cols > unrolls * 8 || remaining_cols > 0) {
        __ addq(matrix, Immediate(unrolls * 32));
      }
      if (main_cols > unrolls * 8) {
        __ addq(colofs, Immediate(unrolls * 32));
        __ cmpq(colofs, Immediate(main_cols * sizeof(float)));
        __ j(less, &l1);
      }
    }

    // Compute remaining columns.
    if (remaining_cols > 0) {
      CHECK_LE(remaining_cols, 7);
      // Initialize remaining columns with bias or zero.
      int coldisp = main_cols * sizeof(float);
      if (remaining_cols & 4) {
        if (bias_ && !strict) {
          __ vmovaps(sum[0].xmm(), Operand(vector, coldisp));
        } else {
          __ vxorps(sum[0].xmm(), sum[0].xmm(), sum[0].xmm());
        }
        coldisp += 4 * sizeof(float);
      }
      for (int i = 0; i < (remaining_cols & 3); ++i) {
        int reg = i + 1;
        if (bias_ && !strict) {
          __ vmovss(sum[reg].xmm(), Operand(vector, coldisp));
        } else {
          __ vxorps(sum[reg].xmm(), sum[reg].xmm(), sum[reg].xmm());
        }
        coldisp += sizeof(float);
      }

      // Loop over rows.
      __ movq(m, matrix);
      __ xorq(rowofs, rowofs);
      __ LoopStart(&l3);

      // Load x[row].
      if (remaining_cols & 4) {
        __ vbroadcastss(elem, Operand(input, rowofs));
      } else {
        __ vmovss(elem, Operand(input, rowofs));
      }

      // Compute first four remaining columns using SSE.
      int left = remaining_cols;
      int disp = 0;
      if (left >= 4) {
        if (fma) {
          __ vfmadd231ps(sum[0].xmm(), elem.xmm(), Operand(m, disp));
        } else {
          __ vmulps(acc[0].xmm(), elem.xmm(), Operand(m, disp));
          __ vaddps(sum[0].xmm(), sum[0].xmm(), acc[0].xmm());
        }
        left -= 4;
        disp += 4 * sizeof(float);
      }

      // Compute up to three remaining columns as scalars.
      int reg = 1;
      while (left > 0) {
        if (fma) {
          __ vfmadd231ss(sum[reg].xmm(), elem.xmm(), Operand(m, disp));
        } else {
          __ vmulss(acc[0].xmm(), elem.xmm(), Operand(m, disp));
          __ vaddss(sum[reg].xmm(), sum[reg].xmm(), acc[0].xmm());
        }
        left--;
        reg++;
        disp += sizeof(float);
      }

      // Next row.
      if (rows > 1) {
        __ addq(m, Immediate(W->stride(0)));
        __ addq(rowofs, Immediate(sizeof(float)));
        __ cmpq(rowofs, Immediate(rows * sizeof(float)));
        __ j(less, &l3);
      }

      // Compute relu and save remaining columns.
      coldisp = main_cols * sizeof(float);
      if (remaining_cols & 4) {
        if (bias_ && strict) {
          __ vaddps(sum[0].xmm(), sum[0].xmm(), Operand(vector, coldisp));
        }
        if (relu_) {
          __ vmaxps(sum[0].xmm(), sum[0].xmm(), zero.xmm());
        }
        __ vmovaps(Operand(output, coldisp), sum[0].xmm());
        coldisp += 4 * sizeof(float);
      }
      for (int i = 0; i < (remaining_cols & 3); ++i) {
        int reg = i + 1;
        if (bias_ && strict) {
          __ vaddss(sum[reg].xmm(), sum[reg].xmm(), Operand(vector, coldisp));
        }
        if (relu_) {
          __ vmaxss(sum[reg].xmm(), sum[reg].xmm(), zero.xmm());
        }
        __ vmovss(Operand(output, coldisp), sum[reg].xmm());
        coldisp += sizeof(float);
      }
    }
  }
};

class AVXFltVecMatMulV : public AVXFltVecMatMulVBase {
 public:
  AVXFltVecMatMulV() : AVXFltVecMatMulVBase(false, false) {}

  string Name() override { return "AVXFltVecMatMulV"; }
  string Operation() override { return "MatMul"; }
};

class AVXFltVecMatMulAddV : public AVXFltVecMatMulVBase {
 public:
  AVXFltVecMatMulAddV() : AVXFltVecMatMulVBase(true, false) {}

  string Name() override { return "AVXFltVecMatMulAddV"; }
  string Operation() override { return "MatMulAdd"; }
};

class AVXFltVecMatMulReluV : public AVXFltVecMatMulVBase {
 public:
  AVXFltVecMatMulReluV() : AVXFltVecMatMulVBase(false, true) {}

  string Name() override { return "AVXFltVecMatMulReluV"; }
  string Operation() override { return "MatMulRelu"; }
};

class AVXFltVecMatMulAddReluV : public AVXFltVecMatMulVBase {
 public:
  AVXFltVecMatMulAddReluV() : AVXFltVecMatMulVBase(true, true) {}

  string Name() override { return "AVXFltVecMatMulAddReluV"; }
  string Operation() override { return "MatMulAddRelu"; }
};

// Horizontal float vector-matrix multiplication for CPUs with AVX.
class AVXFltVecMatMulHBase : public AVXVecMatMulBase {
 public:
  // Maximum number of loop unrolls.
  static const int kMaxUnrolls = 4;

  // Maximum number of adder registers.
  static const int kMaxAdders = 4;

  AVXFltVecMatMulHBase(bool bias, bool relu)
      : AVXVecMatMulBase(bias, relu, COLUMN_MAJOR, DT_FLOAT, DT_FLOAT) {}

  bool Supports(Step *step) override {
    if (!AVXVecMatMulBase::Supports(step)) return false;

    // Horizontal summation is not strict math compatible.
    if (step->GetAttr("strict", false)) return false;

    return true;
  }

  void Adjust(Step *step) override {
    // Get input and output tensors.
    Tensor *x = step->input(0);
    Tensor *W = step->input(1);
    Tensor *b = bias_ ? step->input(2) : nullptr;
    Tensor *y = step->output(0);

    // Align to one ymm register (256 bits, 32 bytes).
    int byte_alignment = 256 / 8;
    x->SetMiniumAlignment(byte_alignment);
    W->SetMiniumAlignment(byte_alignment);
    y->SetMiniumAlignment(byte_alignment);
    if (bias_) b->SetMiniumAlignment(byte_alignment);

    W->SetRequiredOrder(COLUMN_MAJOR);
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
    int main_rows = (rows  / 8) * 8;
    int remaining_rows = rows - main_rows;
    int row_size = W->stride(1);

    // Compute the number of unrolls and adders.
    int unrolls = 0;
    for (int i = 1; i <= kMaxUnrolls; ++i) {
      int batch_size = i * 8;
      if (main_rows >= batch_size && main_rows % batch_size == 0) unrolls = i;
    }
    int adders = unrolls;
    if (adders < 1) adders = 1;
    if (adders > kMaxAdders) adders = kMaxAdders;
    string variant = "U" + std::to_string(unrolls);
    variant += "A" + std::to_string(adders);
    if (remaining_rows > 0) variant += "R" + std::to_string(remaining_rows);
    step->set_variant(variant);

    // Allocate general registers.
    Register row = rr.alloc();
    Register col = rr.alloc();
    Register matrix = rr.alloc();
    Register input = rr.alloc();
    Register output = rr.alloc();
    Register vector = bias_ ? rr.alloc() : no_reg;

    // Allocate SIMD registers.
    std::vector<YMMRegister> elem;
    for (int i = 0; i < std::max(unrolls, 1); ++i) {
      elem.push_back(mm.allocy());
    }
    std::vector<YMMRegister> sum;
    for (int i = 0; i < adders; ++i) {
      sum.push_back(mm.allocy());
    }
    YMMRegister acc = mm.allocy();
    YMMRegister zero = relu_ ? mm.allocy() : no_ymm_reg;

    // Load tensor locations.
    __ LoadTensorAddress(input, x);
    __ LoadTensorAddress(matrix, W);
    if (bias_) {
      __ LoadTensorAddress(vector, b);
    }
    __ LoadTensorAddress(output, y);
    __ xorq(col, col);
    if (relu_) {
      __ vxorps(zero, zero, zero);
    }

    // Outer loop over columns.
    __ LoopStart(&l1);

    if (unrolls == 0) {
      // Unroll dot product for small rows up to seven elements.
      XMMRegister s = sum[0].xmm();
      XMMRegister e = elem[0].xmm();
      if (bias_) {
        __ vmovss(s, Operand(vector, col, times_4));
      } else {
        __ vmovss(e, Operand(input));
        __ vmulss(s, e, Operand(matrix));
      }
      for (int r = (bias_ ? 0 : 1); r < rows; ++r) {
        int disp = r * sizeof(float);
        __ vmovss(e, Operand(input, disp));
        if (masm->Enabled(FMA3)) {
          __ vfmadd231ss(s, e, Operand(matrix, disp));
        } else {
          __ vmulss(e, e, Operand(matrix, disp));
          __ vaddss(s, s, e);
        }
      }
    } else {
      // Inner loop over main rows.
      __ xorq(row, row);
      if (bias_) {
        __ vmovss(sum[0], Operand(vector, col, times_4));
      }
      for (int i = (bias_ ? 1 : 0); i < adders; ++i) {
        __ vxorps(sum[i], sum[i], sum[i]);
      }

      __ LoopStart(&l2);
      for (int i = 0; i < unrolls; ++i) {
        // Load x[row:row+8].
        int disp = 8 * i * sizeof(float);
        __ vmovaps(elem[i], Operand(input, row, times_4, disp));
      }
      for (int i = 0; i < unrolls; ++i) {
        int disp = 8 * i * sizeof(float);
        if (masm->Enabled(FMA3)) {
          // Multiply x[row:row+8] with W[row:row+8,col] and add to sum.
          __ vfmadd231ps(sum[i % adders], elem[i],
                         Operand(matrix, row, times_4, disp));
        } else {
          // Multiply x[row:row+8] with W[row:row+8,col].
          __ vmulps(elem[i], elem[i], Operand(matrix, row, times_4, disp));

          // Sum dot product in parallel.
          __ vaddps(sum[i % adders], sum[i % adders], elem[i]);
        }
      }

      // Move to next row batch.
      if (main_rows > 8 * unrolls) {
        __ addq(row, Immediate(8 * unrolls));
        __ cmpq(row, Immediate(main_rows));
        __ j(less, &l2);
      }

      // Add remaining rows.
      if (remaining_rows > 0) {
        XMMRegister s = acc.xmm();
        XMMRegister e = elem[0].xmm();
        int disp = main_rows * sizeof(float);
        int left = remaining_rows;

        // Add first four remaining elements using SSE.
        if (left >= 4) {
          __ vmovaps(s, Operand(input, disp));
          __ vmulps(s, s, Operand(matrix, disp));
          __ vaddps(sum[0], sum[0], acc);
          left -= 4;
          disp += 4 * sizeof(float);
        }

        // Add up to three remaining elements as scalars.
        if (left > 0) {
          __ vmovss(s, Operand(input, disp));
          __ vmulss(s, s, Operand(matrix, disp));
          left--;
          disp += sizeof(float);
          while (left > 0) {
            __ vmovss(e, Operand(input, disp));
            if (masm->Enabled(FMA3)) {
              __ vfmadd231ss(s, e, Operand(matrix, disp));
            } else {
              __ vmulss(e, e, Operand(matrix, disp));
              __ vaddss(s, s, e);
            }
            left--;
            disp += sizeof(float);
          }
          __ vperm2f128(acc, acc, acc, 0x80);
          __ vaddps(sum[0], sum[0], acc);
        }
      }

      // Sum adders in sum[0].
      if (adders == 4) {
        __ vaddps(sum[0], sum[0], sum[2]);
        __ vaddps(sum[1], sum[1], sum[3]);
        __ vaddps(sum[0], sum[0], sum[1]);
      } else {
        for (int i = 1; i < adders; ++i) {
          __ vaddps(sum[0], sum[0], sum[i]);
        }
      }

      // Add elements in sum[0] horizontally.
      __ vperm2f128(acc, sum[0], sum[0], 1);
      __ vhaddps(sum[0], sum[0], acc);
      __ vhaddps(sum[0], sum[0], sum[0]);
      __ vhaddps(sum[0], sum[0], sum[0]);
    }

    // Compute relu.
    if (relu_) {
      __ vmaxss(sum[0], sum[0], zero);
    }

    // Save to y[col].
    __ vmovss(Operand(output, col, times_4), sum[0]);

    // Move to next column.
    if (cols > 1) {
      __ addq(col, Immediate(1));
      __ addq(matrix, Immediate(row_size));
      __ cmpq(col, Immediate(cols));
      __ j(less, &l1);
    }
  }
};

class AVXFltVecMatMulH : public AVXFltVecMatMulHBase {
 public:
  AVXFltVecMatMulH() : AVXFltVecMatMulHBase(false, false) {}

  string Name() override { return "AVXFltVecMatMulH"; }
  string Operation() override { return "MatMul"; }
};

class AVXFltVecMatMulAddH : public AVXFltVecMatMulHBase {
 public:
  AVXFltVecMatMulAddH() : AVXFltVecMatMulHBase(true, false) {}

  string Name() override { return "AVXFltVecMatMulAddH"; }
  string Operation() override { return "MatMulAdd"; }
};

class AVXFltVecMatMulReluH : public AVXFltVecMatMulHBase {
 public:
  AVXFltVecMatMulReluH() : AVXFltVecMatMulHBase(false, true) {}

  string Name() override { return "AVXFltVecMatMulReluH"; }
  string Operation() override { return "MatMulRelu"; }
};

class AVXFltVecMatMulAddReluH : public AVXFltVecMatMulHBase {
 public:
  AVXFltVecMatMulAddReluH() : AVXFltVecMatMulHBase(true, true) {}

  string Name() override { return "AVXFltVecMatMulAddReluH"; }
  string Operation() override { return "MatMulAddRelu"; }
};

// AVX float matrix-matrix multiplication, C = A * B.
class AVXFltMatMatMul : public Kernel {
 public:
  // Maximum number of loop unrolls.
  static const int kMaxUnrolls = 4;

  // Maximum number of adder registers.
  static const int kMaxAdders = 4;

  string Name() override { return "AVXFltMatMatMul"; }
  string Operation() override { return "MatMul"; }

  bool Supports(Step *step) override {
    // Requires CPU with AVX support.
    if (!CPU::Enabled(AVX)) return false;

    // Two float 2D tensor inputs and one 2D tensor output.
    if (step->indegree() != 2) return false;
    if (step->outdegree() != 1) return false;
    Tensor *A = step->input(0);
    Tensor *B = step->input(1);
    Tensor *C = step->output(0);
    if (A->rank() != 2 || A->type() != DT_FLOAT) return false;
    if (B->rank() != 2 || B->type() != DT_FLOAT) return false;
    if (C->rank() != 2 || C->type() != DT_FLOAT) return false;

    // Check shape.
    bool transpose_a = step->GetAttr("transpose_a", false);
    bool transpose_b = step->GetAttr("transpose_b", false);
    Shape a = A->shape();
    Shape b = B->shape();
    Shape c = C->shape();
    if (transpose_a) a.transpose();
    if (transpose_b) b.transpose();

    if (a.dim(0) != c.dim(0)) return false;
    if (a.dim(1) != b.dim(0)) return false;
    if (b.dim(1) != c.dim(1)) return false;

    // Check order.
    if (!A->SupportsOrder(transpose_a ? COLUMN_MAJOR : ROW_MAJOR)) return false;
    if (!B->SupportsOrder(transpose_b ? ROW_MAJOR : COLUMN_MAJOR)) return false;
    if (!C->SupportsOrder(ROW_MAJOR)) return false;

    return true;
  }

  void Adjust(Step *step) override {
    // Get input and output tensors.
    Tensor *A = step->input(0);
    Tensor *B = step->input(1);
    Tensor *C = step->output(0);

    // Set alignment requirements.
    bool transpose_a = step->GetAttr("transpose_a", false);
    bool transpose_b = step->GetAttr("transpose_b", false);
    if (transpose_a) {
      A->MinAlign({8, 1});
    } else {
      A->MinAlign({1, 8});
    }
    if (transpose_b) {
      B->MinAlign({1, 8});
    } else {
      B->MinAlign({8, 1});
    }

    A->SetMiniumAlignment(32);
    B->SetMiniumAlignment(32);

    // Set order requirements.
    A->SetRequiredOrder(transpose_a ? COLUMN_MAJOR : ROW_MAJOR);
    B->SetRequiredOrder(transpose_b ? ROW_MAJOR : COLUMN_MAJOR);
    C->SetRequiredOrder(ROW_MAJOR);
  }

  void Generate(Step *step, MacroAssembler *masm) override {
    Registers &rr = masm->rr();
    SIMDRegisters &mm = masm->mm();
    Label l1, l2, l3;

    // Get input and output tensors.
    Tensor *A = step->input(0);
    Tensor *B = step->input(1);
    Tensor *C = step->output(0);

    // Get dimensions for matrices.
    bool transpose_a = step->GetAttr("transpose_a", false);
    bool transpose_b = step->GetAttr("transpose_b", false);
    int a_row_dim = transpose_a ? 1 : 0;
    int a_col_dim = transpose_a ? 0 : 1;
    int b_row_dim = transpose_b ? 1 : 0;
    int b_col_dim = transpose_b ? 0 : 1;
    int c_col_dim = 1;

    // Compute the number of unrolls and adders.
    int unrolls = 1;
    for (int i = 2; i <= kMaxUnrolls; ++i) {
      if (B->aligned(b_row_dim) % (i * 8) == 0) unrolls = i;
    }
    int adders = unrolls;
    if (adders > kMaxAdders) adders = kMaxAdders;

    // Allocate general registers.
    Register a = rr.alloc();
    Register b = rr.alloc();
    Register b_row = rr.alloc();
    Register b_end = rr.alloc();
    Register c = rr.alloc();
    Register c_end = rr.alloc();
    Register k = rr.alloc();

    // Allocate SIMD registers.
    std::vector<YMMRegister> elem;
    for (int n = 0; n < unrolls; ++n) {
      elem.push_back(mm.allocy());
    }
    std::vector<YMMRegister> sum;
    for (int n = 0; n < adders; ++n) {
      sum.push_back(mm.allocy());
    }
    YMMRegister acc = mm.allocy();

    // Load tensor locations.
    __ LoadTensorAddress(a, A);
    __ LoadTensorAddress(b, B);
    __ LoadTensorAddress(c, C);

    // Compute end of B and C.
    __ movq(b_end, b);
    __ addq(b_end, Immediate(B->size()));
    __ movq(c_end, c);
    __ addq(c_end, Immediate(C->size()));

    // Loop over all rows in C.
    __ LoopStart(&l1);
    __ movq(b_row, b);

    // Loop over all columns in C.
    __ LoopStart(&l2);
    __ xorq(k, k);
    for (int n = 0; n < adders; ++n) {
      __ vxorps(sum[n], sum[n], sum[n]);
    }

    // Compute dot product of row in A and column in B.
    // C[i,j] = sum_k A[i,k] * B[k,j].
    __ LoopStart(&l3);
    for (int n = 0; n < unrolls; ++n) {
      // Load A[i,k:k+8].
      int disp = 8 * n * sizeof(float);
      __ vmovaps(elem[n], Operand(a, k, times_4, disp));
    }

    for (int n = 0; n < unrolls; ++n) {
      // Multiply A[i,k:k+8] with B[k:k+8,j] and add to sum.
      int disp = 8 * n * sizeof(float);
      if (masm->Enabled(FMA3)) {
        __ vfmadd231ps(sum[n % adders], elem[n],
                       Operand(b_row, k, times_4, disp));
      } else {
        __ vmulps(elem[n], elem[n], Operand(b_row, k, times_4, disp));
        __ vaddps(sum[n % adders], sum[n % adders], elem[n]);
      }
    }

    __ addq(k, Immediate(8 * unrolls));
    __ cmpq(k, Immediate(A->dim(a_col_dim)));
    __ j(less, &l3);

    // Sum adders in sum[0].
    if (adders == 4) {
      __ vaddps(sum[0], sum[0], sum[2]);
      __ vaddps(sum[1], sum[1], sum[3]);
      __ vaddps(sum[0], sum[0], sum[1]);
    } else {
      for (int n = 1; n < adders; ++n) {
        __ vaddps(sum[0], sum[0], sum[n]);
      }
    }

    // Add elements in sum[0] horizontally.
    __ vperm2f128(acc, sum[0], sum[0], 1);
    __ vhaddps(sum[0], sum[0], acc);
    __ vhaddps(sum[0], sum[0], sum[0]);
    __ vhaddps(sum[0], sum[0], sum[0]);

    // Save to C[i,j].
    __ vmovss(Operand(c), sum[0]);
    __ addq(c, Immediate(C->stride(c_col_dim)));

    // Move to next column in B
    __ addq(b_row, Immediate(B->stride(b_col_dim)));
    __ cmpq(b_row, b_end);
    __ j(less, &l2);

    // Move to next row in A.
    __ addq(a, Immediate(A->stride(a_row_dim)));

    // Move to next row in C.
    if (C->padding(1) != 0) {
      __ addq(c, Immediate(C->padding(c_col_dim)));
    }
    __ cmpq(c, c_end);
    __ j(less, &l1);
  }

  int64 Complexity(const Step *step) override {
    return step->input(0)->dim(0) * step->input(1)->elements() * 2;
  }
};

// Horizontal integer vector-matrix multiplication for CPUs with AVX2.
class AVXIntVecMatMulHBase : public AVXVecMatMulBase {
 public:
  AVXIntVecMatMulHBase(bool bias, bool relu)
      : AVXVecMatMulBase(bias, relu, COLUMN_MAJOR, DT_INT8, DT_INT16) {}

  bool Supports(Step *step) override {
    // Requires CPU with AVX2 support.
    if (!CPU::Enabled(AVX2)) return false;
    return AVXVecMatMulBase::Supports(step);
  }

  void Adjust(Step *step) override {
    // Get input and output tensors.
    Tensor *x = step->input(0);
    Tensor *W = step->input(1);
    Tensor *b = bias_ ? step->input(2) : nullptr;
    Tensor *y = step->output(0);

    // Align to one ymm register (256 bits, 32 bytes).
    int byte_alignment = 256 / 8;
    x->SetMiniumAlignment(byte_alignment);
    W->SetMiniumAlignment(byte_alignment);
    y->SetMiniumAlignment(byte_alignment);
    if (bias_) b->SetMiniumAlignment(byte_alignment);

    W->SetRequiredOrder(COLUMN_MAJOR);
    x->MinAlign({1, 32});
    W->MinAlign({32, 1});
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
    bool unroll = W->aligned(0) % 64 == 0;

    // Allocate general registers.
    Register acc = rr.alloc();
    Register row = rr.alloc();
    Register col = rr.alloc();
    Register matrix = rr.alloc();
    Register input = rr.alloc();
    Register output = rr.alloc();
    Register vector = bias_ ? rr.alloc() : no_reg;

    // Allocate SIMD registers.
    YMMRegister zero = mm.allocy();

    YMMRegister xval0 = mm.allocy();
    YMMRegister xpos0 = mm.allocy();
    YMMRegister xneg0 = mm.allocy();
    YMMRegister wval0 = mm.allocy();
    YMMRegister sump0 = mm.allocy();
    YMMRegister sumn0 = mm.allocy();

    YMMRegister xval1 = mm.allocy();
    YMMRegister xpos1 = mm.allocy();
    YMMRegister xneg1 = mm.allocy();
    YMMRegister wval1 = mm.allocy();
    YMMRegister sump1 = mm.allocy();
    YMMRegister sumn1 = mm.allocy();

    // Load tensor locations.
    __ LoadTensorAddress(input, x);
    __ LoadTensorAddress(matrix, W);
    if (bias_) {
      __ LoadTensorAddress(vector, b);
    }
    __ LoadTensorAddress(output, y);
    __ vxorps(zero, zero, zero);
    __ xorq(col, col);

    // Outer loop over columns.
    __ LoopStart(&l1);
    __ xorq(row, row);

    // Initialize positive and negative parts of dot product.
    if (bias_) {
      __ movb(acc, Operand(vector, col));
      __ vmovq(sump0.xmm(), acc);
    } else {
       __ vxorps(sump0, sump0, sump0);
    }
    __ vxorps(sumn0, sumn0, sumn0);
    if (unroll) {
      __ vxorps(sump1, sump1, sump1);
      __ vxorps(sumn1, sumn1, sumn1);
    }

    // Inner loop over rows.
    __ LoopStart(&l2);

    // Load next 32 or 64 elements from x and W and split x into positive and
    // negative parts.
    __ vmovdqa(xval0, Operand(input, row));
    __ vmovdqa(wval0, Operand(matrix, row));

    __ vpminsb(xneg0, xval0, zero);
    __ vpsubb(xneg0, zero, xneg0);
    __ vpmaxsb(xpos0, xval0, zero);

    if (unroll) {
      __ vmovdqa(xval1, Operand(input, row, times_1, 32));
      __ vmovdqa(wval1, Operand(matrix, row, times_1, 32));

      __ vpminsb(xneg1, xval1, zero);
      __ vpsubb(xneg1, zero, xneg1);
      __ vpmaxsb(xpos1, xval1, zero);

      __ addq(row, Immediate(64));
    } else {
      __ addq(row, Immediate(32));
    }

    // Multiply and add positive and negative parts.
    __ vpmaddubsw(xneg0, xneg0, wval0);
    __ vpaddsw(sumn0, sumn0, xneg0);
    __ vpmaddubsw(xpos0, xpos0, wval0);
    __ vpaddsw(sump0, sump0, xpos0);
    if (unroll) {
      __ vpmaddubsw(xpos1, xpos1, wval1);
      __ vpaddsw(sump1, sump1, xpos1);
      __ vpmaddubsw(xneg1, xneg1, wval1);
      __ vpaddsw(sumn1, sumn1, xneg1);
    }

    // Move to next row.
    __ cmpq(row, Immediate(rows));
    __ j(less, &l2);

    // Add elements horizontally.
    YMMRegister sum = sump0;
    YMMRegister hi = sumn0;
    __ vpsubw(sump0, sump0, sumn0);
    if (unroll) {
      __ vpsubw(sump1, sump1, sumn1);
      __ vpaddw(sum, sump0, sump1);
    }
    __ vperm2i128(hi, sum, sum, 1);
    __ vphaddsw(sum, sum, hi);
    __ vphaddsw(sum, sum, sum);
    __ vphaddsw(sum, sum, sum);

    // Compute relu.
    if (relu_) {
      __ vpmaxsw(sum, sum, zero);
    }

    // Save to y[col].
    __ movq(acc, sum.xmm());
    __ movw(Operand(output, col, times_2), acc);

    // Move to next column.
    __ addq(col, Immediate(1));
    __ addq(matrix, Immediate(row_size));
    __ cmpq(col, Immediate(cols));
    __ j(less, &l1);
  }
};

class AVXIntVecMatMulH : public AVXIntVecMatMulHBase {
 public:
  AVXIntVecMatMulH() : AVXIntVecMatMulHBase(false, false) {}

  string Name() override { return "AVXIntVecMatMulH"; }
  string Operation() override { return "MatMul"; }
};

class AVXIntVecMatMulAddH : public AVXIntVecMatMulHBase {
 public:
  AVXIntVecMatMulAddH() : AVXIntVecMatMulHBase(true, false) {}

  string Name() override { return "AVXIntVecMatMulAddH"; }
  string Operation() override { return "MatMulAdd"; }
};

class AVXIntVecMatMulReluH : public AVXIntVecMatMulHBase {
 public:
  AVXIntVecMatMulReluH() : AVXIntVecMatMulHBase(false, true) {}

  string Name() override { return "AVXIntVecMatMulReluH"; }
  string Operation() override { return "MatMulRelu"; }
};

class AVXIntVecMatMulAddReluH : public AVXIntVecMatMulHBase {
 public:
  AVXIntVecMatMulAddReluH() : AVXIntVecMatMulHBase(true, true) {}

  string Name() override { return "AVXIntVecMatMulAddReluH"; }
  string Operation() override { return "MatMulAddRelu"; }
};

void RegisterAVXMatMul(Library *library) {
  // Computes  : C = A * B
  // Input     : A: float32[k,n] row-major
  //             B: float32[n,m] column-major
  // Output    : C: float32[k,m] row-major
  // Requires  : AVX
  // Supports  : FMA3
  library->Register(new AVXFltMatMatMul());

  // Computes  : y = x * W
  // Input     : x: float32[1,n]
  //             W: float32[n,m] column-major
  // Output    : y: float32[1,m]
  // Requires  : AVX
  // Supports  : FMA3
  library->Register(new AVXFltVecMatMulH());

  // Computes  : y = x * W + b
  // Input     : x: float32[1,n]
  //             W: float32[n,m] column-major
  //             b: float32[1,n]
  // Output    : y: float32[1,m]
  // Requires  : AVX
  // Supports  : FMA3
  library->Register(new AVXFltVecMatMulAddH());

  // Computes  : y = max(0, x * W)
  // Input     : x: float32[1,n]
  //             W: float32[n,m] column-major
  // Output    : y: float32[1,m]
  // Requires  : AVX
  // Supports  : FMA3
  library->Register(new AVXFltVecMatMulReluH());

  // Computes  : y = max(0, x * W + b)
  // Input     : x: float32[1,n]
  //             W: float32[n,m] column-major
  //             b: float32[1,n]
  // Output    : y: float32[1,m]
  // Requires  : AVX
  // Supports  : FMA3
  library->Register(new AVXFltVecMatMulAddReluH());

  // Computes  : y = x * W
  // Input     : x: float32[1,n]
  //             W: float32[n,m] row-major
  // Output    : y: float32[1,m]
  // Requires  : AVX
  library->Register(new AVXFltVecMatMulV());

  // Computes  : y = x * W + b
  // Input     : x: float32[1,n]
  //             W: float32[n,m] row-major
  //             b: float32[1,n]
  // Output    : y: float32[1,m]
  // Requires  : AVX
  library->Register(new AVXFltVecMatMulAddV());

  // Computes  : y = max(0, x * W)
  // Input     : x: float32[1,n]
  //             W: float32[n,m] row-major
  // Output    : y: float32[1,m]
  // Requires  : AVX
  library->Register(new AVXFltVecMatMulReluV());

  // Computes  : y = max(0, x * W + b)
  // Input     : x: float32[1,n]
  //             W: float32[n,m] row-major
  //             b: float32[1,n]
  // Output    : y: float32[1,m]
  // Requires  : AVX
  library->Register(new AVXFltVecMatMulAddReluV());

  // Computes  : y = x * W
  // Input     : x: int8[1,n]
  //             W: int8[n,m] column-major
  // Output    : y: int16[1,m]
  // Requires  : AVX2
  library->Register(new AVXIntVecMatMulH());

  // Computes  : y = x * W + b
  // Input     : x: int8[1,n]
  //             W: int8[n,m] column-major
  //             b: int8[1,n]
  // Output    : y: int16[1,m]
  // Requires  : AVX2
  library->Register(new AVXIntVecMatMulAddH());

  // Computes  : y = max(0, x * W)
  // Input     : x: int8[1,n]
  //             W: int8[n,m] column-major
  // Output    : y: int16[1,m]
  // Requires  : AVX2
  library->Register(new AVXIntVecMatMulReluH());

  // Computes  : y = max(0, x * W + b)
  // Input     : x: int8[1,n]
  //             W: int8[n,m] column-major
  //             b: int8[1,n]
  // Output    : y: int16[1,m]
  // Requires  : AVX2
  library->Register(new AVXIntVecMatMulAddReluH());
}

}  // namespace myelin
}  // namespace sling

