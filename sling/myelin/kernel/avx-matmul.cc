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
    if (step->GetAttr("transpose_c", false)) return false;

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

// Float vector-matrix multiplication for CPUs with AVX.
class AVXFltVecMatMulBase : public AVXVecMatMulBase {
 public:
  // Maximum number of loop unrolls.
  static const int kMaxUnrolls = 8;

  // Maximum number of adder registers.
  static const int kMaxAdders = 4;

  AVXFltVecMatMulBase(bool bias, bool relu)
      : AVXVecMatMulBase(bias, relu, ANY_ORDER, DT_FLOAT, DT_FLOAT) {}

  void Adjust(Step *step) override {
    // Get input and output tensors.
    Tensor *x = step->input(0);
    Tensor *W = step->input(1);
    Tensor *b = bias_ ? step->input(2) : nullptr;
    Tensor *y = step->output(0);

    // Use vertical summations for small matrices and horizontal summation for
    // large matrices. Use the on-chip cache size for selection.
    static const int kCacheHitRatio = 2;
    int cache_size = CPU::L3CacheSize();
    if (cache_size == 0) cache_size = 1024 * 1024;
    int elements = x->elements() + W->elements() + y->elements();
    int footprint = elements * sizeof(float);
    if (footprint * kCacheHitRatio > cache_size) {
      W->RequireOrder(COLUMN_MAJOR_PREFERRED);
    } else {
      W->RequireOrder(ROW_MAJOR_PREFERRED);
    }

    // Align to one SIMD register.
    bool avx512 = CPU::Enabled(AVX512F);
    int byte_alignment = avx512 ? 64 : 32;
    x->SetMiniumAlignment(byte_alignment);
    W->SetMiniumAlignment(byte_alignment);
    y->SetMiniumAlignment(byte_alignment);
    if (bias_) b->SetMiniumAlignment(byte_alignment);

    // Rows must be aligned to SIMD boundaries to support aligned loads.
    W->MinAlign({avx512 ? 16 : 8, 1});
  }

  void Generate(Step *step, MacroAssembler *masm) override {
    Tensor *W = step->input(1);
    if (W->order() == ROW_MAJOR) {
      GenerateVertical(step, masm);
    } else {
      GenerateHorizontal(step, masm);
    }
  }

  // Compute matrix multiplication with vertical summation.
  void GenerateVertical(Step *step, MacroAssembler *masm) {
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
    bool avx512 = masm->Enabled(AVX512F);
    bool strict = step->GetAttr("strict", false);
    if (strict) {
      fma = false;
      avx512 = false;
      step->set_variant("strict");
    }
    int vecsize = avx512 ? 16 : 8;
    int vecbytes = vecsize * sizeof(float);

    // Get matrix dimensions.
    int rows = W->dim(0);
    int cols = W->dim(1);
    int main_cols = (cols  / vecsize) * vecsize;
    int remaining_cols = cols - main_cols;

    // Compute the number of unrolls.
    int unrolls = 0;
    for (int i = 1; i <= kMaxUnrolls; ++i) {
      int batch_size = i * vecsize;
      if (main_cols >= batch_size && main_cols % batch_size == 0) unrolls = i;
    }
    if (step->variant().empty()) {
      string variant = "U" + std::to_string(unrolls);
      if (remaining_cols > 0) variant += "R" + std::to_string(remaining_cols);
      variant += "V";
      if (avx512) variant += "Z";
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
    std::vector<ZMMRegister> sum;
    for (int i = 0; i < std::max(unrolls, 4); ++i) {
      sum.push_back(mm.allocz(avx512));
    }
    std::vector<ZMMRegister> acc;
    for (int i = 0; i < 4; ++i) {
      acc.push_back(mm.allocz(avx512));
    }
    ZMMRegister elem = mm.allocz(avx512);
    ZMMRegister zero = relu_ ? mm.allocz(avx512) : no_zmm_reg;

    // Load tensor locations.
    __ LoadTensorAddress(input, x);
    __ LoadTensorAddress(matrix, W);
    if (bias_) {
      __ LoadTensorAddress(vector, b);
    }
    __ LoadTensorAddress(output, y);

    // Initialize SIMD register to zero for relu.
    if (relu_) {
      if (avx512) {
        __ vxorps(zero, zero, zero);
      } else {
        __ vxorps(zero.ymm(), zero.ymm(), zero.ymm());
      }
    }

    // Compute main columns.
    if (unrolls > 0) {
      // Outer loop over matrix column blocks.
      __ xorq(colofs, colofs);
      __ LoopStart(&l1);

      // Initialize block with bias or zero.
      for (int i = 0; i < unrolls; ++i) {
        if (bias_ && !strict) {
          if (avx512) {
            __ vmovaps(sum[i],
                       Operand(vector, colofs, times_1, i * vecbytes));
          } else {
            __ vmovaps(sum[i].ymm(),
                       Operand(vector, colofs, times_1, i * vecbytes));
          }
        } else {
          if (avx512) {
            __ vxorps(sum[i], sum[i], sum[i]);
          } else {
            __ vxorps(sum[i].ymm(), sum[i].ymm(), sum[i].ymm());
          }
        }
      }
      __ movq(m, matrix);
      __ xorq(rowofs, rowofs);

      // Inner loop over rows.
      __ LoopStart(&l2);

      // Load x[row].
      if (avx512) {
        __ vbroadcastss(elem, Operand(input, rowofs));
      } else {
        __ vbroadcastss(elem.ymm(), Operand(input, rowofs));
      }

      // Multiply x[row] with W[row,col:col+n] and add to sum.
      for (int i = 0; i < unrolls; ++i) {
        int a = i % 4;
        if (avx512) {
          __ vfmadd231ps(sum[i], elem, Operand(m, i * vecbytes));
        } else if (fma) {
          __ vfmadd231ps(sum[i].ymm(), elem.ymm(), Operand(m, i * vecbytes));
        } else {
          __ vmulps(acc[a].ymm(), elem.ymm(), Operand(m, i * vecbytes));
          __ vaddps(sum[i].ymm(), sum[i].ymm(), acc[a].ymm());
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
          if (avx512) {
            __ vaddps(sum[i], sum[i],
                      Operand(vector, colofs, times_1, i * vecbytes));
          } else {
            __ vaddps(sum[i].ymm(), sum[i].ymm(),
                      Operand(vector, colofs, times_1, i * vecbytes));
          }
        }

        // Compute relu.
        if (relu_) {
          if (avx512) {
            __ vmaxps(sum[i], sum[i], zero);
          } else {
            __ vmaxps(sum[i].ymm(), sum[i].ymm(), zero.ymm());
          }
        }
        if (avx512) {
          __ vmovaps(Operand(output, colofs, times_1, i * vecbytes),
                     sum[i]);
        } else {
          __ vmovaps(Operand(output, colofs, times_1, i * vecbytes),
                     sum[i].ymm());
        }
      }

      // Next matrix column block.
      if (main_cols > unrolls * vecsize || remaining_cols > 0) {
        __ addq(matrix, Immediate(unrolls * vecbytes));
      }
      if (main_cols > unrolls * vecsize) {
        __ addq(colofs, Immediate(unrolls * vecbytes));
        __ cmpq(colofs, Immediate(main_cols * sizeof(float)));
        __ j(less, &l1);
      }
    }

    // Compute remaining columns.
    if (remaining_cols > 0) {
      if (avx512) {
        CHECK_LT(remaining_cols, 16);
        int coldisp = main_cols * sizeof(float);

        // Initialize mask for residual.
        OpmaskRegister mask = masm->kk().alloc();
        __ LoadMask(remaining_cols, mask);

        // Initialize remaining columns with bias or zero.
        if (bias_) {
          __ vmovaps(sum[0], Operand(vector, coldisp), Mask(mask, zeroing));
        } else {
          __ vxorps(sum[0], sum[0], sum[0]);
        }

        // Loop over rows.
        __ movq(m, matrix);
        __ xorq(rowofs, rowofs);
        __ LoopStart(&l3);

        // Load x[row].
        __ vbroadcastss(elem, Operand(input, rowofs));

        // Compute remaining columns using AVX512 masking.
        __ vfmadd231ps(sum[0], elem, Operand(m), Mask(mask, zeroing));

        // Next row.
        if (rows > 1) {
          __ addq(m, Immediate(W->stride(0)));
          __ addq(rowofs, Immediate(sizeof(float)));
          __ cmpq(rowofs, Immediate(rows * sizeof(float)));
          __ j(less, &l3);
        }

        // Compute relu and save remaining columns.
        if (relu_) {
          __ vmaxps(sum[0], sum[0], zero);
        }
        __ vmovaps(Operand(output, coldisp), sum[0], Mask(mask, merging));
      } else {
        // Initialize remaining columns with bias or zero.
        CHECK_LT(remaining_cols, 8);
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
          __ vbroadcastss(elem.xmm(), Operand(input, rowofs));
        } else {
          __ vmovss(elem.xmm(), Operand(input, rowofs));
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
  }

  // Compute matrix multiplication with horizontal summation.
  void GenerateHorizontal(Step *step, MacroAssembler *masm) {
    Registers &rr = masm->rr();
    SIMDRegisters &mm = masm->mm();
    Label l1, l2;

    // Get input and output tensors.
    Tensor *x = step->input(0);
    Tensor *W = step->input(1);
    Tensor *b = bias_ ? step->input(2) : nullptr;
    Tensor *y = step->output(0);

    // Get matrix dimensions.
    bool avx512 = masm->Enabled(AVX512F);
    int vecsize = avx512 ? 16 : 8;
    int rows = W->dim(0);
    int cols = W->dim(1);
    int main_rows = (rows  / vecsize) * vecsize;
    int remaining_rows = rows - main_rows;
    int row_size = W->stride(1);

    // Compute the number of unrolls and adders.
    int unrolls = 0;
    for (int i = 1; i <= kMaxUnrolls; ++i) {
      int batch_size = i * vecsize;
      if (main_rows >= batch_size && main_rows % batch_size == 0) unrolls = i;
    }
    int adders = unrolls;
    if (adders < 1) adders = 1;
    if (adders > kMaxAdders) adders = kMaxAdders;
    string variant = "U" + std::to_string(unrolls);
    variant += "A" + std::to_string(adders);
    if (remaining_rows > 0) variant += "R" + std::to_string(remaining_rows);
    variant += "H";
    if (avx512) variant += "Z";
    step->set_variant(variant);

    // Allocate general registers.
    Register row = rr.alloc();
    Register col = rr.alloc();
    Register matrix = rr.alloc();
    Register input = rr.alloc();
    Register output = rr.alloc();
    Register vector = bias_ ? rr.alloc() : no_reg;

    // Allocate SIMD registers.
    std::vector<ZMMRegister> elem;
    for (int i = 0; i < std::max(unrolls, 1); ++i) {
      elem.push_back(mm.allocz(avx512));
    }
    std::vector<ZMMRegister> sum;
    for (int i = 0; i < adders; ++i) {
      sum.push_back(mm.allocz(avx512));
    }
    ZMMRegister acc = mm.allocz(avx512);
    ZMMRegister zero = relu_ ? mm.allocz(avx512) : no_zmm_reg;

    // Load tensor locations.
    __ LoadTensorAddress(input, x);
    __ LoadTensorAddress(matrix, W);
    if (bias_) {
      __ LoadTensorAddress(vector, b);
    }
    __ LoadTensorAddress(output, y);
    __ xorq(col, col);
    if (relu_) {
      if (avx512) {
        __ vxorps(zero, zero, zero);
      } else {
        __ vxorps(zero.ymm(), zero.ymm(), zero.ymm());
      }
    }

    // Initialize mask.
    OpmaskRegister mask = masm->kk().alloc();
    if (avx512 && remaining_rows > 0) {
      __ LoadMask(remaining_rows, mask);
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
        if (avx512) {
          __ vmovss(sum[0], Operand(vector, col, times_4));
        } else {
          __ vmovss(sum[0].ymm(), Operand(vector, col, times_4));
        }
      }
      for (int i = (bias_ ? 1 : 0); i < adders; ++i) {
        if (avx512) {
          __ vxorps(sum[i], sum[i], sum[i]);
        } else {
          __ vxorps(sum[i].ymm(), sum[i].ymm(), sum[i].ymm());
        }
      }

      __ LoopStart(&l2);
      for (int i = 0; i < unrolls; ++i) {
        // Load x[row:row+n].
        int disp = vecsize * i * sizeof(float);
        if (avx512) {
          __ vmovaps(elem[i], Operand(input, row, times_4, disp));
        } else {
          __ vmovaps(elem[i].ymm(), Operand(input, row, times_4, disp));
        }
      }
      for (int i = 0; i < unrolls; ++i) {
        int disp = vecsize * i * sizeof(float);
        int a = i % adders;
        if (avx512) {
          // Multiply x[row:row+16] with W[row:row+16,col] and add to sum.
          __ vfmadd231ps(sum[a], elem[i], Operand(matrix, row, times_4, disp));
        } else if (masm->Enabled(FMA3)) {
          // Multiply x[row:row+8] with W[row:row+8,col] and add to sum.
          __ vfmadd231ps(sum[a].ymm(), elem[i].ymm(),
                         Operand(matrix, row, times_4, disp));
        } else {
          // Multiply x[row:row+8] with W[row:row+8,col].
          __ vmulps(elem[i].ymm(), elem[i].ymm(),
                    Operand(matrix, row, times_4, disp));

          // Sum dot product in parallel.
          __ vaddps(sum[a].ymm(), sum[a].ymm(), elem[i].ymm());
        }
      }

      // Move to next row batch.
      if (main_rows > vecsize * unrolls) {
        __ addq(row, Immediate(vecsize * unrolls));
        __ cmpq(row, Immediate(main_rows));
        __ j(less, &l2);
      }

      // Add remaining rows.
      if (remaining_rows > 0) {
        int disp = main_rows * sizeof(float);
        if (avx512) {
          __ vmovaps(elem[0], Operand(input, disp), Mask(mask, zeroing));
          __ vfmadd231ps(sum[0], elem[0], Operand(matrix, disp),
                         Mask(mask, merging));
        } else {
          XMMRegister s = acc.xmm();
          XMMRegister e = elem[0].xmm();
          int left = remaining_rows;

          // Add first four remaining elements using SSE.
          if (left >= 4) {
            __ vmovaps(s, Operand(input, disp));
            __ vmulps(s, s, Operand(matrix, disp));
            __ vaddps(sum[0].ymm(), sum[0].ymm(), acc.ymm());
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
            __ vperm2f128(acc.ymm(), acc.ymm(), acc.ymm(), 0x80);
            __ vaddps(sum[0].ymm(), sum[0].ymm(), acc.ymm());
          }
        }
      }

      // Sum adders in sum[0].
      if (avx512) {
        if (adders == 4) {
          __ vaddps(sum[0], sum[0], sum[2]);
          __ vaddps(sum[1], sum[1], sum[3]);
          __ vaddps(sum[0], sum[0], sum[1]);
        } else {
          for (int n = 1; n < adders; ++n) {
            __ vaddps(sum[0], sum[0], sum[n]);
          }
        }
      } else {
        if (adders == 4) {
          __ vaddps(sum[0].ymm(), sum[0].ymm(), sum[2].ymm());
          __ vaddps(sum[1].ymm(), sum[1].ymm(), sum[3].ymm());
          __ vaddps(sum[0].ymm(), sum[0].ymm(), sum[1].ymm());
        } else {
          for (int n = 1; n < adders; ++n) {
            __ vaddps(sum[0].ymm(), sum[0].ymm(), sum[n].ymm());
          }
        }
      }

      // Add elements in sum[0] horizontally.
      if (avx512) {
        __ Reduce(REDUCE_ADD, DT_FLOAT, sum[0], acc);
      } else {
        __ Reduce(REDUCE_ADD, DT_FLOAT, sum[0].ymm(), acc.ymm());
      }
    }

    // Compute relu.
    if (relu_) {
      if (avx512) {
        __ vmaxss(sum[0], sum[0], zero);
      } else {
        __ vmaxss(sum[0].ymm(), sum[0].ymm(), zero.ymm());
      }
    }

    // Save to y[col].
    __ vmovss(Operand(output, col, times_4), sum[0].ymm());

    // Move to next column.
    if (cols > 1) {
      __ addq(col, Immediate(1));
      __ addq(matrix, Immediate(row_size));
      __ cmpq(col, Immediate(cols));
      __ j(less, &l1);
    }
  }
};

class AVXFltVecMatMul : public AVXFltVecMatMulBase {
 public:
  AVXFltVecMatMul() : AVXFltVecMatMulBase(false, false) {}

  string Name() override { return "AVXFltVecMatMul"; }
  string Operation() override { return "MatMul"; }
};

class AVXFltVecMatMulAdd : public AVXFltVecMatMulBase {
 public:
  AVXFltVecMatMulAdd() : AVXFltVecMatMulBase(true, false) {}

  string Name() override { return "AVXFltVecMatMulAdd"; }
  string Operation() override { return "MatMulAdd"; }
};

class AVXFltVecMatMulRelu : public AVXFltVecMatMulBase {
 public:
  AVXFltVecMatMulRelu() : AVXFltVecMatMulBase(false, true) {}

  string Name() override { return "AVXFltVecMatMulRelu"; }
  string Operation() override { return "MatMulRelu"; }
};

class AVXFltVecMatMulAddRelu : public AVXFltVecMatMulBase {
 public:
  AVXFltVecMatMulAddRelu() : AVXFltVecMatMulBase(true, true) {}

  string Name() override { return "AVXFltVecMatMulAddRelu"; }
  string Operation() override { return "MatMulAddRelu"; }
};

// Float dot product for CPUs with AVX.
class AVXFltDotProduct : public Kernel {
 public:
  // Maximum number of loop unrolls.
  static const int kMaxUnrolls = 4;

  // Maximum number of adder registers.
  static const int kMaxAdders = 4;

  string Name() override { return "AVXFltDotProduct"; }
  string Operation() override { return "MatMul"; }

  bool Supports(Step *step) override {
    // Requires CPU with AVX support.
    if (!CPU::Enabled(AVX)) return false;

    // Two tensor inputs and one tensor output.
    if (step->indegree() != 2 || step->outdegree() != 1) return false;
    Tensor *a = step->input(0);
    Tensor *b = step->input(1);
    Tensor *c = step->output(0);
    if (a->type() != DT_FLOAT) return false;
    if (b->type() != DT_FLOAT) return false;
    if (c->type() != DT_FLOAT) return false;
    if (a->elements() != b->elements()) return false;
    if (c->elements() != 1) return false;

    // Size of be multiple of YMM register size.
    if (a->elements() % 8  != 0) return false;

    // Horizontal summation is not strict math compatible.
    if (step->GetAttr("strict", false)) return false;

    return true;
  }

  int64 Complexity(const Step *step) override {
    return step->output(0)->elements() * 2;
  }

  void Adjust(Step *step) override {
    // Get input and output tensors.
    Tensor *a = step->input(0);
    Tensor *b = step->input(1);

    // Align to one SIMD register (256 bits, 32 bytes).
    bool avx512 = CPU::Enabled(AVX512F) && a->elements() % 16 == 0;
    a->SetMiniumAlignment(avx512 ? 64 : 32);
    b->SetMiniumAlignment(avx512 ? 64 : 32);
  }

  void Generate(Step *step, MacroAssembler *masm) override {
    Registers &rr = masm->rr();
    SIMDRegisters &mm = masm->mm();

    // Get input and output tensors.
    Tensor *a = step->input(0);
    Tensor *b = step->input(1);
    Tensor *c = step->output(0);

    // Get number of elements.
    int n = a->elements();

    // Compute the number of unrolls and adders.
    bool avx512 = masm->Enabled(AVX512F) && n % 16 == 0;
    int vecsize = avx512 ? 16 : 8;
    int unrolls = 0;
    for (int i = 1; i <= kMaxUnrolls; ++i) {
      int batch_size = i * vecsize;
      if (n >= batch_size && n % batch_size == 0) unrolls = i;
    }
    int adders = unrolls;
    if (adders < 1) adders = 1;
    if (adders > kMaxAdders) adders = kMaxAdders;
    string variant = "U" + std::to_string(unrolls);
    variant += "A" + std::to_string(adders);
    if (avx512) variant += "Z";
    step->set_variant(variant);

    // Allocate general registers.
    Register idx = rr.alloc();
    Register aptr = rr.alloc();
    Register bptr = rr.alloc();
    Register cptr = rr.alloc();

    // Allocate SIMD registers.
    std::vector<ZMMRegister> elem;
    for (int i = 0; i < std::max(unrolls, 1); ++i) {
      elem.push_back(mm.allocz(avx512));
    }
    std::vector<ZMMRegister> sum;
    for (int i = 0; i < adders; ++i) {
      sum.push_back(mm.allocz(avx512));
    }
    ZMMRegister acc = mm.allocz(avx512);

    // Load tensor locations.
    __ LoadTensorAddress(aptr, a);
    __ LoadTensorAddress(bptr, b);
    __ xorq(idx, idx);
    for (int i = 0; i < adders; ++i) {
      if (avx512) {
        __ vxorps(sum[i], sum[i], sum[i]);
      } else {
        __ vxorps(sum[i].ymm(), sum[i].ymm(), sum[i].ymm());
      }
    }

    // Outer loop over elements.
    Label l;
    __ LoopStart(&l);

    // Multiply and sum next batch.
    for (int i = 0; i < unrolls; ++i) {
      // Load a[idx:idx+n].
      int disp = vecsize * i * sizeof(float);
      if (avx512) {
        __ vmovaps(elem[i], Operand(aptr, idx, times_4, disp));
      } else {
        __ vmovaps(elem[i].ymm(), Operand(aptr, idx, times_4, disp));
      }
    }
    for (int i = 0; i < unrolls; ++i) {
      // Multiply a[idx:idx+n] with b[idx:idx+n] and add to sum.
      int disp = vecsize * i * sizeof(float);
      int a = i % adders;
      if (avx512) {
        __ vfmadd231ps(sum[a], elem[i], Operand(bptr, idx, times_4, disp));
      } else if (masm->Enabled(FMA3)) {
        __ vfmadd231ps(sum[a].ymm(), elem[i].ymm(),
                       Operand(bptr, idx, times_4, disp));
      } else {
        __ vmulps(elem[i].ymm(), elem[i].ymm(),
                  Operand(bptr, idx, times_4, disp));
        __ vaddps(sum[a].ymm(), sum[a].ymm(), elem[i].ymm());
      }
    }

    // Move to next batch.
    if (n > vecsize * unrolls) {
      __ addq(idx, Immediate(vecsize * unrolls));
      __ cmpq(idx, Immediate(n));
      __ j(less, &l);
    }

    // Sum adders in sum[0].
    if (avx512) {
      if (adders == 4) {
        __ vaddps(sum[0], sum[0], sum[2]);
        __ vaddps(sum[1], sum[1], sum[3]);
        __ vaddps(sum[0], sum[0], sum[1]);
      } else {
        for (int n = 1; n < adders; ++n) {
          __ vaddps(sum[0], sum[0], sum[n]);
        }
      }
    } else {
      if (adders == 4) {
        __ vaddps(sum[0].ymm(), sum[0].ymm(), sum[2].ymm());
        __ vaddps(sum[1].ymm(), sum[1].ymm(), sum[3].ymm());
        __ vaddps(sum[0].ymm(), sum[0].ymm(), sum[1].ymm());
      } else {
        for (int n = 1; n < adders; ++n) {
          __ vaddps(sum[0].ymm(), sum[0].ymm(), sum[n].ymm());
        }
      }
    }

    // Add elements in sum[0] horizontally.
    if (avx512) {
      __ Reduce(REDUCE_ADD, DT_FLOAT, sum[0], acc);
    } else {
      __ Reduce(REDUCE_ADD, DT_FLOAT, sum[0].ymm(), acc.ymm());
    }

    // Save result to c.
    __ LoadTensorAddress(cptr, c);
    if (avx512) {
      __ vmovss(Operand(cptr), sum[0]);
    } else {
      __ vmovss(Operand(cptr), sum[0].ymm());
    }
  }
};

// Float accumulating outer product for CPUs with AVX (C += A * B).
class AVXFltAssignAddOuter : public Kernel {
 public:
  // Block size.
  static const int kRowRegs = 4;
  static const int kColRegs = 4;

  string Name() override { return "AVXFltAssignAddOuter"; }
  string Operation() override { return "AssignAddMatMul"; }

  bool Supports(Step *step) override {
    // Requires CPU with AVX support.
    if (!CPU::Enabled(AVX)) return false;

    // Three matrix inputs.
    if (step->indegree() != 3 || step->outdegree() != 0) return false;
    Tensor *c = step->input(0);
    Tensor *a = step->input(1);
    Tensor *b = step->input(2);
    if (a->type() != DT_FLOAT || a->rank() != 2) return false;
    if (b->type() != DT_FLOAT || b->rank() != 2) return false;
    if (c->type() != DT_FLOAT || c->rank() != 2) return false;
    if (a->dim(0) != 1 || a->dim(1) != c->dim(0)) return false;
    if (b->dim(0) != 1 || b->dim(1) != c->dim(1)) return false;

    if (step->GetAttr("transpose_a", false)) return false;
    if (step->GetAttr("transpose_b", false)) return false;
    if (step->GetAttr("transpose_c", false)) return false;

    return true;
  }

  void Adjust(Step *step) override {
    // Get tensors.
    Tensor *c = step->input(0);
    Tensor *a = step->input(1);
    Tensor *b = step->input(2);

    // Align to SIMD register.
    bool avx512 = CPU::Enabled(AVX512F);
    int byte_alignment = avx512 ? 64 : 32;
    a->SetMiniumAlignment(byte_alignment);
    b->SetMiniumAlignment(byte_alignment);
    c->SetMiniumAlignment(byte_alignment);

    // Output must be row-major.
    c->RequireOrder(ROW_MAJOR);
  }

  void Generate(Step *step, MacroAssembler *masm) override {
    // Get tensors.
    Tensor *c = step->input(0);
    Tensor *a = step->input(1);
    Tensor *b = step->input(2);

    // FMA is not strict math compatible.
    bool fma = masm->Enabled(FMA3) && !step->GetAttr("strict", false);
    bool avx512 = masm->Enabled(AVX512F);

    // Get matrix dimensions.
    int vecsize = avx512 ? 16 : 8;
    int rows = c->dim(0);
    int cols = c->dim(1);
    int rowsize = c->stride(0);
    int colblk = vecsize * kColRegs;
    int main_cols = (cols / colblk) * colblk;
    int remaining_cols = cols - main_cols;
    int main_rows = (rows / kRowRegs) * kRowRegs;

    // Allocate general registers.
    Register cptr = masm->rr().alloc();
    Register aptr = masm->rr().alloc();
    Register bptr = masm->rr().alloc();
    Register col = masm->rr().alloc();
    Register row = masm->rr().alloc();

    // Allocate SIMD registers.
    std::vector<ZMMRegister> areg;
    std::vector<ZMMRegister> breg;
    std::vector<ZMMRegister> creg;
    std::vector<ZMMRegister> acc;
    for (int i = 0; i < kRowRegs; ++i) {
      areg.push_back(masm->mm().allocz(avx512));
    }
    for (int i = 0; i < kColRegs; ++i) {
      breg.push_back(masm->mm().allocz(avx512));
      creg.push_back(masm->mm().allocz(avx512));
      acc.push_back(masm->mm().allocz(avx512));
    }

    // Load tensor locations.
    __ LoadTensorAddress(cptr, c);
    __ LoadTensorAddress(aptr, a);
    __ LoadTensorAddress(bptr, b);

    // Initialize mask.
    OpmaskRegister mask = masm->kk().alloc();
    if (avx512 && remaining_cols % 16 != 0) {
      __ LoadMask(remaining_cols % 16, mask);
    }

    // First compute rows in blocks (stage 0) and then the remaining ones one
    // row at a time (stage 1).
    __ xorq(row, row);
    for (int stage = 0; stage < 2; ++stage) {
      // Determine the row block size.
      int rowblk;
      bool single;
      if (stage == 0) {
        if (rows < kRowRegs) continue;
        rowblk = kRowRegs;
        single = (rows == kRowRegs);
      } else {
        if (rows % kRowRegs == 0) continue;
        rowblk = 1;
        single = (rows % kRowRegs == 1);
      }

      // Outer loop over row blocks.
      Label l1;
      __ bind(&l1);

      // Load a[row] block.
      for (int r = 0; r < rowblk; ++r) {
        int disp = r * sizeof(float);
        if (avx512) {
          __ vbroadcastss(areg[r], Operand(aptr, row, times_4, disp));
        } else {
          __ vbroadcastss(areg[r].ymm(), Operand(aptr, row, times_4, disp));
        }
      }

      // Compute columns in blocks.
      if (main_cols > 0) {
        // Inner loop over column blocks.
        __ xorq(col, col);
        Label l2;
        __ bind(&l2);

        // Load b[col] block.
        for (int c = 0; c < kColRegs; ++c) {
          int disp = c * vecsize * sizeof(float);
          if (avx512) {
            __ vmovups(breg[c], Operand(bptr, col, times_4, disp));
          } else {
            __ vmovups(breg[c].ymm(), Operand(bptr, col, times_4, disp));
          }
        }

        // Multiply a[row] block with b[col] block and add to c[row,col] block.
        for (int r = 0; r < rowblk; ++r) {
          for (int c = 0; c < kColRegs; ++c) {
            int disp = r * rowsize + c * vecsize * sizeof(float);
            if (avx512) {
              __ vmovups(creg[c], Operand(cptr, col, times_4, disp));
              __ vfmadd231ps(creg[c], areg[r], breg[c]);
              __ vmovups(Operand(cptr, col, times_4, disp), creg[c]);
            } else {
              __ vmovups(creg[c].ymm(), Operand(cptr, col, times_4, disp));
              if (fma) {
                __ vfmadd231ps(creg[c].ymm(), areg[r].ymm(), breg[c].ymm());
              } else {
                __ vmulps(acc[c].ymm(), areg[r].ymm(), breg[c].ymm());
                __ vaddps(creg[c].ymm(), creg[c].ymm(), acc[c].ymm());
              }
              __ vmovups(Operand(cptr, col, times_4, disp), creg[c].ymm());
            }
          }
        }

        if (main_cols > vecsize * rowblk) {
          __ addq(col, Immediate(vecsize * rowblk));
          __ cmpq(col, Immediate(main_cols));
          __ j(less, &l2);
        }
      }

      // Compute remaining columns.
      int coldisp = main_cols * sizeof(float);
      int left = remaining_cols;
      if (avx512) {
        // First 16 floats at a time using AVX512 without masking.
        while (left >= 16) {
          // Load b[col].
          __ vmovups(breg[0], Operand(bptr, coldisp));

          // Multiply a[row] block with b[col] and add to c[row,col] block.
          for (int r = 0; r < rowblk; ++r) {
            int disp = r * rowsize + coldisp;
            __ vmovups(creg[0], Operand(cptr, disp));
            __ vfmadd231ps(creg[0], areg[r], breg[0]);
            __ vmovups(Operand(cptr, disp), creg[0]);
          }

          left -= 16;
          coldisp += 16 * sizeof(float);
        }

        // Compute remaining columns using AVX512 with masking.
        if (left > 0) {
          // Load b[col].
          __ vmovups(breg[0], Operand(bptr, coldisp), Mask(mask, zeroing));

          // Multiply a[row] block with b[col] and add to c[row,col] block.
          for (int r = 0; r < rowblk; ++r) {
            int disp = r * rowsize + coldisp;
            __ vmovups(creg[0], Operand(cptr, disp), Mask(mask, zeroing));
            __ vfmadd231ps(creg[0], areg[r], breg[0]);
            __ vmovups(Operand(cptr, disp), creg[0], Mask(mask, merging));
          }
        }
      } else {
        // First 8 floats at a time using AVX.
        while (left >= 8) {
          // Load b[col].
          __ vmovups(breg[0].ymm(), Operand(bptr, coldisp));

          // Multiply a[row] block with b[col] and add to c[row,col] block.
          for (int r = 0; r < rowblk; ++r) {
            int disp = r * rowsize + coldisp;
            __ vmovups(creg[0].ymm(), Operand(cptr, disp));
            if (fma) {
              __ vfmadd231ps(creg[0].ymm(), areg[r].ymm(), breg[0].ymm());
            } else {
              __ vmulps(acc[0].ymm(), areg[r].ymm(), breg[0].ymm());
              __ vaddps(creg[0].ymm(), creg[0].ymm(), acc[0].ymm());
            }
            __ vmovups(Operand(cptr, disp), creg[0].ymm());
          }

          left -= 8;
          coldisp += 8 * sizeof(float);
        }

        // Compute next four columns using SSE.
        if (left >= 4) {
          // Load b[col].
          __ vmovups(breg[0].xmm(), Operand(bptr, coldisp));

          // Multiply a[row] block with b[col] and add to c[row,col] block.
          for (int r = 0; r < rowblk; ++r) {
            int disp = r * rowsize + coldisp;
            __ vmovups(creg[0].xmm(), Operand(cptr, disp));
            if (fma) {
              __ vfmadd231ps(creg[0].xmm(), areg[r].xmm(), breg[0].xmm());
            } else {
              __ vmulps(acc[0].xmm(), areg[r].xmm(), breg[0].xmm());
              __ vaddps(creg[0].xmm(), creg[0].xmm(), acc[0].xmm());
            }
            __ vmovups(Operand(cptr, disp), creg[0].xmm());
          }

          left -= 4;
          coldisp += 4 * sizeof(float);
        }

        // Compute remaining remaining columns (0-3).
        while (left > 0) {
          // Load b[col].
          __ vmovss(breg[0].xmm(), Operand(bptr, coldisp));

          // Multiply a[row] block with b[col] and add to c[row,col] block.
          for (int r = 0; r < rowblk; ++r) {
            int disp = r * rowsize + coldisp;
            __ vmovss(creg[0].xmm(), Operand(cptr, disp));
            if (fma) {
              __ vfmadd231ss(creg[0].xmm(), areg[r].xmm(), breg[0].xmm());
            } else {
              __ vmulss(acc[0].xmm(), areg[r].xmm(), breg[0].xmm());
              __ vaddss(creg[0].xmm(), creg[0].xmm(), acc[0].xmm());
            }
            __ vmovss(Operand(cptr, disp), creg[0].xmm());
          }

          left -= 1;
          coldisp += sizeof(float);
        }
      }

      // Next row block.
      __ addq(cptr, Immediate(rowblk * rowsize));
      if (!single) {
        __ addq(row, Immediate(rowblk));
        __ cmpq(row, Immediate(stage == 0 ? main_rows : rows));
        __ j(less, &l1);
      }
    }
  }

  int64 Complexity(const Step *step) override {
    return step->input(0)->elements() * 2;
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

    W->RequireOrder(COLUMN_MAJOR);
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
  // Computes  : y = x * W
  // Input     : x: float32[1,n]
  //             W: float32[n,m]
  // Output    : y: float32[1,m]
  // Supports  : FMA3, AVX512
  // Requires  : AVX
  library->Register(new AVXFltVecMatMul());

  // Computes  : y = x * W + b
  // Input     : x: float32[1,n]
  //             W: float32[n,m]
  //             b: float32[1,n]
  // Output    : y: float32[1,m]
  // Supports  : FMA3, AVX512
  // Requires  : AVX
  library->Register(new AVXFltVecMatMulAdd());

  // Computes  : y = max(0, x * W)
  // Input     : x: float32[1,n]
  //             W: float32[n,m]
  // Output    : y: float32[1,m]
  // Requires  : AVX
  // Supports  : FMA3, AVX512
  library->Register(new AVXFltVecMatMulRelu());

  // Computes  : y = max(0, x * W + b)
  // Input     : x: float32[1,n]
  //             W: float32[n,m]
  //             b: float32[1,n]
  // Output    : y: float32[1,m]
  // Requires  : AVX
  // Supports  : FMA3, AVX512
  library->Register(new AVXFltVecMatMulAddRelu());

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

  // Computes  : c = a * c
  // Input     : a: float32[1,n]
  //             b: float32[n,1]
  // Output    : c: float32[1]
  // Requires  : AVX
  // Supports  : FMA3, AVX512
  library->Register(new AVXFltDotProduct());

  // Computes  : C += A * B
  // Input     : A: float32[n,1]
  //             B: float32[1,m]
  // Output    : C: float32[n,m]
  // Requires  : AVX
  // Supports  : FMA3, AVX512
  library->Register(new AVXFltAssignAddOuter());
}

}  // namespace myelin
}  // namespace sling

