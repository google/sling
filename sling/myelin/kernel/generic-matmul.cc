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

#include "sling/myelin/kernel/generic.h"

#include <string>

#include "sling/myelin/compute.h"
#include "sling/myelin/macro-assembler.h"

#define __ masm->

namespace sling {
namespace myelin {

using namespace jit;

// Generic float vector matrix multiplication, y = Relu(x * W + b).
class GenericFltVecMatMulBase : public Kernel {
 public:
  GenericFltVecMatMulBase(bool bias, bool relu) : bias_(bias), relu_(relu) {}

  bool Supports(Step *step) override {
    // Requires CPU with SSE support.
    if (!CPU::Enabled(SSE)) return false;

    // Two or three float 2D tensor inputs and one 2D tensor output.
    if (step->indegree() != (bias_ ? 3 : 2)) return false;
    if (step->outdegree() != 1) return false;
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

    return true;
  }

  void Adjust(Step *step) override {
    step->input(1)->RequireOrder(COLUMN_MAJOR);
  }

  void Generate(Step *step, MacroAssembler *masm) override {
    Registers &rr = masm->rr();
    SIMDRegisters &mm = masm->mm();
    Label l1, l2;

    Tensor *x = step->input(0);
    Tensor *W = step->input(1);
    Tensor *b = bias_ ? step->input(2) : nullptr;
    Tensor *y = step->output(0);

    int rows = W->dim(0);
    int cols = W->dim(1);
    int row_size = W->stride(1);

    Register row = rr.alloc();
    Register col = rr.alloc();
    Register matrix = rr.alloc();
    Register input = rr.alloc();
    Register output = rr.alloc();
    Register vector = bias_ ? rr.alloc() : no_reg;

    XMMRegister elem = mm.allocx();
    XMMRegister sum = mm.allocx();
    XMMRegister zero = relu_ ? mm.allocx() : no_xmm_reg;

    bool strict = step->GetAttr("strict", false);
    if (strict) step->set_variant("strict");

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

    __ LoopStart(&l1);
    if (bias_ && !strict) {
      __ movss(sum, Operand(vector, col, times_4));
    } else {
      __ xorps(sum, sum);
    }
    __ xorq(row, row);

    __ LoopStart(&l2);
    __ movss(elem, Operand(input, row, times_4));
    __ mulss(elem, Operand(matrix, row, times_4));
    __ addq(row, Immediate(1));
    __ cmpq(row, Immediate(rows));
    __ addss(sum, elem);
    __ j(not_equal, &l2);

    if (bias_ && strict) {
      __ addss(sum, Operand(vector, col, times_4));
    }

    if (relu_) {
      __ maxss(sum, zero);
    }
    __ movss(Operand(output, col, times_4), sum);
    __ addq(col, Immediate(1));
    __ addq(matrix, Immediate(row_size));
    __ cmpq(col, Immediate(cols));
    __ j(not_equal, &l1);
  }

  int64 Complexity(const Step *step) override {
    int64 ops = step->input(1)->elements() * 2;
    if (bias_) ops += step->input(2)->elements();
    if (relu_) ops += step->output(0)->elements();
    return ops;
  }

 private:
  bool bias_;  // add bias vector to result, y=Wx+b
  bool relu_;  // apply rectified linear unit, y=max(0,Wx+b)
};

class GenericFltVecMatMul : public GenericFltVecMatMulBase {
 public:
  GenericFltVecMatMul() : GenericFltVecMatMulBase(false, false) {}

  string Name() override { return "GenFltVecMatMul"; }
  string Operation() override { return "MatMul"; }
};

class GenericFltVecMatMulAdd : public GenericFltVecMatMulBase {
 public:
  GenericFltVecMatMulAdd() : GenericFltVecMatMulBase(true, false) {}

  string Name() override { return "GenFltVecMatMulAdd"; }
  string Operation() override { return "MatMulAdd"; }
};

class GenericFltVecMatMulRelu : public GenericFltVecMatMulBase {
 public:
  GenericFltVecMatMulRelu() : GenericFltVecMatMulBase(false, true) {}

  string Name() override { return "GenFltVecMatMulRelu"; }
  string Operation() override { return "MatMulRelu"; }
};

class GenericFltVecMatMulAddRelu : public GenericFltVecMatMulBase {
 public:
  GenericFltVecMatMulAddRelu() : GenericFltVecMatMulBase(true, true) {}

  string Name() override { return "GenFltVecMatMulAddRelu"; }
  string Operation() override { return "MatMulAddRelu"; }
};

// Generic float matrix multiplication, C = A * B.
class GenericFltMatMatMul : public Kernel {
 public:
  string Name() override { return "GenFltMatMatMul"; }
  string Operation() override { return "MatMul"; }

  bool Supports(Step *step) override {
    // Requires CPU with SSE support.
    if (!CPU::Enabled(SSE)) return false;

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
    if (step->GetAttr("transpose_c", false)) return false;
    bool transpose_a = step->GetAttr("transpose_a", false);
    bool transpose_b = step->GetAttr("transpose_b", false);
    Shape a = A->shape();
    Shape b = B->shape();
    Shape c = C->shape();
    if (transpose_a) a = a.transposed();
    if (transpose_b) b = b.transposed();

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
    bool transpose_a = step->GetAttr("transpose_a", false);
    bool transpose_b = step->GetAttr("transpose_b", false);
    step->input(0)->RequireOrder(transpose_a ? COLUMN_MAJOR : ROW_MAJOR);
    step->input(1)->RequireOrder(transpose_b ? ROW_MAJOR : COLUMN_MAJOR);
    step->output(0)->RequireOrder(ROW_MAJOR);
  }

  void Generate(Step *step, MacroAssembler *masm) override {
    Registers &rr = masm->rr();
    SIMDRegisters &mm = masm->mm();
    Label l1, l2, l3;

    // Get inputs and outputs.
    Tensor *A = step->input(0);
    Tensor *B = step->input(1);
    Tensor *C = step->output(0);

    // Get dimensions for matrices.
    bool transpose_a = step->GetAttr("transpose_a", false);
    bool transpose_b = step->GetAttr("transpose_b", false);
    int a_row_dim = transpose_a ? 1 : 0;
    int a_col_dim = transpose_a ? 0 : 1;
    int b_col_dim = transpose_b ? 0 : 1;
    int c_col_dim = 1;

    // Allocate registers.
    Register a = rr.alloc();
    Register b = rr.alloc();
    Register b_row = rr.alloc();
    Register b_end = rr.alloc();
    Register c = rr.alloc();
    Register c_end = rr.alloc();
    Register k = rr.alloc();

    XMMRegister elem = mm.allocx();
    XMMRegister sum = mm.allocx();

    // Load tensor addresses.
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
    __ xorps(sum, sum);

    // Compute dot product of row in A and column in B.
    // C[i,j] = sum_k A[i,k] * B[k,j].
    __ LoopStart(&l3);
    __ movss(elem, Operand(a, k, times_4));
    __ mulss(elem, Operand(b_row, k, times_4));
    __ addq(k, Immediate(1));
    __ cmpq(k, Immediate(A->dim(a_col_dim)));
    __ addss(sum, elem);
    __ j(not_equal, &l3);

    // Store result in C.
    __ movss(Operand(c), sum);
    __ addq(c, Immediate(C->stride(c_col_dim)));

    // Move to next column in B
    __ addq(b_row, Immediate(B->stride(b_col_dim)));
    __ cmpq(b_row, b_end);
    __ j(not_equal, &l2);

    // Move to next row in A.
    __ addq(a, Immediate(A->stride(a_row_dim)));

    // Move to next row in C.
    if (C->padding(1) != 0) {
      __ addq(c, Immediate(C->padding(c_col_dim)));
    }
    __ cmpq(c, c_end);
    __ j(not_equal, &l1);
  }

  int64 Complexity(const Step *step) override {
    return step->input(0)->dim(0) * step->input(1)->elements() * 2;
  }
};

// Generic integer vector matrix multiplication, y = Relu(x * W + b).
class GenericIntVecMatMulBase : public Kernel {
 public:
  GenericIntVecMatMulBase(bool bias, bool relu) : bias_(bias), relu_(relu) {}

  static bool IsIntType(Type t) {
    return t == DT_INT8 || t == DT_INT16 || t == DT_INT32 || t == DT_INT64;
  }

  bool Supports(Step *step) override {
    // Two or three integer 2D tensor inputs and one 2D tensor output.
    if (step->indegree() != (bias_ ? 3 : 2)) return false;
    if (step->outdegree() != 1) return false;
    Tensor *x = step->input(0);
    Tensor *W = step->input(1);
    Tensor *y = step->output(0);
    if (x->rank() != 2 || !IsIntType(x->type())) return false;
    if (W->rank() != 2 || !IsIntType(W->type())) return false;
    if (y->rank() != 2 || !IsIntType(y->type())) return false;

    // Check shape. First input must be a row vector.
    if (x->dim(0) != 1 || x->dim(1) != W->dim(0)) return false;
    if (y->dim(0) != x->dim(0) || y->dim(1) != W->dim(1)) return false;

    // Transpose not supported.
    if (step->GetAttr("transpose_a", false)) return false;
    if (step->GetAttr("transpose_b", false)) return false;
    if (step->GetAttr("transpose_c", false)) return false;

    // The matrix must support column-major order.
    if (!W->SupportsOrder(COLUMN_MAJOR)) return false;

    // Check bias vector.
    if (bias_) {
      Tensor *b = step->input(2);
      if (!IsIntType(b->type())) return false;
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

  void Adjust(Step *step) override {
    // Reserve registers.
    bool saturate = step->GetAttr("saturate", false);
    if (step->output(0)->type() == DT_INT64) saturate = false;
    int num_regs = 8;
    if (bias_) num_regs++;
    if (saturate) num_regs += 2;
    step->SetRegisterUsage(num_regs);

    // Matrix must be column major.
    step->input(1)->RequireOrder(COLUMN_MAJOR);
  }

  void Generate(Step *step, MacroAssembler *masm) override {
    Registers &rr = masm->rr();
    Label l1, l2;

    // Get input and output tensors.
    Tensor *x = step->input(0);
    Tensor *W = step->input(1);
    Tensor *b = bias_ ? step->input(2) : nullptr;
    Tensor *y = step->output(0);

    // Get dimensions.
    int rows = W->dim(0);
    int cols = W->dim(1);
    int row_size = W->stride(1);

    // Check bounds if saturate attribute is set.
    bool saturate = step->GetAttr("saturate", false);
    if (y->type() == DT_INT64) saturate = false;

    // Allocate registers.
    Register v = rr.alloc();
    Register m = rr.alloc();
    Register sum = rr.alloc();

    Register row = rr.alloc();
    Register col = rr.alloc();
    Register matrix = rr.alloc();
    Register input = rr.alloc();
    Register output = rr.alloc();
    Register vector = bias_ ? rr.alloc() : no_reg;

    Register min = saturate ? rr.alloc() : no_reg;
    Register max = saturate ? rr.alloc() : no_reg;

    // Initialize overflow bounds.
    if (saturate) {
      switch (y->type()) {
        case DT_INT8:
          __ movq(min, Immediate(relu_ ? 0 : -0x80));
          __ movq(max, Immediate(0x7f));
          break;

        case DT_INT16:
          __ movq(min, Immediate(relu_ ? 0 : -0x8000));
          __ movq(max, Immediate(0x7fff));
          break;

        case DT_INT32:
          __ movq(min, Immediate(relu_ ? 0 : -0x80000000));
          __ movq(max, Immediate(0x7fffffff));
          break;

        default: ; // no overflow check
      }
    }

    // Load tensor locations.
    __ LoadTensorAddress(input, x);
    __ LoadTensorAddress(matrix, W);
    if (bias_) {
      __ LoadTensorAddress(vector, b);
    }
    __ LoadTensorAddress(output, y);
    __ xorq(col, col);

    // Initialize dot product.
    __ LoopStart(&l1);
    if (bias_) {
      __ LoadInteger(sum, vector, col, b->type());
    } else {
      __ xorq(sum, sum);
    }
    __ xorq(row, row);

    // Compute dot product.
    __ LoopStart(&l2);
    __ LoadInteger(v, input, row, x->type());
    __ LoadInteger(m, matrix, row, W->type());
    __ imulq(v, m);
    __ addq(sum, v);
    __ addq(row, Immediate(1));
    __ cmpq(row, Immediate(rows));
    __ j(not_equal, &l2);

    // Apply relu.
    if (relu_) {
      Label nonneg;
      __ testq(sum, sum);
      __ j(positive, &nonneg);
      __ xorq(sum, sum);
      __ bind(&nonneg);
    }

    // Check for overflow.
    if (saturate) {
      __ cmpq(sum, min);
      __ cmovq(less, sum, min);
      __ cmpq(sum, max);
      __ cmovq(greater, sum, max);
    }

    // Store result in output.
    __ StoreInteger(output, col, sum, y->type());
    __ addq(col, Immediate(1));
    __ addq(matrix, Immediate(row_size));
    __ cmpq(col, Immediate(cols));
    __ j(not_equal, &l1);
  }

  int64 Complexity(const Step *step) override {
    int ops = step->input(1)->elements() * 2;
    if (bias_) ops += step->input(2)->elements();
    if (relu_) ops += step->output(0)->elements();
    return ops;
  }

 private:
  bool bias_;  // add bias vector to result, y=Wx+b
  bool relu_;  // apply rectified linear unit, y=max(0,Wx+b)
};

class GenericIntVecMatMul : public GenericIntVecMatMulBase {
 public:
  GenericIntVecMatMul() : GenericIntVecMatMulBase(false, false) {}

  string Name() override { return "GenIntVecMatMul"; }
  string Operation() override { return "MatMul"; }
};

class GenericIntVecMatMulAdd : public GenericIntVecMatMulBase {
 public:
  GenericIntVecMatMulAdd() : GenericIntVecMatMulBase(true, false) {}

  string Name() override { return "GenIntVecMatMulAdd"; }
  string Operation() override { return "MatMulAdd"; }
};

class GenericIntVecMatMulRelu : public GenericIntVecMatMulBase {
 public:
  GenericIntVecMatMulRelu() : GenericIntVecMatMulBase(false, true) {}

  string Name() override { return "GenFltIntMatMulRelu"; }
  string Operation() override { return "MatMulRelu"; }
};

class GenericIntVecMatMulAddRelu : public GenericIntVecMatMulBase {
 public:
  GenericIntVecMatMulAddRelu() : GenericIntVecMatMulBase(true, true) {}

  string Name() override { return "GenIntVecMatMulAddRelu"; }
  string Operation() override { return "MatMulAddRelu"; }
};

// Combine matrix multiplication with add and relu.
class MatMulTransformer : public Transformer {
 public:
  string Name() override { return "MatMulTransformer"; }

  bool Transform(Flow *flow) override {
    int combines = 0;
    while (Combine(flow, "MatMul", "Add", "MatMulAdd") ||
           Combine(flow, "MatMul", "Relu", "MatMulRelu") ||
           Combine(flow, "MatMulAdd", "Relu", "MatMulAddRelu")) {
      combines++;
    }
    return combines > 0;
  }

  // Try to find combinations and replace them with a combined op.
  bool Combine(Flow *flow, const string &first, const string &second,
               const string &combined) {
    // Find operations that can be combined.
    for (Flow::Operation *op : flow->ops()) {
      if (op->type != first) continue;
      if (op->outputs.size() != 1) continue;
      Flow::Variable *var = op->outputs[0];
      if (var->usages() != 1) continue;
      if (var->consumers[0]->type != second) continue;
      if (var->consumers[0]->task != op->task) continue;
      if (var->out()) continue;
      if (!var->shape.defined()) continue;
      if (var->type == DT_DOUBLE) continue;
      if (op->indegree() >= 1) {
        // Only combine for vector inputs.
        Flow::Variable *input = op->inputs[0];
        if (input->rank() == 2 && input->dim(0) > 1) continue;
      }
      if (op->GetAttr("transpose_a", false)) continue;
      if (op->GetAttr("transpose_b", false)) continue;
      if (op->GetAttr("transpose_c", false)) continue;

      flow->Fuse(op, var->consumers[0], combined);
      return true;
    }
    return false;
  }
};

// Merge transpose into matmul attributes.
class TransposeTransformer : public Transformer {
 public:
  string Name() override { return "TransposeTransformer"; }

  bool Transform(Flow *flow) override {
    int updates = 0;

    // Eliminate double transpose.
    for (Flow::Operation *op : flow->Find("Transpose|Transpose")) {
      Flow::Operation *t1 = op;
      Flow::Operation *t2 = t1->inputs[0]->producer;
      if (t1->outputs[0]->out()) continue;
      if (t1->outputs[0]->usages() != 1) continue;
      if (t1->HasAttr("perm")) continue;
      if (t2->HasAttr("perm")) continue;

      t2->outputs[0]->shape = t2->inputs[0]->shape;
      t1->outputs[0]->shape = t1->inputs[0]->shape;
      flow->Eliminate(t1);
      flow->Eliminate(t2);
      updates++;
    }

    // Eliminate double transpose through reference.
    for (Flow::Operation *op : flow->Find("Reference|Transpose")) {
      Flow::Operation *transpose = op;
      Flow::Operation *reference = transpose->inputs[0]->producer;
      if (transpose->outputs[0]->usages() != 1) continue;
      if (transpose->outputs[0]->out()) continue;
      if (reference->outputs[0]->usages() != 1) continue;
      if (reference->outputs[0]->out()) continue;
      if (transpose->HasAttr("perm")) continue;

      Flow::Variable *var = flow->Var(reference->GetAttr("var"));
      if (var == nullptr || var->producer == nullptr) continue;
      if (var->producer->type != "Transpose") continue;
      if (var->producer->HasAttr("perm")) continue;

      // Move reference to the input of the referenced transpose and eliminate
      // transpose.
      Flow::Variable *tin = var->producer->inputs[0];
      reference->SetAttr("var", tin->name);
      tin->set_out();
      transpose->inputs[0]->shape = transpose->outputs[0]->shape;
      flow->Eliminate(transpose);

      // Check if the referenced transpose is still an output.
      if (var->out() && !var->consumers.empty()) {
        int var_refs = 0;
        for (auto *op : flow->ops()) {
          if (op->type == "Reference" && op->GetAttr("var") == var->name) {
            var_refs++;
          }
        }
        if (var_refs == 0) var->set_out(false);
      }

      updates++;
    }

    // Merge double transpose.
    for (Flow::Operation *op : flow->Find("Transpose|Transpose")) {
      Flow::Operation *t1 = op;
      Flow::Operation *t2 = t1->inputs[0]->producer;
      if (t2->outputs[0]->out()) continue;
      if (t2->outputs[0]->usages() != 1) continue;

      int rank1 = t1->outputs[0]->rank();
      int rank2 = t2->outputs[0]->rank();
      if (rank1 != rank2) continue;

      Shape perm1;
      Shape perm2;
      if (!t1->GetAttr("perm", &perm1)) perm1.reverse(rank1);
      if (!t2->GetAttr("perm", &perm2)) perm2.reverse(rank2);
      Shape perm = perm2.permuted(perm1);
      t1->SetAttr("perm", perm);

      t2->outputs[0]->shape = t2->inputs[0]->shape;
      flow->Eliminate(t2);
      updates++;
    }

    // Fold transpose of first argument into matmul.
    for (Flow::Operation *op : flow->Find("Transpose|MatMul")) {
      Flow::Operation *matmul = op;
      Flow::Operation *transpose = matmul->inputs[0]->producer;
      if (transpose->outputs[0]->usages() != 1) continue;
      if (transpose->outputs[0]->out()) continue;
      if (transpose->HasAttr("perm")) continue;

      transpose->outputs[0]->shape = transpose->inputs[0]->shape;
      flow->Eliminate(transpose);
      matmul->SetAttr("transpose_a", !matmul->GetAttr("transpose_a", false));
      updates++;
    }

    // Fold transpose of second argument into matmul.
    for (Flow::Operation *op : flow->Find("Transpose|1:MatMul")) {
      Flow::Operation *matmul = op;
      Flow::Operation *transpose = matmul->inputs[1]->producer;
      if (transpose->outputs[0]->usages() != 1) continue;
      if (transpose->outputs[0]->out()) continue;
      if (transpose->HasAttr("perm")) continue;

      transpose->outputs[0]->shape = transpose->inputs[0]->shape;
      flow->Eliminate(transpose);
      matmul->SetAttr("transpose_b", !matmul->GetAttr("transpose_b", false));
      updates++;
    }

    // Fold transpose of output into matmul.
    for (Flow::Operation *op : flow->Find("MatMul|Transpose")) {
      Flow::Operation *transpose = op;
      Flow::Operation *matmul = transpose->inputs[0]->producer;
      if (matmul->outputs[0]->usages() != 1) continue;
      if (matmul->outputs[0]->out()) continue;
      if (transpose->HasAttr("perm")) continue;

      matmul->outputs[0]->shape = transpose->outputs[0]->shape;
      flow->Eliminate(transpose);
      matmul->SetAttr("transpose_c", !matmul->GetAttr("transpose_c", false));
      updates++;
    }

    // Factor transpose out of MatMul result by applying C^T=A*B => C=B^T*A^T.
    for (Flow::Operation *op : flow->Find("MatMul")) {
      if (!op->GetAttr("transpose_c", false)) continue;
      if (op->indegree() != 2 || op->outdegree() != 1) continue;
      op->SwapInputs();
      bool ta = op->GetAttr("transpose_a", false);
      bool tb = op->GetAttr("transpose_b", false);
      op->SetAttr("transpose_a", !tb);
      op->SetAttr("transpose_b", !ta);
      op->RemoveAttr("transpose_c");
    }

    return updates > 0;
  }
};

void RegisterGenericMatMul(Library *library) {
  // Transformations.
  library->RegisterTransformer(new MatMulTransformer());
  library->RegisterTransformer(new TransposeTransformer());

  // Computes  : C = A * B
  // Input     : A: float32[k,n] row-major
  //             B: float32[n,m] column-major
  // Output    : C: float32[k,m] row-major
  library->Register(new GenericFltMatMatMul());

  // Computes  : y = x * W
  // Input     : x: float32[1,n]
  //             W: float32[n,m] column-major
  // Output    : y: float32[1,m]
  library->Register(new GenericFltVecMatMul());

  // Computes  : y = x * W + b
  // Input     : x: float32[1,n]
  //             W: float32[n,m] column-major
  //             b: float32[1,n]
  // Output    : y: float32[1,m]
  library->Register(new GenericFltVecMatMulAdd());

  // Computes  : y = max(0, x * W)
  // Input     : x: float32[1,n]
  //             W: float32[n,m] column-major
  // Output    : y: float32[1,m]
  library->Register(new GenericFltVecMatMulRelu());

  // Computes  : y = max(0, x * W + b)
  // Input     : x: float32[1,n]
  //             W: float32[n,m] column-major
  //             b: float32[1,n]
  // Output    : y: float32[1,m]
  library->Register(new GenericFltVecMatMulAddRelu());

  // Computes  : y = x * W
  // Input     : x: int8/16/32/64[1,n]
  //             W: int8/16/32/64[n,m] column-major
  // Output    : y: int8/16/32/64[1,m]
  library->Register(new GenericIntVecMatMul());

  // Computes  : y = x * W + b
  // Input     : x: int8/16/32/64[1,n]
  //             W: int8/16/32/64[n,m] column-major
  //             b: int8/16/32/64[1,n]
  // Output    : y: int8/16/32/64[1,m]
  library->Register(new GenericIntVecMatMulAdd());

  // Computes  : y = max(0, x * W)
  // Input     : x: int8/16/32/64[1,n]
  //             W: int8/16/32/64[n,m] column-major
  // Output    : y: int8/16/32/64[1,m]
  library->Register(new GenericIntVecMatMulRelu());

  // Computes  : y = max(0, x * W + b)
  // Input     : x: int8/16/32/64[1,n]
  //             W: int8/16/32/64[n,m] column-major
  //             b: int8/16/32/64[1,n]
  // Output    : y: int8/16/32/64[1,m]
  library->Register(new GenericIntVecMatMulAddRelu());
}

}  // namespace myelin
}  // namespace sling

