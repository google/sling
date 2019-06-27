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

#include "sling/myelin/kernel/cuda.h"

#include <string>

#include "sling/myelin/cuda/cuda-kernel.h"

namespace sling {
namespace myelin {

// Matrix multiplication using CUDA.
class CUDAMatMulBase : public CUDAKernel {
 public:
  // Maximum number of loop unrolls.
  static const int MAX_UNROLLS = 8;

  CUDAMatMulBase(bool bias, bool relu) : bias_(bias), relu_(relu) {}

  bool Supports(Step *step) override {
    // Requires CUDA support.
    if (!CUDAKernel::Supports(step)) return false;

    // Two or three 2D tensor inputs and one 2D tensor output.
    if (step->inputs().size() != (bias_ ? 3 : 2)) return false;
    if (step->outputs().size() != 1) return false;
    Args args(step);
    Matrix &A = args.A;
    Matrix &B = args.B;
    Matrix &C = args.C;

    // Check rank.
    if (A.rank() != 2) return false;
    if (B.rank() != 2) return false;
    if (C.rank() != 2) return false;

    // Check shape.
    if (A.height() != C.height()) return false;
    if (A.width() != B.height()) return false;
    if (B.width() != C.width()) return false;

    // Types must match and be supported by CUDA.
    Type type = A.type();
    if (TypeTraits::of(type).ptx() == nullptr) return false;
    if (B.type() != type || C.type() != type) return false;

    // Check bias vector.
    if (bias_) {
      Matrix &v = args.v;
      if (v.type() != type) return false;
      if (v.rank() == 1) {
        if (v.tensor()->dim(0) != C.width()) return false;
      } else if (v.rank() == 2) {
        if (v.height() != 1 || v.width() != C.width()) return false;
      } else {
        return false;
      }
    }

    return true;
  }

  void Adjust(Step *step) override {
    // Prefer row-major.
    Args args(step);
    args.A.prefer_row_major();
    args.B.prefer_row_major();
    args.C.prefer_row_major();
  }

  void GeneratePTX(Step *step, PTXMacroAssembler *ptx) override {
    // Get input and output tensors.
    Args args(step);

    // Use tiled matrix multiplication if matrices consist of whole tiles.
    if (args.A.tileable(16) && args.B.tileable(16)) {
      GenerateTiled(step, ptx, 16);
    } else if (args.A.tileable(8) && args.B.tileable(8)) {
      GenerateTiled(step, ptx, 8);
    } else {
      GenerateSingle(step, ptx);
    }
  }

  void GenerateSingle(Step *step, PTXMacroAssembler *ptx) {
    // Get input and output tensors.
    Args args(step);
    Matrix &A = args.A;
    Matrix &B = args.B;
    Matrix &C = args.C;
    Matrix &v = args.v;

    int width = C.width();
    int height = C.height();
    int depth = A.width();

    Type dtype = A.type();
    const TypeTraits &traits = TypeTraits::of(dtype);
    const char *type = traits.ptx();
    bool fp = dtype == DT_FLOAT || dtype == DT_DOUBLE || dtype == DT_HALF;
    bool vec = height == 1;
    int dsize = traits.size();

    // Compute number of unrolls.
    int unrolls = 0;
    for (int i = 1; i <= MAX_UNROLLS; ++i) {
      if (depth % i == 0) unrolls = i;
    }
    if (step->variant().empty()) {
      string variant = "U" + std::to_string(unrolls);
      step->set_variant(variant);
    }

    // Set grid size. Use one thread for each output element in C.
    ptx->set_grid_dims(width, height);

    // Get output row and column in C.
    ptx_decl(u32, col);
    ptx->LoadThreadIndex(col, 0);
    ptx_decl(u32, row);
    if (!vec) {
      ptx->LoadThreadIndex(row, 1);
    }

    // Check bounds.
    if (vec) {
      ptx_decl(pred, outside);
      ptx_emit(setp.ge.u32, outside, col, PTXImm(width));
      ptx_if(outside);
      ptx_jump(done);
      ptx_endif();
    } else {
      ptx_decl(pred, outside_col);
      ptx_emit(setp.ge.u32, outside_col, col, PTXImm(width));
      ptx_decl(pred, outside_row);
      ptx_emit(setp.ge.u32, outside_row, row, PTXImm(height));
      ptx_decl(pred, outside);
      ptx_emit(or.pred, outside, outside_col, outside_row);
      ptx_if(outside);
      ptx_jump(done);
      ptx_endif();
    }

    // Compute address of row in A.
    ptx_decl(b64, aptr);
    ptx->LoadTensorAddress(aptr, A.tensor());
    if (!vec) {
      ptx_emit(mad.wide.u32, aptr, row, PTXImm(A.row(1)), aptr);
    }

    // Compute address of column in B.
    ptx_decl(b64, bptr);
    ptx->LoadTensorAddress(bptr, B.tensor());
    ptx_emit(mad.wide.u32, bptr, col, PTXImm(B.column(1)), bptr);

    // Compute dot product.
    PTXConst zero(PTXConst::ZERO, type);
    ptx_decl(u32, idx);
    ptx_emit(mov.u32, idx, PTXImm(0));
    PTXReg sum = ptx->reg(type, "sum");
    ptx->emit(PTXInstr("mov", type), sum, zero);
    ptx_label(loop);

    // Compute sum += A[row,idx] * B[idx,col].
    PTXReg a = ptx->reg(type, "a");
    PTXReg b = ptx->reg(type, "b");
    for (int i = 0; i < unrolls; ++i) {
      ptx->emit(PTXInstr("ld.global", type), a, PTXAddr(aptr, A.column(i)));
      ptx->emit(PTXInstr("ld.global", type), b, PTXAddr(bptr, B.row(i)));
      ptx->emit(PTXInstr(fp ? "fma.rn" : "mad.lo", type), sum, a, b, sum);
    }

    // Next element.
    if (unrolls != depth) {
      ptx_emit(add.u32, idx, idx, PTXImm(unrolls));
      ptx_emit(add.u64, aptr, aptr, PTXImm(A.column(unrolls)));
      ptx_emit(add.u64, bptr, bptr, PTXImm(B.row(unrolls)));

      ptx_decl(pred, more);
      ptx_emit(setp.lt.u32, more, idx, PTXImm(depth));
      ptx_if(more);
      ptx_jump(loop);
      ptx_endif();
    }

    // Optionally add bias.
    if (bias_) {
      ptx_decl(b64, vptr);
      ptx->LoadTensorAddress(vptr, v.tensor());
      ptx_emit(mad.wide.u32, vptr, col, PTXImm(dsize), vptr);

      PTXReg bias = ptx->reg(type, "bias");
      ptx->emit(PTXInstr("ld.global", type), bias, PTXAddr(vptr));
      ptx->emit(PTXInstr("add", type), sum, sum, bias);
    }

    // Optionally compute relu.
    if (relu_) {
      ptx->emit(PTXInstr("max", type), sum, sum, zero);
    }

    // Save result in C[row,col].
    ptx_decl(b64, cptr);
    ptx->LoadTensorAddress(cptr, C.tensor());
    if (!vec) {
      ptx_emit(mad.wide.u32, cptr, row, PTXImm(C.row(1)), cptr);
    }
    ptx_emit(mad.wide.u32, cptr, col, PTXImm(C.column(1)), cptr);
    ptx->emit(PTXInstr("st.global", type), PTXAddr(cptr), sum);

    // Done.
    ptx_label(done);
    ptx_ret();
  }

  void GenerateTiled(Step *step, PTXMacroAssembler *ptx, int tile_size) {
    // Get input and output tensors.
    Args args(step);
    Matrix &A = args.A;
    Matrix &B = args.B;
    Matrix &C = args.C;
    Matrix &v = args.v;

    int width = C.width();
    int height = C.height();
    int depth = A.width();

    Type dtype = A.type();
    const TypeTraits &traits = TypeTraits::of(dtype);
    const char *type = traits.ptx();
    bool fp = dtype == DT_FLOAT || dtype == DT_DOUBLE || dtype == DT_HALF;
    int dsize = traits.size();
    step->set_variant("T" + std::to_string(tile_size));

    // Set the block size to the title size and use one tread per output
    // element.
    ptx->set_block_dims(tile_size, tile_size);
    ptx->set_grid_dims(width, height);

    // Declare shared memory for tiles of A and B.
    char str[128];
    sprintf(str, ".shared .%s ablock[%d];\n", type, tile_size * tile_size);
    ptx->emit(str);
    sprintf(str, ".shared .%s bblock[%d];\n", type, tile_size * tile_size);
    ptx->emit(str);
    ptx_decl(b64, atile);
    ptx_decl(b64, btile);
    ptx_emit(mov.u64, atile, PTXLiteral("ablock"));
    ptx_emit(mov.u64, btile, PTXLiteral("bblock"));

    // Get block x and y indices.
    ptx_decl(u32, bx);
    ptx_decl(u32, by);
    ptx->LoadBlockIndex(bx, 0);
    ptx->LoadBlockIndex(by, 1);

    // Get tile x and y indices.
    ptx_decl(u32, tx);
    ptx_decl(u32, ty);
    ptx->LoadBlockThreadIndex(tx, 0);
    ptx->LoadBlockThreadIndex(ty, 1);

    // Compute output row and column.
    ptx_decl(u32, x);
    ptx_decl(u32, y);
    ptx_emit(mad.lo.u32, x, bx, PTXImm(tile_size), tx);
    ptx_emit(mad.lo.u32, y, by, PTXImm(tile_size), ty);

    // Compute address of first A tile row.
    ptx_decl(b64, aptr);
    ptx->LoadTensorAddress(aptr, A.tensor());
    ptx_emit(mad.wide.u32, aptr, y, PTXImm(A.row(1)), aptr);
    ptx_emit(mad.wide.u32, aptr, tx, PTXImm(A.column(1)), aptr);

    // Compute address of first B tile column.
    ptx_decl(b64, bptr);
    ptx->LoadTensorAddress(bptr, B.tensor());
    ptx_emit(mad.wide.u32, bptr, ty, PTXImm(B.row(1)), bptr);
    ptx_emit(mad.wide.u32, bptr, x, PTXImm(B.column(1)), bptr);

    // Compute offset of (col,row) element in tiles. These are pointers to
    // the elements in the shared memory blocks that the thread is responsible
    // for loading.
    ptx_decl(b64, ain);
    ptx_emit(mad.wide.u32, ain, ty, PTXImm(dsize * tile_size), atile);
    ptx_emit(mad.wide.u32, ain, tx, PTXImm(dsize), ain);

    ptx_decl(b64, bin);
    ptx_emit(mad.wide.u32, bin, ty, PTXImm(dsize * tile_size), btile);
    ptx_emit(mad.wide.u32, bin, tx, PTXImm(dsize), bin);

    // Sub-matrices of A and B that thread is summing over.
    ptx_decl(b64, as);
    ptx_decl(b64, bs);
    ptx_emit(mad.wide.u32, as, ty, PTXImm(dsize * tile_size), atile);
    ptx_emit(mad.wide.u32, bs, tx, PTXImm(dsize), btile);

    // Accumulate result for C[row,col].
    PTXConst zero(PTXConst::ZERO, type);
    PTXReg c = ptx->reg(type, "c");
    ptx->emit(PTXInstr("mov", type), c, zero);

    // Loop over all the tiles of A and B that are required to compute the
    // tile in C.
    ptx_decl(u32, idx);
    ptx_emit(mov.u32, idx, PTXImm(0));
    ptx_label(loop);

    // Load A and B tiles from device memory to shared memory. Each thread loads
    // one element from each of the A and B tiles.
    PTXReg a = ptx->reg(type, "a");
    PTXReg b = ptx->reg(type, "b");
    ptx->emit(PTXInstr("ld.global", type), a, PTXAddr(aptr));
    ptx->emit(PTXInstr("st.shared", type), PTXAddr(ain), a);
    ptx->emit(PTXInstr("ld.global", type), b, PTXAddr(bptr));
    ptx->emit(PTXInstr("st.shared", type), PTXAddr(bin), b);

    // Synchronize to make sure the tiles are loaded before starting the
    // computation.
    ptx_emit(bar.sync, PTXImm(0));

    // Multiply A and B tiles together.
    for (int i = 0; i < tile_size; ++i) {
      // c += a[row,i] * b[i,col].
      int aofs = i * dsize;
      int bofs = i * dsize * tile_size;
      ptx->emit(PTXInstr("ld.shared", type), a, PTXAddr(as, aofs));
      ptx->emit(PTXInstr("ld.shared", type), b, PTXAddr(bs, bofs));
      ptx->emit(PTXInstr(fp ? "fma.rn" : "mad.lo", type), c, a, b, c);
    }

    // Synchronize to make sure that the preceding computation is done before
    // loading new tiles of A and B in the next iteration.
    ptx_emit(bar.sync, PTXImm(0));

    // Next tile.
    ptx_emit(add.u64, aptr, aptr, PTXImm(A.column(tile_size)));
    ptx_emit(add.u64, bptr, bptr, PTXImm(B.row(tile_size)));
    ptx_emit(add.u32, idx, idx, PTXImm(tile_size));

    ptx_decl(pred, more);
    ptx_emit(setp.lt.u32, more, idx, PTXImm(depth));
    ptx_if(more);
    ptx_jump(loop);
    ptx_endif();

    // Optionally add bias.
    if (bias_) {
      ptx_decl(b64, vptr);
      ptx->LoadTensorAddress(vptr, v.tensor());
      ptx_emit(mad.wide.u32, vptr, x, PTXImm(dsize), vptr);

      PTXReg bias = ptx->reg(type, "bias");
      ptx->emit(PTXInstr("ld.global", type), bias, PTXAddr(vptr));
      ptx->emit(PTXInstr("add", type), c, c, bias);
    }

    // Optionally compute relu.
    if (relu_) {
      ptx->emit(PTXInstr("max", type), c, c, zero);
    }

    // Store result in C[row,col].
    ptx_decl(b64, cptr);
    ptx->LoadTensorAddress(cptr, C.tensor());
    ptx_emit(mad.wide.u32, cptr, y, PTXImm(C.row(1)), cptr);
    ptx_emit(mad.wide.u32, cptr, x, PTXImm(C.column(1)), cptr);
    ptx->emit(PTXInstr("st.global", type), PTXAddr(cptr), c);

    // Done.
    ptx_label(done);
    ptx_ret();
  }

  int64 Complexity(const Step *step) override {
    Args args(step);
    int64 ops = args.A.height();
    ops *= args.B.tensor()->elements() * 2;
    if (bias_) ops += args.v.tensor()->elements();
    if (relu_) ops += args.C.tensor()->elements();
    return ops;
  }

 protected:
  // Matrix argument with optional transpose.
  class Matrix {
   public:
    Matrix(Tensor *tensor, bool transposed)
        : tensor_(tensor), transposed_(transposed) {}

    // Outer dimension of tensor array.
    int outer() const { return transposed_ ? 1 : 0; }

    // Inner dimension in tensor array.
    int inner() const { return transposed_ ? 0 : 1; }

    // Height (outer dimension) of matrix.
    int height() const { return tensor_->dim(outer()); }

    // Width (inner dimension) of matrix.
    int width() const { return tensor_->dim(inner()); }

    // Offset of row.
    int row(int n) const { return tensor_->stride(outer()) * n; }

    // Offset of column.
    int column(int n) const { return tensor_->stride(inner()) * n; }

    // Tensor for matrix.
    Tensor *tensor() const { return tensor_; }

    // Element data type.
    Type type() const { return tensor_->type(); }

    // Tensor rank.
    int rank() const { return tensor_->rank(); }

    // Prefer row-major layout for matrix.
    void prefer_row_major() {
      Order order = transposed_ ? COLUMN_MAJOR_PREFERRED : ROW_MAJOR_PREFERRED;
      tensor_->RequireOrder(order);
    }

    // Check if matrix can be broken down into tiles.
    bool tileable(int n) const { return height() % n == 0 && width() % n == 0; }

   private:
    Tensor *tensor_;   // underlying tensor for matrix
    bool transposed_;  // transposed matrix
  };

  // Arguments for MatMul kernel.
  struct Args {
    Args(const Step *step)
      : A(step->input(0), step->GetAttr("transpose_a", false)),
        B(step->input(1), step->GetAttr("transpose_b", false)),
        C(step->output(0), step->GetAttr("transpose_c", false)),
        v(step->indegree() < 3 ? nullptr : step->input(2), false) {}

    Matrix A;
    Matrix B;
    Matrix C;
    Matrix v;
  };

  bool bias_;    // add bias vector to result, y=Wx+b
  bool relu_;    // apply rectified linear unit, y=max(0,Wx+b)
};

class CUDAMatMul : public CUDAMatMulBase {
 public:
  CUDAMatMul() : CUDAMatMulBase(false, false) {}

  string Name() override { return "CUDAMatMul"; }
  string Operation() override { return "MatMul"; }
};

class CUDAMatMulAdd : public CUDAMatMulBase {
 public:
  CUDAMatMulAdd() : CUDAMatMulBase(true, false) {}

  string Name() override { return "CUDAMatMulAdd"; }
  string Operation() override { return "MatMulAdd"; }
};

class CUDAMatMulRelu : public CUDAMatMulBase {
 public:
  CUDAMatMulRelu() : CUDAMatMulBase(false, true) {}

  string Name() override { return "CUDAMatMulRelu"; }
  string Operation() override { return "MatMulRelu"; }
};

class CUDAMatMulAddRelu : public CUDAMatMulBase {
 public:
  CUDAMatMulAddRelu() : CUDAMatMulBase(true, true) {}

  string Name() override { return "CUDAMatMulAddRelu"; }
  string Operation() override { return "MatMulAddRelu"; }
};

void RegisterCUDAMatMulLibrary(Library *library) {
  library->Register(new CUDAMatMul());
  library->Register(new CUDAMatMulAdd());
  library->Register(new CUDAMatMulRelu());
  library->Register(new CUDAMatMulAddRelu());
}

}  // namespace myelin
}  // namespace sling

