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

#include "sling/myelin/cuda/cuda-kernel.h"

namespace sling {
namespace myelin {

// Concatenation of input tensors along first dimension using CUDA.
class CUDABasicConcat : public CUDAKernel {
 public:
  string Name() override { return "CUDABasicConcat"; }
  string Operation() override { return "ConcatV2"; }

  bool Supports(Step *step) override {
    // Requires CUDA support.
    if (!CUDAKernel::Supports(step)) return false;

    // Check inputs and outputs.
    if (step->indegree() < 2 || step->outdegree() != 1) return false;

    // Only concatenation along a singular prefix supported.
    int n = step->GetAttr("N", step->indegree() - 1);
    if (step->indegree() < n + 1) return false;
    Tensor *axis = step->input(n);
    if (!axis->IsConstant()) return false;
    int a = axis->value<int32>();
    if (step->output(0)->shape().outer(a) != 1) return false;

    return true;
  }

  void GeneratePTX(Step *step, PTXMacroAssembler *ptx) override {
    // Get the number of tensors to concatenate.
    int n = step->GetAttr("N", step->indegree() - 1);

    // Find maximum number of 32-bit words to copy.
    static const int WORD_SIZE = 4;
    int max_words = 0;
    bool has_residuals = false;
    for (int i = 0; i < n; ++i) {
      int bytes = step->input(i)->size();
      int words = bytes / WORD_SIZE;
      if (words > max_words) max_words = words;
      if (bytes != words * WORD_SIZE) has_residuals = true;
    }
    ptx->set_grid_dims(max_words);

    // Get thread index.
    ptx_decl(b32, idx);
    ptx->GetThreadIndex(idx, 0);

    // Compute block offset.
    ptx_decl(b64, ofs);
    ptx_emit(mul.wide.u32, ofs, idx, PTXImm(WORD_SIZE));

    // Load output tensor address.
    ptx_decl(b64, out);
    ptx->LoadTensorAddress(out, step->output(0));

    // Residuals are copied in the first thread.
    ptx_decl(pred, first);
    if (has_residuals) {
      ptx_emit(setp.eq.u32, first, idx, PTXImm(0));
    }

    // Copy input tensors to output.
    int offset = 0;
    ptx_decl(b64, in);
    ptx_decl(b64, src);
    ptx_decl(b64, dst);
    ptx_decl(u32, data32);
    ptx_decl(u16, data16);
    ptx_decl(u8, data8);
    for (int i = 0; i < n; ++i) {
      // Load input tensor address.
      int size = step->input(i)->size();
      ptx->LoadTensorAddress(in, step->input(i));

      // Copy main block in parallel; one word per thread.
      int words = size / WORD_SIZE;
      if (words > 0) {
        PTXReg copy = ptx->reg("pred", "copy", i);
        ptx_emit(setp.lt.u32, copy, idx, PTXImm(words));
        ptx_if(copy);
        ptx_emit(add.u64, src, in, ofs);
        ptx_emit(add.u64, dst, out, ofs);
        ptx_emit(ld.global.u32, data32, PTXAddr(src));
        ptx_emit(st.global.u32, PTXAddr(dst, offset), data32);
        ptx_endif();
      }

      // Copy residual in first thread.
      int residual = size - words * WORD_SIZE;
      if (residual > 0) {
        int res_ofs_in = words * WORD_SIZE;
        int res_ofs_out = res_ofs_in + offset;
        ptx_if(first);
        switch (residual) {
          case 1:
            ptx_emit(ld.global.u8, data8, PTXAddr(in, res_ofs_in));
            ptx_emit(st.global.u8, PTXAddr(out, res_ofs_out), data8);
            break;

          case 2:
            ptx_emit(ld.global.u16, data16, PTXAddr(in, res_ofs_in));
            ptx_emit(st.global.u16, PTXAddr(out, res_ofs_out), data16);
            break;

          case 3:
            ptx_emit(ld.global.u16, data16, PTXAddr(in, res_ofs_in));
            ptx_emit(st.global.u16, PTXAddr(out, res_ofs_out), data16);
            ptx_emit(ld.global.u8, data8, PTXAddr(in, res_ofs_in + 2));
            ptx_emit(st.global.u8, PTXAddr(out, res_ofs_out + 2), data8);
            break;
        }
        ptx_endif();
      }

      // Move to next destination in output tensor.
      offset += size;
    }
    ptx_ret();
  }

  int64 Complexity(const Step *step) override {
    return 0;
  }
};

// CUDA-based embedding lookup for single feature.
class CUDALookupSingle : public CUDAKernel {
 public:
  string Name() override { return "CUDALookupSingle"; }
  string Operation() override { return "Lookup"; }

  bool Supports(Step *step) override {
    // Requires CUDA support.
    if (!CUDAKernel::Supports(step)) return false;

    // Check inputs and outputs.
    if (step->indegree() != 2 || step->outdegree() != 1) return false;

    // Check types.
    Tensor *f = step->input(0);
    Tensor *M = step->input(1);
    Tensor *v = step->output(0);
    if (f->type() != DT_INT32 || f->elements() != 1) return false;
    if (M->type() != DT_FLOAT || M->rank() != 2) return false;
    if (v->type() != DT_FLOAT || v->rank() != 2) return false;
    if (v->dim(0) != 1 || v->dim(1) != M->dim(1)) return false;

    // Check that the output is not already a reference or a cell output.
    if (v->ref() || v->out()) return false;

    return true;
  }

  void Adjust(Step *step) override {
    // Make output a reference into the embedding matrix.
    Tensor *v = step->output(0);
    CHECK(!v->ref());
    CHECK(!v->out());
    v->set_ref(true);

    // Embedding matrix must be row-major.
    step->input(1)->SetRequiredOrder(ROW_MAJOR);
  }

  void GeneratePTX(Step *step, PTXMacroAssembler *ptx) override {
    // Get inputs and outputs.
    Tensor *f = step->input(0);
    Tensor *M = step->input(1);
    Tensor *v = step->output(0);

    // Get embedding size. The last element is the OOV element.
    int embedding_size = M->dim(0) - 1;

    // Set grid size.
    ptx->set_grid_dims(1);

    // Get feature index.
    ptx_decl(b64, fptr);
    ptx->LoadTensorAddress(fptr, f);
    ptx_decl(u32, fidx);
    ptx_emit(ld.global.u32, fidx, PTXAddr(fptr));

    // Use OOV for negative index.
    ptx_decl(pred, oov);
    ptx_emit(setp.eq.s32, oov, fidx, PTXImm(-1));
    ptx_if(oov);
    ptx_emit(mov.s32, fidx, PTXImm(embedding_size));
    ptx_endif();

    // Compute offset in embedding.
    ptx_decl(b64, ofs);
    ptx_emit(mul.wide.s32, ofs, fidx, PTXImm(M->stride(0)));

    // Lookup element in embedding.
    ptx_decl(b64, mptr);
    ptx->LoadTensorAddress(mptr, M);
    ptx_emit(add.u64, mptr, mptr, ofs);

    // Save reference to embedding vector.
    ptx_emit(st.global.b64, PTXAddr(ptx->data(), v->device_offset()), mptr);

    // Done.
    ptx_label(done);
    ptx_ret();
  }

  int64 Complexity(const Step *step) override {
    return 0;
  }
};

// CUDA-based embedding lookup for multiple features.
class CUDALookupMultiple : public CUDAKernel {
 public:
  string Name() override { return "CUDALookupMultiple"; }
  string Operation() override { return "Lookup"; }

  bool Supports(Step *step) override {
    // Requires CUDA support.
    if (!CUDAKernel::Supports(step)) return false;

    // Check inputs and outputs.
    if (step->indegree() != 2 || step->outdegree() != 1) return false;

    // Check types.
    Tensor *f = step->input(0);
    Tensor *M = step->input(1);
    Tensor *v = step->output(0);
    if (f->type() != DT_INT32) return false;
    if (M->type() != DT_FLOAT || M->rank() != 2) return false;
    if (v->type() != DT_FLOAT || v->rank() != 2) return false;
    if (v->dim(0) != 1 || v->dim(1) != M->dim(1)) return false;

    return true;
  }

  void Adjust(Step *step) override {
    // Embedding matrix must be row-major.
    step->input(1)->SetRequiredOrder(ROW_MAJOR);
  }

  void GeneratePTX(Step *step, PTXMacroAssembler *ptx) override {
    // Get inputs and outputs.
    Tensor *f = step->input(0);
    Tensor *M = step->input(1);
    Tensor *v = step->output(0);

    // Get embedding size and dimension. The last element is the OOV element.
    int embedding_size = M->dim(0) - 1;
    int embedding_dims = v->dim(1);

    // Get number of input features.
    int num_features = f->dim(1);

    // Use one thread for each element in the embedding.
    ptx->set_grid_dims(embedding_dims);

    // Get thread index.
    ptx_decl(b32, idx);
    ptx->GetThreadIndex(idx, 0);

    // Check bounds.
    ptx_decl(pred, outside);
    ptx_emit(setp.ge.u32, outside, idx, PTXImm(embedding_dims));
    ptx_if(outside);
    ptx_jump(done);
    ptx_endif();

    // Get embedding.
    ptx_decl(u64, embedding);
    ptx->LoadTensorAddress(embedding, M);
    ptx_emit(mad.wide.u32, embedding, idx, PTXImm(M->stride(1)), embedding);

    // Loop over input features.
    ptx_decl(f32, sum);
    ptx_emit(mov.f32, sum, PTXFloat(0));
    ptx_decl(b64, fptr);
    ptx->LoadTensorAddress(fptr, f);
    ptx_decl(u32, fidx);
    ptx_emit(mov.u32, fidx, PTXImm(0));
    ptx_label(loop1);

    // Get feature from feature vector.
    ptx_decl(s32, feature);
    ptx_emit(ld.global.s32, feature, PTXAddr(fptr));

    // Use OOV if feature value is -1.
    ptx_decl(pred, oov);
    ptx_emit(setp.eq.s32, oov, feature, PTXImm(-1));
    ptx_if(oov);
    ptx_emit(mov.s32, feature, PTXImm(embedding_size));
    ptx_endif();

    // Skip if feature value is -2.
    ptx_decl(pred, empty);
    ptx_emit(setp.eq.s32, empty, feature, PTXImm(-2));
    ptx_if(empty);
    ptx_jump(skip);
    ptx_endif();

    // Add embedding for feature to sum.
    ptx_decl(b64, mptr);
    ptx_emit(mad.wide.u32, mptr, feature, PTXImm(M->stride(0)), embedding);
    ptx_decl(f32, value);
    ptx_emit(ld.global.f32, value, PTXAddr(mptr));
    ptx_emit(add.f32, sum, sum, value);

    // Next feature.
    ptx_label(skip);
    ptx_emit(add.u32, fidx, fidx, PTXImm(1));
    ptx_emit(add.u64, fptr, fptr, PTXImm(sizeof(int)));
    ptx_decl(pred, more);
    ptx_emit(setp.lt.u32, more, fidx, PTXImm(num_features));
    ptx_if(more);
    ptx_jump(loop1);
    ptx_endif();

    // Save sum to output.
    ptx_decl(b64, vptr);
    ptx->LoadTensorAddress(vptr, v);
    ptx_emit(mad.wide.u32, vptr, idx, PTXImm(v->stride(1)), vptr);
    ptx_emit(st.global.f32, PTXAddr(vptr), sum);

    // Done.
    ptx_label(done);
    ptx_ret();
  }

  int64 Complexity(const Step *step) override {
    return step->input(0)->elements() * step->output(0)->elements();
  }
};

// CUDA-based feature collect operation for recurrent features mapped through
// an embedding matrix.
class CUDACollect : public CUDAKernel {
 public:
  string Name() override { return "CUDACollect"; }
  string Operation() override { return "Collect"; }

  bool Supports(Step *step) override {
    // Requires CUDA support.
    if (!CUDAKernel::Supports(step)) return false;

    // Check inputs and outputs.
    if (step->indegree() != 2 || step->outdegree() != 1) return false;

    // Check types.
    Tensor *f = step->input(0);
    Tensor *M = step->input(1);
    Tensor *R = step->output(0);
    if (f->type() != DT_INT32) return false;
    if (M->type() != DT_FLOAT || M->rank() != 2) return false;
    if (R->type() != DT_FLOAT || R->rank() != 2) return false;

    if (f->dim(0) != 1 || f->dim(1) != R->dim(0)) return false;
    if (R->dim(1) != M->dim(1) + 1) return false;

    return true;
  }

  void Adjust(Step *step) override {
    step->input(1)->SetRequiredOrder(ROW_MAJOR);
    step->output(0)->SetRequiredOrder(ROW_MAJOR);
  }

  void GeneratePTX(Step *step, PTXMacroAssembler *ptx) override {
    // Get inputs and outputs.
    Tensor *f = step->input(0);
    Tensor *M = step->input(1);
    Tensor *R = step->output(0);

    // Get size of activation vectors.
    int dims = M->dim(1);

    // Get number input features.
    int num_features = f->dim(1);

    // Use a thread for each feature and embedding dim.
    ptx->set_grid_dims(dims, num_features);

    // Get feature and embedding position thread index.
    ptx_decl(b32, midx);
    ptx->GetThreadIndex(midx, 0);
    ptx_decl(b32, fidx);
    ptx->GetThreadIndex(fidx, 1);

    // Check bounds.
    ptx_decl(pred, outside_x);
    ptx_emit(setp.ge.u32, outside_x, midx, PTXImm(dims));
    ptx_decl(pred, outside_y);
    ptx_emit(setp.ge.u32, outside_y, fidx, PTXImm(num_features));
    ptx_decl(pred, outside);
    ptx_emit(or.pred, outside, outside_x, outside_y);
    ptx_if(outside);
    ptx_jump(done);
    ptx_endif();

    // Get feature id.
    ptx_decl(b64, fptr);
    ptx->LoadTensorAddress(fptr, f);
    ptx_decl(s32, fval);
    ptx_emit(mad.wide.u32, fptr, fidx, PTXImm(f->stride(1)), fptr);
    ptx_emit(ld.global.s32, fval, PTXAddr(fptr));

    // Output zero if feature value is -1.
    ptx_decl(f32, value);
    ptx_emit(mov.f32, value, PTXFloat(0));
    ptx_decl(pred, oov);
    ptx_emit(setp.eq.s32, oov, fval, PTXImm(-1));
    ptx_if(oov);
    ptx_jump(l1);
    ptx_endif();

    // Lookup embedding.
    ptx_decl(b64, mptr);
    ptx->LoadTensorAddress(mptr, M);
    ptx_emit(mad.wide.u32, mptr, fval, PTXImm(M->stride(0)), mptr);
    ptx_emit(mad.wide.u32, mptr, midx, PTXImm(M->stride(1)), mptr);
    ptx_emit(ld.global.f32, value, PTXAddr(mptr));

    // Get output address.
    ptx_label(l1);
    ptx_decl(b64, result);
    ptx->LoadTensorAddress(result, R);
    ptx_decl(b64, rrow);
    if (num_features > 1) {
      ptx_emit(mad.wide.u32, rrow, fidx, PTXImm(R->stride(0)), result);
    } else {
      rrow = result;
    }
    ptx_decl(b64, rptr);
    ptx_emit(mad.wide.u32, rptr, midx, PTXImm(R->stride(1)), rrow);

    // Store value in result.
    ptx_emit(st.global.f32, PTXAddr(rptr), value);

    // The OOV indicator is set in the first thread for the feature.
    ptx_decl(pred, first);
    ptx_emit(setp.eq.u32, first, midx, PTXImm(0));
    ptx_ifnot(first);
    ptx_jump(done);
    ptx_endif();

    // Set OOV indicator to 1.0 if feature is -1.
    ptx_decl(f32, oovval);
    ptx_if(oov);
    ptx_emit(mov.f32, oovval, PTXFloat(1.0));
    ptx_else();
    ptx_emit(mov.f32, oovval, PTXFloat(0));
    ptx_endif();
    ptx_emit(st.global.f32, PTXAddr(rrow, dims * R->stride(1)), oovval);

    // Done.
    ptx_label(done);
    ptx_ret();
  }

  int64 Complexity(const Step *step) override {
    return 0;
  }
};

// Register CUDA array library.
void RegisterCUDAArrayLibrary(Library *library) {
  library->Register(new CUDABasicConcat());
  library->Register(new CUDALookupMultiple());
  library->Register(new CUDALookupSingle());
  library->Register(new CUDACollect());
}

}  // namespace myelin
}  // namespace sling

