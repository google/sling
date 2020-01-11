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

#include <math.h>

#include "sling/myelin/cuda/cuda-kernel.h"

namespace sling {
namespace myelin {

// Concatenation of input tensors along first dimension using CUDA.
class CUDABasicConcat : public CUDAKernel {
 public:
  static const int WORD_SIZE = 4;

  string Name() override { return "CUDABasicConcat"; }
  string Operation() override { return "Concat"; }

  bool Supports(Step *step) override {
    // Requires CUDA support.
    if (!CUDAKernel::Supports(step)) return false;

    // Check inputs and outputs.
    if (step->indegree() < 2 || step->outdegree() != 1) return false;

    // Only concatenation along a singular prefix supported.
    int n = step->GetAttr("N", step->indegree() - 1);
    if (step->indegree() < n + 1) return false;
    Tensor *axis = step->input(n);
    if (!axis->constant()) return false;
    int a = axis->value<int32>();
    if (step->output(0)->shape().outer(a) != 1) return false;

    return true;
  }

  void Adjust(Step *step) override {
    // Input and output tensors must be word aligned.
    int n = step->GetAttr("N", step->indegree() - 1);
    for (int i = 0; i < n; ++i) {
      step->input(i)->SetMiniumAlignment(WORD_SIZE);
    }
    step->output(0)->SetMiniumAlignment(WORD_SIZE);
  }

  void GeneratePTX(Step *step, PTXMacroAssembler *ptx) override {
    // Get the number of tensors to concatenate.
    int n = step->GetAttr("N", step->indegree() - 1);

    // Find maximum number of 32-bit words to copy.
    int max_words = 0;
    bool has_residuals = false;
    for (int i = 0; i < n; ++i) {
      int bytes = step->input(i)->size();
      int words = bytes / WORD_SIZE;
      if (words > max_words) max_words = words;
      if (bytes != words * WORD_SIZE) has_residuals = true;
    }
    ptx->set_grid_dims(std::max(max_words, 1));

    // Get thread index.
    ptx_decl(b32, idx);
    ptx->LoadThreadIndex(idx, 0);

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
        int align = offset  % WORD_SIZE;
        switch (align) {
          case 0:
            ptx_emit(ld.global.u32, data32, PTXAddr(src));
            ptx_emit(st.global.u32, PTXAddr(dst, offset), data32);
            break;
          case 2:
            ptx_emit(ld.global.u16, data16, PTXAddr(src));
            ptx_emit(st.global.u16, PTXAddr(dst, offset), data16);
            ptx_emit(ld.global.u16, data16, PTXAddr(src, 2));
            ptx_emit(st.global.u16, PTXAddr(dst, offset + 2), data16);
            break;
          default:
            ptx_emit(ld.global.u8, data8, PTXAddr(src));
            ptx_emit(st.global.u8, PTXAddr(dst, offset), data8);
            ptx_emit(ld.global.u8, data8, PTXAddr(src, 1));
            ptx_emit(st.global.u8, PTXAddr(dst, offset + 1), data8);
            ptx_emit(ld.global.u8, data8, PTXAddr(src, 2));
            ptx_emit(st.global.u8, PTXAddr(dst, offset + 2), data8);
            ptx_emit(ld.global.u8, data8, PTXAddr(src, 3));
            ptx_emit(st.global.u8, PTXAddr(dst, offset + 3), data8);
        }
        ptx_endif();
      }

      // Copy residual in first thread.
      int residual = size - words * WORD_SIZE;
      if (residual > 0) {
        int res_ofs_in = words * WORD_SIZE;
        int res_ofs_out = res_ofs_in + offset;
        ptx_if(first);
        for (int r = 0; r < residual; ++r) {
          ptx_emit(ld.global.u8, data8, PTXAddr(in, res_ofs_in + r));
          ptx_emit(st.global.u8, PTXAddr(out, res_ofs_out + r), data8);
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
class CUDAGatherSingle : public CUDAKernel {
 public:
  string Name() override { return "CUDAGatherSingle"; }
  string Operation() override { return "Gather"; }

  bool Supports(Step *step) override {
    // Requires CUDA support.
    if (!CUDAKernel::Supports(step)) return false;

    // Check inputs and outputs.
    if (step->indegree() != 2 && step->indegree() != 3) return false;
    if (step->outdegree() != 1) return false;

    // Check types.
    Tensor *M = step->input(0);
    Tensor *f = step->input(1);
    Tensor *oov = step->indegree() == 3 ? step->input(2) : nullptr;
    Tensor *v = step->output(0);
    if (f->type() != DT_INT32) return false;
    if (M->type() != DT_FLOAT || M->rank() != 2) return false;
    if (v->type() != DT_FLOAT) return false;
    if (oov != nullptr && oov->type() != DT_FLOAT) return false;
    int n = f->elements();
    int d = M->dim(1);
    int r = v->rank() - 1;
    if (v->shape().outer(r) != n) return false;
    if (v->shape().inner(r) != d) return false;
    if (oov != nullptr && v->shape().inner(r) != oov->elements()) return false;
    if (n != 1) return false;

    // Check that the output is not already a reference.
    if (v->ref()) return false;

    return true;
  }

  void Adjust(Step *step) override {
    // Make output a reference into the embedding matrix.
    Tensor *v = step->output(0);
    DCHECK(!v->ref());
    v->set_ref(true);
    v->Link(step->input(0));
    if (step->indegree() == 3) v->Link(step->input(2));

    // Embedding matrix must be row-major.
    step->input(0)->RequireOrder(ROW_MAJOR);
  }

  void GeneratePTX(Step *step, PTXMacroAssembler *ptx) override {
    // Get inputs and outputs.
    Tensor *M = step->input(0);
    Tensor *f = step->input(1);
    Tensor *oov = step->indegree() == 3 ? step->input(2) : nullptr;
    Tensor *v = step->output(0);

    // Set grid size.
    ptx->set_grid_dims(1);

    // Get feature value.
    ptx_decl(b64, fptr);
    ptx->LoadTensorAddress(fptr, f);
    ptx_decl(u32, feature);
    ptx_emit(ld.global.u32, feature, PTXAddr(fptr));

    // Look up embedding vector. Use OOV for negative index.
    ptx_decl(b64, mptr);
    if (oov) {
      ptx_decl(pred, is_oov);
      ptx_emit(setp.lt.s32, is_oov, feature, PTXImm(0));
      ptx_if(is_oov);
      ptx->LoadTensorAddress(mptr, oov);
      ptx_else();
      ptx_decl(b64, ofs);
      ptx_emit(mul.wide.s32, ofs, feature, PTXImm(M->stride(0)));
      ptx->LoadTensorAddress(mptr, M);
      ptx_emit(add.u64, mptr, mptr, ofs);
      ptx_endif();
    } else {
      // Compute offset in embedding.
      ptx_decl(b64, ofs);
      ptx_emit(mul.wide.s32, ofs, feature, PTXImm(M->stride(0)));

      // Lookup element in embedding.
      ptx->LoadTensorAddress(mptr, M);
      ptx_emit(add.u64, mptr, mptr, ofs);
    }

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
class CUDAGatherMultiple : public CUDAKernel {
 public:
  string Name() override { return "CUDAGatherMultiple"; }
  string Operation() override { return "Gather"; }

  bool Supports(Step *step) override {
    // Requires CUDA support.
    if (!CUDAKernel::Supports(step)) return false;

    // Check inputs and outputs.
    if (step->indegree() != 2 && step->indegree() != 3) return false;
    if (step->outdegree() != 1) return false;

    // Check types.
    Tensor *M = step->input(0);
    Tensor *f = step->input(1);
    Tensor *oov = step->indegree() == 3 ? step->input(2) : nullptr;
    Tensor *v = step->output(0);
    if (f->type() != DT_INT32) return false;
    if (M->type() != DT_FLOAT || M->rank() != 2) return false;
    if (v->type() != DT_FLOAT) return false;
    if (oov != nullptr && oov->type() != DT_FLOAT) return false;
    int n = f->elements();
    int d = M->dim(1);
    int r = v->rank() - 1;
    if (v->shape().outer(r) != n) return false;
    if (v->shape().inner(r) != d) return false;
    if (oov != nullptr && v->shape().inner(r) != oov->elements()) return false;

    return true;
  }

  void Adjust(Step *step) override {
    // Embedding matrix must be row-major.
    step->input(0)->RequireOrder(ROW_MAJOR);
  }

  void GeneratePTX(Step *step, PTXMacroAssembler *ptx) override {
    // Get inputs and outputs.
    Tensor *M = step->input(0);
    Tensor *f = step->input(1);
    Tensor *oov = step->indegree() == 3 ? step->input(2) : nullptr;
    Tensor *v = step->output(0);

    // Get embedding size and dimension-
    int embedding_dims = v->dim(1);

    // Get number of input features.
    int num_features = f->dim(1);

    // Use one thread for each element in the embedding.
    ptx->set_grid_dims(embedding_dims);

    // Get thread index.
    ptx_decl(b32, idx);
    ptx->LoadThreadIndex(idx, 0);

    // Check bounds.
    ptx_decl(pred, outside);
    ptx_emit(setp.ge.u32, outside, idx, PTXImm(embedding_dims));
    ptx_if(outside);
    ptx_jump(done);
    ptx_endif();

    // Get embedding.
    ptx_decl(u64, embedding);
    ptx->LoadTensorAddress(embedding, M);

    // Get OOV vector.
    ptx_decl(u64, oovptr);
    if (oov != nullptr) {
      ptx->LoadTensorAddress(oovptr, oov);
    }

    // Compute offset of element in embedding vector.
    ptx_decl(b64, element_offset);
    ptx_emit(mul.wide.u32, element_offset, idx, PTXImm(M->element_size()));

    // Initialize output pointer.
    ptx_decl(b64, vptr);
    ptx->LoadTensorAddress(vptr, v);
    ptx_emit(add.u64, vptr, vptr, element_offset);

    // Loop over input features.
    ptx_decl(b64, fptr);
    ptx->LoadTensorAddress(fptr, f);
    ptx_decl(u32, fidx);
    ptx_emit(mov.u32, fidx, PTXImm(0));
    ptx_label(loop1);

    // Get feature from feature vector.
    ptx_decl(s32, feature);
    ptx_emit(ld.global.s32, feature, PTXAddr(fptr));

    // Look up embedding vector. Use OOV for negative index.
    ptx_decl(b64, mptr);
    if (oov) {
      ptx_decl(pred, is_oov);
      ptx_emit(setp.lt.s32, is_oov, feature, PTXImm(0));
      ptx_if(is_oov);
      ptx_emit(mov.b64, mptr, oovptr);
      ptx_else();
      ptx_emit(mad.wide.u32, mptr, feature, PTXImm(M->stride(0)), embedding);
      ptx_endif();
    } else {
      ptx_emit(mad.wide.u32, mptr, feature, PTXImm(M->stride(0)), embedding);
    }
    ptx_emit(add.u64, mptr, mptr, element_offset);

    // Get element in embedding.
    ptx_decl(f32, value);
    ptx_emit(ld.global.f32, value, PTXAddr(mptr));

    // Write element to output.
    ptx_emit(st.global.f32, PTXAddr(vptr), value);

    // Next feature.
    ptx_emit(add.u32, fidx, fidx, PTXImm(1));
    ptx_emit(add.u64, fptr, fptr, PTXImm(sizeof(int)));
    ptx_emit(add.u64, vptr, vptr, PTXImm(M->stride(0)));
    ptx_decl(pred, more);
    ptx_emit(setp.lt.u32, more, fidx, PTXImm(num_features));
    ptx_if(more);
    ptx_jump(loop1);
    ptx_endif();

    // Done.
    ptx_label(done);
    ptx_ret();
  }

  int64 Complexity(const Step *step) override {
    return step->input(0)->elements() * step->output(0)->elements();
  }
};

// Look up multiple features in embedding with pooling.
class CUDAPoolingGather : public CUDAKernel {
 public:
  // Pooling operations.
  enum Pooling {SUM, AVG, MAX};

  CUDAPoolingGather(Pooling pooling) : pooling_(pooling) {}

  string Name() override { return "CUDA" + Operation(); }
  string Operation() override {
    switch (pooling_) {
      case SUM: return "GatherSum";
      case AVG: return "GatherAvg";
      case MAX: return "GatherMax";
      default: return "???";
    }
  }

  bool Supports(Step *step) override {
    // Requires CUDA support.
    if (!CUDAKernel::Supports(step)) return false;

    // Check inputs and outputs.
    if (step->indegree() != 2 || step->outdegree() != 1) return false;

    // Check types.
    Tensor *M = step->input(0);
    Tensor *f = step->input(1);
    Tensor *v = step->output(0);
    if (M->type() != DT_FLOAT || M->rank() != 2) return false;
    if (f->type() != DT_INT32 || f->rank() != 2) return false;
    if (v->type() != DT_FLOAT || v->elements() != M->dim(1)) return false;

    return true;
  }

  void Adjust(Step *step) override {
    // Embedding matrix must be row-major.
    step->input(0)->RequireOrder(ROW_MAJOR);
  }

  void GeneratePTX(Step *step, PTXMacroAssembler *ptx) override {
    // Get inputs and outputs.
    Tensor *M = step->input(0);
    Tensor *f = step->input(1);
    Tensor *v = step->output(0);

    // Get embedding size and dimension-
    int embedding_dims = v->dim(1);

    // Get number of input features.
    int num_features = f->dim(1);

    // Use one thread for each element in the embedding.
    ptx->set_grid_dims(embedding_dims);

    // Get thread index.
    ptx_decl(b32, idx);
    ptx->LoadThreadIndex(idx, 0);

    // Check bounds.
    ptx_decl(pred, outside);
    ptx_emit(setp.ge.u32, outside, idx, PTXImm(embedding_dims));
    ptx_if(outside);
    ptx_jump(done);
    ptx_endif();

    // Get embedding.
    ptx_decl(u64, embedding);
    ptx->LoadTensorAddress(embedding, M);

    // Compute offset of element in embedding vector.
    ptx_decl(b64, element_offset);
    ptx_emit(mul.wide.u32, element_offset, idx, PTXImm(M->element_size()));

    // Initialize neutral value for pooling operation.
    ptx_decl(f32, accum);
    if (pooling_ == MAX) {
      ptx_emit(mov.f32, accum, PTXFloat(-INFINITY));
    } else {
      ptx_emit(mov.f32, accum, PTXFloat(0.0));
    }

    // Loop over input features.
    ptx_decl(b64, fptr);
    ptx->LoadTensorAddress(fptr, f);
    ptx_decl(u32, fidx);
    ptx_emit(mov.u32, fidx, PTXImm(0));
    ptx_label(loop1);

    // Get feature from feature vector.
    ptx_decl(s32, feature);
    ptx_emit(ld.global.s32, feature, PTXAddr(fptr));

    // Stop if feature is negative.
    ptx_decl(pred, negative);
    ptx_emit(setp.lt.s32, negative, feature, PTXImm(0));
    ptx_if(negative);
    ptx_jump(end);
    ptx_endif();

    // Look up embedding vector. Use OOV for negative index.
    ptx_decl(b64, mptr);
    ptx_emit(mad.wide.u32, mptr, feature, PTXImm(M->stride(0)), embedding);
    ptx_emit(add.u64, mptr, mptr, element_offset);

    // Get element in embedding.
    ptx_decl(f32, value);
    ptx_emit(ld.global.f32, value, PTXAddr(mptr));

    // Accumulate values.
    if (pooling_ == MAX) {
      ptx_emit(max.f32, accum, accum, value);
    } else {
      ptx_emit(add.f32, accum, accum, value);
    }

    // Next feature.
    ptx_emit(add.u32, fidx, fidx, PTXImm(1));
    ptx_emit(add.u64, fptr, fptr, PTXImm(sizeof(int)));
    ptx_decl(pred, more);
    ptx_emit(setp.lt.u32, more, fidx, PTXImm(num_features));
    ptx_if(more);
    ptx_jump(loop1);
    ptx_endif();

    // Store result.
    ptx_label(end);
    if (pooling_ == AVG) {
      // Compute average.
      ptx_decl(f32, count);
      ptx_emit(cvt.u32.f32, count, fidx);
      ptx_emit(div.rnd.f32, accum, count);
    }

    ptx_decl(b64, vptr);
    ptx->LoadTensorAddress(vptr, v);
    ptx_emit(add.u64, vptr, vptr, element_offset);
    ptx_emit(st.global.f32, PTXAddr(vptr), accum);

    // Done.
    ptx_label(done);
    ptx_ret();
  }

  int64 Complexity(const Step *step) override {
    Tensor *M = step->input(0);
    Tensor *f = step->input(1);
    return M->dim(1) * f->elements() + (pooling_ == AVG ? M->dim(1) : 0);
  }

 private:
  Pooling pooling_;  // pooling operation for combining vectors
};

// Register CUDA array library.
void RegisterCUDAArrayLibrary(Library *library) {
  library->Register(new CUDABasicConcat());
  library->Register(new CUDAGatherMultiple());
  library->Register(new CUDAGatherSingle());
  library->Register(new CUDAPoolingGather(CUDAPoolingGather::SUM));
  library->Register(new CUDAPoolingGather(CUDAPoolingGather::AVG));
  library->Register(new CUDAPoolingGather(CUDAPoolingGather::MAX));
}

}  // namespace myelin
}  // namespace sling
