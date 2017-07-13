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

#include "myelin/compute.h"
#include "myelin/macro-assembler.h"

#define __ masm->

namespace sling {
namespace myelin {

using namespace jit;

// Reshape tensor while preserving the underlying data.
class Reshape : public Kernel {
 public:
  string Name() override { return "Reshape"; }
  string Operation() override { return "Reshape"; }

  bool Supports(Step *step) override {
    // Check inputs and outputs.
    if (step->indegree() != 2 || step->outdegree() != 1) return false;
    Tensor *x = step->input(0);
    Tensor *y = step->output(0);
    if (x->type() != y->type()) return false;
    if (x->elements() != y->elements()) return false;
    return true;
  }

  void Adjust(Step *step) override {
    step->output(0)->set_ref(step->input(0)->ref());
    CHECK(step->AllowInPlace(0, 0, true));
  }

  void Generate(Step *step, MacroAssembler *masm) override {
    CHECK(step->input(0)->SharedWith(step->output(0)));
  }

  int64 Complexity(const Step *step) override {
    return 0;
  }
};

// Removes dimensions of size 1 from the shape of a tensor while preserving the
// underlying data.
class Squeeze : public Kernel {
 public:
  string Name() override { return "Squeeze"; }
  string Operation() override { return "Squeeze"; }

  bool Supports(Step *step) override {
    // Check inputs and outputs.
    if (step->indegree() != 1 || step->outdegree() != 1) return false;
    Tensor *x = step->input(0);
    Tensor *y = step->output(0);
    if (x->type() != y->type()) return false;
    if (x->elements() != y->elements()) return false;
    return true;
  }

  void Adjust(Step *step) override {
    step->output(0)->set_ref(step->input(0)->ref());
    CHECK(step->AllowInPlace(0, 0, true));
  }

  void Generate(Step *step, MacroAssembler *masm) override {
    CHECK(step->input(0)->SharedWith(step->output(0)));
  }

  int64 Complexity(const Step *step) override {
    return 0;
  }
};

// Inserts a dimension of 1 into a tensor's shape while preserving the
// underlying data.
class ExpandDims : public Kernel {
 public:
  string Name() override { return "ExpandDims"; }
  string Operation() override { return "ExpandDims"; }

  bool Supports(Step *step) override {
    // Check inputs and outputs.
    if (step->indegree() != 2 || step->outdegree() != 1) return false;
    Tensor *x = step->input(0);
    Tensor *y = step->output(0);
    if (x->type() != y->type()) return false;
    if (x->elements() != y->elements()) return false;
    return true;
  }

  void Adjust(Step *step) override {
    step->output(0)->set_ref(step->input(0)->ref());
    CHECK(step->AllowInPlace(0, 0, true));
  }

  void Generate(Step *step, MacroAssembler *masm) override {
    CHECK(step->input(0)->SharedWith(step->output(0)));
  }

  int64 Complexity(const Step *step) override {
    return 0;
  }
};

// Kernel for resizing the input by padding or cropping.
class Resize : public Kernel {
 public:
  bool Supports(Step *step) override {
    // Check inputs and outputs.
    if (step->indegree() != 3 || step->outdegree() != 1) return false;
    Tensor *x = step->input(0);
    Tensor *y = step->output(0);
    if (x->type() != y->type()) return false;
    return true;
  }

  void Adjust(Step *step) override {
    Tensor *x = step->input(0);
    Tensor *y = step->output(0);
    step->AllowInPlace(0, 0, x->elements() == y->elements());
  }

  void Generate(Step *step, MacroAssembler *masm) override {
    // Check if resize is a no-op.
    Tensor *x = step->input(0);
    Tensor *y = step->output(0);
    bool shared = x->SharedWith(y);
    bool pad = y->size() > x->size();
    bool crop = y->size() < x->size();
    if (shared && !pad && !crop) {
      step->set_variant("nop");
      return;
    } else if (!shared) {
      step->set_variant("copy");
    } else if (pad) {
      step->set_variant("pad");
    } else if (crop) {
      step->set_variant("crop");
    }

    // Allocate registers.
    Register src = masm->rr().alloc_fixed(rsi);
    Register dst = masm->rr().alloc_fixed(rdi);
    Register cnt = masm->rr().alloc_fixed(rcx);
    Register acc = masm->rr().alloc_fixed(rax);

    if (shared) {
     // Pad output if needed.
     if (pad) {
       __ LoadTensorAddress(dst, y);
       __ addq(dst, Immediate(x->size()));
       __ xorq(acc, acc);
       __ movq(cnt, Immediate(y->size() - x->size()));
       __ repstosb();
     }
    } else {
      // Load tensors.
      __ LoadTensorAddress(src, x);
      __ LoadTensorAddress(dst, y);

      // Copy input to output.
      __ movq(cnt, Immediate(std::min(x->size(), y->size())));
      __ repmovsb();

      // Pad output if needed.
      if (pad) {
        __ xorq(acc, acc);
        __ movq(cnt, Immediate(y->size() - x->size()));
        __ repstosb();
      }
    }
  }

  int64 Complexity(const Step *step) override {
    return 0;
  }
};

// Divide "spatial" dimensions [1, ..., M] of the input, and interleaves these
// with the "batch" dimension (0).
class SpaceToBatch : public Resize {
 public:
  string Name() override { return "SpaceToBatch"; }
  string Operation() override { return "SpaceToBatchND"; }
};

// Reshapes the "batch" dimension 0 into M + 1 dimensions, and interleaves these
// back into the spatial dimensions [1, ..., M].
class BatchToSpace : public Resize {
 public:
  string Name() override { return "BatchToSpace"; }
  string Operation() override { return "BatchToSpaceND"; }
};

// Packs an array of rank-R tensors into one rank-(R+1) tensor.
class Pack : public Kernel {
 public:
  string Name() override { return "Pack"; }
  string Operation() override { return "Pack"; }

  bool Supports(Step *step) override {
    // Check inputs and outputs.
    if (step->indegree() != 1 || step->outdegree() != 1) return false;
    Tensor *x = step->input(0);
    Tensor *y = step->output(0);
    if (x->type() != y->type()) return false;
    if (x->elements() != y->elements()) return false;
    return true;
  }

  void Adjust(Step *step) override {
    step->output(0)->set_ref(step->input(0)->ref());
    CHECK(step->AllowInPlace(0, 0, true));
  }

  void Generate(Step *step, MacroAssembler *masm) override {
    CHECK(step->input(0)->SharedWith(step->output(0)));
  }

  int64 Complexity(const Step *step) override {
    return 0;
  }
};

// Unpacks an array of a rank-R tensor into rank-(R-1) tensors.
class Unpack : public Kernel {
 public:
  string Name() override { return "Unpack"; }
  string Operation() override { return "Unpack"; }

  bool Supports(Step *step) override {
    // Check inputs and outputs.
    if (step->indegree() != 2 || step->outdegree() != 1) return false;
    Tensor *x = step->input(0);
    Tensor *y = step->output(0);
    if (x->type() != y->type()) return false;
    if (x->elements() != y->elements()) return false;
    return true;
  }

  void Adjust(Step *step) override {
    step->output(0)->set_ref(step->input(0)->ref());
    CHECK(step->AllowInPlace(0, 0, true));
  }

  void Generate(Step *step, MacroAssembler *masm) override {
    CHECK(step->input(0)->SharedWith(step->output(0)));
  }

  int64 Complexity(const Step *step) override {
    return 0;
  }
};

// Output concatenation of input tensors along first dimension.
class BasicConcat : public Kernel {
 public:
  string Name() override { return "BasicConcat"; }
  string Operation() override { return "ConcatV2"; }

  bool Supports(Step *step) override {
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

  void Adjust(Step *step) override {
  }

  void Generate(Step *step, MacroAssembler *masm) override {
    // Get the number of tensors to concatenate.
    int n = step->GetAttr("N", step->indegree() - 1);

    // Allocate registers.
    Register src = masm->rr().alloc_fixed(rsi);
    Register dst = masm->rr().alloc_fixed(rdi);
    Register cnt = masm->rr().alloc_fixed(rcx);
    Register acc = masm->rr().alloc_fixed(rax);
    Register in = masm->rr().alloc();
    Register out = masm->rr().alloc();

    // Load output tensor.
    __ LoadTensorAddress(out, step->output(0));

    // Copy input tensors to output.
    int offset = 0;
    for (int i = 0; i < n; ++i) {
      int size = step->input(i)->size();
      if (size > 0 && size < 16) {
        __ LoadTensorAddress(in, step->input(i));
        int disp = offset;
        int left = size;
        while (left >= 8) {
          __ movq(acc, Operand(in, disp));
          __ movq(Operand(out, disp), acc);
          disp += 8;
          left -= 8;
        }
        while (left >= 4) {
          __ movl(acc, Operand(in, disp));
          __ movl(Operand(out, disp), acc);
          disp += 4;
          left -= 4;
        }
        while (left >= 2) {
          __ movw(acc, Operand(in, disp));
          __ movw(Operand(out, disp), acc);
          disp += 2;
          left -= 2;
        }
        while (left >= 1) {
          __ movb(acc, Operand(in, disp));
          __ movb(Operand(out, disp), acc);
          disp += 1;
          left -= 1;
        }
      } else {
        __ LoadTensorAddress(src, step->input(i));
        __ leaq(dst, Operand(out, offset));
        __ movq(cnt, Immediate(size));
        __ repmovsb();
      }
      offset += size;
    }
    CHECK_EQ(offset, step->output(0)->size());
  }

  int64 Complexity(const Step *step) override {
    return 0;
  }
};

// Output concatenation of input tensors along any axis.
class GeneralConcat : public Kernel {
 public:
  string Name() override { return "GeneralConcat"; }
  string Operation() override { return "ConcatV2"; }

  bool Supports(Step *step) override {
    // Check inputs and outputs.
    if (step->indegree() < 2 || step->outdegree() != 1) return false;

    // Check concatenation axis.
    int n = step->GetAttr("N", step->indegree() - 1);
    if (step->indegree() < n + 1) return false;
    if (!step->input(n)->IsConstant()) return false;
    int axis = step->input(n)->value<int32>();

    // Check outer prefix has same size for all inputs.
    Tensor *output = step->output(0);
    if (output->rank() < axis) return false;
    int prefix = output->shape().outer(axis);
    for (int i = 0; i < n; ++i) {
      Tensor *input = step->input(i);
      if (input->rank() < axis) return false;
      if (input->shape().outer(axis) != prefix) return false;
      if (input->type() != output->type()) return false;
    }

    return true;
  }

  void Adjust(Step *step) override {
  }

  void Generate(Step *step, MacroAssembler *masm) override {
    // Get the number of tensors to concatenate.
    int n = step->GetAttr("N", step->indegree() - 1);

    // Allocate registers.
    Register src = masm->rr().alloc_fixed(rsi);
    Register dst = masm->rr().alloc_fixed(rdi);
    Register cnt = masm->rr().alloc_fixed(rcx);
    Register acc = masm->rr().alloc_fixed(rax);
    Register out = masm->rr().alloc();
    Register idx = masm->rr().alloc();
    std::vector<Register> in(n);
    for (int i = 0; i < n; ++i) in[i] = masm->rr().alloc();

    // Load input tensors.
    for (int i = 0; i < n; ++i) {
      __ LoadTensorAddress(in[i], step->input(i));
    }

    // Load output tensor.
    __ LoadTensorAddress(out, step->output(0));
    __ xorq(idx, idx);

    // Loop over outer prefix.
    Label l;
    int axis = step->input(n)->value<int32>();
    int prefix = step->output(0)->shape().outer(axis);
    LOG(INFO) << "Prefix size " << prefix;
    __ bind(&l);

    // Copy input tensors to output.
    Tensor *output = step->output(0);
    for (int i = 0; i < n; ++i) {
      Tensor *input = step->input(i);
      int size = axis > 0 ? input->stride(axis - 1) : input->size();
      if (size > 0 && size < 16) {
        int disp = 0;
        int left = size;
        while (left >= 8) {
          __ movq(acc, Operand(in[i], disp));
          __ movq(Operand(out, disp), acc);
          disp += 8;
          left -= 8;
        }
        while (left >= 4) {
          __ movl(acc, Operand(in[i], disp));
          __ movl(Operand(out, disp), acc);
          disp += 4;
          left -= 4;
        }
        while (left >= 2) {
          __ movw(acc, Operand(in[i], disp));
          __ movw(Operand(out, disp), acc);
          disp += 2;
          left -= 2;
        }
        while (left >= 1) {
          __ movb(acc, Operand(in[i], disp));
          __ movb(Operand(out, disp), acc);
          disp += 1;
          left -= 1;
        }
      } else {
        __ movq(src, in[i]);
        __ movq(dst, out);
        __ movq(cnt, Immediate(size));
        __ repmovsb();
      }
      __ addq(in[i], Immediate(size));
    }

    // Next chunk.
    int size = axis > 0 ? output->stride(axis - 1) : output->size();
    __ addq(out, Immediate(size));
    __ incq(idx);
    __ cmpq(idx, Immediate(prefix));
    __ j(less, &l);
  }

  int64 Complexity(const Step *step) override {
    return 0;
  }
};

// Look up single embedding.
class SingleGather : public Kernel {
 public:
  string Name() override { return "SingleGather"; }
  string Operation() override { return "Gather"; }

  bool Supports(Step *step) override {
    // Check inputs and outputs.
    if (step->indegree() != 2 || step->outdegree() != 1) return false;

    // Check types.
    Tensor *M = step->input(0);
    Tensor *f = step->input(1);
    Tensor *v = step->output(0);
    int r = f->rank();
    if (f->type() != DT_INT32 || f->elements() != 1) return false;
    if (M->type() != DT_FLOAT || M->rank() != 2) return false;
    if (v->type() != DT_FLOAT || v->rank() != r + 1) return false;
    if (v->shape().outer(r) != 1 || v->dim(r) != M->dim(1)) return false;

    return true;
  }

  void Adjust(Step *step) override {
    // Make output a reference into the embedding matrix.
    step->output(0)->set_ref(true);
    step->output(0)->set_link(step->input(0));

    // Embedding matrix must be row-major.
    step->input(0)->SetRequiredOrder(ROW_MAJOR);
  }

  void Generate(Step *step, MacroAssembler *masm) override {
    // Get inputs and outputs.
    Tensor *M = step->input(0);
    Tensor *f = step->input(1);
    Tensor *v = step->output(0);
    CHECK(f->IsLocal());
    CHECK(v->IsLocal());

    // Allocate registers.
    Register acc = masm->rr().alloc();
    Register addr = masm->rr().alloc();
    Register embeddings = masm->rr().alloc();

    // Get feature index.
    if (f->ref()) {
      __ movq(addr, Operand(masm->instance(), f->offset()));
      __ movsxlq(acc, Operand(addr));
    } else {
      __ movsxlq(acc, Operand(masm->instance(), f->offset()));
    }

    // Compute offset in embedding.
    __ Multiply(acc, M->stride(0));

    // Lookup element in embedding.
    __ LoadTensorAddress(embeddings, M);
    __ addq(acc, embeddings);

    // Save reference to embedding vector.
    if (f->ref()) {
      __ movq(addr, Operand(masm->instance(), v->offset()));
      __ movsxlq(acc, Operand(addr));
    } else {
      __ movq(Operand(masm->instance(), v->offset()), acc);
    }
  }

  int64 Complexity(const Step *step) override {
    return 0;
  }
};

// Look up multiple features in embedding.
class MultiGather : public Kernel {
 public:
  string Name() override { return "MultiGather"; }
  string Operation() override { return "Gather"; }

  bool Supports(Step *step) override {
    // Check inputs and outputs.
    if (step->indegree() != 2 || step->outdegree() != 1) return false;

    // Check types.
    Tensor *M = step->input(0);
    Tensor *f = step->input(1);
    Tensor *v = step->output(0);
    int r = f->rank();
    int n = f->elements();
    if (f->type() != DT_INT32) return false;
    if (M->type() != DT_FLOAT || M->rank() != 2) return false;
    if (v->type() != DT_FLOAT || v->rank() != r + 1) return false;
    if (v->shape().outer(r) != n || v->dim(r) != M->dim(1)) return false;

    return true;
  }

  void Adjust(Step *step) override {
    // Embedding matrix must be row-major.
    step->input(0)->SetRequiredOrder(ROW_MAJOR);
  }

  void Generate(Step *step, MacroAssembler *masm) override {
    // Get inputs and outputs.
    Tensor *M = step->input(0);
    Tensor *f = step->input(1);
    Tensor *v = step->output(0);
    CHECK(f->IsLocal());
    CHECK(v->IsLocal());

    // Allocate registers.
    Register src = masm->rr().alloc_fixed(rsi);
    Register dst = masm->rr().alloc_fixed(rdi);
    Register cnt = masm->rr().alloc_fixed(rcx);
    Register acc = masm->rr().alloc();
    Register index = masm->rr().alloc();
    Register input = masm->rr().alloc();
    Register embeddings = masm->rr().alloc();

    // Load tensor locations.
    __ LoadTensorAddress(embeddings, M);
    __ LoadTensorAddress(input, f);
    __ LoadTensorAddress(dst, v);

    // Loop over all feature indices.
    Label l;
    __ xorq(index, index);
    __ bind(&l);

    // Get feature index.
    __ movsxlq(acc, Operand(input, index, times_4));

    // Compute address in embedding.
    __ movq(src, embeddings);
    __ Multiply(acc, M->stride(0));
    __ addq(src, acc);

    // Copy embedding vector to output.
    __ movq(cnt, Immediate(M->stride(0)));
    __ repmovsb();

    // Next feature index.
    __ incq(index);
    __ cmpq(index, Immediate(f->elements()));
    __ j(less, &l);
  }

  int64 Complexity(const Step *step) override {
    return 0;
  }
};

// Register array kernels.
void RegisterArrayKernels(Library *library) {
  library->Register(new Reshape());
  library->Register(new Squeeze());
  library->Register(new ExpandDims());
  library->Register(new SpaceToBatch());
  library->Register(new BatchToSpace());
  library->Register(new Pack());
  library->Register(new Unpack());
  library->Register(new GeneralConcat());
  library->Register(new BasicConcat());
  library->Register(new MultiGather());
  library->Register(new SingleGather());
}

}  // namespace myelin
}  // namespace sling

