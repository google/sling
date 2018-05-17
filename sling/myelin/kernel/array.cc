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

#include "sling/myelin/compute.h"
#include "sling/myelin/macro-assembler.h"

#define __ masm->

namespace sling {
namespace myelin {

using namespace jit;

// Allocate registers for unrolling.
static int SIMDUnrolls(int size, int vecsize, int max_unrolls) {
  int unrolls = 0;
  for (int i = 1; i <= max_unrolls; ++i) {
    int batch_size = i * vecsize;
    if (size >= batch_size && size % batch_size == 0) unrolls = i;
  }
  return unrolls;
}

static int AllocateYMMUnrolls(MacroAssembler *masm,
                              int size,
                              int max_unrolls,
                              std::vector<YMMRegister> *regs) {
  int unrolls = SIMDUnrolls(size, 8, max_unrolls);
  for (int i = 0; i < std::max(unrolls, 1); ++i) {
    regs->push_back(masm->mm().allocy());
  }
  return unrolls;
}

static int AllocateZMMUnrolls(MacroAssembler *masm,
                              int size,
                              int max_unrolls,
                              std::vector<ZMMRegister> *regs) {
  int unrolls = SIMDUnrolls(size, 16, max_unrolls);
  for (int i = 0; i < std::max(unrolls, 1); ++i) {
    regs->push_back(masm->mm().allocz());
  }
  return unrolls;
}

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
    CHECK(step->AllowInPlace(0, 0, true));
  }

  void Generate(Step *step, MacroAssembler *masm) override {
    CHECK(step->input(0)->SharedWith(step->output(0)));
  }

  Placement Location() override { return NOWHERE; }

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

  Placement Location() override { return NOWHERE; }

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

  Placement Location() override { return NOWHERE; }

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

  Placement Location() override { return NOWHERE; }

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

  Placement Location() override { return NOWHERE; }

  int64 Complexity(const Step *step) override {
    return 0;
  }
};

// Output a one-hot vector.
class OneHot : public Kernel {
 public:
  string Name() override { return "OneHot"; }
  string Operation() override { return "OneHot"; }

  bool Supports(Step *step) override {
    // Check inputs and outputs.
    if (step->indegree() != 1 || step->outdegree() != 1) return false;
    Tensor *index = step->input(0);
    Tensor *onehot = step->output(0);
    if (index->type() != DT_INT32) return false;
    if (onehot->type() != DT_FLOAT) return false;
    return true;
  }

  void Generate(Step *step, MacroAssembler *masm) override {
    Tensor *index = step->input(0);
    Tensor *onehot = step->output(0);

    // Allocate registers.
    Register dst = masm->rr().alloc_fixed(rdi);
    Register cnt = masm->rr().alloc_fixed(rcx);
    Register acc = masm->rr().alloc_fixed(rax);
    Register input = masm->rr().alloc();
    Register output = masm->rr().alloc();

    // Zero output tensor.
    __ LoadTensorAddress(output, onehot);
    __ movq(dst, output);
    __ movq(cnt, Immediate(onehot->size()));
    __ xorq(acc, acc);
    __ repstosb();

    // Set one-hot index.
    __ LoadTensorAddress(input, index);
    __ movsxlq(acc, Operand(input));
    __ movq(Operand(output, acc, times_4), Immediate(0x3F800000));
  }

  int64 Complexity(const Step *step) override {
    return 0;
  }
};

// Slice input tensors along first dimension.
class Slice : public Kernel {
 public:
  string Name() override { return "Slice"; }
  string Operation() override { return "Slice"; }

  bool Supports(Step *step) override {
    // Check inputs and outputs.
    if (step->indegree() != 3 || step->outdegree() != 1) return false;

    // Check arguments.
    Tensor *input = step->input(0);
    Tensor *begin = step->input(1);
    Tensor *size = step->input(2);
    Tensor *output = step->output(0);
    if (begin->rank() > 1 || begin->type() != DT_INT32) return false;
    if (size->rank() > 1 || size->type() != DT_INT32) return false;
    std::vector<int> s;
    CHECK(size->GetData(&s));
    if (Shape(s) != output->shape()) return false;
    if (input->type() != output->type()) return false;

    return true;
  }

  void Adjust(Step *step) override {
  }

  void Generate(Step *step, MacroAssembler *masm) override {
    // Get inputs and output.
    Tensor *source = step->input(0);
    Tensor *begin = step->input(1);
    Tensor *size = step->input(2);
    Tensor *destination = step->output(0);

    // Compute size of slice.
    std::vector<int> size_tensor;
    CHECK(size->GetData(&size_tensor));
    int bytes = source->element_size();
    for (int i = 0; i < size_tensor.size(); ++i) {
      bytes *= size_tensor[i];
    }

    // Allocate registers.
    Register src = masm->rr().alloc_fixed(rsi);
    Register dst = masm->rr().alloc_fixed(rdi);
    Register cnt = masm->rr().alloc_fixed(rcx);
    Register acc = masm->rr().alloc_fixed(rax);

    // Get source and destination addresses.
    __ LoadTensorAddress(src, source, begin);
    __ LoadTensorAddress(dst, destination);

    // Copy input to output.
    if (bytes > 0 && bytes < 16) {
      int disp = 0;
      int left = bytes;
      while (left >= 8) {
        __ movq(acc, Operand(src, disp));
        __ movq(Operand(dst, disp), acc);
        disp += 8;
        left -= 8;
      }
      while (left >= 4) {
        __ movl(acc, Operand(src, disp));
        __ movl(Operand(dst, disp), acc);
        disp += 4;
        left -= 4;
      }
      while (left >= 2) {
        __ movw(acc, Operand(src, disp));
        __ movw(Operand(dst, disp), acc);
        disp += 2;
        left -= 2;
      }
      while (left >= 1) {
        __ movb(acc, Operand(src, disp));
        __ movb(Operand(dst, disp), acc);
        disp += 1;
        left -= 1;
      }
    } else {
      __ movq(cnt, Immediate(bytes));
      __ repmovsb();
    }
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
    if (!axis->constant()) return false;
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
        int disp = 0;
        int left = size;
        while (left >= 8) {
          __ movq(acc, Operand(in, disp));
          __ movq(Operand(out, offset + disp), acc);
          disp += 8;
          left -= 8;
        }
        while (left >= 4) {
          __ movl(acc, Operand(in, disp));
          __ movl(Operand(out, offset + disp), acc);
          disp += 4;
          left -= 4;
        }
        while (left >= 2) {
          __ movw(acc, Operand(in, disp));
          __ movw(Operand(out, offset + disp), acc);
          disp += 2;
          left -= 2;
        }
        while (left >= 1) {
          __ movb(acc, Operand(in, disp));
          __ movb(Operand(out, offset + disp), acc);
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
    CHECK_EQ(offset, step->output(0)->size()) << step->name();
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
    if (!step->input(n)->constant()) return false;
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
    int r = f->rank() - 1;
    if (f->type() != DT_INT32 || f->elements() != 1) return false;
    if (M->type() != DT_FLOAT || M->rank() != 2) return false;
    if (v->type() != DT_FLOAT || v->rank() != r + 1) return false;
    if (v->shape().outer(r) != 1 || v->dim(r) != M->dim(1)) return false;

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
    CHECK(v->ref());

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
    __ movq(Operand(masm->instance(), v->offset()), acc);
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
    int r = f->rank() - 1;
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

// Look up multiple features in embedding with pooling.
class PoolingGather : public Kernel {
 public:
  // Pooling operations.
  enum Pooling {SUM, AVG, MAX};

  PoolingGather(Pooling pooling) : pooling_(pooling) {}

  string Name() override { return Operation(); }
  string Operation() override {
    switch (pooling_) {
      case SUM: return "GatherSum";
      case AVG: return "GatherAvg";
      case MAX: return "GatherMax";
      default: return "???";
    }
  }

  bool Supports(Step *step) override {
    // Requires SSE or AVX support.
    if (!CPU::Enabled(AVX) && !CPU::Enabled(SSE)) return false;

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
    Tensor *M = step->input(0);
    Tensor *v = step->output(0);

    // Align to one ymm/xmm register.
    int align = 4;
    if (CPU::Enabled(AVX)) align = 8;
    if (CPU::Enabled(AVX512F)) align = 16;
    M->SetMiniumAlignment(align * sizeof(float));
    v->SetMiniumAlignment(align * sizeof(float));

    // Embedding matrix must be row-major.
    M->SetRequiredOrder(ROW_MAJOR);
    if (M->dim(1) >= align) M->MinAlign({1, align});
  }

  void Generate(Step *step, MacroAssembler *masm) override {
    // Get inputs and outputs.
    Tensor *M = step->input(0);
    Tensor *f = step->input(1);
    Tensor *v = step->output(0);
    CHECK(f->IsLocal()) << f->name();
    CHECK(v->IsLocal()) << v->name();
    int n = v->elements();

    // Allocate registers.
    Register acc = masm->rr().alloc_fixed(rax);
    Register src = masm->rr().alloc_fixed(rsi);
    Register dst = masm->rr().alloc_fixed(rdi);
    Register cnt = masm->rr().alloc_fixed(rcx);
    Register ofs = cnt;
    Register fidx = masm->rr().alloc();
    Register fcnt = masm->rr().alloc();
    Register embeddings = masm->rr().alloc();
    Register input = masm->rr().alloc();
    Register output = masm->rr().alloc();

    // Load tensor locations.
    __ LoadTensorAddress(embeddings, M);
    __ LoadTensorAddress(input, f);
    __ LoadTensorAddress(output, v);

    // Zero feature index and feature count.
    __ xorq(fidx, fidx);
    if (pooling_ == AVG) {
      __ xorq(fcnt, fcnt);
    }

    // Set up mask.
    OpmaskRegister mask = masm->kk().alloc();
    if (CPU::Enabled(AVX512F) && n % 16 != 0) {
      __ LoadMask(n % 16, mask);
    }

    // Find first (non-negative) feature.
    Label l1, l2, done;
    __ bind(&l1);
    __ movsxlq(acc, Operand(input, fidx, times_4));
    __ testq(acc, acc);
    __ j(positive, &l2);
    __ incq(fidx);
    __ cmpq(fidx, Immediate(f->elements()));
    __ j(less, &l1);

    // No feature found; zero output vector.
    __ xorq(acc, acc);
    __ movq(dst, output);
    __ movq(cnt, Immediate(v->size()));
    __ repstosb();
    __ jmp(&done);

    // First non-negative feature found; copy its embedding vector to output.
    __ bind(&l2);
    __ movq(src, embeddings);
    __ Multiply(acc, M->stride(0));
    __ addq(src, acc);
    __ movq(dst, output);
    __ movq(cnt, Immediate(M->stride(0)));
    __ repmovsb();
    if (pooling_ == AVG) {
      __ incq(fcnt);
    }

    // Go over the remaining features.
    Label l3, l4;
    __ bind(&l3);
    __ incq(fidx);
    __ cmpq(fidx, Immediate(f->elements()));
    __ j(equal, &l4);
    __ movsxlq(acc, Operand(input, fidx, times_4));
    __ testq(acc, acc);
    __ j(negative, &l4);

    // Combine embedding vector for feature with current result.
    if (pooling_ == AVG) {
      __ incq(fcnt);
    }
    __ movq(src, embeddings);
    __ Multiply(acc, M->stride(0));
    __ addq(src, acc);

    // Update output vector with embedding vector for feature.
    if (masm->Enabled(AVX512F)) {
      // Combine elements using AVX512 vectors.
      std::vector<ZMMRegister> elem;
      int main = (n / 16) * 16;
      int unrolls = AllocateZMMUnrolls(masm, main, 4, &elem);
      if (unrolls > 0) {
        Label next;
        __ xorq(ofs, ofs);
        __ bind(&next);
        for (int i = 0; i < unrolls; ++i) {
          int disp = i * 16 * sizeof(float);
          __ vmovaps(elem[i], Operand(src, ofs, times_1, disp));
        }
        for (int i = 0; i < unrolls; ++i) {
          int disp = i * 16 * sizeof(float);
          if (pooling_ == MAX) {
            __ vmaxps(elem[i], elem[i], Operand(output, ofs, times_1, disp));
          } else {
            __ vaddps(elem[i], elem[i], Operand(output, ofs, times_1, disp));
          }
        }
        for (int i = 0; i < unrolls; ++i) {
          int disp = i * 16 * sizeof(float);
          __ vmovaps(Operand(output, ofs, times_1, disp), elem[i]);
        }

        if (16 * unrolls > main) {
          __ addq(ofs, Immediate(8 * unrolls * sizeof(float)));
          __ cmpq(ofs, Immediate(main * sizeof(float)));
          __ j(less, &next);
        }
      }

      // Combine residual elements.
      if (n % 16 > 0) {
        int disp = main * sizeof(float);
        __ vmovaps(elem[0], Operand(src, disp), Mask(mask, zeroing));
        if (pooling_ == MAX) {
          __ vmaxps(elem[0], elem[0], Operand(output, disp),
                    Mask(mask, zeroing));
        } else {
          __ vaddps(elem[0], elem[0], Operand(output, disp),
                    Mask(mask, zeroing));
        }
        __ vmovaps(Operand(output, disp), elem[0], Mask(mask, merging));
      }
    } else if (masm->Enabled(AVX)) {
      // Combine elements using AVX vectors.
      std::vector<YMMRegister> elem;
      int main = (n / 8) * 8;
      int unrolls = AllocateYMMUnrolls(masm, main, 4, &elem);
      if (unrolls > 0) {
        Label next;
        __ xorq(ofs, ofs);
        __ bind(&next);
        for (int i = 0; i < unrolls; ++i) {
          int disp = i * 8 * sizeof(float);
          __ vmovaps(elem[i], Operand(src, ofs, times_1, disp));
        }
        for (int i = 0; i < unrolls; ++i) {
          int disp = i * 8 * sizeof(float);
          if (pooling_ == MAX) {
            __ vmaxps(elem[i], elem[i], Operand(output, ofs, times_1, disp));
          } else {
            __ vaddps(elem[i], elem[i], Operand(output, ofs, times_1, disp));
          }
        }
        for (int i = 0; i < unrolls; ++i) {
          int disp = i * 8 * sizeof(float);
          __ vmovaps(Operand(output, ofs, times_1, disp), elem[i]);
        }

        if (8 * unrolls > main) {
          __ addq(ofs, Immediate(8 * unrolls * sizeof(float)));
          __ cmpq(ofs, Immediate(main * sizeof(float)));
          __ j(less, &next);
        }
      }

      // Combine residual elements.
      int disp = main * sizeof(float);
      for (int i = 0; i < n % 8; ++i) {
        int r = i % std::max(unrolls, 1);
        __ vmovss(elem[r], Operand(src, disp));
        if (pooling_ == MAX) {
          __ vmaxss(elem[r], elem[r], Operand(output, disp));
        } else {
          __ vaddss(elem[r], elem[r], Operand(output, disp));
        }
        __ vmovss(Operand(output, disp), elem[r]);
        disp += sizeof(float);
      }
    } else {
      // Combine elements using SSE vectors.
      int main = (n / 4) * 4;
      XMMRegister elem = masm->mm().allocx();
      if (n >= 4) {
        Label next;
        __ xorq(ofs, ofs);
        __ bind(&next);
        __ movaps(elem, Operand(src, ofs));
        if (pooling_ == MAX) {
          __ maxps(elem, Operand(output, ofs));
        } else {
          __ addps(elem, Operand(output, ofs));
        }
        __ movaps(Operand(output, ofs), elem);
        __ addq(ofs, Immediate(4 * sizeof(float)));
        __ cmpq(ofs, Immediate(main * sizeof(float)));
        __ j(less, &next);
      }

      // Combine residual elements.
      int disp = main * sizeof(float);
      for (int i = 0; i < n % 4; ++i) {
        __ movss(elem, Operand(src, disp));
        if (pooling_ == MAX) {
          __ maxss(elem, Operand(output, disp));
        } else {
          __ addss(elem, Operand(output, disp));
        }
        __ movss(Operand(output, disp), elem);
        disp += sizeof(float);
      }
    }

    // Next feature.
    __ jmp(&l3);
    __ bind(&l4);

    // Compute average.
    if (pooling_ == AVG) {
      if (masm->Enabled(AVX512F)) {
        // Compute 1/fcnt.
        ZMMRegister scalar = masm->mm().allocz();
        __ vcvtqsi2ss(scalar.xmm(), scalar.xmm(), fcnt);
        __ vrcpss(scalar.xmm(), scalar.xmm(), scalar.xmm());
        __ vbroadcastss(scalar, scalar);

        // Multiply all output elements with scalar to get the average.
        std::vector<ZMMRegister> elem;
        int main = (n / 16) * 16;
        int unrolls = AllocateZMMUnrolls(masm, main, 4, &elem);
        if (unrolls > 0) {
          Label next;
          __ xorq(ofs, ofs);
          __ bind(&next);
          for (int i = 0; i < unrolls; ++i) {
            int disp = i * 16 * sizeof(float);
            __ vmulps(elem[i], scalar, Operand(output, ofs, times_1, disp));
          }
          for (int i = 0; i < unrolls; ++i) {
            int disp = i * 16 * sizeof(float);
            __ vmovaps(Operand(output, ofs, times_1, disp), elem[i]);
          }
          __ addq(ofs, Immediate(16 * unrolls * sizeof(float)));
          __ cmpq(ofs, Immediate(main * sizeof(float)));
          __ j(less, &next);
        }
        if (n % 16 > 0) {
          int disp = main * sizeof(float);
          __ vmulps(elem[0], scalar, Operand(output, disp),
                    Mask(mask, zeroing));
          __ vmovaps(Operand(output, disp), elem[0], Mask(mask, merging));
        }
      } else if (masm->Enabled(AVX)) {
        // Compute 1/fcnt.
        YMMRegister scalar = masm->mm().allocy();
        __ vcvtqsi2ss(scalar.xmm(), scalar.xmm(), fcnt);
        __ vrcpss(scalar.xmm(), scalar.xmm(), scalar.xmm());
        if (masm->Enabled(AVX2)) {
          __ vbroadcastss(scalar, scalar);
        } else {
          __ vshufps(scalar, scalar, scalar, 0);
          __ vperm2f128(scalar, scalar, scalar, 0);
        }

        // Multiply all output elements with scalar to get the average.
        std::vector<YMMRegister> elem;
        int main = (n / 8) * 8;
        int unrolls = AllocateYMMUnrolls(masm, main, 4, &elem);
        if (unrolls > 0) {
          Label next;
          __ xorq(ofs, ofs);
          __ bind(&next);
          for (int i = 0; i < unrolls; ++i) {
            int disp = i * 8 * sizeof(float);
            __ vmulps(elem[i], scalar, Operand(output, ofs, times_1, disp));
          }
          for (int i = 0; i < unrolls; ++i) {
            int disp = i * 8 * sizeof(float);
            __ vmovaps(Operand(output, ofs, times_1, disp), elem[i]);
          }
          __ addq(ofs, Immediate(8 * unrolls * sizeof(float)));
          __ cmpq(ofs, Immediate(main * sizeof(float)));
          __ j(less, &next);
        }
        int disp = main * sizeof(float);
        for (int i = 0; i < n % 8; ++i) {
          int r = i % std::max(unrolls, 1);
          __ vmulss(elem[r].xmm(), scalar.xmm(), Operand(output, disp));
          __ vmovss(Operand(output, disp), elem[r].xmm());
          disp += sizeof(float);
        }
      } else {
        // Compute 1/fcnt.
        XMMRegister scalar = masm->mm().allocx();
        __ cvtqsi2ss(scalar, fcnt);
        __ rcpss(scalar, scalar);

        // Multiply all output elements with scalar to get the average.
        XMMRegister elem = masm->mm().allocx();
        Label next;
        __ xorq(ofs, ofs);
        __ bind(&next);
        __ movss(elem, Operand(output, ofs));
        __ mulss(elem, scalar);
        __ movss(Operand(output, ofs), elem);
        __ addq(ofs, Immediate(sizeof(float)));
        __ cmpq(ofs, Immediate(v->size()));
        __ j(less, &next);
      }
    }

    __ bind(&done);
  }

  int64 Complexity(const Step *step) override {
    Tensor *M = step->input(0);
    Tensor *f = step->input(1);
    return M->dim(1) * f->elements() + (pooling_ == AVG ? M->dim(1) : 0);
  }

 private:
  Pooling pooling_;  // pooling operation for combining vectors
};

// Add sparse (scaled) input to variable.
class ScatterAdd : public Kernel {
 public:
  ScatterAdd(bool scale) : scale_(scale) {}

  string Name() override { return Operation(); }
  string Operation() override {
    return scale_ ? "ScatterMulAdd" : "ScatterAdd";
  }

  bool Supports(Step *step) override {
    // Requires SSE or AVX support.
    if (!CPU::Enabled(AVX) && !CPU::Enabled(SSE)) return false;

    // Check inputs and outputs.
    if (step->indegree() != (scale_ ? 4 : 3)) return false;
    if (step->outdegree() != 0) return false;

    // Check types.
    Tensor *var = step->input(0);
    Tensor *indices = step->input(1);
    Tensor *value = step->input(2);
    Tensor *scaler = scale_ ? step->input(3) : nullptr;
    if (var->type() != DT_FLOAT || var->rank() != 2) return false;
    if (var->constant()) return false;
    if (indices->type() != DT_INT32 || indices->rank() != 2) return false;
    if (value->type() != DT_FLOAT) return false;
    if (value->elements() != var->dim(1)) return false;
    if (scale_) {
      if (scaler->type() != DT_FLOAT) return false;
      if (scaler->elements() != 1) return false;
    }

    return true;
  }

  void Adjust(Step *step) override {
    Tensor *var = step->input(0);
    Tensor *value = step->input(2);

    // Align to one SIMD register.
    int align = 4;
    if (CPU::Enabled(AVX)) align = 8;
    if (CPU::Enabled(AVX512F)) align = 16;
    var->SetMiniumAlignment(align * sizeof(float));
    value->SetMiniumAlignment(align * sizeof(float));

    // Embedding matrix must be row-major.
    var->SetRequiredOrder(ROW_MAJOR);
    int minalign = 1;
    if (var->dim(1) >= 4) minalign = 4;
    if (CPU::Enabled(AVX) && var->dim(1) >= 8) minalign = 8;
    if (CPU::Enabled(AVX512F) && var->dim(1) >= 16) minalign = 16;
    var->MinAlign({1, minalign});
  }

  void Generate(Step *step, MacroAssembler *masm) override {
    // Get inputs.
    Tensor *var = step->input(0);
    Tensor *indices = step->input(1);
    Tensor *value = step->input(2);
    Tensor *scaler = scale_ ? step->input(3) : nullptr;
    bool single = indices->elements() == 1;
    int n = value->elements();

    // Allocate registers.
    Register acc = masm->rr().alloc();
    Register ofs = masm->rr().alloc();
    Register varaddr = masm->rr().alloc();
    Register idxaddr = masm->rr().alloc();
    Register valaddr = masm->rr().alloc();
    Register fidx = masm->rr().alloc();
    Register src = masm->rr().alloc();
    ZMMRegister factor = masm->mm().allocz(false);

    // Load tensor locations.
    __ LoadTensorAddress(varaddr, var);
    __ LoadTensorAddress(idxaddr, indices);
    __ LoadTensorAddress(valaddr, value);

    // Load scaling value.
    if (scaler) {
      __ LoadTensorAddress(src, scaler);
      if (masm->Enabled(AVX512F)) {
        __ vbroadcastss(factor, Operand(src));
      } else if (masm->Enabled(AVX)) {
        __ vbroadcastss(factor.ymm(), Operand(src));
      } else {
        __ movss(factor.xmm(), Operand(src));
        __ shufps(factor.xmm(), factor.xmm(), 0);
      }
    }

    // Set up mask.
    OpmaskRegister mask = masm->kk().alloc();
    if (CPU::Enabled(AVX512F) && n % 16 != 0) {
      __ LoadMask(n % 16, mask);
    }

    // Loop over features.
    if (!single) {
      __ xorq(fidx, fidx);
    }
    Label l1, l2;
    __ bind(&l1);
    if (single) {
      __ movsxlq(acc, Operand(idxaddr));
    } else {
      __ movsxlq(acc, Operand(idxaddr, fidx, times_4));
    }
    __ testq(acc, acc);
    __ j(negative, &l2);

    //  look up address of index in embedding
    __ Multiply(acc, var->stride(0));
    __ addq(acc, varaddr);

    // Add (scaled) input vector for feature to embedding vector.
    if (masm->Enabled(AVX512F)) {
      // Update elements using AVX-512 vectors.
      std::vector<ZMMRegister> elem;
      int main = (n / 16) * 16;
      int unrolls = AllocateZMMUnrolls(masm, main, 4, &elem);
      if (unrolls > 0) {
        Label next;
        __ xorq(ofs, ofs);
        __ bind(&next);
        for (int i = 0; i < unrolls; ++i) {
          int disp = i * 16 * sizeof(float);
          if (scale_) {
            __ vmulps(elem[i], factor, Operand(valaddr, ofs, times_1, disp));
          } else {
            __ vmovaps(elem[i], Operand(valaddr, ofs, times_1, disp));
          }
        }
        for (int i = 0; i < unrolls; ++i) {
          int disp = i * 16 * sizeof(float);
          __ vaddps(elem[i], elem[i], Operand(acc, ofs, times_1, disp));
        }
        for (int i = 0; i < unrolls; ++i) {
          int disp = i * 16 * sizeof(float);
          __ vmovaps(Operand(acc, ofs, times_1, disp), elem[i]);
        }
        if (16 * unrolls > main) {
          __ addq(ofs, Immediate(8 * unrolls * sizeof(float)));
          __ cmpq(ofs, Immediate(main * sizeof(float)));
          __ j(less, &next);
        }
      }

      // Update residual elements.
      if (n % 16 != 0) {
        int disp = main * sizeof(float);
        if (scale_) {
          __ vmulps(elem[0], factor, Operand(valaddr, disp),
                    Mask(mask, zeroing));
        } else {
          __ vmovups(elem[0], Operand(valaddr, disp), Mask(mask, zeroing));
        }
        __ vaddps(elem[0], elem[0], Operand(acc, disp),
                  Mask(mask, zeroing));
        __ vmovups(Operand(acc, disp), elem[0], Mask(mask, merging));
      }
    } else if (masm->Enabled(AVX)) {
      // Update elements using AVX vectors.
      std::vector<YMMRegister> elem;
      int main = (n / 8) * 8;
      int unrolls = AllocateYMMUnrolls(masm, main, 4, &elem);
      if (unrolls > 0) {
        Label next;
        __ xorq(ofs, ofs);
        __ bind(&next);
        for (int i = 0; i < unrolls; ++i) {
          int disp = i * 8 * sizeof(float);
          if (scale_) {
            __ vmulps(elem[i], factor.ymm(),
                      Operand(valaddr, ofs, times_1, disp));
          } else {
            __ vmovaps(elem[i], Operand(valaddr, ofs, times_1, disp));
          }
        }
        for (int i = 0; i < unrolls; ++i) {
          int disp = i * 8 * sizeof(float);
          __ vaddps(elem[i], elem[i], Operand(acc, ofs, times_1, disp));
        }
        for (int i = 0; i < unrolls; ++i) {
          int disp = i * 8 * sizeof(float);
          __ vmovaps(Operand(acc, ofs, times_1, disp), elem[i]);
        }
        if (8 * unrolls > main) {
          __ addq(ofs, Immediate(8 * unrolls * sizeof(float)));
          __ cmpq(ofs, Immediate(main * sizeof(float)));
          __ j(less, &next);
        }
      }

      // Update residual elements.
      int disp = main * sizeof(float);
      if (n % 8 >= 4) {
        if (scale_) {
          __ vmulps(elem[0].xmm(), factor.xmm(), Operand(valaddr, disp));
        } else {
          __ vmovaps(elem[0].xmm(), Operand(valaddr, disp));
        }
        __ vaddps(elem[0].xmm(), elem[0].xmm(), Operand(acc, disp));
        __ vmovaps(Operand(acc, disp), elem[0].xmm());
        disp += 4 * sizeof(float);
      }
      for (int i = 0; i < n % 4; ++i) {
        int r = i % std::max(unrolls, 1);
        if (scale_) {
          __ vmulss(elem[r].xmm(), factor.xmm(), Operand(valaddr, disp));
        } else {
          __ vmovss(elem[r].xmm(), Operand(valaddr, disp));
        }
        __ vaddss(elem[r].xmm(), elem[r].xmm(), Operand(acc, disp));
        __ vmovss(Operand(acc, disp), elem[r].xmm());
        disp += sizeof(float);
      }
    } else {
      // Update elements using SSE vectors.
      XMMRegister elem = masm->mm().allocx();
      int main = (n / 4) * 4;
      if (n >= 4) {
        Label next;
        __ xorq(ofs, ofs);
        __ bind(&next);
        __ movaps(elem, Operand(valaddr, ofs));
        if (scale_) {
          __ mulps(elem, factor.xmm());
        }
        __ addps(elem, Operand(acc, ofs));
        __ movaps(Operand(acc, ofs), elem);
        __ addq(ofs, Immediate(4 * sizeof(float)));
        __ cmpq(ofs, Immediate(main * sizeof(float)));
        __ j(less, &next);
      }

      // Update residual elements.
      int disp = main * sizeof(float);
      for (int i = 0; i < n % 4; ++i) {
        __ movss(elem, Operand(valaddr, disp));
        if (scale_) {
          __ mulss(elem, factor.xmm());
        }
        __ addss(elem, Operand(acc, disp));
        __ movss(Operand(acc, disp), elem);
        disp += sizeof(float);
      }
    }

    if (!single) {
      __ incq(fidx);
      __ cmpq(fidx, Immediate(indices->elements()));
      __ j(less, &l1);
    }
    __ bind(&l2);
  }

  int64 Complexity(const Step *step) override {
    Tensor *indices = step->input(1);
    Tensor *value = step->input(2);
    return value->elements() * indices->elements() * (scale_ ? 2 : 1);
  }

 private:
  bool scale_;  // scale input
};

// Fold scaling or outer product into update ops.
class UpdateTransformer : public Transformer {
 public:
  bool Transform(Flow *flow) override {
    int updates = 0;

    // Transform outer product update.
    for (Flow::Operation *op : flow->Find("MatMul|1:Add|1:Assign")) {
      Flow::Operation *assign = op;
      Flow::Operation *add = assign->inputs[1]->producer;
      Flow::Operation *matmul = add->inputs[1]->producer;
      if (assign->inputs[0] != add->inputs[0]) continue;
      if (add->outputs[0]->usages() != 1) continue;
      if (matmul->outputs[0]->usages() != 1) continue;

      // Only fuse if matrix multiplication is an outer product.
      if (matmul->inputs.size() != 2) continue;
      Shape a = matmul->inputs[0]->shape;
      Shape b = matmul->inputs[1]->shape;
      if (a.rank() != 2 || b.rank() != 2) continue;
      if (matmul->GetAttr("transpose_a", false)) a = a.transpose();
      if (matmul->GetAttr("transpose_b", false)) b = b.transpose();
      if (a.dim(1) != 1 || b.dim(0) != 1) continue;

      flow->Fuse(assign, flow->Fuse(add, matmul, ""), "AssignAddMatMul", true);
      updates++;
    }

    // Transform sparse update.
    for (Flow::Operation *op : flow->Find("Scatter|1:Add|1:Assign")) {
      Flow::Operation *assign = op;
      Flow::Operation *add = assign->inputs[1]->producer;
      Flow::Operation *scatter = add->inputs[1]->producer;
      if (assign->inputs[0] != add->inputs[0]) continue;
      if (add->outputs[0]->usages() != 1) continue;
      if (scatter->outputs[0]->usages() != 1) continue;

      flow->Fuse(assign, flow->Fuse(add, scatter, ""), "ScatterAdd", true);
      updates++;
    }

    // Transform sparse update scaling.
    for (Flow::Operation *op : flow->Find("Mul|2:ScatterAdd")) {
      Flow::Operation *scatter = op;
      Flow::Operation *mul = scatter->inputs[2]->producer;
      if (scatter->indegree() != 3) continue;
      if (mul->outputs[0]->usages() != 1) continue;
      flow->Fuse(scatter, mul, "ScatterMulAdd");
      updates++;
    }

    return updates > 0;
  }
};

// Propagate tensor references across reshapes.
class ReshapeRefTransformer : public Transformer {
 public:
  bool Transform(Flow *flow) override {
    bool updated = false;
    for (Flow::Operation *op : flow->ops()) {
      if (op->type != "Reshape") continue;
      if (op->indegree() != 2 || op->outdegree() != 1) return false;
      if (op->inputs[0]->ref() && !op->outputs[0]->ref()) {
        op->outputs[0]->set_ref();
        updated = true;
      }
      if (op->outputs[0]->ref() && !op->inputs[0]->ref()) {
        op->inputs[0]->set_ref();
        updated = true;
      }
    }

    return updated;
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
  library->Register(new OneHot());
  library->Register(new GeneralConcat());
  library->Register(new BasicConcat());
  library->Register(new Slice());
  library->Register(new MultiGather());
  library->Register(new SingleGather());
  library->Register(new PoolingGather(PoolingGather::SUM));
  library->Register(new PoolingGather(PoolingGather::AVG));
  library->Register(new PoolingGather(PoolingGather::MAX));
  library->Register(new ScatterAdd(false));
  library->Register(new ScatterAdd(true));

  library->RegisterTransformer(new UpdateTransformer());
  library->RegisterTransformer(new ReshapeRefTransformer());
}

}  // namespace myelin
}  // namespace sling
