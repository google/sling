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

#include <set>

#include "sling/myelin/compute.h"
#include "sling/myelin/macro-assembler.h"
#include "sling/myelin/simd-assembler.h"

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
    CHECK(step->AllowInPlace(0, 0, true)) << step->name();
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
    if (step->indegree() != 1 || step->outdegree() != 1) return false;
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
    if (step->indegree() != 1 && step->indegree() != 2) return false;
    if (step->outdegree() != 1) return false;
    Tensor *index = step->input(0);
    Tensor *value = step->indegree() > 1 ? step->input(1) : nullptr;
    Tensor *onehot = step->output(0);
    if (index->type() != DT_INT32) return false;
    if (index->elements() != 1) return false;
    if (onehot->type() != DT_FLOAT) return false;
    if (value != nullptr) {
      if (value->type() != DT_FLOAT) return false;
      if (value->elements() != 1) return false;
    }
    return true;
  }

  void Generate(Step *step, MacroAssembler *masm) override {
    Tensor *index = step->input(0);
    Tensor *value = step->indegree() > 1 ? step->input(1) : nullptr;
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
    if (value != nullptr) {
      Register vaddr = masm->rr().alloc();
      __ LoadTensorAddress(vaddr, value);
      XMMRegister v = masm->mm().allocx();
      __ movss(v, Operand(vaddr));
      __ movss(Operand(output, acc, times_4), v);
    } else {
      __ movq(Operand(output, acc, times_4), Immediate(0x3F800000));
    }
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

    // Get source and destination addresses.
    __ LoadTensorAddress(src, source, begin);
    __ LoadTensorAddress(dst, destination);

    // Copy input to output.
    __ Copy(dst, 0, src, 0, bytes);
  }

  int64 Complexity(const Step *step) override {
    return 0;
  }
};

// Output concatenation of input tensors along first dimension.
class BasicConcat : public Kernel {
 public:
  string Name() override { return "BasicConcat"; }
  string Operation() override { return "Concat"; }

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
    if (step->output(0)->dynamic()) return false;

    return true;
  }

  void Generate(Step *step, MacroAssembler *masm) override {
    // Get the number of tensors to concatenate.
    int n = step->GetAttr("N", step->indegree() - 1);

    // Allocate registers.
    Register src = masm->rr().alloc_preferred(rsi);
    Register dst = masm->rr().alloc_preferred(rdi);
    Register out = masm->rr().alloc_preferred(rdx);

    // Load output tensor.
    __ LoadTensorAddress(out, step->output(0));

    // Copy input tensors to output.
    int offset = 0;
    for (int i = 0; i < n; ++i) {
      int size = step->input(i)->size();
        __ LoadTensorAddress(src, step->input(i));
        __ leaq(dst, Operand(out, offset));
      __ Copy(dst, 0, src, 0, size);
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
  string Operation() override { return "Concat"; }

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
      if (input->dynamic() != output->dynamic()) return false;
    }

    return true;
  }

  void Generate(Step *step, MacroAssembler *masm) override {
    // Get the number of tensors to concatenate.
    int n = step->GetAttr("N", step->indegree() - 1);
    Tensor *output = step->output(0);

    // Allocate registers.
    Register src = masm->rr().alloc_preferred(rsi);
    Register dst = masm->rr().alloc_preferred(rdi);
    Register cnt = masm->rr().alloc_preferred(rcx);
    Register idx = masm->rr().alloc();
    std::vector<Register> in(n);
    for (int i = 0; i < n; ++i) in[i] = masm->rr().alloc();

    // Load input tensors.
    for (int i = 0; i < n; ++i) {
      __ LoadTensorAddress(in[i], step->input(i));
    }

    // Load output tensor.
    __ LoadTensorAddress(dst, output);

    // Loop over outer prefix.
    Label l;
    int axis = step->input(n)->value<int32>();
    int repeat = output->shape().outer(axis);
    if (output->dynamic()) {
      __ LoadDynamicSize(idx, output, repeat);
      step->set_variant("DYN");
    } else {
      __ movq(idx, Immediate(repeat));
    }
    __ bind(&l);

    // Copy input tensors to output.
    int copied = 0;
    for (int i = 0; i < n; ++i) {
      Tensor *input = step->input(i);
      int size = input->AxisSize(axis);
      __ movq(src, in[i]);
      __ movq(cnt, Immediate(size));
      __ repmovsb();
      __ addq(in[i], Immediate(size));
      copied += size;
    }

    // Next chunk.
    int size = output->AxisSize(axis);
    if (copied != size) {
      __ addq(dst, Immediate(size - copied));
    }
    __ decq(idx);
    __ j(not_zero, &l);
  }

  int64 Complexity(const Step *step) override {
    return 0;
  }
};

// Split input tensors into chunks along a dimension.
class Split : public Kernel {
 public:
  string Name() override { return "Split"; }
  string Operation() override { return "Split"; }

  bool Supports(Step *step) override {
    // Check inputs and outputs.
    if (step->indegree() != 3) return false;

    // Only constant number of splits along a singular prefix supported.
    Tensor *input = step->input(0);
    Tensor *splits = step->input(1);
    Tensor *axis = step->input(2);

    // Check splits.
    if (splits->type() != DT_INT32 || !splits->constant()) return false;
    int n = splits->value<int32>();
    if (n != step->outdegree()) return false;

    // Check axis.
    if (axis->type() != DT_INT32 || !axis->constant()) return false;
    int a = axis->value<int32>();
    if (a > input->rank() - 1) return false;

    // Check that outputs match the input.
    Type dt = input->type();
    int size = input->shape().inner(a);
    if (size % n != 0) return false;
    for (int i = 0; i < n; ++i) {
      Tensor *output = step->output(i);
      if (output->type() != dt) return false;
      if (output->rank() != input->rank()) return false;
      if (output->shape().inner(a) != size / n) return false;
      if (output->dynamic() != input->dynamic()) return false;
    }
    return true;
  }

  void Generate(Step *step, MacroAssembler *masm) override {
    // Get input.
    Tensor *input = step->input(0);
    int n = step->input(1)->value<int32>();
    int axis = step->input(2)->value<int32>();
    int repeat = input->shape().outer(axis);

    // Allocate registers.
    Register src = masm->rr().alloc_preferred(rsi);
    Register dst = masm->rr().alloc_preferred(rdi);
    Register cnt = masm->rr().alloc_preferred(rcx);
    Register idx = masm->rr().alloc_preferred(rcx);

    // Load input tensor.
    __ LoadTensorAddress(src, input);

    if (input->dynamic() || repeat > 1) {
      // Load output tensors.
      step->set_variant("REP");
      std::vector<Register> out(n);
      for (int i = 0; i < n; ++i) {
        out[i] = masm->rr().alloc();
        __ LoadTensorAddress(out[i], step->output(i));
      }

      // Loop over outer prefix.
      Label l;
      if (input->dynamic()) {
        __ LoadDynamicSize(idx, input, repeat);
        step->set_variant("DYN");
      } else {
        __ movq(idx, Immediate(repeat));
      }
      __ bind(&l);

      // Split input to output.
      int copied = 0;
      for (int i = 0; i < n; ++i) {
        Tensor *output = step->output(i);
        int size = output->AxisSize(axis);
        __ movq(dst, out[i]);
        __ movq(cnt, Immediate(size));
        __ repmovsb();
        __ addq(out[i], Immediate(size));
        copied += size;
      }

      // Next chunk.
      int size = input->AxisSize(axis);
      if (copied != size) {
        __ addq(src, Immediate(size - copied));
      }
      __ decq(idx);
      __ j(not_zero, &l);
    } else {
      // Simple non-repeated split.
      for (int i = 0; i < n; ++i) {
        int size = step->output(i)->AxisSize(axis);
        __ LoadTensorAddress(dst, step->output(i));
        __ movq(cnt, Immediate(size));
        __ repmovsb();
      }
    }
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
    if (step->indegree() != 2 && step->indegree() != 3) return false;
    if (step->outdegree() != 1) return false;

    // Check types.
    Tensor *M = step->input(0);
    Tensor *f = step->input(1);
    Tensor *oov = step->indegree() == 3 ? step->input(2) : nullptr;
    Tensor *v = step->output(0);
    Type type = M->type();
    if (f->type() != DT_INT32) return false;
    if (M->rank() != 2) return false;
    if (v->type() != type) return false;
    if (oov != nullptr && oov->type() != type) return false;
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

  void Generate(Step *step, MacroAssembler *masm) override {
    // Get inputs and outputs.
    Tensor *M = step->input(0);
    Tensor *f = step->input(1);
    Tensor *oov = step->indegree() == 3 ? step->input(2) : nullptr;
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

    // Check for OOV feature.
    Label l1;
    if (oov != nullptr) {
      __ testq(acc, acc);
      __ j(negative, &l1);
    }

    // Compute offset in embedding.
    __ Multiply(acc, M->stride(0));

    // Lookup element in embedding.
    __ LoadTensorAddress(embeddings, M);
    __ addq(acc, embeddings);

    // Use oov vector for negative features.
    if (oov != nullptr) {
      Label l2;
      __ jmp(&l2);
      __ bind(&l1);
      __ LoadTensorAddress(acc, oov);
      __ bind(&l2);
    }

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
    if (step->indegree() != 2 && step->indegree() != 3) return false;
    if (step->outdegree() != 1) return false;

    // Check types.
    Tensor *M = step->input(0);
    Tensor *f = step->input(1);
    Tensor *oov = step->indegree() == 3 ? step->input(2) : nullptr;
    Tensor *v = step->output(0);
    Type type = M->type();
    if (f->type() != DT_INT32) return false;
    if (M->rank() != 2) return false;
    if (v->type() != type) return false;
    if (oov != nullptr && oov->type() != type) return false;
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

  void Generate(Step *step, MacroAssembler *masm) override {
    // Get inputs and outputs.
    Tensor *M = step->input(0);
    Tensor *f = step->input(1);
    Tensor *oov = step->indegree() == 3 ? step->input(2) : nullptr;
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

    // Check for OOV feature.
    Label l1;
    if (oov != nullptr) {
      __ testq(acc, acc);
      __ j(negative, &l1);
    }

    // Compute address in embedding.
    __ movq(src, embeddings);
    __ Multiply(acc, M->stride(0));
    __ addq(src, acc);

    // Use oov vector for negative features.
    if (oov != nullptr) {
      Label l2;
      __ jmp(&l2);
      __ bind(&l1);
      __ LoadTensorAddress(src, oov);
      __ bind(&l2);
    }

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
    // Check inputs and outputs.
    if (step->indegree() != 2 || step->outdegree() != 1) return false;

    // Check types.
    Tensor *M = step->input(0);
    Tensor *f = step->input(1);
    Tensor *v = step->output(0);
    if (!SIMDAssembler::Supports(M->type()) || M->rank() != 2) return false;
    if (f->type() != DT_INT32 || f->rank() != 2) return false;
    if (v->type() != M->type() || v->elements() != M->dim(1)) return false;
    if (pooling_ == AVG) {
      if (M->type() != DT_FLOAT && M->type() != DT_DOUBLE) return false;
      if (!CPU::Enabled(SSE2)) return false;
    }

    return true;
  }

  void Adjust(Step *step) override {
    Tensor *M = step->input(0);
    Tensor *v = step->output(0);

    // Align to one vector register.
    Type type = M->type();
    int vecbytes = SIMDAssembler::VectorBytes(type);
    M->SetMiniumAlignment(vecbytes);
    v->SetMiniumAlignment(vecbytes);

    // Embedding matrix must be row-major.
    M->RequireOrder(ROW_MAJOR);

    // Reserve registers.
    int regs = SIMDAssembler::RegisterUsage(type) + 9;
    step->SetRegisterUsage(regs);
  }

  void Generate(Step *step, MacroAssembler *masm) override {
    // Get inputs and outputs.
    Tensor *M = step->input(0);
    Tensor *f = step->input(1);
    Tensor *v = step->output(0);
    int n = v->elements();

    // Create SIMD code generators.
    Type type = M->type();
    int dsize = TypeTraits::of(type).size();
    int vecbytes = SIMDAssembler::VectorBytes(type);
    bool aligned = M->stride(0) % vecbytes == 0;
    SIMDAssembler sasm(masm, type, aligned);
    step->set_variant(sasm.name());

    // Compute vector processing strategy.
    SIMDStrategy strategy(&sasm, n);
    strategy.PreloadMasks();

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
    auto elem = sasm.alloc(strategy.MaxUnrolls());

    // Load tensor locations.
    __ LoadTensorAddress(embeddings, M);
    __ LoadTensorAddress(input, f);
    __ LoadTensorAddress(output, v);

    // Zero feature index and feature count.
    __ xorq(fidx, fidx);
    if (pooling_ == AVG) {
      __ xorq(fcnt, fcnt);
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
    Reduction op = pooling_ == MAX ? REDUCE_MAX : REDUCE_ADD;
    for (auto &phase : strategy.phases()) {
      auto *gen = phase.generator;
      int vecsize = gen->VectorSize();
      int blkstart = phase.offset * dsize;
      int blksize = phase.unrolls * vecsize * dsize;

      if (phase.repeat > 1) {
        // Repeated phase.
        Label lu;
        if (blkstart == 0) {
          __ xorq(ofs, ofs);
        } else {
          __ movq(ofs, Immediate(blkstart));
        }
        __ bind(&lu);
        for (int i = 0; i < phase.unrolls; ++i) {
          int disp = i * vecsize * dsize;
          gen->Load(elem[i], Operand(src, ofs, times_1, disp));
          gen->Accumulate(op, elem[i], Operand(output, ofs, times_1, disp));
          gen->Store(Operand(output, ofs, times_1, disp), elem[i]);
        }
        __ addq(ofs, Immediate(blksize));
        __ cmpq(ofs, Immediate(blkstart + phase.repeat * blksize));
        __ j(less, &lu);
      } else if (phase.masked == 0) {
        // Residual phase.
        for (int i = 0; i < phase.unrolls; ++i) {
          int disp = blkstart + i * vecsize * dsize;
          gen->Load(elem[i], Operand(src, disp));
          gen->Accumulate(op, elem[i], Operand(output, disp));
          gen->Store(Operand(output, disp), elem[i]);
        }
      } else {
        // Masked phase.
        CHECK_EQ(phase.unrolls, 1);
        gen->MaskedLoad(elem[0], Operand(src, blkstart));
        gen->MaskedAccumulate(op, elem[0], Operand(output, blkstart));
        gen->MaskedStore(Operand(output, blkstart), elem[0]);
      }
    }

    // Next feature.
    __ jmp(&l3);
    __ bind(&l4);

    // Compute average.
    if (pooling_ == AVG) {
      // Compute 1/fcnt.
      int scalar = sasm.alloc();
      XMMRegister sr = jit::XMMRegister::from_code(scalar);
      if (masm->Enabled(AVX)) {
        __ vcvtqsi2ss(sr, sr, fcnt);
        __ vrcpss(sr, sr, sr);
        if (type == DT_DOUBLE) {
          __ vcvtss2sd(sr, sr, sr);
        }
      } else {
        __ cvtqsi2ss(sr, fcnt);
        __ rcpss(sr, sr);
        if (type == DT_DOUBLE) {
          CHECK(masm->Enabled(SSE2));
          __ cvtss2sd(sr, sr);
        }
      }
      sasm.main()->Broadcast(scalar, scalar);

      // Multiply all output elements with scalar to get the average.
      for (auto &phase : strategy.phases()) {
        auto *gen = phase.generator;
        int vecsize = gen->VectorSize();
        int blkstart = phase.offset * dsize;
        int blksize = phase.unrolls * vecsize * dsize;

        if (phase.repeat > 1) {
          // Repeated phase.
          Label lu;
          if (blkstart == 0) {
            __ xorq(ofs, ofs);
          } else {
            __ movq(ofs, Immediate(blkstart));
          }
          __ bind(&lu);
          for (int i = 0; i < phase.unrolls; ++i) {
            int disp = i * vecsize * dsize;
            gen->Mul(elem[i], scalar, Operand(output, ofs, times_1, disp));
            gen->Store(Operand(output, ofs, times_1, disp), elem[i]);
          }
          __ addq(ofs, Immediate(blksize));
          __ cmpq(ofs, Immediate(blkstart + phase.repeat * blksize));
          __ j(less, &lu);
        } else if (phase.masked == 0) {
          // Residual phase.
          for (int i = 0; i < phase.unrolls; ++i) {
            int disp = blkstart + i * vecsize * dsize;
            gen->Mul(elem[i], scalar, Operand(output, disp));
            gen->Store(Operand(output, disp), elem[i]);
          }
        } else {
          // Masked phase.
          CHECK_EQ(phase.unrolls, 1);
          gen->MaskedMul(elem[0], scalar, Operand(output, blkstart));
          gen->MaskedStore(Operand(output, blkstart), elem[0]);
        }
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

// Accumulate sparse (scaled) input.
class AssignAddScatter : public Kernel {
 public:
  AssignAddScatter(bool scale) : scale_(scale) {}

  string Name() override { return Operation(); }
  string Operation() override {
    return scale_ ? "AssignAddMulScatter" : "AssignAddScatter";
  }

  bool Supports(Step *step) override {
    // Check inputs and outputs.
    Args args(step, scale_);
    if (!args.valid) return false;

    // Check arguments.
    Type type = args.var->type();
    if (!SIMDAssembler::Supports(type)) return false;
    if (args.var->rank() != 2) return false;
    if (args.var->constant()) return false;
    if (args.indices->type() != DT_INT32) return false;
    if (args.indices->rank() != 2) return false;
    if (args.value->type() != type || args.value->rank() != 2) return false;
    if (args.value->dim(1) != args.var->dim(1)) return false;
    if (args.value->dim(0) != 1 &&
        args.value->dim(0) != args.indices->dim(1)) {
      return false;
    }
    if (scale_) {
      if (args.scaler->type() != type) return false;
      if (args.scaler->elements() != 1) return false;
    }
    if (args.ref) {
      if (args.ref->type() != type) return false;
      if (args.ref->shape() != args.var->shape()) return false;
      if (!args.ref->ref()) return false;
    }

    return true;
  }

  void Adjust(Step *step, const Options &options) override {
    Args args(step, scale_);

    // Add sparsity bitmap index.
    if (options.sparse_threshold > 0 &&
        args.var->dim(0) >= options.sparse_threshold &&
        step->GetAttr("sparse", true)) {
      Tensor *sparse = args.var->MakeSparse();
      if (args.ref) args.ref->set_sparse(sparse);
    }

    // Link output reference to input variable.
    if (args.ref) args.var->Link(args.ref);

    // Align to one vector register.
    Type type = args.var->type();
    int vecbytes = SIMDAssembler::VectorBytes(type);
    args.var->SetMiniumAlignment(vecbytes);
    args.value->SetMiniumAlignment(vecbytes);

    // Embedding matrix must be row-major.
    args.var->RequireOrder(ROW_MAJOR);

    // Reserve registers.
    int regs = SIMDAssembler::RegisterUsage(type) + 8;
    if (args.scaler) regs++;
    step->SetRegisterUsage(regs);
  }

  void Generate(Step *step, MacroAssembler *masm) override {
    // Get inputs.
    Args args(step, scale_);
    Tensor *sparse = args.var->sparse();
    bool single = args.indices->elements() == 1;
    int n = args.value->dim(1);

    // Create SIMD code generators.
    Type type = args.var->type();
    int dsize = TypeTraits::of(type).size();
    int vecbytes = SIMDAssembler::VectorBytes(type);
    bool aligned = args.var->stride(0) % vecbytes == 0;
    SIMDAssembler sasm(masm, type, aligned);
    step->set_variant(sasm.name());

    // Compute vector processing strategy.
    SIMDStrategy strategy(&sasm, n);
    strategy.PreloadMasks();

    // Allocate registers.
    Register bit = masm->rr().alloc_fixed(rcx);
    Register acc = masm->rr().alloc();
    Register varaddr = masm->rr().alloc();
    Register idxaddr = masm->rr().alloc();
    Register valaddr = masm->rr().alloc();
    Register bmaddr = masm->rr().alloc();
    Register fidx = masm->rr().alloc();
    Register ofs = masm->rr().alloc();
    Register src = bit;
    Register aux = ofs;
    auto elem = sasm.alloc(strategy.MaxUnrolls());
    int factor = args.scaler ? sasm.alloc() : -1;

    // Load tensor locations.
    __ LoadTensorAddress(varaddr, args.var);
    __ LoadTensorAddress(idxaddr, args.indices);
    __ LoadTensorAddress(valaddr, args.value);
    if (sparse) {
      __ LoadTensorAddress(bmaddr, sparse);
    }

    // Optionally output reference to assigned variable.
    if (args.ref != nullptr) {
      CHECK(args.ref->IsLocal());
      CHECK(args.ref->ref());
      __ movq(Operand(masm->instance(), args.ref->offset()), varaddr);
    }

    // Load scaling value.
    if (args.scaler) {
      __ LoadTensorAddress(src, args.scaler);
      sasm.main()->Broadcast(factor, Operand(src));
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

    // Update sparsity bitmap.
    if (sparse) {
      __ movq(bit, acc);
      __ movq(aux, Immediate(1));
      __ shlq_cl(aux);
      __ shrq(bit, Immediate(6));
      __ orq(Operand(bmaddr, bit, times_8), aux);
    }

    //  Look up address of index in embedding.
    __ Multiply(acc, args.var->stride(0));
    __ addq(acc, varaddr);

    // Update OOV vector for missing features.
    if (args.oov) {
      Label l3;
      __ jmp(&l3);
      __ bind(&l2);
      __ LoadTensorAddress(acc, args.oov);
      __  bind(&l3);
    }

    // Add (scaled) input vector for feature to embedding vector.
    for (auto &phase : strategy.phases()) {
      auto *gen = phase.generator;
      int vecsize = gen->VectorSize();
      int blkstart = phase.offset * dsize;
      int blksize = phase.unrolls * vecsize * dsize;

      if (phase.repeat > 1) {
        // Repeated phase.
        Label lu;
        if (blkstart == 0) {
          __ xorq(ofs, ofs);
        } else {
          __ movq(ofs, Immediate(blkstart));
        }
        __ bind(&lu);
        for (int i = 0; i < phase.unrolls; ++i) {
          int disp = i * vecsize * dsize;
          gen->Load(elem[i], Operand(acc, ofs, times_1, disp));
          if (scale_) {
            gen->MulAdd(elem[i], factor, Operand(valaddr, ofs, times_1, disp),
                        true);
          } else {
            gen->Add(elem[i], elem[i], Operand(valaddr, ofs, times_1, disp));
          }
          gen->Store(Operand(acc, ofs, times_1, disp), elem[i]);
        }
        __ addq(ofs, Immediate(blksize));
        __ cmpq(ofs, Immediate(blkstart + phase.repeat * blksize));
        __ j(less, &lu);
      } else if (phase.masked == 0) {
        // Residual phase.
        for (int i = 0; i < phase.unrolls; ++i) {
          int disp = blkstart + i * vecsize * dsize;
          gen->Load(elem[i], Operand(acc, disp));
          if (scale_) {
            gen->MulAdd(elem[i], factor, Operand(valaddr, disp), true);
          } else {
            gen->Add(elem[i], elem[i], Operand(valaddr, disp));
          }
          gen->Store(Operand(acc, disp), elem[i]);
        }
      } else {
        // Masked phase.
        CHECK_EQ(phase.unrolls, 1);
        gen->MaskedLoad(elem[0], Operand(acc, blkstart));
        if (scale_) {
          gen->MaskedMulAdd(elem[0], factor, Operand(valaddr, blkstart));
        } else {
          gen->MaskedAdd(elem[0], elem[0], Operand(valaddr, blkstart));
        }
        gen->MaskedStore(Operand(acc, blkstart), elem[0]);
      }
    }

    if (args.value->dim(0) != 1) {
      __ addq(valaddr, Immediate(args.value->stride(0)));
    }

    if (!single) {
      __ incq(fidx);
      __ cmpq(fidx, Immediate(args.indices->elements()));
      __ j(less, &l1);
    }
    if (args.oov == nullptr) {
      __ bind(&l2);
    }
  }

  int64 Complexity(const Step *step) override {
    Tensor *indices = step->input(1);
    Tensor *value = step->input(2);
    return value->elements() * indices->elements() * (scale_ ? 2 : 1);
  }

 private:
  // Arguments to scatter op.
  struct Args {
    Args(Step *step, bool scale) {
      if (step->indegree() < 3) return;
      if (step->outdegree() > 1) return;
      var = step->input(0);
      indices = step->input(1);
      value = step->input(2);
      if (step->outdegree() > 0) ref = step->output(0);

      if (scale) {
        if (step->indegree() != 4 && step->indegree() != 5) return;
        if (step->indegree() > 3) scaler = step->input(3);
        if (step->indegree() > 4) oov = step->input(4);
      } else {
        if (step->indegree() != 3 && step->indegree() != 4) return;
        if (step->indegree() > 3) oov = step->input(3);
      }
      valid = true;
    }

    bool valid = false;
    Tensor *var = nullptr;
    Tensor *indices = nullptr;
    Tensor *value = nullptr;
    Tensor *scaler = nullptr;
    Tensor *ref = nullptr;
    Tensor *oov = nullptr;
  };

  bool scale_;  // scale input
};

// Reduction over an axis.
class Reduce : public Kernel {
 public:
  Reduce(const string &name, Reduction op) : name_(name), op_(op) {}

  string Name() override { return name_; }
  string Operation() override { return name_; }

  bool Supports(Step *step) override {
    // Check inputs and outputs.
    if (step->indegree() != 1 || step->outdegree() != 1) return false;
    Tensor *x = step->input(0);
    Tensor *y = step->output(0);

    // Check type.
    if (x->type() != y->type()) return false;
    if (!SIMDAssembler::Supports(x->type())) return false;

    // Check shape.
    int axis = step->GetAttr("axis", -1);
    bool keepdims = step->GetAttr("keepdims", false);
    if (axis < 0 || axis >= x->rank()) return false;
    if (x->shape().reduced(axis, keepdims) != y->shape()) return false;

    return true;
  }

  void Adjust(Step *step) override {
    // Get input and output.
    Tensor *x = step->input(0);
    Tensor *y = step->output(0);

    // Require dense standard layout.
    x->RequireStandardOrder();
    y->RequireStandardOrder();
    x->RequireDense();
    y->RequireDense();

    // Set alignment.
    Type type = x->type();
    int vecbytes = SIMDAssembler::VectorBytes(type);
    x->SetMiniumAlignment(vecbytes);
    y->SetMiniumAlignment(vecbytes);
  }

  void Generate(Step *step, MacroAssembler *masm) override {
    // Get input and output.
    Tensor *x = step->input(0);
    Tensor *y = step->output(0);
    int axis = step->GetAttr("axis", -1);

    // Compute dimensions.
    Type type = x->type();
    int dsize = TypeTraits::of(type).size();
    int vecbytes = SIMDAssembler::VectorBytes(type);

    int outer_size = x->shape().outer(axis);
    int reduction_size = x->dim(axis);
    int inner_size = x->shape().inner(axis + 1);

    // Allocate registers.
    Register in = masm->rr().alloc();
    Register out = masm->rr().alloc();
    Register ofs = masm->rr().alloc();

    // Load tensor addresses.
    __ LoadTensorAddress(in, x);
    __ LoadTensorAddress(out, y);

    // Reduction over the last axis is done using horizontal reduction whereas
    // reduction over other axes is done using vertical reduction.
    if (inner_size == 1) {
      // Create SIMD code generators.
      bool aligned = x->stride(axis - 1) % vecbytes == 0;
      SIMDAssembler sasm(masm, type, aligned);

      // Compute vector processing strategy.
      step->set_variant(sasm.name() + "H");
      SIMDStrategy strategy(&sasm, reduction_size);
      strategy.PreloadMasks();

      // Loop over batches.
      Register batch = masm->rr().alloc();
      Label lb;
      if (outer_size > 1) {
        __ xorq(batch, batch);
        __ bind(&lb);
      }

      // Initialize reduction with neutral element.
      auto acc = sasm.alloc(strategy.MaxUnrolls());
      for (auto r : acc) sasm.main()->LoadNeutral(op_, r);

      // Reduce inner vector.
      bool scalar = true;
      for (auto &phase : strategy.phases()) {
        auto *gen = phase.generator;
        int vecsize = gen->VectorSize();
        int blkstart = phase.offset * dsize;
        int blksize = phase.unrolls * vecsize * dsize;
        if (vecsize > 1) scalar = false;

        if (phase.repeat > 1) {
          // Repeated phase.
          Label lu;
          if (blkstart == 0) {
            __ xorq(ofs, ofs);
          } else {
            __ movq(ofs, Immediate(blkstart));
          }
          __ bind(&lu);
          for (int i = 0; i < phase.unrolls; ++i) {
            int disp = i * vecsize * dsize;
            gen->Accumulate(op_, acc[i], Operand(in, ofs, times_1, disp));
          }
          __ addq(ofs, Immediate(blksize));
          __ cmpq(ofs, Immediate(blkstart + phase.repeat * blksize));
          __ j(less, &lu);
        } else if (phase.masked == 0) {
          // Residual phase.
          if (phase.offset == 0 || vecsize == sasm.main()->VectorSize()) {
            // Same vector size as bulk; unroll directly into accumulators.
            for (int i = 0; i < phase.unrolls; ++i) {
              int disp = blkstart + i * vecsize * dsize;
              gen->Accumulate(op_, acc[i], Operand(in, disp));
            }
          } else {
            // Accumulate unrolled residual and merge into first accumulator.
            auto residual = sasm.alloc();
            sasm.main()->LoadNeutral(op_, residual);
            for (int i = 0; i < phase.unrolls; ++i) {
              int disp = blkstart + i * vecsize * dsize;
              gen->Accumulate(op_, residual, Operand(in, disp));
            }
            sasm.main()->Accumulate(op_, acc[0], residual);
          }
        } else {
          // Masked phase.
          CHECK_EQ(phase.unrolls, 1);
          gen->MaskedAccumulate(op_, acc[0], Operand(in, blkstart));
        }
      }

      // Horizontal reduction of results.
      sasm.Reduce(op_, acc);
      if (!scalar) sasm.main()->Reduce(op_, acc[0]);

      // Save result in y.
      sasm.scalar()->Store(Operand(out), acc[0]);

      // Next batch.
      if (outer_size > 1) {
        __ addq(in, Immediate(reduction_size * dsize));
        __ addq(out, Immediate(dsize));
        __ incq(batch);
        __ cmpq(batch, Immediate(outer_size));
        __ j(less, &lb);
      }
    } else {
      // Create SIMD code generators.
      bool aligned = x->stride(axis) % vecbytes == 0;
      SIMDAssembler sasm(masm, type, aligned);

      // Compute vector processing strategy.
      step->set_variant(sasm.name() + "V");
      SIMDStrategy strategy(&sasm, inner_size);
      strategy.PreloadMasks();
      auto acc = sasm.alloc(strategy.MaxUnrolls());

      // Loop over batches.
      Register batch = masm->rr().alloc();
      Label lb;
      if (outer_size > 1) {
        __ xorq(batch, batch);
        __ bind(&lb);
      }

      // Vertically reduction.
      for (auto &phase : strategy.phases()) {
        auto *gen = phase.generator;
        int vecsize = gen->VectorSize();
        int blkstart = phase.offset * dsize;
        int blksize = phase.unrolls * vecsize * dsize;
        int stride = axis > 0 ? x->stride(axis - 1) : x->size();

        if (phase.masked == 0) {
          // Repeated/residial phase.
          Label l2;
          if (phase.offset == 0) {
            __ xorq(ofs, ofs);
          } else {
            __ movq(ofs, Immediate(blkstart));
          }
          __ bind(&l2);

          // Initialize accumulators with neutral element.
          for (int r = 0; r < phase.unrolls; ++r) {
            gen->LoadNeutral(op_, acc[r]);
          }

          // Loop over reduction axis and reduce block vertically.
          Label l3;
          __ bind(&l3);
          for (int i = 0; i < phase.unrolls; ++i) {
            int disp = i * vecsize * dsize;
            gen->Accumulate(op_, acc[i], Operand(in, ofs, times_1, disp));
          }
          __ addq(ofs, Immediate(inner_size * dsize));
          __ cmpq(ofs, Immediate(stride));
          __ j(less, &l3);

          // Store result for block.
          for (int i = 0; i < phase.unrolls; ++i) {
            gen->Store(Operand(out, i * vecsize * dsize), acc[i]);
          }
          __ addq(out, Immediate(blksize));

          if (phase.repeat > 1) {
            // Next block.
            __ subq(ofs, Immediate(stride - blksize));
            __ cmpq(ofs, Immediate(blkstart + phase.repeat * blksize));
            __ j(less, &l2);
          }
        } else {
          // Masked phase.
          CHECK_EQ(phase.unrolls, 1);
          CHECK_EQ(phase.repeat, 1);
          if (phase.offset == 0) {
            __ xorq(ofs, ofs);
          } else {
            __ movq(ofs, Immediate(blkstart));
          }

          // Initialize accumulator with neutral element.
          gen->LoadNeutral(op_, acc[0]);

          // Loop over reduction axis and reduce block vertically.
          Label l3;
          __ bind(&l3);
            gen->MaskedAccumulate(op_, acc[0], Operand(in, ofs, times_1));
          __ addq(ofs, Immediate(inner_size * dsize));
          __ cmpq(ofs, Immediate(stride));
          __ j(less, &l3);

          // Store result for block.
          gen->MaskedStore(Operand(out), acc[0]);
          __ addq(out, Immediate(phase.masked * dsize));
        }
      }

      // Next batch.
      if (outer_size > 1) {
        __ addq(in, Immediate(reduction_size * inner_size * dsize));
        __ incq(batch);
        __ cmpq(batch, Immediate(outer_size));
        __ j(less, &lb);
      }
    }
  }

  int64 Complexity(const Step *step) override {
    return step->input(0)->elements();
  }

 private:
  string name_;
  Reduction op_;
};

// Transpose tensor by permuting dimensions.
class Transpose : public Kernel {
 public:
  string Name() override { return "Transpose"; }
  string Operation() override { return "Transpose"; }

  bool Supports(Step *step) override {
    // Check inputs and outputs.
    if (step->indegree() != 1 || step->outdegree() != 1) return false;
    Tensor *x = step->input(0);
    Tensor *y = step->output(0);
    if (x->type() != y->type()) return false;

    // Check permutation.
    Shape perm = GetPerm(step);
    if (x->shape().permuted(perm) != y->shape()) return false;

    return true;
  }

  void Adjust(Step *step) override {
    // Get input and output.
    Tensor *x = step->input(0);
    Tensor *y = step->output(0);
    Shape perm = GetPerm(step);
    int shuffled = Shuffled(perm);

    // Trivial permutation is a no-op.
    if (shuffled <= 0 && step->AllowInPlace(0, 0, true)) return;

    // Require dense standard layout.
    x->RequireStandardOrder();
    y->RequireStandardOrder();
    x->RequireDense();
    y->RequireDense();

    // Reserve registers.
    step->SetRegisterUsage(5 + shuffled);
  }

  void Generate(Step *step, MacroAssembler *masm) override {
    // Get input and output.
    Tensor *x = step->input(0);
    Tensor *y = step->output(0);
    Shape perm = GetPerm(step);

    // Find the number of outer and inner dimensions.
    int outer_dims = Outer(perm);
    int inner_dims = Inner(perm);
    int shuffle_dims = Shuffled(perm);
    if (shuffle_dims <= 0) {
      CHECK(x->SharedWith(y));
      return;
    }

    // Set kernel variant.
    string variant = step->variant();
    if (outer_dims > 0) variant += "O" + std::to_string(outer_dims);
    if (shuffle_dims > 0) variant += "S" + std::to_string(shuffle_dims);
    if (inner_dims > 0) variant += "I" + std::to_string(inner_dims);
    step->set_variant(variant);

    // Allocate registers.
    Register src = masm->rr().alloc_fixed(rsi);
    Register dst = masm->rr().alloc_fixed(rdi);
    Register cnt = masm->rr().alloc_fixed(rcx);
    Register in = masm->rr().alloc();
    Register ofs = masm->rr().alloc();
    Register aux = cnt;

    // Load tensor addresses.
    __ LoadTensorAddress(in, x);
    __ LoadTensorAddress(dst, y);

    // Loop over outer dimensions.
    Register batch = masm->rr().alloc();
    Label lb;
    if (outer_dims > 0) {
      __ xorq(batch, batch);
      __ bind(&lb);
    }

    // Loop over shuffled dimensions.
    std::vector<Label> shuffle_loop(shuffle_dims);
    std::vector<Register> shuffle_index(shuffle_dims);
    for (int i = 0; i < shuffle_dims; ++i) {
      shuffle_index[i] = masm->rr().alloc();
      __ xorq(shuffle_index[i], shuffle_index[i]);
      __ bind(&shuffle_loop[i]);
    }

    // Compute offset of shuffled element/block in input.
    CHECK_GE(shuffle_dims, 2);
    __ leaq(ofs, Operand(shuffle_index[0], shuffle_index[1]));
    for (int i = 2; i < shuffle_dims; ++i) {
      __ addq(ofs, shuffle_index[i]);
    }

    // Copy element/block from input to output.
    int block_size = y->stride(y->rank() - inner_dims - 1);
    if (block_size == 1) {
      __ movb(aux, Operand(in, ofs));
      __ movb(Operand(dst), aux);
      __ addq(dst, Immediate(1));
    } else if (block_size == 2) {
      __ movw(aux, Operand(in, ofs));
      __ movw(Operand(dst), aux);
      __ addq(dst, Immediate(2));
    } else if (block_size == 4) {
      __ movl(aux, Operand(in, ofs));
      __ movl(Operand(dst), aux);
      __ addq(dst, Immediate(4));
    } else if (block_size == 8) {
      __ movq(aux, Operand(in, ofs));
      __ movq(Operand(dst), aux);
      __ addq(dst, Immediate(8));
    } else {
      __ leaq(src, Operand(in, ofs));
      __ movq(cnt, Immediate(block_size));
      __ repmovsb();
    }

    // Next shuffled element/block.
    for (int i = shuffle_dims - 1; i >= 0; --i) {
      int d = perm[i + outer_dims];
      int stride = x->stride(d);
      int size = x->dim(d);
      __ addq(shuffle_index[i], Immediate(stride));
      __ cmpq(shuffle_index[i], Immediate(stride * size));
      __ j(less, &shuffle_loop[i]);
    }

    // Next outer batch.
    if (outer_dims > 0) {
      __ addq(in, Immediate(x->stride(outer_dims - 1)));
      __ incq(batch);
      __ cmpq(batch, Immediate(x->shape().outer(outer_dims)));
      __ j(less, &lb);
    }
  }

  int64 Complexity(const Step *step) override {
    return step->input(0)->elements();
  }

 private:
  // Get permutation attribute.
  static Shape GetPerm(Step *step) {
    Shape perm;
    if (!step->GetAttr("perm", &perm)) {
      perm.reverse(step->input(0)->rank());
    }
    return perm;
  }

  // Number of preserved outer dimensions in permutation.
  static int Outer(const Shape &perm) {
    int r = perm.rank();
    int outer = 0;
    for (int d = 0; d < r; ++d) {
      if (perm[d] != d) break;
      outer++;
    }
    return outer;
  }

  // Number of preserved inner dimensions in permutation.
  static int Inner(const Shape &perm) {
    int r = perm.rank();
    int inner = 0;
    for (int d = r - 1; d >= 0; --d) {
      if (perm[d] != d) break;
      inner++;
    }
    return inner;
  }

  // Number of shuffled dimensions in permutation.
  static int Shuffled(const Shape &perm) {
    return perm.rank() - Outer(perm) - Inner(perm);
  }
};

// Fold multiplication into update ops.
class UpdateTransformer : public Transformer {
 public:
  string Name() override { return "UpdateTransformer"; }

  bool Transform(Flow *flow) override {
    bool updated = false;
    bool again = true;
    while (again) {
      again = false;
      if (TransformMatMul(flow)) {
        again = true;
        updated = true;
      }
      if (TransformDistributiveUpdate(flow)) {
        again = true;
        updated = true;
      }
      if (TransformSparseUpdate(flow)) {
        again = true;
        updated = true;
      }
      if (TransformScaledSparseUpdate(flow)) {
        again = true;
        updated = true;
      }
    }
    return updated;
  }

  // Transform matrix multiplication updates.
  bool TransformMatMul(Flow *flow) {
    int updates = 0;
    for (Flow::Operation *op : flow->Find("MatMul|1:Add|1:Assign")) {
      Flow::Operation *assign = op;
      Flow::Operation *add = assign->inputs[1]->producer;
      Flow::Operation *matmul = add->inputs[1]->producer;

      if (assign->inputs[0] != add->inputs[0]) continue;
      if (add->outputs[0]->usages() != 1) continue;
      if (matmul->outputs[0]->usages() != 1) continue;

      flow->Fuse(assign, flow->Fuse(add, matmul, ""), "AssignAddMatMul", true);
      updates++;
    }
    return updates > 0;
  }

  // Transform distributive scatter udates.
  bool TransformDistributiveUpdate(Flow *flow) {
    // Find assignments for scatter operations.
    std::set<Flow::Operation *> scatter_assigns;
    for (Flow::Operation *op : flow->Find("Scatter")) {
      while (op->outdegree() == 1 && op->outputs[0]->usages() == 1) {
        op = op->outputs[0]->consumers[0];
      }
      if (op->type == "Assign") scatter_assigns.insert(op);
    }

    // Split additive updates.
    int updates = 0;
    for (Flow::Operation *op : flow->Find("Add|1:Add|1:Assign")) {
      Flow::Operation *assign1 = op;
      Flow::Operation *add1 = assign1->inputs[1]->producer;
      Flow::Operation *add2 = add1->inputs[1]->producer;
      Flow::Variable *target = assign1->inputs[0];

      if (add1->outputs[0]->usages() != 1) continue;
      if (add2->outputs[0]->usages() != 1) continue;
      if (add1->inputs[0] != target) continue;
      if (scatter_assigns.count(assign1) == 0) continue;

      // Split into two accumulative updates.
      Flow::Function *func = assign1->func;
      Flow::Operation *assign2 = flow->AddOperation(func, "", "Assign");
      assign2->AddInput(target);
      assign2->AddInput(add2->outputs[0]);
      add1->ReplaceInput(add1->inputs[1], add2->inputs[0]);
      add2->ReplaceInput(add2->inputs[0], target);
      updates++;
    }
    return updates > 0;
  }

  // Transform sparse updates.
  bool TransformSparseUpdate(Flow *flow) {
    int updates = 0;
    for (Flow::Operation *op : flow->Find("Scatter|1:Add|1:Assign")) {
      Flow::Operation *assign = op;
      Flow::Operation *add = assign->inputs[1]->producer;
      Flow::Operation *scatter = add->inputs[1]->producer;
      if (assign->inputs[0] != add->inputs[0]) continue;
      if (add->outputs[0]->usages() != 1) continue;
      if (scatter->outputs[0]->usages() != 1) continue;

      Flow::Operation *add_scatter = flow->Fuse(add, scatter, "");
      flow->Fuse(assign, add_scatter, "AssignAddScatter", true);
      updates++;
    }
    return updates > 0;
  }

  // Transform sparse update scalings.
  bool TransformScaledSparseUpdate(Flow *flow) {
    int updates = 0;
    for (Flow::Operation *op : flow->Find("Mul|2:AssignAddScatter")) {
      Flow::Operation *scatter = op;
      Flow::Operation *mul = scatter->inputs[2]->producer;
      if (scatter->indegree() != 3) continue;
      if (mul->outputs[0]->usages() != 1) continue;
      flow->Fuse(scatter, mul, "AssignAddMulScatter");
      updates++;
    }
    return updates > 0;
  }
};

// Propagate tensor references across reshapes.
class ReshapeRefTransformer : public Transformer {
 public:
  string Name() override { return "ReshapeRefTransformer"; }

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
  library->Register(new Split());
  library->Register(new Slice());
  library->Register(new MultiGather());
  library->Register(new SingleGather());
  library->Register(new PoolingGather(PoolingGather::SUM));
  library->Register(new PoolingGather(PoolingGather::AVG));
  library->Register(new PoolingGather(PoolingGather::MAX));
  library->Register(new AssignAddScatter(false));
  library->Register(new AssignAddScatter(true));

  library->Register(new Reduce("Sum", REDUCE_ADD));
  library->Register(new Reduce("Product", REDUCE_MUL));
  library->Register(new Reduce("Max", REDUCE_MAX));
  library->Register(new Reduce("Min", REDUCE_MIN));
  library->Register(new Reduce("All", REDUCE_AND));
  library->Register(new Reduce("Any", REDUCE_OR));
  library->Register(new Transpose());

  library->RegisterTransformer(new UpdateTransformer());
  library->RegisterTransformer(new ReshapeRefTransformer());
}

}  // namespace myelin
}  // namespace sling
