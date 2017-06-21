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
    if (x->shape().elements() != y->shape().elements()) return false;
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
    if (x->shape().elements() != y->shape().elements()) return false;
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
    if (x->shape().elements() != y->shape().elements()) return false;
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

// Divide "spatial" dimensions [1, ..., M] of the input, and interleaves these
// with the "batch" dimension (0).
class SpaceToBatch : public Kernel {
 public:
  string Name() override { return "SpaceToBatch"; }
  string Operation() override { return "SpaceToBatchND"; }

  bool Supports(Step *step) override {
    // Check inputs and outputs.
    if (step->indegree() != 3 || step->outdegree() != 1) return false;
    Tensor *x = step->input(0);
    Tensor *y = step->output(0);
    if (x->type() != y->type()) return false;
    if (x->shape().elements() != y->shape().elements()) {
      LOG(WARNING) << step->name() << " needs padding: "
                   << x->shape().ToString() << " -> " <<  y->shape().ToString();
    }
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

// Reshapes the "batch" dimension 0 into M + 1 dimensions, and interleaves these
// back into the spatial dimensions [1, ..., M].
class BatchToSpace : public Kernel {
 public:
  string Name() override { return "BatchToSpace"; }
  string Operation() override { return "BatchToSpaceND"; }

  bool Supports(Step *step) override {
    // Check inputs and outputs.
    if (step->indegree() != 3 || step->outdegree() != 1) return false;
    Tensor *x = step->input(0);
    Tensor *y = step->output(0);
    if (x->type() != y->type()) return false;
    if (x->shape().elements() != y->shape().elements()) {
      LOG(WARNING) << step->name() << " needs padding: "
                   << x->shape().ToString() << " -> " <<  y->shape().ToString();
    }
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
    if (x->shape().elements() != y->shape().elements()) return false;
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
    if (x->shape().elements() != y->shape().elements()) return false;
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

// Output concatenation of input tensors.
class SimpleConcat : public Kernel {
 public:
  string Name() override { return "SimpleConcat"; }
  string Operation() override { return "ConcatV2"; }

  bool Supports(Step *step) override {
    // Check inputs and outputs.
    if (step->indegree() < 2 || step->outdegree() != 1) return false;

    // Only concatenation along first axis supported.
    int n = step->indegree() - 1;
    Tensor *axis = step->input(n);
    if (!axis->IsConstant()) return false;
    if (axis->value<int32>() != 1) return false;

    return true;
  }

  void Adjust(Step *step) override {
  }

  void Generate(Step *step, MacroAssembler *masm) override {
    // The last input is the axis.
    int n = step->indegree() - 1;

    // Allocate registers.
    Register src = masm->rr().alloc_preferred(r8);
    Register dst = masm->rr().alloc_preferred(r9);

    // Load output tensor.
    __ LoadTensorAddress(dst, step->output(0));

    // Copy input tensors to output.
    int offset = 0;
    for (int i = 0; i < n; ++i) {
      int size = step->input(i)->size();
      __ LoadTensorAddress(src, step->input(i));
      __ Copy(dst, offset, src, 0, size);
      offset += size;
    }
    CHECK_EQ(offset, step->output(0)->size());
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
  library->Register(new SimpleConcat());
}

}  // namespace myelin
}  // namespace sling

