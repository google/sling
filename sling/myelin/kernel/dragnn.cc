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

#include "sling/myelin/kernel/dragnn.h"

#include "sling/myelin/compute.h"
#include "sling/myelin/macro-assembler.h"

#define __ masm->

namespace sling {
namespace myelin {

using namespace jit;

// Stub for Dragnn initializer.
class DragnnInitializer : public Kernel {
 public:
  string Name() override { return "DragnnInitializerDummy"; }
  string Operation() override { return "DragnnEmbeddingInitializer"; }

  bool Supports(Step *step) override {
    return true;
  }

  void Generate(Step *step, MacroAssembler *masm) override {
  }
};

// Dragnn feature collect operation for recurrent features mapped through an
// embedding matrix.
class DragnnCollect : public Kernel {
 public:
  string Name() override { return "DragnnCollect"; }
  string Operation() override { return "Collect"; }

  bool Supports(Step *step) override {
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
    step->input(1)->RequireOrder(ROW_MAJOR);
    step->output(0)->RequireOrder(ROW_MAJOR);
  }

  void Generate(Step *step, MacroAssembler *masm) override {
    Registers &rr = masm->rr();
    Label l1, l2, l3;

    // Get inputs and outputs.
    Tensor *f = step->input(0);
    Tensor *M = step->input(1);
    Tensor *R = step->output(0);

    // Get size of activation vectors.
    int dims = M->dim(1);

    // Get number input features.
    int num_features = f->dim(1);

    // Allocate registers.
    rr.use(rsi);
    rr.use(rdi);
    rr.use(rcx);
    Register acc = rr.alloc();
    Register input = rr.alloc();
    Register activations = rr.alloc();
    Register output = rr.alloc();
    Register index = rr.alloc();
    Register one = rr.alloc();
    rr.release(rsi);
    rr.release(rdi);
    rr.release(rcx);

    // Load tensor locations.
    __ LoadTensorAddress(input, f);
    __ LoadTensorAddress(activations, M);
    __ LoadTensorAddress(output, R);

    // Loop over input features.
    if (num_features != 1) {
      __ xorq(index, index);
      __ LoopStart(&l1);
    }

    // Get next feature index.
    if (num_features == 1) {
      __ movsxlq(acc, Operand(input));
    } else {
      __ movsxlq(acc, Operand(input, index, times_4));
    }

    // Check for OOV feature.
    __ testq(acc, acc);
    __ j(negative, &l2);

    // Copy activation vector to output.
    __ Multiply(acc, M->stride(0));
    __ addq(acc, activations);
    __ Copy(output, 0, acc, 0, dims * sizeof(float));
    __ jmp(&l3);

    // Set OOV indicator to 1.0 if feature is -1.
    __ bind(&l2);
    __ cmpq(acc, Immediate(-1));
    __ j(not_equal, &l3);
    __ movl(one, Immediate(0x3f800000));
    __ movl(Operand(output, dims * sizeof(float)), one);

    // Next feature.
    __ bind(&l3);
    if (num_features != 1) {
      __ addq(output, Immediate(R->stride(0)));
      __ incq(index);
      __ cmpq(index, Immediate(num_features));
      __ j(not_equal, &l1);
    }
  }

  int64 Complexity(const Step *step) override {
    return 0;
  }
};

// Dragnn feature lookup operation for fixed features mapped through an
// embedding matrix.
class DragnnLookup : public Kernel {
 public:
  string Name() override { return "DragnnLookup"; }
  string Operation() override { return "Lookup"; }

  bool Supports(Step *step) override {
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
    step->input(1)->RequireOrder(ROW_MAJOR);
  }

  void Generate(Step *step, MacroAssembler *masm) override {
    Registers &rr = masm->rr();
    SIMDRegisters &mm = masm->mm();
    Label l1, l2, l3, l4;

    // Get inputs and outputs.
    Tensor *f = step->input(0);
    Tensor *M = step->input(1);
    Tensor *v = step->output(0);

    // Get embedding size and dimension. The last element is the OOV element.
    int embedding_size = M->dim(0) - 1;
    int embedding_dims = v->dim(1);

    // Get number input features.
    int num_features = f->dim(1);

    // Allocate registers.
    Register acc = rr.alloc();
    Register input = rr.alloc();
    Register embeddings = rr.alloc();
    Register output = rr.alloc();
    Register col = rr.alloc();
    Register row = rr.alloc();
    Register oov = rr.alloc();
    XMMRegister elem = mm.allocx();

    // Load tensor locations.
    __ LoadTensorAddress(input, f);
    __ LoadTensorAddress(embeddings, M);
    __ LoadTensorAddress(output, v);

    // Loop over input features.
    __ movq(oov, Immediate(embedding_size));
    __ xorq(col, col);
    __ LoopStart(&l1);

    // Get next feature index.
    __ movsxlq(acc, Operand(input, col, times_4));

    // Use OOV for if feature is -1, otherwise skip feature if it is negative.
    __ testq(acc, acc);
    __ j(positive, &l2);
    __ cmpq(acc, Immediate(-1));
    __ j(not_equal, &l4);
    __ movq(acc, oov);

    // Compute address of embedding vector.
    __ bind(&l2);
    __ Multiply(acc, M->stride(0));
    __ leaq(acc, Operand(embeddings, acc));

    // Add embedding vector to output.
    __ xorq(row, row);
    __ LoopStart(&l3);
    __ movss(elem, Operand(output, row, times_4));
    __ addss(elem, Operand(acc, row, times_4));
    __ movss(Operand(output, row, times_4), elem);
    __ incq(row);
    __ cmpq(row, Immediate(embedding_dims));
    __ j(not_equal, &l3);

    // Next feature.
    __ bind(&l4);
    __ incq(col);
    __ cmpq(col, Immediate(num_features));
    __ j(not_equal, &l1);
  }

  int64 Complexity(const Step *step) override {
    return step->input(0)->elements() * step->output(0)->elements();
  }
};

// Dragnn feature lookup operation for single fixed features mapped through an
// embedding matrix. This just outputs a reference to the row in the embedding
// matrix.
class DragnnLookupSingle : public Kernel {
 public:
  string Name() override { return "DragnnLookupSingle"; }
  string Operation() override { return "Lookup"; }

  bool Supports(Step *step) override {
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
    v->Link(step->input(1));

    // Embedding matrix must be row-major.
    step->input(1)->RequireOrder(ROW_MAJOR);
  }

  void Generate(Step *step, MacroAssembler *masm) override {
    // Get inputs and outputs.
    Tensor *f = step->input(0);
    Tensor *M = step->input(1);
    Tensor *v = step->output(0);

    // Get embedding size. The last element is the OOV element.
    int embedding_size = M->dim(0) - 1;

    // Allocate registers.
    Register acc = masm->rr().alloc();
    Register oov = masm->rr().alloc();
    Register embeddings = masm->rr().alloc();

    // Get feature index.
    CHECK(!f->ref());
    __ movsxlq(acc, Operand(masm->instance(), f->offset()));

    // Use OOV for negative index.
    __ movq(oov, Immediate(embedding_size));
    __ testq(acc, acc);
    __ cmovq(negative, acc, oov);

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

// Dragnn feature lookup operation for fixed features mapped through an
// embedding matrix. This can be used when the size of the embedding is small
// enough to fit into registers.
class DragnnLookupUnrolled : public Kernel {
 public:
  string Name() override { return "DragnnLookupUnrolled"; }
  string Operation() override { return "Lookup"; }

  static const int kBlockSize = 8;
  static const int kMaxEmbeddingDim = kBlockSize * 16;

  bool Supports(Step *step) override {
    // Requires CPU with AVX support.
    if (!CPU::Enabled(AVX)) return false;

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

    // Check if embedding dimension allows us to unroll.
    int embedding_dims = M->dim(1);
    if (embedding_dims > kMaxEmbeddingDim) return false;
    if (embedding_dims % kBlockSize != 0) return false;

    return true;
  }

  void Adjust(Step *step) override {
    // Align embeddings and output.
    int align = kBlockSize * sizeof(float);
    step->input(1)->MinAlign({1, kBlockSize});
    step->input(1)->SetMiniumAlignment(align);
    step->output(0)->MinAlign({1, kBlockSize});
    step->output(0)->SetMiniumAlignment(align);

    // Embedding matrix must be row-major.
    step->input(1)->RequireOrder(ROW_MAJOR);
  }

  void Generate(Step *step, MacroAssembler *masm) override {
    Registers &rr = masm->rr();
    SIMDRegisters &mm = masm->mm();
    Label l1, l2, l3;

    // Get inputs and outputs.
    Tensor *f = step->input(0);
    Tensor *M = step->input(1);
    Tensor *v = step->output(0);

    // Get embedding size and dimension. The last element is the OOV element.
    int embedding_size = M->dim(0) - 1;
    int embedding_dims = v->dim(1);

    // Get number of input features.
    int num_features = f->dim(1);

    // Allocate registers.
    Register acc = rr.alloc();
    Register input = rr.alloc();
    Register embeddings = rr.alloc();
    Register output = rr.alloc();
    Register col = rr.alloc();
    Register oov = rr.alloc();

    // Allocate registers for summing embedding vectors.
    std::vector<YMMRegister> sum;
    int blocks = embedding_dims / kBlockSize;
    for (int i = 0; i < blocks; ++i) sum.push_back(mm.allocy());

    // Load tensor locations.
    __ LoadTensorAddress(input, f);
    __ LoadTensorAddress(embeddings, M);
    __ LoadTensorAddress(output, v);

    // Clear output vector.
    for (int i = 0; i < blocks; ++i) {
      __ vxorps(sum[i], sum[i], sum[i]);
    }

    // Loop over input features.
    __ movq(oov, Immediate(embedding_size));
    __ xorq(col, col);
    __ LoopStart(&l1);

    // Get next feature index.
    __ movsxlq(acc, Operand(input, col, times_4));

    // Use OOV for negative index.
    __ testq(acc, acc);
    __ j(positive, &l2);
    __ cmpq(acc, Immediate(-1));
    __ j(not_equal, &l3);
    __ movq(acc, oov);

    // Compute address of embedding vector.
    __ bind(&l2);
    __ Multiply(acc, M->stride(0));
    __ addq(acc, embeddings);

    // Add embedding vector to sum.
    for (int i = 0; i < blocks; ++i) {
      __ vaddps(sum[i], sum[i], Operand(acc, i * kBlockSize * sizeof(float)));
    }

    // Next feature.
    __ bind(&l3);
    __ incq(col);
    __ cmpq(col, Immediate(num_features));
    __ j(not_equal, &l1);

    // Store sum.
    for (int i = 0; i < blocks; ++i) {
      __ vmovaps(Operand(output, i * kBlockSize * sizeof(float)), sum[i]);
    }
  }

  int64 Complexity(const Step *step) override {
    return step->input(0)->elements() * step->output(0)->elements();
  }
};

// Type inference for Dragnn ops.
class DragnnTyper : public Typer {
 public:
  string Name() override { return "DragnnTyper"; }

  bool InferTypes(Flow *flow, Flow::Operation *op) override {
    // Infer shape for lookup operation.
    if (op->type == "Lookup") {
      if (op->indegree() == 2 && op->outdegree() == 1) {
        Flow::Variable *features = op->inputs[0];
        Flow::Variable *embeddings = op->inputs[1];
        Flow::Variable *result = op->outputs[0];
        if (features->rank() == 2 && embeddings->rank() == 2) {
          result->shape.assign(1, embeddings->dim(1));
          return true;
        }
      }
    }

    // Infer shape for collect operation.
    if (op->type == "Collect") {
      if (op->indegree() == 2 && op->outdegree() == 1) {
        Flow::Variable *features = op->inputs[0];
        Flow::Variable *embeddings = op->inputs[1];
        Flow::Variable *result = op->outputs[0];
        if (features->rank() == 2 && embeddings->rank() == 2) {
          // Add extra element for OOV indicator.
          result->shape.assign(features->dim(1), embeddings->dim(1) + 1);
          return true;
        }
      }
    }

    if (op->type == "DragnnEmbeddingInitializer") {
      if (op->outdegree() == 1) {
        Flow::Variable *result = op->outputs[0];
        result->type = DT_INT32;
        result->shape.clear();
      }
    }

    return false;
  }
};

// Flow transformations for Dragnn ops.
class DragnnTransformer : public Transformer {
 public:
  string Name() override { return "DragnnTransformer"; }

  bool Transform(Flow *flow) override {
    std::vector<Flow::Operation *> noops;
    for (Flow::Operation *op : flow->ops()) {
      if (op->type == "FeatureVector" || op->type == "Enter") {
        noops.push_back(op);
      }
    }
    for (Flow::Operation *op : noops) {
      flow->Eliminate(op);
    }
    return !noops.empty();
  }
};

// Precompute embeddings with a linear transform.
class PrecomputedEmbeddings : public Transformer {
 public:
  string Name() override { return "PrecomputedEmbeddings"; }

  bool Transform(Flow *flow) override {
    int num_precompute = 0;
    for (auto *op : flow->Find({"Lookup", "Reshape", "MatMul"})) {
      Flow::Operation *matmul = op;
      Flow::Operation *reshape = matmul->inputs[0]->producer;
      Flow::Operation *lookup = reshape->inputs[0]->producer;
      if (matmul->indegree() != 2 || !matmul->inputs[1]->constant()) continue;
      if (lookup->indegree() != 2 || !lookup->inputs[1]->constant()) continue;
      if (lookup->outputs[0]->out()) continue;
      if (reshape->outputs[0]->out()) continue;
      Flow::Variable *feature = lookup->inputs[0];
      Flow::Variable *embedding = lookup->inputs[1];
      Flow::Variable *transform = matmul->inputs[1];
      if (embedding->type != transform->type) continue;
      if (embedding->rank() != 2 || transform->rank() != 2) continue;

      // Multiply the embeddings with the linear transform.
      string name = embedding->name + "/" + transform->name;
      Flow::Variable *precomputed =
        flow->AddVariable(name, transform->type,
                          {embedding->dim(0), transform->dim(1)});
      flow->AddOperation(lookup->func, name + "/Precompute", "MatMul",
                         {embedding, transform}, {precomputed});

      // Convert the MatMul to a Lookup on the precomputed embeddings.
      matmul->type = "Lookup";
      matmul->ReplaceInput(matmul->inputs[0], feature);
      matmul->ReplaceInput(transform, precomputed);

      // Remove old Lookup if it is no longer used.
      if (reshape->outputs[0]->consumers.empty()) {
        flow->RemoveOperation(reshape);
      }
      if (lookup->outputs[0]->consumers.empty()) {
        flow->RemoveOperation(lookup);
      }

      num_precompute++;
    }
    return num_precompute > 0;
  }
};

// Register Dragnn library.
void RegisterDragnnLibrary(Library *library) {
  library->RegisterTyper(new DragnnTyper());
  library->RegisterTransformer(new PrecomputedEmbeddings());
  library->RegisterTransformer(new DragnnTransformer());
  library->Register(new DragnnInitializer());
  library->Register(new DragnnLookup());
  library->Register(new DragnnLookupUnrolled());
  library->Register(new DragnnLookupSingle());
  library->Register(new DragnnCollect());
}

}  // namespace myelin
}  // namespace sling

