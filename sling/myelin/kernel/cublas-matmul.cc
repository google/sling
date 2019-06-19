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

#include <string>

#include "sling/myelin/cuda/cuda-kernel.h"
#include "sling/myelin/cuda/cuda-runtime.h"

#define __ masm->

namespace sling {
namespace myelin {

// Allow use of offsetof on non-POD types.
#pragma GCC diagnostic ignored "-Winvalid-offsetof"

using namespace jit;

// GPU matrix multiplication using CUBLAS.
class CUBLASMatMul : public Kernel {
 public:
  string Name() override { return "CUBLASMatMul"; }
  string Operation() override { return "MatMul"; }
  Placement Location() override { return DEVICE; }

  bool Supports(Step *step) override {
    // Requires CUDA support with CUBLAS Lt.
    if (!HasCuBLASLt()) return false;

    // Two 2D tensor inputs and one 2D tensor output.
    if (step->inputs().size() != 2) return false;
    if (step->outputs().size() != 1) return false;

    // Check rank.
    if (step->input(0)->rank() != 2) return false;
    if (step->input(1)->rank() != 2) return false;
    if (step->output(0)->rank() != 2) return false;

    // Types must match and be supported by CUDA.
    Args args(step);
    if (args.traits.cuda() == -1) return false;
    if (args.A.type() != args.type || args.B.type() != args.type) return false;
    if (!args.A.compatible()) return false;
    if (!args.B.compatible()) return false;
    if (!args.C.compatible()) return false;

    // Check if matrix multiplication is supported by CUBLAS.
    CUDADevice *device = step->cell()->runtime()->Device();
    Descriptor desc;
    if (!args.Setup(&desc, device->lthandle())) return false;

    return true;
  }

  void Generate(Step *step, MacroAssembler *masm) override {
    // Get input and output tensors.
    Args args(step);

    // Set up descriptors.
    Descriptor *desc = new Descriptor();
    step->cell()->network()->AddResource(desc);
    CUDADevice *device = step->cell()->runtime()->Device();
    cublasLtHandle_t handle = device->lthandle();
    CHECK(args.Setup(desc, handle));

    // Set up arguments.
    Register tmp = masm->rr().alloc_temp();
    Register descaddr = masm->rr().alloc_temp();
    __ movp(descaddr, desc);

    __ movp(arg_reg_1, handle);
    __ movq(arg_reg_2, Operand(descaddr, offsetof(Descriptor, op)));
    __ movp(arg_reg_3, args.traits.one());  // alpha
    __ LoadTensorDeviceAddress(arg_reg_4, args.A.tensor);
    __ movq(arg_reg_5, Operand(descaddr, offsetof(Descriptor, a)));
    __ LoadTensorDeviceAddress(arg_reg_6, args.B.tensor);

    __ pushq(Operand(masm->instance(), StreamOffset(step)));
    __ pushq(Immediate(0));  // workspace size
    __ pushq(Immediate(0));  // workspace
    __ leaq(tmp, Operand(descaddr, offsetof(Descriptor, heuristics.algo)));
    __ pushq(tmp);
    __ pushq(Operand(descaddr, offsetof(Descriptor, c)));
    __ LoadTensorDeviceAddress(tmp, args.C.tensor);
    __ pushq(tmp);
    __ pushq(Operand(descaddr, offsetof(Descriptor, c)));
    __ pushq(tmp);
    __ movp(tmp, args.traits.zero());  // beta
    __ pushq(tmp);
    __ pushq(Operand(descaddr, offsetof(Descriptor, b)));

    // Call cublasLtMatmul.
    __ load_extern(tmp, reinterpret_cast<void *>(cublasLtMatmul),
                   "cublasLtMatmul");
    __ call(tmp);
    __ addq(rsp, Immediate(10 * 8));
    CUDARuntime::EmitStatusCheck("cublasLtMatmul", masm);
  }

  int64 Complexity(const Step *step) override {
    Tensor *A = step->input(0);
    Tensor *B = step->input(1);
    int64 ops = A->dim(step->GetAttr("transpose_a", false) ? 1 : 0);
    ops *= B->elements() * 2;
    return ops;
  }

 protected:
  // Matrix argument with optional transpose.
  struct Matrix {
    Matrix(Tensor *tensor, bool t) : tensor(tensor), transpose(t) {
      if (tensor->order() == COLUMN_MAJOR) {
        cols = tensor->dim(0);
        rows = tensor->dim(1);
        stride = tensor->stride(1);
      } else {
        rows = tensor->dim(0);
        cols = tensor->dim(1);
        stride = tensor->stride(0);
        transpose = !transpose;
      }
    }

    // Element data type.
    Type type() const { return tensor->type(); }

    // Tensor rank.
    int rank() const { return tensor->rank(); }

    // Swap row/column order.
    void swap() {
      std::swap(rows, cols);
      transpose = !transpose;
    }

    // Matrix operation.
    cublasOperation_t op() const {
      return transpose ? CUBLAS_OP_T : CUBLAS_OP_N;
    }

    // Set up CUBLAS matrix layout descriptor.
    cublasLtMatrixLayout_t describe() const {
      const auto &traits = TypeTraits::of(type());
      cudaDataType dt = static_cast<cudaDataType>(traits.cuda());
      int ld = stride / traits.size();
      if (ld == 0) ld = rows;
      cublasLtMatrixLayout_t layout;
      CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&layout, dt, rows, cols, ld));
      return layout;
    }

    // Check that tensor is compatible with CUBLAS.
    bool compatible() {
      // References must be placed on the host.
      if (tensor->ref() && (tensor->ref_placement() & HOST) == 0) return false;

      return true;
    }

    Tensor *tensor;   // underlying tensor for matrix
    int rows;         // number of rows in maxtrix
    int cols;         // number of columns in maxtrix
    int stride;       // size of leading dimension in bytes
    bool transpose;   // transposed matrix
  };

  // CUBLAS matrix multiplication descriptors.
  struct Descriptor : public Network::Resource {
    ~Descriptor() override {
      if (preference) cublasLtMatmulPreferenceDestroy(preference);
      if (c) cublasLtMatrixLayoutDestroy(c);
      if (b) cublasLtMatrixLayoutDestroy(b);
      if (a) cublasLtMatrixLayoutDestroy(a);
      if (op) cublasLtMatmulDescDestroy(op);
    }

    cublasLtMatmulDesc_t op = nullptr;
    cublasLtMatrixLayout_t a = nullptr;
    cublasLtMatrixLayout_t b = nullptr;
    cublasLtMatrixLayout_t c = nullptr;
    cublasLtMatmulPreference_t preference = nullptr;
    cublasLtMatmulHeuristicResult_t heuristics;
  };

  // Arguments for MatMul kernel.
  struct Args {
    Args(const Step *step)
      : A(step->input(0), step->GetAttr("transpose_a", false)),
        B(step->input(1), step->GetAttr("transpose_b", false)),
        C(step->output(0), step->GetAttr("transpose_c", false)),
        type(C.type()),
        traits(TypeTraits::of(type)),
        dt(static_cast<cudaDataType>(traits.cuda())) {
      // The output matrix cannot be transposed.
      if (C.transpose) {
        // C^T=B^T*A^T.
        std::swap(A, B);
        A.swap();
        B.swap();
        C.swap();
      }
    }

    // Set up descriptors.
    bool Setup(Descriptor *desc, cublasLtHandle_t handle) {
      // Get matrix layout descriptors.
      desc->a = A.describe();
      desc->b = B.describe();
      desc->c = C.describe();

      // Create matmul descriptor.
      cublasStatus_t status;
      status = cublasLtMatmulDescCreate(&desc->op, dt);
      if (status != CUBLAS_STATUS_SUCCESS) return false;

      cublasOperation_t opa = A.op();
      status = cublasLtMatmulDescSetAttribute(
          desc->op, CUBLASLT_MATMUL_DESC_TRANSA, &opa, sizeof(opa));
      if (status != CUBLAS_STATUS_SUCCESS) return false;

      cublasOperation_t opb = B.op();
      status = cublasLtMatmulDescSetAttribute(
          desc->op, CUBLASLT_MATMUL_DESC_TRANSB, &opb, sizeof(opb));
      if (status != CUBLAS_STATUS_SUCCESS) return false;

      if (C.transpose) return false;

      // Select algorithm.
      status = cublasLtMatmulPreferenceCreate(&desc->preference);
      if (status != CUBLAS_STATUS_SUCCESS) return false;

      int num_results;
      status = cublasLtMatmulAlgoGetHeuristic(
          handle, desc->op, desc->a, desc->b, desc->c, desc->c,
          desc->preference, 1, &desc->heuristics, &num_results);
      if (status != CUBLAS_STATUS_SUCCESS) return false;
      if (num_results != 1) return false;

      return true;
    }

    // Arguments.
    Matrix A;
    Matrix B;
    Matrix C;

    // Datatype.
    Type type;
    const TypeTraits &traits;
    cudaDataType dt;
  };
};

void RegisterCUBLASMatMulLibrary(Library *library) {
  library->Register(new CUBLASMatMul());
}

}  // namespace myelin
}  // namespace sling

