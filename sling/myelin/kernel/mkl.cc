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

#include "sling/myelin/kernel/mkl.h"

#include <dlfcn.h>
#include <map>
#include <mutex>
#include <string>
#include <vector>

#include "sling/base/flags.h"
#include "sling/base/types.h"
#include "sling/myelin/compute.h"
#include "sling/myelin/macro-assembler.h"

DEFINE_string(mklrt, "", "Intel MKL runtime model");

#define __ masm->

namespace sling {
namespace myelin {

using namespace jit;

// Definitions from mkl_cblas.h in Intel Math Kernel Library.

typedef int64 mkl_int_t;

enum MKLLayout : mkl_int_t {
  MKL_ROW_MAJOR = 101,
  MKL_COL_MAJOR = 102,
};

enum MKLTranspose : mkl_int_t {
  MKL_NO_TRANS   = 111,
  MKL_TRANS      = 112,
  MKL_CONJ_TRANS = 113,
};

enum MKLJITStatus : mkl_int_t {
  MKL_JIT_SUCCESS  = 0,
  MKL_NO_JIT       = 1,
  MKL_JIT_ERROR    = 2,
};

typedef void *gemm_jit_kernel_t;

// MKL support.
static bool mkl_support = false;
static bool mkl_batch_support = false;
static bool mkl_jit_support = false;

// MKL functions.
const void *cblas_sgemm = nullptr;
const void *cblas_dgemm = nullptr;
const void *cblas_sgemm_batch = nullptr;
const void *cblas_dgemm_batch = nullptr;

// MKL JIT functions.

MKLJITStatus (*mkl_cblas_jit_create_sgemm)(
  void **jitter,
  const MKLLayout layout,
  const MKLTranspose transa, const MKLTranspose transb,
  const mkl_int_t m, const mkl_int_t n, const mkl_int_t k,
  const float alpha, const mkl_int_t lda, const mkl_int_t ldb,
  const float beta, const mkl_int_t ldc) = nullptr;

MKLJITStatus (*mkl_cblas_jit_create_dgemm)(
  void **jitter,
  const MKLLayout layout,
  const MKLTranspose transa, const MKLTranspose transb,
  const mkl_int_t m, const mkl_int_t n, const mkl_int_t k,
  const double alpha, const mkl_int_t lda, const mkl_int_t ldb,
  const double beta, const mkl_int_t ldc) = nullptr;

MKLJITStatus (*mkl_jit_destroy)(void *jitter) = nullptr;

gemm_jit_kernel_t (*mkl_jit_get_sgemm_ptr)(const void *jitter) = nullptr;
gemm_jit_kernel_t (*mkl_jit_get_dgemm_ptr)(const void *jitter) = nullptr;

// Flag to check that we only try to initialize the MKL library once.
static std::once_flag mkl_initialized;

// Intel MKL runtime models.
std::map<string, std::vector<const char *>> mkl_runtimes = {
  // Default model.
  {"", {
    "libmkl_core.so",
    "libmkl_sequential.so",
    "libmkl_intel_ilp64.so"}
  },

  // Sequential model.
  {"seq", {
    "libmkl_core.so",
    "libmkl_sequential.so",
    "libmkl_intel_ilp64.so"}
  },

  // Intel OMP threading model.
  {"intel", {
    "libmkl_core.so",
    "libiomp5.so",
    "libmkl_intel_thread.so",
    "libmkl_intel_ilp64.so"}
  },

  // Intel Threading Building Blocks (TBB) model.
  {"tbb", {
    "libmkl_core.so",
    "libtbb.so",
    "libmkl_tbb_thread.so",
    "libmkl_intel_ilp64.so"}
  },

  // GNU OpenMP threading model.
  {"gnu", {
    "libmkl_core.so",
    "libgomp.so",
    "libmkl_gnu_thread.so",
    "libmkl_intel_ilp64.so"}
  },

  // Google MKL model.
  {"g3", {
    "libmklml_gnu.so",
    "libmklml_intel.so"}
  },

  // MKL local model.
  {"local", {
    "local/mkl/libmklml_gnu.so",
    "local/mkl/libmklml_intel.so"}
  },
};

#define LOAD_MKL_FUNCTION(name) \
  name = reinterpret_cast<decltype(name)>(dlsym(lib , #name));

// Load Intel MKL library.
static bool LoadMKLLibrary() {
  // Set up list of libraries to load.
  auto f = mkl_runtimes.find(FLAGS_mklrt);
  if (f == mkl_runtimes.end()) {
    LOG(ERROR) << "Unknown MKL runtime model: " << FLAGS_mklrt;
    return false;
  }

  // Try to load MKL libraries.
  void *lib = nullptr;
  for (auto *libname : f->second) {
    VLOG(2) << "Loading MKL runtime: " << libname;
    lib = dlopen(libname, RTLD_LAZY | RTLD_GLOBAL);
    if (lib == nullptr) {
      VLOG(1) << "Error loading " << libname << ": " << dlerror();
      return false;
    }
  }

  // Resolve library functions.
  LOAD_MKL_FUNCTION(cblas_sgemm);
  LOAD_MKL_FUNCTION(cblas_dgemm);
  LOAD_MKL_FUNCTION(cblas_sgemm_batch);
  LOAD_MKL_FUNCTION(cblas_dgemm_batch);
  LOAD_MKL_FUNCTION(mkl_cblas_jit_create_sgemm);
  LOAD_MKL_FUNCTION(mkl_cblas_jit_create_dgemm);
  LOAD_MKL_FUNCTION(mkl_jit_destroy);
  LOAD_MKL_FUNCTION(mkl_jit_get_sgemm_ptr);
  LOAD_MKL_FUNCTION(mkl_jit_get_dgemm_ptr);

  mkl_support = (cblas_sgemm != nullptr);
  mkl_batch_support = (cblas_sgemm_batch != nullptr);
  mkl_jit_support = (mkl_cblas_jit_create_sgemm != nullptr);

  return true;
}

// Check if MKL is supported.
static bool SupportsMKL() {
  std::call_once(mkl_initialized, []() { LoadMKLLibrary(); });
  return mkl_support;
}

// Matrix multiplication using Intel Math Kernel Library, C = A * B.
class MKLMatMul : public Kernel {
 public:
  MKLMatMul(bool accumulate) : accumulate_(accumulate)  {}

  string Name() override {
    return accumulate_ ? "MKLAccMatMul" : "MKLMatMul";
  }
  string Operation() override {
    return accumulate_ ? "AssignAddMatMul" : "MatMul";
  }

  bool Supports(Step *step, const Options &options) override {
    // Check arguments.
    if (accumulate_) {
      if (step->indegree() != 3) return false;
      if (step->outdegree() != 0) return false;
    } else {
      if (step->indegree() != 2) return false;
      if (step->outdegree() != 1) return false;
    }
    Args args(step, accumulate_);

    // Check type, shape and order.
    if (!args.Compatible()) return false;
    if (!args.A.tensor->SupportsOrder(ROW_MAJOR)) return false;
    if (!args.B.tensor->SupportsOrder(ROW_MAJOR)) return false;
    if (!args.C.tensor->SupportsOrder(ROW_MAJOR)) return false;

    // Check that MKL is supported.
    if (!options.aot) {
      if (!SupportsMKL()) return false;
      if (args.C.batchsize() > 1 && !mkl_batch_support) return false;
    }

    return true;
  }

  void Adjust(Step *step, const Options &options) override {
    Args args(step, accumulate_);

    // Only row-major supported for now.
    args.A.tensor->RequireOrder(ROW_MAJOR);
    args.B.tensor->RequireOrder(ROW_MAJOR);
    args.C.tensor->RequireOrder(ROW_MAJOR);

    // Set alignment to largest vector size supported by CPU. Assume max
    // alignment in AOT mode.
    int alignment = args.traits.size();
    if (CPU::Enabled(SSE)) alignment = 16;
    if (CPU::Enabled(AVX)) alignment = 32;
    if (CPU::Enabled(AVX512F)) alignment = 64;
    if (options.aot) alignment = 64;
    args.A.tensor->SetMiniumAlignment(alignment);
    args.B.tensor->SetMiniumAlignment(alignment);
    args.C.tensor->SetMiniumAlignment(alignment);
  }

  void Generate(Step *step, MacroAssembler *masm) override {
    // Get arguments.
    Args args(step, accumulate_);

    // Get dimensions for matrices.
    int dsize = args.traits.size();
    int m = args.C.rows();
    int n = args.C.cols();
    int k = args.A.cols();
    int lda = args.A.stride() / dsize;
    int ldb = args.B.stride() / dsize;
    int ldc = args.C.stride() / dsize;
    int batch = args.C.batchsize();

    if (batch == 1) {
      // Try to get JIT-compiled MKL kernel. Never use JIT in AOT mode.
      bool jitted = false;
      if (mkl_jit_support && !masm->options().aot) {
        // Get jitter for MKL function.
        MKLJITStatus status;
        void *jitter;
        if (args.type == DT_FLOAT) {
          status = mkl_cblas_jit_create_sgemm(
              &jitter, MKL_ROW_MAJOR, args.A.op(), args.B.op(),
              m, n, k, 1.0, lda, ldb, accumulate_ ? 1.0 : 0.0, ldc);
        } else {
          status = mkl_cblas_jit_create_dgemm(
              &jitter, MKL_ROW_MAJOR, args.A.op(), args.B.op(),
              m, n, k, 1.0, lda, ldb, accumulate_ ? 1.0 : 0.0, ldc);
        }
        if (status == MKL_JIT_SUCCESS || status == MKL_NO_JIT) {
          // Get pointer to JIT function.
          gemm_jit_kernel_t kernel;
          if (args.type == DT_FLOAT) {
            kernel = mkl_jit_get_sgemm_ptr(jitter);
          } else {
            kernel = mkl_jit_get_dgemm_ptr(jitter);
          }

          // Generate call to JIT function.
        __ movp(arg_reg_1, jitter);
        __ LoadTensorAddress(arg_reg_2, args.A.tensor);
        __ LoadTensorAddress(arg_reg_3, args.B.tensor);
        __ LoadTensorAddress(arg_reg_4, args.C.tensor);
        __ call_extern(kernel, "");

          jitted = true;
          step->set_variant(status == MKL_NO_JIT ? "STDJIT" : "JIT");
          step->cell()->network()->AddResource(new MKLJitter(jitter));
        }
      }

      // Generate call to standard gemm function if no JIT was provided.
      if (!jitted) {
        // Set up arguments to gemm routine.
        Register a = masm->rr().alloc();
        Register b = masm->rr().alloc();
        Register c = masm->rr().alloc();
        __ LoadTensorAddress(a, args.A.tensor);
        __ LoadTensorAddress(b, args.B.tensor);
        __ LoadTensorAddress(c, args.C.tensor);

        __ pushq(Immediate(ldc));
        __ pushq(c);
        __ pushq(Immediate(ldb));
        __ pushq(b);
        __ pushq(Immediate(lda));
        __ pushq(a);

        if (args.type == DT_FLOAT) {
          auto *one = masm->GetConstant<float>(1.0);
          __ movss(xmm0, one->address());  // alpha=1.0
          if (accumulate_) {
            __ movss(xmm1, xmm0);          // beta=1.0
          } else {
            __ pxor(xmm1, xmm1);           // beta=0.0
          }
        } else {
          auto *one = masm->GetConstant<double>(1.0);
          __ movsd(xmm0, one->address());  // alpha=1.0
          if (accumulate_) {
            __ movsd(xmm1, xmm0);          // beta=1.0
          } else {
            __ pxor(xmm1, xmm1);           // beta=0.0
          }
        }

        __ movq(arg_reg_1, Immediate(MKL_ROW_MAJOR));
        __ movq(arg_reg_2, Immediate(args.A.op()));
        __ movq(arg_reg_3, Immediate(args.B.op()));
        __ movq(arg_reg_4, Immediate(m));
        __ movq(arg_reg_5, Immediate(n));
        __ movq(arg_reg_6, Immediate(k));

        // Call MKL cblas_gemm.
        if (args.type == DT_FLOAT) {
          __ call_extern(cblas_sgemm, "cblas_sgemm");
        } else {
          __ call_extern(cblas_dgemm, "cblas_dgemm");
        }
        __ addq(rsp, Immediate(6 * 8));

        step->set_variant("STD");
      }
    } else {
      // Build arrays of batched matrices on the stack.
      Register a = args.A.GenerateArray(masm);
      Register b = args.B.GenerateArray(masm);
      Register c = args.C.GenerateArray(masm);

      // Generate constant arrays.
      auto *group_size = masm->GetConstant<mkl_int_t>(batch);
      auto *lda_array = masm->GetConstant<mkl_int_t>(lda);
      auto *ldb_array = masm->GetConstant<mkl_int_t>(ldb);
      auto *ldc_array = masm->GetConstant<mkl_int_t>(ldc);

      StaticData *alpha_array;
      StaticData *beta_array;
      if (args.type == DT_FLOAT) {
        alpha_array = masm->GetConstant<float>(1.0);
        beta_array = masm->GetConstant<float>(accumulate_ ? 1.0 : 0.0);
      } else {
        alpha_array = masm->GetConstant<double>(1.0);
        beta_array = masm->GetConstant<double>(accumulate_ ? 1.0 : 0.0);
      }

      auto *transa_array = masm->GetConstant<mkl_int_t>(args.A.op());
      auto *transb_array = masm->GetConstant<mkl_int_t>(args.B.op());
      auto *m_array = masm->GetConstant<mkl_int_t>(m);
      auto *n_array = masm->GetConstant<mkl_int_t>(n);
      auto *k_array = masm->GetConstant<mkl_int_t>(k);

      // Set up arguments for batch gemm.
      Register tmp = masm->rr().alloc();
      __ leaq(tmp, group_size->address());
      __ pushq(tmp);
      __ pushq(Immediate(1));  // group count
      __ leaq(tmp, ldc_array->address());
      __ pushq(tmp);
      __ pushq(c);
      __ leaq(tmp, beta_array->address());
      __ pushq(tmp);
      __ leaq(tmp, ldb_array->address());
      __ pushq(tmp);
      __ pushq(b);
      __ leaq(tmp, lda_array->address());
      __ pushq(tmp);
      __ pushq(a);
      __ leaq(tmp, alpha_array->address());
      __ pushq(tmp);

      __ movq(arg_reg_1, Immediate(MKL_ROW_MAJOR));
      __ leaq(arg_reg_2, transa_array->address());
      __ leaq(arg_reg_3, transb_array->address());
      __ leaq(arg_reg_4, m_array->address());
      __ leaq(arg_reg_5, n_array->address());
      __ leaq(arg_reg_6, k_array->address());

      // Call MKL cblas_gemm.
      if (args.type == DT_FLOAT) {
        __ call_extern(cblas_sgemm_batch, "cblas_sgemm_batch");
      } else {
        __ call_extern(cblas_dgemm_batch, "cblas_dgemm_batch");
      }
      __ addq(rsp, Immediate(batch * 3 * 8 + 10 * 8));
      step->set_variant("*" + std::to_string(batch));
    }
  }

  int64 Complexity(const Step *step) override {
    Args args(step, accumulate_);
    int64 ops = args.C.rows() * args.C.cols();
    ops *= args.A.cols() * 2;
    ops *= args.C.batchsize();
    return ops;
  }

 private:
  // Matrix argument with optional transpose.
  struct Matrix {
    Matrix(Tensor *tensor, bool t) : tensor(tensor), transpose(t) {}

    // Element data type.
    Type type() const { return tensor->type(); }

    // Tensor rank.
    int rank() const { return tensor->rank(); }

    // Number of batch dimensions.
    int batchdims() const { return rank() - 2; }

    // Batch size.
    int batchsize() const { return tensor->shape().outer(batchdims()); }

    // Checked for batch of matrices.
    bool batched() const { return batchsize() > 1; }

    // Rows and comlumns after optional transpose.
    int rows() const { return tensor->dim(rank() - (transpose ? 1 : 2)); }
    int cols() const { return tensor->dim(rank() - (transpose ? 2 : 1)); }

    // Matrix stride for outer dimension.
    int stride() const { return tensor->stride(batchdims()); }

    // Size of (each) matrix.
    int size() const {
      return batched() ? tensor->stride(batchdims() - 1) : tensor->size();
    }

    // Matrix operation.
    MKLTranspose op() const {
      return transpose ? MKL_TRANS : MKL_NO_TRANS;
    }

    // Generate array of pointers to matrices on the stack. Returns register
    // pointing to the array.
    Register GenerateArray(MacroAssembler *masm) {
      Register mat = masm->rr().alloc();
      Register cnt = masm->rr().alloc();
      Label l;
      __ LoadTensorAddress(mat, tensor);
      __ addq(mat, Immediate(tensor->size()));
      __ xorq(cnt, cnt);
      __ bind(&l);
      __ subq(mat, Immediate(size()));
      __ pushq(mat);
      __ incq(cnt);
      __ cmpq(cnt, Immediate(batchsize()));
      __ j(less, &l);
      __ movq(mat, rsp);
      masm->rr().release(cnt);
      return mat;
    }

    Tensor *tensor;   // underlying tensor for matrix
    bool transpose;   // transposed matrix
  };

  // Arguments for MatMul kernel.
  struct Args {
    Args(const Step *step, bool accumulate)
      : A(accumulate ? step->input(1) : step->input(0),
          step->GetAttr("transpose_a", false)),
        B(accumulate ? step->input(2) : step->input(1),
          step->GetAttr("transpose_b", false)),
        C(accumulate ? step->input(0) : step->output(0),
          step->GetAttr("transpose_c", false)),
        type(C.type()),
        traits(TypeTraits::of(type)) {}

    // Check that shapes and types are compatible.
    bool Compatible() const {
      // Check types.
      if (type != DT_FLOAT && type != DT_DOUBLE) return false;
      if (A.type() != type || B.type() != type) return false;

      // Output cannot be transposed.
      if (C.transpose) return false;

      // Check ranks.
      if (C.rank() < 2) return false;
      if (A.rank() != C.rank()) return false;
      if (B.rank() != C.rank()) return false;

      // Check shapes.
      if (A.rows() != C.rows()) return false;
      if (A.cols() != B.rows()) return false;
      if (B.cols() != C.cols()) return false;

      // Check batch size.
      if (C.batchsize() != A.batchsize()) return false;
      if (C.batchsize() != B.batchsize()) return false;

      return true;
    }

    // Arguments.
    Matrix A;
    Matrix B;
    Matrix C;

    // Datatype.
    Type type;
    const TypeTraits &traits;
  };

  // Network resource for MKL jitter.
  struct MKLJitter : public Network::Resource {
    MKLJitter(void *jitter) : jitter(jitter) {}
    ~MKLJitter() override { mkl_jit_destroy(jitter); }
    void *jitter;
  };

 private:
  bool accumulate_;  // matmul with assignment
};

void RegisterMKLLibrary(Library *library) {
  library->Register(new MKLMatMul(false));
  library->Register(new MKLMatMul(true));
}

}  // namespace myelin
}  // namespace sling

