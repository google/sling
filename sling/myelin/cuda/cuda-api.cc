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

#include "sling/myelin/cuda/cuda-api.h"

#include <dlfcn.h>

#include "sling/base/logging.h"

namespace sling {
namespace myelin {

// Handle to CUDA library.
static void *cuda_lib = nullptr;

// Handle to cuBLASLt library.
static void *cublaslt_lib = nullptr;

// CUDA driver API functions.
CUresult (*cuDriverGetVersion)(int *version);
CUresult (*cuInit)(unsigned int flags);
CUresult (*cuDeviceGetCount)(int *count);
CUresult (*cuDeviceGet)(CUdevice *device, int ordinal);
CUresult (*cuDeviceGetName)(char *name, int len, CUdevice dev);
CUresult (*cuDeviceComputeCapability)(int *major,
                                      int *minor,
                                      CUdevice dev);
CUresult (*cuDeviceTotalMem)(size_t *bytes, CUdevice dev);
CUresult (*cuDeviceGetAttribute)(int *pi,
                                 CUdevice_attribute attrib,
                                 CUdevice dev);
CUresult (*cuCtxCreate)(CUcontext *pctx,
                        unsigned int flags,
                        CUdevice dev);
CUresult (*cuCtxDestroy)(CUcontext ctx);
CUresult (*cuCtxGetCurrent)(CUcontext* pctx);
CUresult (*cuDevicePrimaryCtxRetain)(CUcontext *pctx, CUdevice dev);
CUresult (*cuDevicePrimaryCtxRelease)(CUdevice dev);
CUresult (*cuModuleLoadDataEx)(CUmodule *module,
                               const void *image,
                               unsigned int num_options,
                               CUjit_option *options,
                               void **option_values);
CUresult (*cuModuleUnload)(CUmodule hmod);
CUresult (*cuModuleGetFunction)(CUfunction *hfunc,
                                CUmodule hmod,
                                const char *name);
CUresult (*cuFuncGetAttribute)(int *pi,
                               CUfunction_attribute attrib,
                               CUfunction hfunc);
CUresult (*cuOccupancyMaxPotentialBlockSize)(int *min_grid_size,
                                             int *block_size,
                                             CUfunction func,
                                             CUoccupancyB2DSize msfunc,
                                             size_t dynamic_smem_size,
                                             int block_size_limit);
CUresult (*cuMemAlloc)(CUdeviceptr *dptr, size_t size);
CUresult (*cuMemFree)(CUdeviceptr dptr);
CUresult (*cuMemAllocHost)(void **pdata, size_t size);
CUresult (*cuMemFreeHost)(void *ptr);
CUresult (*cuMemsetD8)(CUdeviceptr dptr, unsigned char uc, size_t n);
CUresult (*cuMemcpyHtoD)(CUdeviceptr dst,
                         const void *src,
                         size_t size);
CUresult (*cuMemcpyDtoH)(void *dst,
                         CUdeviceptr src,
                         size_t size);
CUresult (*cuMemcpyHtoDAsync)(CUdeviceptr dst,
                              const void *src,
                              size_t size,
                              CUstream hstream);
CUresult (*cuMemcpyDtoHAsync)(void *dst,
                              CUdeviceptr src,
                              size_t size,
                              CUstream hstream);
CUresult (*cuMemcpyDtoD)(CUdeviceptr dst, CUdeviceptr src, size_t size);
CUresult (*cuStreamCreate)(CUstream *hstream, unsigned int flags);
CUresult (*cuStreamDestroy)(CUstream hstream);
CUresult (*cuStreamSynchronize)(CUstream hstream);
CUresult (*cuLaunchKernel)(CUfunction f,
                           unsigned int grid_dim_x,
                           unsigned int grid_dim_y,
                           unsigned int grid_dim_z,
                           unsigned int block_dim_x,
                           unsigned int block_dim_y,
                           unsigned int block_dim_z,
                           unsigned int shared_mem_bytes,
                           CUstream hstream,
                           void **kernelParams,
                           void **extra);
CUresult (*cuProfilerInitialize)(const char *config_file,
                                 const char *output_file,
                                 CUoutput_mode output_mode);
CUresult (*cuProfilerStart)();
CUresult (*cuProfilerStop)();

// cuBLASLt API functions.
cublasStatus_t (*cublasLtCreate)(cublasLtHandle_t *handle);
cublasStatus_t (*cublasLtDestroy)(cublasLtHandle_t handle);

cublasStatus_t (*cublasLtCtxInit)(void);
cublasStatus_t (*cublasLtShutdownCtx)(void);

cublasStatus_t (*cublasLtMatmulDescCreate)(
    cublasLtMatmulDesc_t *desc,
    cudaDataType type);

cublasStatus_t (*cublasLtMatmulDescDestroy)(cublasLtMatmulDesc_t desc);

cublasStatus_t (*cublasLtMatmulDescSetAttribute)(
    cublasLtMatmulDesc_t desc,
    cublasLtMatmulDescAttributes_t attr,
    const void *buf,
    size_t size);

cublasStatus_t (*cublasLtMatrixLayoutCreate)(
    cublasLtMatrixLayout_t *layout,
    cudaDataType type,
    uint64_t rows,
    uint64_t cols,
    int64_t ld);

cublasStatus_t (*cublasLtMatrixLayoutDestroy)(
    cublasLtMatrixLayout_t layout);

cublasStatus_t (*cublasLtMatrixLayoutSetAttribute)(
  cublasLtMatrixLayout_t layout,
  cublasLtMatrixLayoutAttribute_t attr,
  void *buf,
  size_t size);

cublasStatus_t (*cublasLtMatmulPreferenceCreate)(
    cublasLtMatmulPreference_t *pref);

cublasStatus_t (*cublasLtMatmulPreferenceDestroy)(
    cublasLtMatmulPreference_t pref);

cublasStatus_t (*cublasLtMatmulPreferenceSetAttribute)(
    cublasLtMatmulPreference_t pref,
    cublasLtMatmulPreferenceAttributes_t attr,
    const void *buf,
    size_t size);

cublasStatus_t (*cublasLtMatmulAlgoGetHeuristic)(
    cublasLtHandle_t handle,
    cublasLtMatmulDesc_t opdesc,
    cublasLtMatrixLayout_t adesc,
    cublasLtMatrixLayout_t bdesc,
    cublasLtMatrixLayout_t cdesc,
    cublasLtMatrixLayout_t ddesc,
    cublasLtMatmulPreference_t preference,
    int requested_algo_count,
    cublasLtMatmulHeuristicResult_t heuristic_results[],
    int *algo_count);

cublasStatus_t (*cublasLtMatmul)(
    cublasLtHandle_t handle,
    cublasLtMatmulDesc_t desc,
    const void *alpha,
    const void *A,
    cublasLtMatrixLayout_t adesc,
    const void *B,
    cublasLtMatrixLayout_t bdesc,
    const void *beta,
    const void *C,
    cublasLtMatrixLayout_t cdesc,
    void *D,
    cublasLtMatrixLayout_t ddesc,
    const cublasLtMatmulAlgo_t *algo,
    void *workspace,
    size_t workspace_size,
    cudaStream_t stream);

#define LOAD_CUDA_FUNCTION(name, version) \
  name = reinterpret_cast<decltype(name)>(dlsym(cuda_lib , #name version)); \
  if (!name) LOG(WARNING) << #name version " not found in CUDA library"

#define LOAD_CUBLASLT_FUNCTION(name) \
  name = reinterpret_cast<decltype(name)>(dlsym(cublaslt_lib , #name)); \
  if (!name) LOG(WARNING) << #name " not found in cuBLASLt library"

#define LIBPATH "/usr/lib/x86_64-linux-gnu"

bool LoadCUDALibrary() {
  // Try to load CUDA library.
  CHECK(cuda_lib == nullptr) << "CUDA library already loaded";
  cuda_lib = dlopen("libcuda.so", RTLD_NOW);
  if (cuda_lib == nullptr) {
    cuda_lib = dlopen(LIBPATH "/libcuda.so.1", RTLD_NOW);
  }
  if (cuda_lib == nullptr) return false;

  // Resolve library functions.
  LOAD_CUDA_FUNCTION(cuDriverGetVersion, "");
  LOAD_CUDA_FUNCTION(cuInit, "");
  LOAD_CUDA_FUNCTION(cuDeviceGetCount, "");
  LOAD_CUDA_FUNCTION(cuDeviceGet, "");
  LOAD_CUDA_FUNCTION(cuDeviceGetName, "");
  LOAD_CUDA_FUNCTION(cuDeviceComputeCapability, "");
  LOAD_CUDA_FUNCTION(cuDeviceTotalMem, "_v2");
  LOAD_CUDA_FUNCTION(cuDeviceGetAttribute, "");
  LOAD_CUDA_FUNCTION(cuCtxCreate, "_v2");
  LOAD_CUDA_FUNCTION(cuCtxDestroy, "");
  LOAD_CUDA_FUNCTION(cuCtxGetCurrent, "");
  LOAD_CUDA_FUNCTION(cuDevicePrimaryCtxRetain, "");
  LOAD_CUDA_FUNCTION(cuDevicePrimaryCtxRelease, "");
  LOAD_CUDA_FUNCTION(cuModuleLoadDataEx, "");
  LOAD_CUDA_FUNCTION(cuModuleUnload, "");
  LOAD_CUDA_FUNCTION(cuModuleGetFunction, "");
  LOAD_CUDA_FUNCTION(cuFuncGetAttribute, "");
  LOAD_CUDA_FUNCTION(cuOccupancyMaxPotentialBlockSize, "");
  LOAD_CUDA_FUNCTION(cuMemAlloc, "_v2");
  LOAD_CUDA_FUNCTION(cuMemFree, "_v2");
  LOAD_CUDA_FUNCTION(cuMemAllocHost, "_v2");
  LOAD_CUDA_FUNCTION(cuMemFreeHost, "");
  LOAD_CUDA_FUNCTION(cuMemsetD8, "_v2");
  LOAD_CUDA_FUNCTION(cuMemcpyHtoD, "_v2");
  LOAD_CUDA_FUNCTION(cuMemcpyDtoH, "_v2");
  LOAD_CUDA_FUNCTION(cuMemcpyHtoDAsync, "_v2");
  LOAD_CUDA_FUNCTION(cuMemcpyDtoHAsync, "_v2");
  LOAD_CUDA_FUNCTION(cuMemcpyDtoD, "_v2");
  LOAD_CUDA_FUNCTION(cuStreamCreate, "");
  LOAD_CUDA_FUNCTION(cuStreamDestroy, "_v2");
  LOAD_CUDA_FUNCTION(cuStreamSynchronize, "");
  LOAD_CUDA_FUNCTION(cuLaunchKernel, "");
  LOAD_CUDA_FUNCTION(cuProfilerInitialize, "");
  LOAD_CUDA_FUNCTION(cuProfilerStart, "");
  LOAD_CUDA_FUNCTION(cuProfilerStop, "");

  // Try to load cuBLASLt library.
  cublaslt_lib = dlopen("libcublasLt.so", RTLD_NOW);
  if (cublaslt_lib == nullptr) {
    cublaslt_lib = dlopen(LIBPATH "/libcublasLt.so", RTLD_NOW);
  }
  if (cublaslt_lib != nullptr) {
    LOAD_CUBLASLT_FUNCTION(cublasLtCreate);
    LOAD_CUBLASLT_FUNCTION(cublasLtDestroy);
    LOAD_CUBLASLT_FUNCTION(cublasLtCtxInit);
    LOAD_CUBLASLT_FUNCTION(cublasLtShutdownCtx);
    LOAD_CUBLASLT_FUNCTION(cublasLtMatmulDescCreate);
    LOAD_CUBLASLT_FUNCTION(cublasLtMatmulDescDestroy);
    LOAD_CUBLASLT_FUNCTION(cublasLtMatmulDescSetAttribute);
    LOAD_CUBLASLT_FUNCTION(cublasLtMatrixLayoutCreate);
    LOAD_CUBLASLT_FUNCTION(cublasLtMatrixLayoutDestroy);
    LOAD_CUBLASLT_FUNCTION(cublasLtMatrixLayoutSetAttribute);
    LOAD_CUBLASLT_FUNCTION(cublasLtMatmulPreferenceCreate);
    LOAD_CUBLASLT_FUNCTION(cublasLtMatmulPreferenceDestroy);
    LOAD_CUBLASLT_FUNCTION(cublasLtMatmulPreferenceSetAttribute);
    LOAD_CUBLASLT_FUNCTION(cublasLtMatmulAlgoGetHeuristic);
    LOAD_CUBLASLT_FUNCTION(cublasLtMatmul);
  }

  return true;
}

bool HasCuBLASLt() {
  return cublaslt_lib != nullptr;
}

}  // namespace myelin
}  // namespace sling

