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

#include "myelin/cuda/cuda-api.h"

#include <dlfcn.h>

#include "base/logging.h"

namespace sling {
namespace myelin {

// Handle to CUDA library.
static void *cuda_lib = nullptr;

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
CUresult (*cuCtxDetach)(CUcontext ctx);
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

#define LOAD_CUDA_FUNCTION(name, version) \
  name = reinterpret_cast<decltype(name)>(dlsym(cuda_lib , #name version)); \
  if (!name) LOG(WARNING) << #name version " not found in CUDA library"

bool LoadCUDALibrary() {
  // Try to load CUDA library.
  CHECK(cuda_lib == nullptr) << "CUDA library already loaded";
  cuda_lib = dlopen("libcuda.so", RTLD_LAZY);
  if (cuda_lib == nullptr) {
    cuda_lib = dlopen("/usr/lib/x86_64-linux-gnu/libcuda.so.1", RTLD_LAZY);
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
  LOAD_CUDA_FUNCTION(cuCtxDetach, "");
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

  return true;
}

}  // namespace myelin
}  // namespace sling

