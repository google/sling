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


// Nvidia CUDA driver API.
// See also: http://docs.nvidia.com/cuda/cuda-driver-api

#ifndef SLING_MYELIN_CUDA_CUDA_API_H_
#define SLING_MYELIN_CUDA_CUDA_API_H_

#include <stdlib.h>
#include <stdint.h>

namespace sling {
namespace myelin {

// CUDA handles.
typedef uint64_t CUdeviceptr;
typedef int CUdevice;
typedef struct CUctx_st *CUcontext;
typedef struct CUmod_st *CUmodule;
typedef struct CUfunc_st *CUfunction;
typedef struct CUstream_st *CUstream;

// CUDA error codes.
enum CUresult {
  CUDA_SUCCESS                              = 0,
  CUDA_ERROR_INVALID_VALUE                  = 1,
  CUDA_ERROR_OUT_OF_MEMORY                  = 2,
  CUDA_ERROR_NOT_INITIALIZED                = 3,
  CUDA_ERROR_DEINITIALIZED                  = 4,
  CUDA_ERROR_PROFILER_DISABLED              = 5,
  CUDA_ERROR_PROFILER_NOT_INITIALIZED       = 6,
  CUDA_ERROR_PROFILER_ALREADY_STARTED       = 7,
  CUDA_ERROR_PROFILER_ALREADY_STOPPED       = 8,
  CUDA_ERROR_NO_DEVICE                      = 100,
  CUDA_ERROR_INVALID_DEVICE                 = 101,
  CUDA_ERROR_INVALID_IMAGE                  = 200,
  CUDA_ERROR_INVALID_CONTEXT                = 201,
  CUDA_ERROR_CONTEXT_ALREADY_CURRENT        = 202,
  CUDA_ERROR_MAP_FAILED                     = 205,
  CUDA_ERROR_UNMAP_FAILED                   = 206,
  CUDA_ERROR_ARRAY_IS_MAPPED                = 207,
  CUDA_ERROR_ALREADY_MAPPED                 = 208,
  CUDA_ERROR_NO_BINARY_FOR_GPU              = 209,
  CUDA_ERROR_ALREADY_ACQUIRED               = 210,
  CUDA_ERROR_NOT_MAPPED                     = 211,
  CUDA_ERROR_NOT_MAPPED_AS_ARRAY            = 212,
  CUDA_ERROR_NOT_MAPPED_AS_POINTER          = 213,
  CUDA_ERROR_ECC_UNCORRECTABLE              = 214,
  CUDA_ERROR_UNSUPPORTED_LIMIT              = 215,
  CUDA_ERROR_CONTEXT_ALREADY_IN_USE         = 216,
  CUDA_ERROR_PEER_ACCESS_UNSUPPORTED        = 217,
  CUDA_ERROR_INVALID_PTX                    = 218,
  CUDA_ERROR_INVALID_GRAPHICS_CONTEXT       = 219,
  CUDA_ERROR_INVALID_SOURCE                 = 300,
  CUDA_ERROR_FILE_NOT_FOUND                 = 301,
  CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND = 302,
  CUDA_ERROR_SHARED_OBJECT_INIT_FAILED      = 303,
  CUDA_ERROR_OPERATING_SYSTEM               = 304,
  CUDA_ERROR_INVALID_HANDLE                 = 400,
  CUDA_ERROR_NOT_FOUND                      = 500,
  CUDA_ERROR_NOT_READY                      = 600,
  CUDA_ERROR_ILLEGAL_ADDRESS                = 700,
  CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES        = 701,
  CUDA_ERROR_LAUNCH_TIMEOUT                 = 702,
  CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING  = 703,
  CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED    = 704,
  CUDA_ERROR_PEER_ACCESS_NOT_ENABLED        = 705,
  CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE         = 708,
  CUDA_ERROR_CONTEXT_IS_DESTROYED           = 709,
  CUDA_ERROR_ASSERT                         = 710,
  CUDA_ERROR_TOO_MANY_PEERS                 = 711,
  CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED = 712,
  CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED     = 713,
  CUDA_ERROR_HARDWARE_STACK_ERROR           = 714,
  CUDA_ERROR_ILLEGAL_INSTRUCTION            = 715,
  CUDA_ERROR_MISALIGNED_ADDRESS             = 716,
  CUDA_ERROR_INVALID_ADDRESS_SPACE          = 717,
  CUDA_ERROR_INVALID_PC                     = 718,
  CUDA_ERROR_LAUNCH_FAILED                  = 719,
  CUDA_ERROR_NOT_PERMITTED                  = 800,
  CUDA_ERROR_NOT_SUPPORTED                  = 801,
  CUDA_ERROR_UNKNOWN                        = 999,
};

// CUDA device attributes.
enum CUdevice_attribute {
  CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK = 1,
  CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X = 2,
  CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y = 3,
  CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z = 4,
  CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X = 5,
  CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y = 6,
  CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z = 7,
  CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK = 8,
  CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY = 9,
  CU_DEVICE_ATTRIBUTE_WARP_SIZE = 10,
  CU_DEVICE_ATTRIBUTE_MAX_PITCH = 11,
  CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK = 12,
  CU_DEVICE_ATTRIBUTE_CLOCK_RATE = 13,
  CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT = 14,
  CU_DEVICE_ATTRIBUTE_GPU_OVERLAP = 15,
  CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT = 16,
  CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT = 17,
  CU_DEVICE_ATTRIBUTE_INTEGRATED = 18,
  CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY = 19,
  CU_DEVICE_ATTRIBUTE_COMPUTE_MODE = 20,
  CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_WIDTH = 21,
  CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_WIDTH = 22,
  CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_HEIGHT = 23,
  CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH = 24,
  CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT = 25,
  CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH = 26,
  CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_WIDTH = 27,
  CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_HEIGHT = 28,
  CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_LAYERS = 29,
  CU_DEVICE_ATTRIBUTE_SURFACE_ALIGNMENT = 30,
  CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS = 31,
  CU_DEVICE_ATTRIBUTE_ECC_ENABLED = 32,
  CU_DEVICE_ATTRIBUTE_PCI_BUS_ID = 33,
  CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID = 34,
  CU_DEVICE_ATTRIBUTE_TCC_DRIVER = 35,
  CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE = 36,
  CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH = 37,
  CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE = 38,
  CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR = 39,
  CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT = 40,
  CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING = 41,
  CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_WIDTH = 42,
  CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_LAYERS = 43,
  CU_DEVICE_ATTRIBUTE_CAN_TEX2D_GATHER = 44,
  CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_GATHER_WIDTH = 45,
  CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_GATHER_HEIGHT = 46,
  CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH_ALTERNATE = 47,
  CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT_ALTERNATE = 48,
  CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH_ALTERNATE = 49,
  CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID = 50,
  CU_DEVICE_ATTRIBUTE_TEXTURE_PITCH_ALIGNMENT = 51,
  CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_WIDTH = 52,
  CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_WIDTH = 53,
  CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_LAYERS = 54,
  CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_WIDTH = 55,
  CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_WIDTH = 56,
  CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_HEIGHT = 57,
  CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_WIDTH = 58,
  CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_HEIGHT = 59,
  CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_DEPTH = 60,
  CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_WIDTH = 61,
  CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_LAYERS = 62,
  CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_WIDTH = 63,
  CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_HEIGHT = 64,
  CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_LAYERS = 65,
  CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_WIDTH = 66,
  CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_WIDTH = 67,
  CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_LAYERS = 68,
  CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LINEAR_WIDTH = 69,
  CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_WIDTH = 70,
  CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_HEIGHT = 71,
  CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_PITCH = 72,
  CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_WIDTH = 73,
  CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_HEIGHT = 74,
  CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR = 75,
  CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR = 76,
  CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_MIPMAPPED_WIDTH = 77,
  CU_DEVICE_ATTRIBUTE_STREAM_PRIORITIES_SUPPORTED = 78,
  CU_DEVICE_ATTRIBUTE_GLOBAL_L1_CACHE_SUPPORTED = 79,
  CU_DEVICE_ATTRIBUTE_LOCAL_L1_CACHE_SUPPORTED = 80,
  CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR = 81,
  CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR = 82,
  CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY = 83,
  CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD = 84,
  CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD_GROUP_ID = 85,
  CU_DEVICE_ATTRIBUTE_MAX
};

// CUDA JIT options.
enum CUjit_option {
  CU_JIT_MAX_REGISTERS = 0,
  CU_JIT_THREADS_PER_BLOCK,
  CU_JIT_WALL_TIME,
  CU_JIT_INFO_LOG_BUFFER,
  CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES,
  CU_JIT_ERROR_LOG_BUFFER,
  CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES,
  CU_JIT_OPTIMIZATION_LEVEL,
  CU_JIT_TARGET_FROM_CUCONTEXT,
  CU_JIT_TARGET,
  CU_JIT_FALLBACK_STRATEGY,
  CU_JIT_GENERATE_DEBUG_INFO,
  CU_JIT_LOG_VERBOSE,
  CU_JIT_GENERATE_LINE_INFO,
  CU_JIT_CACHE_MODE,
  CU_JIT_NUM_OPTIONS
};

// CUDA function attributes.
enum  CUfunction_attribute {
  CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK = 0,
  CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES = 1,
  CU_FUNC_ATTRIBUTE_CONST_SIZE_BYTES = 2,
  CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES = 3,
  CU_FUNC_ATTRIBUTE_NUM_REGS = 4,
  CU_FUNC_ATTRIBUTE_PTX_VERSION = 5,
  CU_FUNC_ATTRIBUTE_BINARY_VERSION = 6,
  CU_FUNC_ATTRIBUTE_CACHE_MODE_CA = 7,
  CU_FUNC_ATTRIBUTE_MAX
};

// CUDA cubin fallback strategies.
enum CUjit_fallback {
  CU_PREFER_PTX = 0,
  CU_PREFER_BINARY,
};

// CUDA context creation flags.
enum CUctx_flags {
  CU_CTX_SCHED_AUTO          = 0x00,
  CU_CTX_SCHED_SPIN          = 0x01,
  CU_CTX_SCHED_YIELD         = 0x02,
  CU_CTX_SCHED_BLOCKING_SYNC = 0x04,
  CU_CTX_SCHED_MASK          = 0x07,

  CU_CTX_MAP_HOST            = 0x08,
  CU_CTX_LMEM_RESIZE_TO_MAX  = 0x10,
  CU_CTX_FLAGS_MASK          = 0x1f,
};

// CUDA stream creation flags.
enum CUstream_flags {
  CU_STREAM_DEFAULT      = 0x0,
  CU_STREAM_NON_BLOCKING = 0x1,
};

// CUDA profiler output mode.
enum CUoutput_mode {
  CU_OUT_KEY_VALUE_PAIR  = 0x00,
  CU_OUT_CSV             = 0x01,
};

// Per-block dynamic shared memory mapping function.
typedef size_t (*CUoccupancyB2DSize)(int block_size);

// CUDA driver API functions.
extern CUresult (*cuDriverGetVersion)(int *version);
extern CUresult (*cuInit)(unsigned int flags);
extern CUresult (*cuDeviceGetCount)(int *count);
extern CUresult (*cuDeviceGet)(CUdevice *device, int ordinal);
extern CUresult (*cuDeviceGetName)(char *name, int len, CUdevice dev);
extern CUresult (*cuDeviceComputeCapability)(int *major,
                                             int *minor,
                                             CUdevice dev);
extern CUresult (*cuDeviceTotalMem)(size_t *bytes, CUdevice dev);
extern CUresult (*cuDeviceGetAttribute)(int *pi,
                                        CUdevice_attribute attrib,
                                        CUdevice dev);
extern CUresult (*cuCtxCreate)(CUcontext *pctx,
                               unsigned int flags,
                               CUdevice dev);
extern CUresult (*cuCtxDetach)(CUcontext ctx);
extern CUresult (*cuModuleLoadDataEx)(CUmodule *module,
                                      const void *image,
                                      unsigned int num_options,
                                      CUjit_option *options,
                                      void **option_values);
extern CUresult (*cuModuleUnload)(CUmodule hmod);
extern CUresult (*cuModuleGetFunction)(CUfunction *hfunc,
                                       CUmodule hmod,
                                       const char *name);
extern CUresult (*cuFuncGetAttribute)(int *pi,
                                      CUfunction_attribute attrib,
                                      CUfunction hfunc);
extern CUresult (*cuOccupancyMaxPotentialBlockSize)(int *min_grid_size,
                                                    int *block_size,
                                                    CUfunction func,
                                                    CUoccupancyB2DSize msfunc,
                                                    size_t dynamic_smem_size,
                                                    int block_size_limit);
extern CUresult (*cuMemAlloc)(CUdeviceptr *dptr, size_t size);
extern CUresult (*cuMemFree)(CUdeviceptr dptr);
extern CUresult (*cuMemAllocHost)(void **pdata, size_t size);
extern CUresult (*cuMemFreeHost)(void *ptr);
extern CUresult (*cuMemsetD8)(CUdeviceptr dptr, unsigned char uc, size_t n);
extern CUresult (*cuMemcpyHtoD)(CUdeviceptr dst,
                                const void *src,
                                size_t size);
extern CUresult (*cuMemcpyDtoH)(void *dst,
                                CUdeviceptr src,
                                size_t size);
extern CUresult (*cuMemcpyHtoDAsync)(CUdeviceptr dst,
                                     const void *src,
                                     size_t size,
                                     CUstream hstream);
extern CUresult (*cuMemcpyDtoHAsync)(void *dst,
                                     CUdeviceptr src,
                                     size_t size,
                                     CUstream hstream);
extern CUresult (*cuMemcpyDtoD)(CUdeviceptr dst, CUdeviceptr src, size_t size);
extern CUresult (*cuStreamCreate)(CUstream *hstream, unsigned int flags);
extern CUresult (*cuStreamDestroy)(CUstream hstream);
extern CUresult (*cuStreamSynchronize)(CUstream hstream);
extern CUresult (*cuLaunchKernel)(CUfunction f,
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
extern CUresult (*cuProfilerInitialize)(const char *config_file,
                                        const char *output_file,
                                        CUoutput_mode output_mode);
extern CUresult (*cuProfilerStart)();
extern CUresult (*cuProfilerStop)();

// Load CUDA library. Return false if CUDA library not found.
bool LoadCUDALibrary();

}  // namespace myelin
}  // namespace sling

#endif  // SLING_MYELIN_CUDA_CUDA_API_H_

