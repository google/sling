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

#ifndef SLING_MYELIN_CUDA_CUDA_KERNEL_H_
#define SLING_MYELIN_CUDA_CUDA_KERNEL_H_

#include <string>

#include "sling/myelin/compute.h"
#include "sling/myelin/macro-assembler.h"
#include "sling/myelin/cuda/cuda.h"

namespace sling {
namespace myelin {

// PTX macro-assembler for generating code for CUDA kernels.
class PTXMacroAssembler : public PTXAssembler {
 public:
  PTXMacroAssembler(const string &name);

  // Parameter register for data parameter.
  const PTXReg &data() const { return data_; }

  // Grid size for kernel.
  int grid_dim(int d) const { return grid_dim_[d]; }
  void set_grid_dim(int d, int size) { grid_dim_[d] = size; }
  void set_grid_dims(int x = 1, int y = 1, int z = 1) {
    grid_dim_[0] = x;
    grid_dim_[1] = y;
    grid_dim_[2] = z;
  }

  // Return grid size for kernel.
  int grid_size() const { return grid_dim_[0] * grid_dim_[1] * grid_dim_[2]; }

  // Block dimensions for kernel.
  int block_dim(int d) const { return block_dim_[d]; }
  void set_block_dim(int d, int size) { block_dim_[d] = size; }
  void set_block_dims(int x = 1, int y = 1, int z = 1) {
    block_dim_[0] = x;
    block_dim_[1] = y;
    block_dim_[2] = z;
  }

  // Return requested block size for kernel.
  int block_size() const {
    return block_dim_[0] * block_dim_[1] * block_dim_[2];
  }

  // Load address of tensor into register.
  void LoadTensorAddress(const PTXReg &reg, Tensor *tensor);

  // Load the kernel thread index for dimension.
  void LoadThreadIndex(const PTXReg &idx, int d);

  // Load the kernel thread index within block for dimension.
  void LoadBlockThreadIndex(const PTXReg &idx, int d);

  // Load the kernel block index for dimension.
  void LoadBlockIndex(const PTXReg &idx, int d);

  // Load the kernel block size for dimension.
  void LoadBlockDim(const PTXReg &idx, int d);

 private:
  // Data instance parameter.
  PTXReg data_;

  // Grid size for x, y, and z dimension.
  int grid_dim_[3];

  // Block size. This is estimated automaticallly if not explicitly set.
  int block_dim_[3];
};

// Kernel for launching CUDA kernels on GPUs.
class CUDAKernel : public Kernel {
 public:
  // Run kernel on CUDA device.
  Placement Location() override { return DEVICE; }

  // Checks if CUDA is supported by runtime.
  bool Supports(Step *step) override;

  // Generate code for launching CUDA kernel.
  void Generate(Step *step, MacroAssembler *masm) override;

  // Generate PTX code for CUDA kernel.
  virtual void GeneratePTX(Step *step, PTXMacroAssembler *ptx) = 0;
};

}  // namespace myelin
}  // namespace sling

#endif  // SLING_MYELIN_CUDA_CUDA_KERNEL_H_

