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

#include "sling/myelin/cuda/cuda-kernel.h"

#include <stdlib.h>

#include "sling/base/logging.h"
#include "sling/myelin/compute.h"
#include "sling/myelin/cuda/cuda.h"
#include "sling/myelin/cuda/cuda-runtime.h"

#define __ masm->

namespace sling {
namespace myelin {

using namespace jit;

// Temporary register.
static Register tmpreg = r10;

PTXMacroAssembler::PTXMacroAssembler(const string &name): PTXAssembler(name) {
  // Kernel functions take one parameter with the address of the device data
  // instance block.
  PTXReg instance = param("b64", "instance");
  data_ = reg("b64", "data");
  emit("ld_param_b64", data_, PTXAddr(instance));

  // Grid and block size for kernel.
  grid_dim_[0] = grid_dim_[1] = grid_dim_[2] = 1;
  block_dim_[0] = block_dim_[1] = block_dim_[2] = 0;
}

void PTXMacroAssembler::LoadTensorAddress(const PTXReg &reg, Tensor *tensor) {
  if (tensor->IsGlobal()) {
    // Read from global tensor.
    emit("ld.const.u64", reg, PTXAddr(abs(tensor->device_data())));
  } else if (tensor->ref()) {
    // Read from reference tensor.
    emit("ld.global.u64", reg, PTXAddr(data(), tensor->device_offset()));
  } else if (tensor->device_offset() == 0) {
    // Read from first instance tensor.
    emit("mov.u64", reg, data());
  } else {
    // Read from instance tensor.
    emit("add.u64", reg, data(), PTXImm(tensor->device_offset()));
  }
}

void PTXMacroAssembler::LoadThreadIndex(const PTXReg &idx, int d) {
  static const char *thridx[] = {"thridxx", "thridxy", "thridxz"};
  static const char *blkidx[] = {"blkidxx", "blkidxy", "blkidxz"};
  static const char *blkdim[] = {"blkdimx", "blkdimy", "blkdimz"};

  PTXReg tidx = reg("b32", thridx[d]);
  PTXReg bidx = reg("b32", blkidx[d]);
  PTXReg bdim = reg("b32", blkdim[d]);
  LoadBlockThreadIndex(tidx, d);
  LoadBlockIndex(bidx, d);
  LoadBlockDim(bdim, d);
  emit("mad.lo.u32", idx, bidx, bdim, tidx);
}

void PTXMacroAssembler::LoadBlockThreadIndex(const PTXReg &idx, int d) {
  static const char *tid[] = {"%tid.x", "%tid.y", "%tid.z"};
  emit("mov.u32", idx, PTXLiteral(tid[d]));
}

void PTXMacroAssembler::LoadBlockIndex(const PTXReg &idx, int d) {
  static const char *ctaid[] = {"%ctaid.x", "%ctaid.y", "%ctaid.z"};
  emit("mov.u32", idx, PTXLiteral(ctaid[d]));
}

void PTXMacroAssembler::LoadBlockDim(const PTXReg &idx, int d) {
  static const char *ntid[] = {"%ntid.x", "%ntid.y", "%ntid.z"};
  emit("mov.u32", idx, PTXLiteral(ntid[d]));
}

bool CUDAKernel::Supports(Step *step) {
  CUDADevice *device = step->cell()->runtime()->Device();
  return device != nullptr;
}

void CUDAKernel::Generate(Step *step, MacroAssembler *masm) {
  // Set up macro-assembler for generating PTX code for kernel.
  CUDADevice *device = step->cell()->runtime()->Device();
  CHECK(device != nullptr);
  string name = step->name();
  for (char &c : name) {
    if (c == '/' || c == '-') c = '_';
  }
  PTXMacroAssembler ptx(name.c_str());
  ptx.set_target(device->capability());

  // Generate PTX code for GPU kernel.
  GeneratePTX(step, &ptx);
  string code;
  ptx.Generate(&code);
  VLOG(9) << step->name() << " PTX code:\n" << code;

  // Compile PTX into a CUDA module.
  CUDAModule *module = device->Compile(code.c_str());
  CUDAFunction func(*module, name.c_str());
  step->cell()->network()->linker()->AddDeviceCode(step, code);

  VLOG(9) << step->name() << " PTX usage: "
          << func.shared_size() << " shared, "
          << func.const_size() << " const, "
          << func.local_size() << " local, "
          << func.num_regs() << " regs";

  // Compute kernel block size.
  int grid_size = ptx.grid_size();
  int min_grid_size;
  int block_size;
  CHECK_CUDA(cuOccupancyMaxPotentialBlockSize (
      &min_grid_size, &block_size, func.handle(),
      nullptr, func.shared_size(), grid_size));

  // Compute block dimensions.
  int x = ptx.grid_dim(0);
  int y = ptx.grid_dim(1);
  int z = ptx.grid_dim(2);
  int block_dim_x = ptx.block_dim(0);
  int block_dim_y = ptx.block_dim(1);
  int block_dim_z = ptx.block_dim(2);
  if (ptx.block_size() == 0) {
    block_dim_x = block_dim_y = block_dim_z = 1;
    if (x >= block_size) {
      // The x dimension takes up the whole block.
      block_dim_x = block_size;
    } else {
      // Distribute block to y dimension.
      block_dim_x = x;
      block_dim_y = block_size / block_dim_x;
      if (y < block_dim_y) {
        // Distribute block to z dimension.
        block_dim_y = y;
        block_dim_z = block_size / (block_dim_x * block_dim_y);
        if (z < block_dim_z) block_dim_z = z;
      }
    }
  }

  // Compute grid dimensions.
  int grid_dim_x = (x + block_dim_x - 1) / block_dim_x;
  int grid_dim_y = (y + block_dim_y - 1) / block_dim_y;
  int grid_dim_z = (z + block_dim_z - 1) / block_dim_z;

  VLOG(5) << step->name() << ", block size " << block_size
          << ", min grid size " << min_grid_size << ", thread ("
          << x << "," << y << "," << z
          << "), grid ("
          << grid_dim_x << "," << grid_dim_y << "," << grid_dim_z
          << "), block ("
          << block_dim_x << "," << block_dim_y << "," << block_dim_z << ")";

  // Build parameter array with device instance address as the only parameter.
  Register params = tmpreg;
  __ pushq(Operand(masm->instance(), offsetof(CUDAInstance, data)));
  __ pushq(rsp);
  __ movq(params, rsp);

  // Set up register-based parameters for launching kernel.
  __ movp(arg_reg_1, func.handle());
  __ movq(arg_reg_2, Immediate(grid_dim_x));
  __ movq(arg_reg_3, Immediate(grid_dim_y));
  __ movq(arg_reg_4, Immediate(grid_dim_z));
  __ movq(arg_reg_5, Immediate(block_dim_x));
  __ movq(arg_reg_6, Immediate(block_dim_y));

  // Set up stack-based parameters for launching kernel.
  __ pushq(Immediate(0));  // extra options
  __ pushq(params);
  __ pushq(Operand(masm->instance(), StreamOffset(step)));
  __ pushq(Immediate(0));  // shared memory
  __ pushq(Immediate(block_dim_z));

  // Call cuLaunchKernel.
  __ load_extern(tmpreg, reinterpret_cast<void *>(cuLaunchKernel),
                 "cuLaunchKernel");
  __ call(tmpreg);
  __ addq(rsp, Immediate(7 * 8));
  CUDARuntime::EmitStatusCheck("cuLaunchKernel", masm);
}

}  // namespace myelin
}  // namespace sling

