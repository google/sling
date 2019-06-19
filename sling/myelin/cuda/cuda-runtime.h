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

#ifndef SLING_MYELIN_CUDA_CUDA_RUNTIME_H_
#define SLING_MYELIN_CUDA_CUDA_RUNTIME_H_

#include <string>

#include "sling/base/types.h"
#include "sling/myelin/compute.h"
#include "sling/myelin/cuda/cuda.h"

namespace sling {
namespace myelin {

// Instance data for cells running on CUDA devices. This is stored at the
// beginning of the host data instance block.
struct CUDAInstance {
  DevicePtr data;       // pointer to instance data allocated on device
  CUstream mainstream;  // stream for synchronizing operations in main task
};

// Runtime for executing kernels on GPUs using the Nvidia CUDA API.
class CUDARuntime : public Runtime {
 public:
  ~CUDARuntime();

  // Connect runtime to CUDA devices. If the device number is -1 the runtime
  // tries to selected the best GPU device for computations.
  void Connect(int device_number = -1, int flags = 0);

  // Disconnect runtime from device.
  void Disconnect();

  // Return description of the GPU device.
  string Description() override;

  // Return CUDA device for runtime.
  CUDADevice *Device() override { return device_; }

  // Check if runtime has been connected to a device.
  bool connected() const { return device_ != nullptr; }

  // Instance data allocation.
  void AllocateInstance(Instance *instance) override;
  void FreeInstance(Instance *instance) override;
  void ClearInstance(Instance *instance) override;

  // Channel allocation.
  char *AllocateChannel(char *data,
                        size_t old_size,
                        size_t new_size,
                        size_t alignment,
                        Placement placement) override;
  void ClearChannel(char *data, size_t pos,
                    size_t size,
                    Placement placement) override;
  void FreeChannel(char *data, Placement placement) override;

  // Asynchronous execution.
  bool SupportsAsync() override { return true; }
  TaskFunc StartTaskFunc() override { return StartTask; }
  TaskFunc WaitTaskFunc() override { return WaitTask; }
  InstanceFunc SyncMainFunc() override { return SyncMain; }

  static void StartTask(Task *task);
  static void WaitTask(Task *task);
  static void SyncMain(void *instance);

  // Allocate CUDA instance in data instance block.
  int ExtraInstanceData(Cell *cell) override { return sizeof(CUDAInstance); }

  // Constant tensor copying.
  DevicePtr CopyTensorToDevice(const Tensor *tensor) override;
  void RemoveTensorFromDevice(const Tensor *tensor) override;

  // Fetch tensor from device.
  char *FetchTensorFromDevice(const Instance *data,
                              const Tensor *tensor) override;

  // Fetch data block from device.
  char *FetchDataFromDevice(DevicePtr data, size_t size) override;

  // Instance tensor copying.
  void EmitTensorTransfers(const Transfers &xfers,
                           Cell *cell,
                           MacroAssembler *masm) override;

  // Emit code for CUDA status check. This is only done for debug builds.
  static void EmitStatusCheck(const char *msg, MacroAssembler *masm);

  // Profiling support.
  static void StartProfiler(void *data);
  static void StopProfiler(void *data);
  InstanceFunc StartProfilerFunc() override { return StartProfiler; }
  InstanceFunc StopProfilerFunc() override { return StopProfiler; }

 private:
  // Instance data block.
  struct Block {
    size_t host_offset;
    size_t device_offset;
    size_t size;
    int taskidx;
  };

  // Coalesce transfers of consecutive data blocks.
  static std::vector<Block> MergedTransfers(const std::vector<Transfer> &xfers);

  // CUDA device for computations.
  CUDADevice *device_ = nullptr;
};

// Get offset of stream in data instance block.
int StreamOffset(Step *step);

}  // namespace myelin
}  // namespace sling

#endif  // SLING_MYELIN_CUDA_CUDA_RUNTIME_H_

