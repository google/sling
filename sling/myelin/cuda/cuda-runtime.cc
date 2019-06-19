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

#include "sling/myelin/cuda/cuda-runtime.h"

#include <algorithm>

#include "sling/base/logging.h"
#include "sling/myelin/compute.h"
#include "sling/myelin/macro-assembler.h"
#include "sling/myelin/cuda/cuda.h"

namespace sling {
namespace myelin {

using namespace jit;

// Use pinned memory for instance data.
static const bool pinned_memory = true;

CUDARuntime::~CUDARuntime() {
  Disconnect();
}

void CUDARuntime::Connect(int device_number, int flags) {
  // Check if device is already connected to another device.
  if (device_ != nullptr) {
    if (device_number != -1 && device_->number() != device_number) {
      LOG(FATAL) << "CUDA runtime already connect to another device";
    }
  } else if (device_number == -1) {
    // Select the CUDA device with the most cores.
    device_ = new CUDADevice(0, flags);
    for (int d = 1; d < CUDA::Devices(); ++d) {
      CUDADevice *candidate = new CUDADevice(d, flags);
      if (candidate->cores() > device_->cores()) {
        delete device_;
        device_ = candidate;
      } else {
        delete candidate;
      }
    }
  } else {
    // Initialize CUDA device.
    device_ = new CUDADevice(device_number, flags);
  }
}

void CUDARuntime::Disconnect() {
  delete device_;
  device_ = nullptr;
}

string CUDARuntime::Description() {
  if (device_ != nullptr) {
    return "CUDA device " + std::to_string(device_->number()) +
           ": " + device_->ToString();
  } else {
    return "No CUDA device";
  }
}

void CUDARuntime::AllocateInstance(Instance *instance) {
  // Allocate pinned host memory for instance.
  void *data;
  if (pinned_memory) {
    CHECK_CUDA(cuMemAllocHost(&data, instance->size()));
    CHECK_EQ(reinterpret_cast<uint64>(data) % instance->alignment(), 0);
  } else {
    int rc = posix_memalign(&data, instance->alignment(), instance->size());
    CHECK_EQ(rc, 0);
  }
  instance->set_data(reinterpret_cast<char *>(data));

  // Set up CUDA runtime instance block which is located at the start of the
  // host instance block.
  CUDAInstance *rt = reinterpret_cast<CUDAInstance *>(data);

  // Allocate device instance block.
  size_t size = instance->cell()->device_instance_size();
  if (size > 0) {
    CHECK_CUDA(cuMemAlloc(&rt->data, size));
  } else {
    rt->data = DEVICE_NULL;
  }

  // Allocate streams for tasks.
  CHECK_CUDA(cuStreamCreate(&rt->mainstream, CU_STREAM_NON_BLOCKING));
  for (int i = 0; i < instance->num_tasks(); ++i) {
    Task *task = instance->task(i);

    // Allocate stream for each asynchronous task and store it in the task
    // state.
    CUstream stream;
    CHECK_CUDA(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING));
    task->state = stream;
  }
}

void CUDARuntime::FreeInstance(Instance *instance) {
  // Deallocate instance memory on device.
  CUDAInstance *rt = reinterpret_cast<CUDAInstance *>(instance->data());
  if (rt->data != DEVICE_NULL) {
    CHECK_CUDA(cuMemFree(rt->data));
  }

  // Destroy CUDA streams for instance.
  CHECK_CUDA(cuStreamDestroy(rt->mainstream));
  for (int i = 0; i < instance->num_tasks(); ++i) {
    Task *task = instance->task(i);
    CUstream stream = static_cast<CUstream>(task->state);
    CHECK_CUDA(cuStreamDestroy(stream));
  }

  // Deallocate host memory for instance.
  if (pinned_memory) {
    cuMemFreeHost(instance->data());
  } else {
    free(instance->data());
  }
}

void CUDARuntime::ClearInstance(Instance *instance) {
  // Do not clear task data at the start of the instance block.
  memset(instance->data() + instance->cell()->data_start(), 0,
         instance->size() - instance->cell()->data_start());

  // Clear instance on device in debug mode.
#if !defined(NDEBUG)
  CUDAInstance *rt = reinterpret_cast<CUDAInstance *>(instance->data());
  if (rt->data != DEVICE_NULL) {
    size_t size = instance->cell()->device_instance_size();
    CHECK_CUDA(cuMemsetD8(rt->data, 0, size));
  }
#endif
}

char *CUDARuntime::AllocateChannel(char *data,
                                   size_t old_size,
                                   size_t new_size,
                                   size_t alignment,
                                   Placement placement) {
  if (placement & DEVICE) {
    // Allocate channel in device memory.
    if (new_size == 0) return nullptr;
    DevicePtr buffer;
    CHECK_CUDA(cuMemAlloc(&buffer, new_size));
    if (data != nullptr) {
      CHECK_CUDA(cuMemcpyDtoD(buffer, reinterpret_cast<DevicePtr>(data),
                              old_size));
    }
    return reinterpret_cast<char *>(buffer);
  } else {
    // Allocate channel in host memory.
    void *buffer;
    CHECK_EQ(posix_memalign(&buffer, alignment, new_size), 0);
    if (data != nullptr) {
      memcpy(buffer, data, old_size);
      free(data);
    }
    return reinterpret_cast<char *>(buffer);
  }
}

void CUDARuntime::ClearChannel(char *data, size_t pos,
                               size_t size, Placement placement) {
  if (placement & DEVICE) {
    CHECK_CUDA(cuMemsetD8(reinterpret_cast<DevicePtr>(data + pos), 0, size));
  } else {
    memset(data + pos, 0, size);
  }
}

void CUDARuntime::FreeChannel(char *data, Placement placement) {
  if (placement & DEVICE) {
    if (data != nullptr) {
      CHECK_CUDA(cuMemFree(reinterpret_cast<DevicePtr>(data)));
    }
  } else {
    free(data);
  }
}

void CUDARuntime::StartTask(Task *task) {
  // The task is run in the calling thread. All the CUDA kernels in the task
  // will be launched asynchronously so they might not yet have completed when
  // returning from the task function.
  task->func(task->arg);
}

void CUDARuntime::WaitTask(Task *task) {
  // Wait until all operations have completed in the task stream.
  CUstream stream = static_cast<CUstream>(task->state);
  CHECK_CUDA(cuStreamSynchronize(stream));
}

void CUDARuntime::SyncMain(void *instance) {
  CUDAInstance *rt = static_cast<CUDAInstance *>(instance);
  CHECK_CUDA(cuStreamSynchronize(rt->mainstream));
}

DevicePtr CUDARuntime::CopyTensorToDevice(const Tensor *tensor) {
  // Allocate memory for constant tensor on device.
  DevicePtr dest;
  CHECK_CUDA(cuMemAlloc(&dest, tensor->space()));

  // Copy tensor data to device.
  CHECK_CUDA(cuMemcpyHtoD(dest, tensor->data(), tensor->space()));

  VLOG(5) << "Allocate tensor " << tensor->name() << " on device at "
          << dest << ", " << tensor->space() << " bytes";

  return dest;
}

void CUDARuntime::RemoveTensorFromDevice(const Tensor *tensor) {
  CHECK_CUDA(cuMemFree(tensor->device_data())) << tensor->name();
}

char *CUDARuntime::FetchTensorFromDevice(const Instance *data,
                                         const Tensor *tensor) {
  // Allocate host memory buffer for tensor.
  char *dest = reinterpret_cast<char *>(malloc(tensor->space()));

  // Copy tensor from device.
  if (tensor->device_data() != DEVICE_NULL) {
    CHECK_CUDA(cuMemcpyDtoH(dest, tensor->device_data(), tensor->space()));
  } else {
    CUDAInstance *rt = reinterpret_cast<CUDAInstance *>(data->data());
    CHECK(rt != nullptr);
    size_t offset = tensor->device_offset();
    CHECK_NE(offset, NOOFFSET);
    CHECK_CUDA(cuMemcpyDtoH(dest, rt->data + offset, tensor->space()));
  }

  return dest;
}

char *CUDARuntime::FetchDataFromDevice(DevicePtr data, size_t size) {
  if (data == DEVICE_NULL) return nullptr;
  char *dest = reinterpret_cast<char *>(malloc(size));
  CHECK_CUDA(cuMemcpyDtoH(dest, data, size));
  return dest;
}

void CUDARuntime::EmitTensorTransfers(const Transfers &xfers,
                                      Cell *cell,
                                      MacroAssembler *masm) {
  // Host to device transfers.
  Register datareg = masm->instance();
  if (!xfers.host_to_device.empty()) {
    for (auto &t : xfers.host_to_device) {
      VLOG(8) << "Transfer " << t.tensor->name() << " from host to device";
    }
    for (Block &blk : MergedTransfers(xfers.host_to_device)) {
      // Set destination device address.
      masm->movq(arg_reg_1, Operand(datareg, offsetof(CUDAInstance, data)));
      if (blk.device_offset != 0) {
        masm->addq(arg_reg_1, Immediate(blk.device_offset));
      }

      // Set source host address.
      masm->leaq(arg_reg_2, Operand(datareg, blk.host_offset));

      // Set size.
      masm->movq(arg_reg_3, Immediate(blk.size));

      // Set stream for task.
      int ofs;
      if (blk.taskidx == -1) {
        // Main task stream is stored in runtime block.
        ofs = offsetof(CUDAInstance, mainstream);
      } else {
        // Parallel task stream is stored in task block.
        ofs = cell->task_offset(blk.taskidx) + offsetof(Task, state);
      }
      masm->movq(arg_reg_4, Operand(datareg, ofs));

      // Call cuMemcpyHtoDAsync(src, dst, size, stream).
      Register acc = masm->rr().alloc();
      masm->load_extern(acc, reinterpret_cast<void *>(cuMemcpyHtoDAsync),
                        "cuMemcpyHtoDAsync");
      masm->call(acc);
      masm->rr().release(acc);
      EmitStatusCheck("cuMemcpyHtoDAsync", masm);

      VLOG(5) << "Copy " << blk.host_offset << "->" << blk.device_offset
              << " size " << blk.size << " from host to device";
    }
  }

  // Device to host transfers.
  if (!xfers.device_to_host.empty()) {
    for (auto &t : xfers.device_to_host) {
      VLOG(8) << "Transfer " << t.tensor->name() << " from device to host";
    }
    for (Block &blk : MergedTransfers(xfers.device_to_host)) {
      // Set destination device address.
      masm->leaq(arg_reg_1, Operand(datareg, blk.host_offset));

      // Set source device address.
      masm->movq(arg_reg_2, Operand(datareg, offsetof(CUDAInstance, data)));
      if (blk.device_offset != 0) {
        masm->addq(arg_reg_2, Immediate(blk.device_offset));
      }

      // Set size.
      masm->movq(arg_reg_3, Immediate(blk.size));

      // Set stream for task.
      int ofs;
      if (blk.taskidx == -1) {
        // Main task stream is stored in runtime block.
        ofs = offsetof(CUDAInstance, mainstream);
      } else {
        // Parallel task stream is stored in task block.
        ofs = cell->task_offset(blk.taskidx) + offsetof(Task, state);
      }
      masm->movq(arg_reg_4, Operand(datareg, ofs));

      // Call cuMemcpyDtoHAsync(src, dst, size, stream).
      Register acc = masm->rr().alloc();
      masm->load_extern(acc, reinterpret_cast<void *>(cuMemcpyDtoHAsync),
                        "cuMemcpyDtoHAsync");
      masm->call(acc);
      masm->rr().release(acc);
      EmitStatusCheck("cuMemcpyDtoHAsync", masm);

      VLOG(5) << "Copy " << blk.device_offset << "->" << blk.host_offset
              << " size " << blk.size << " from device to host";
    }
  }
}

std::vector<CUDARuntime::Block> CUDARuntime::MergedTransfers(
    const std::vector<Transfer> &xfers) {
  // Sort transfers in task and instance offset order.
  std::vector<Transfer> t = xfers;
  std::sort(t.begin(), t.end(), [](const Transfer &a, const Transfer &b) {
    if (a.taskidx == b.taskidx) {
      return a.tensor->offset() < b.tensor->offset();
    } else {
      return a.taskidx < b.taskidx;
    }
  });

  // Merge consecutive blocks.
  std::vector<Block> blocks;
  int start = 0;
  while (start < t.size()) {
    Block blk;
    blk.taskidx = t[start].taskidx;
    blk.size = t[start].tensor->space();
    blk.host_offset = t[start].tensor->offset();
    blk.device_offset = t[start].tensor->device_offset();
    int end = start + 1;
    while (end < t.size() &&
           t[end].taskidx == blk.taskidx &&
           t[end].tensor->offset() == blk.host_offset + blk.size &&
           t[end].tensor->device_offset() == blk.device_offset + blk.size) {
      blk.size += t[end].tensor->space();
      end++;
    }
    blocks.push_back(blk);
    start = end;
  }

  return blocks;
}

void CUDAErrorHandler(int error, const char *msg) {
  LOG(FATAL) << "CUDA error " << error << ": " << msg;
}

void CUDARuntime::EmitStatusCheck(const char *msg, MacroAssembler *masm) {
#if !defined(NDEBUG)
  // Return code from CUDA function is in rax.
  Label l;
  masm->cmpq(rax, Immediate(0));
  masm->j(equal, &l);
  masm->movq(arg_reg_1, rax);
  masm->movp(arg_reg_2, const_cast<char *>(msg));
  masm->movp(r10, reinterpret_cast<void *>(CUDAErrorHandler));
  masm->call(r10);
  masm->bind(&l);
#endif
}

void CUDARuntime::StartProfiler(void *data) {
  cuProfilerStart();
}

void CUDARuntime::StopProfiler(void *data) {
  cuProfilerStop();
}

// Get offset of stream in data instance block.
int StreamOffset(Step *step) {
  if (step->task_index() == -1) {
    // Main task stream is stored in runtime block.
    return offsetof(CUDAInstance, mainstream);
  } else {
    // Parallel task stream is stored in task block.
    int task_offset = step->cell()->task_offset(step->task_index());
    return task_offset + offsetof(Task, state);
  }
}

}  // namespace myelin
}  // namespace sling

