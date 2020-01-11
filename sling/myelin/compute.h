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

#ifndef SLING_MYELIN_COMPUTE_H_
#define SLING_MYELIN_COMPUTE_H_

#include <string>
#include <unordered_map>
#include <vector>

#include "sling/base/types.h"
#include "sling/myelin/flow.h"
#include "sling/string/printf.h"
#include "third_party/jit/code.h"
#include "third_party/jit/cpu.h"

namespace sling {
namespace myelin {

class MacroAssembler;
class Network;
class Cell;
class Step;
class Instance;
class Tensor;
class TensorData;
class CUDADevice;
class CustomKernel;
class InstanceAllocator;
class ProfileSummary;
struct Options;

// Element order.
enum Order {
  ANY_ORDER,
  ROW_MAJOR,
  COLUMN_MAJOR,
  ROW_MAJOR_PREFERRED,
  COLUMN_MAJOR_PREFERRED,
  CONFLICTING_ORDER
};

// Task state.
enum TaskState {PENDING, ACTIVE, COMPLETED};

// Placement for data and code execution.
enum Placement {NOWHERE = 0x0, HOST = 0x1, DEVICE = 0x2, EVERYWHERE = 0x3};

// Pointer to data in device memory.
typedef uint64 DevicePtr;
#define DEVICE_NULL 0
const size_t NOOFFSET = -1;

// Minimum data alignment.
static const int kMinDataAlignment = sizeof(void *);

// Abstract interface for kernel implementing a code generator for an operation.
class Kernel {
 public:
  virtual ~Kernel() = default;

  // Return name of operation supported by kernel.
  virtual string Operation() = 0;

  // Return descriptive name for kernel.
  virtual string Name() { return Operation(); }

  // Return location of kernel computation.
  virtual Placement Location() { return HOST; }

  // Check if kernel supports generating code for step.
  virtual bool Supports(Step *step) { return false; }
  virtual bool Supports(Step *step, const Options &options) {
    return Supports(step);
  }

  // Let kernel adjust alignment constraints for step.
  virtual void Adjust(Step *step) {}
  virtual void Adjust(Step *step, const Options &options) {
    Adjust(step);
  }

  // Generate code for step.
  virtual void Generate(Step *step, MacroAssembler *masm) = 0;

  // Number of numeric operations kernel performs for step.
  virtual int64 Complexity(const Step *step) { return -1; }
};

// Library of kernels for implementing operations.
class Library : public Transformations {
 public:
  typedef std::vector<Kernel *> Kernels;

  ~Library();

  // Registers kernel. Ownership is transferred to the library.
  void Register(Kernel *kernel);

  // Register custom kernel.
  CustomKernel &Register(const string &op, const string &name,
                         void (*func)(const TensorData &arg,
                                      TensorData *output));
  CustomKernel &Register(const string &op, const string &name,
                         void (*func)(const TensorData &arg1,
                                      const TensorData &arg2,
                                      TensorData *output));
  CustomKernel &Register(const string &op, const string &name,
                         void (*func)(const TensorData &arg1,
                                      const TensorData &arg2,
                                      const TensorData &arg3,
                                      TensorData *output));
  CustomKernel &Register(const string &op, const string &name,
                         void (*func)(const TensorData &arg1,
                                      const TensorData &arg2,
                                      const TensorData &arg3,
                                      const TensorData &arg4,
                                      TensorData *output));

  // Find kernels implementing operation.
  const Kernels &Lookup(const string &op) const;

 private:
  // Register custom kernel.
  CustomKernel &RegisterCustomKernel(const string &op, const string &name,
                                     void *func, int indegree, int outdegree);

  // Map from op name to kernels implementing the op.
  std::unordered_map<string, Kernels> kernels_;

  // Empty kernel list.
  Kernels no_kernels_;
};

// A task is an asynchronous function that can be run in parallel with the main
// computation. The task structures are stored in the instance blocks.
struct Task {
  // Function with argument to be executed by task.
  void (*func)(void *arg);
  void *arg;

  // Data field that can be used by runtime for state information.
  void *state;

  // Task id for flow.
  int32 id;

  // Task index for cell.
  int32 index;
};

// Data transfer between host and device.
struct Transfer {
  Transfer(Tensor *tensor, int taskidx) : tensor(tensor), taskidx(taskidx) {}

  Tensor *tensor;  // tensor to be transferred
  int taskidx;     // index of task for performing the transfer
};

// List of data transfers between host and device.
struct Transfers {
  // Add transfer from host to device.
  void add_host_to_device(Tensor *tensor, int taskidx) {
    host_to_device.emplace_back(tensor, taskidx);
  }

  // Add transfer from device to host.
  void add_device_to_host(Tensor *tensor, int taskidx) {
    device_to_host.emplace_back(tensor, taskidx);
  }

  std::vector<Transfer> host_to_device;  // transfers from host to device
  std::vector<Transfer> device_to_host;  // transfers from device to host
};

// Runtime support for network.
class Runtime {
 public:
  typedef void (*TaskFunc)(Task *);
  typedef void (*InstanceFunc)(void *);

  virtual ~Runtime() = default;

  // Return runtime description.
  virtual string Description() { return ""; }

  // Allocate and initialize instance data.
  virtual void AllocateInstance(Instance *instance) = 0;

  // Deallocate instance data.
  virtual void FreeInstance(Instance *instance) = 0;

  // Clear instance data.
  virtual void ClearInstance(Instance *instance) = 0;

  // Allocate or reallocate channel.
  virtual char *AllocateChannel(char *data,
                                size_t old_size,
                                size_t new_size,
                                size_t alignment,
                                Placement placement) = 0;

  // Clear elements in channel.
  virtual void ClearChannel(char *data, size_t pos,
                            size_t size,
                            Placement placement) = 0;

  // Deallocate channel.
  virtual void FreeChannel(char *data, Placement placement) = 0;

  // Generate prologue for cell function.
  virtual void GeneratePrologue(Cell *cell, MacroAssembler *masm) {}

  // Generate epilogue for cell function.
  virtual void GenerateEpilogue(Cell *cell, MacroAssembler *masm) {}

  // Check if runtime supports asynchronous execution of steps.
  virtual bool SupportsAsync() = 0;

  // Return runtime function for starting task.
  virtual TaskFunc StartTaskFunc() = 0;

  // Return runtime function for waiting for task completion.
  virtual TaskFunc WaitTaskFunc() = 0;

  // Return runtime function for synchronizing the main task execution. This
  // can return null if no synchronization is needed.
  virtual InstanceFunc SyncMainFunc() { return nullptr; }

  // Return the size of extra instance data needed by runtime. This extra data
  // will be allocated at the beginning of the instance block at offset 0.
  virtual int ExtraInstanceData(Cell *cell) { return 0; }

  // Copy constant tensor to device.
  virtual DevicePtr CopyTensorToDevice(const Tensor *tensor) {
    return DEVICE_NULL;
  }

  // Remove constant tensor from device.
  virtual void RemoveTensorFromDevice(const Tensor *tensor) {}

  // Fetch copy of tensor from device. Caller takes ownership of the returned
  // data buffer.
  virtual char *FetchTensorFromDevice(const Instance *data,
                                      const Tensor *tensor) {
    return nullptr;
  }

  // Fetch data block from device. Caller takes ownership of the returned
  // data buffer.
  virtual char *FetchDataFromDevice(DevicePtr data, size_t size) {
    return nullptr;
  }

  // Generate code for transferring data between host and device.
  virtual void EmitTensorTransfers(const Transfers &xfers,
                                   Cell *cell,
                                   MacroAssembler *masm) {}

  // Return CUDA device used by runtime.
  virtual CUDADevice *Device() { return nullptr; }

  // Return runtime function for starting profiler.
  virtual InstanceFunc StartProfilerFunc() { return nullptr; }

  // Return runtime function for stopping profiler.
  virtual InstanceFunc StopProfilerFunc() { return nullptr; }
};

// Linker interface for linking code and data in network.
class Linker {
 public:
  virtual ~Linker() = default;

  // Begin compilation of network.
  virtual void BeginNetwork(Network *network) {}

  // Compilation of network complete.
  virtual void EndNetwork(Network *network) {}

  // Start code generation for cell.
  virtual void BeginCell(Cell *cell) {}

  // Compilation of cell completed.
  virtual void EndCell(Cell *cell,
                       jit::CodeGenerator *generator,
                       jit::Code *code,
                       int data_size) {}

  // Start code generation for step.
  virtual void BeginStep(Step *step, int offset) {}

  // Code generation for step completed.
  virtual void EndStep(Step *step, int offset) {}

  // Add tensor data block to linker.
  virtual void AddData(Tensor *data) {}

  // Add device code for step.
  virtual void AddDeviceCode(Step *step, const string &code) {}
};

// A tensor is a multi-dimensional array that can be used for globals and
// parameters.
class Tensor {
 public:
  // Update minimum alignment constraints for tensor by combining new alignment
  // with existing constraints.
  void MinAlign(const Shape &align);

  // Update minimum alignment constraint for last dimension of tensor by
  // combining new alignment with existing constraints.
  void MinAlignLast(int align);

  // Ensure same alignment as other tensor.
  void SameAlign(Tensor *other);

  // Ensure compatible alignment modulo broadcasting with other tensor.
  void CompatibleAlign(Tensor *other);

  // Check if alignment is conflicting with other requirements.
  bool SupportsAlignment(const Shape &align) const;

  // Check if pointer is aligned properly for tensor.
  bool IsAligned(void *ptr) const {
    return (reinterpret_cast<size_t>(ptr) & (byte_alignment_ - 1)) == 0;
  }

  // Check if tensor can support order.
  bool SupportsOrder(Order order);

  // Set required element order.
  void RequireOrder(Order order);

  // Update minimum byte alignment for tensor by combining new alignment
  // with existing constraints.
  void SetMiniumAlignment(int alignment);

  // Require dense encoding with no padding of dimensions.
  void RequireDense() { require_dense_ = true; }

  // Require standard row-major order.
  void RequireStandardOrder() {
    if (rank() > 1 && dim(0) > 1) RequireOrder(ROW_MAJOR);
  }

  // Check if tensor has the same shape as another tensor.
  bool HasSameShape(const Tensor *other) const;

  // Check if tensor has dense layout.
  bool HasDenseLayout() const;

  // Check if tensor has standard layout, i.e. dense row-major.
  bool HasStandardLayout() const {
    return order_ == ROW_MAJOR && HasDenseLayout();
  }

  // Check if tensor shape is broadcast compatible with another tensor.
  bool Compatible(const Tensor *other) const;

  // Check if tensor is a scalar.
  bool IsScalar() const { return rank() == 0; }

  // Check if tensor is a vector.
  bool IsVector() const { return rank() == 1; }

  // Check if tensor is a matrix.
  bool IsMatrix() const { return rank() == 2; }

  // Tensor name for parameter or global.
  const string &name() const { return name_; }

  // Data type for tensor elements.
  Type type() const { return type_; }

  // Reference to tensor.
  bool ref() const { return ref_; }
  void set_ref(bool ref) { ref_ = ref; }

  // Reference to dynamically sized tensor channel.
  bool dynamic() const { return dynamic_; }
  void set_dynamic(bool dynamic) { dynamic_ = dynamic; }

  // Tensor shape.
  const Shape &shape() const { return shape_; }
  int rank() const { return shape_.rank(); }
  int dim(int d) const { return shape_.dim(d); }

  // Minimum alignment requirement for each dimension.
  const Shape &minalign() const { return minalign_; }
  int minalign(int d) const { return minalign_.dim(d); }

  // Tensor shape after alignment.
  const Shape &aligned() const { return aligned_; }
  int aligned(int d) const { return aligned_.dim(d); }

  // Size (in bytes) of each dimension after alignment.
  const Shape &stride() const { return stride_; }
  int stride(int d) const { return stride_.dim(d); }

  // Padding (in bytes) to each dimension.
  int padding(int d) const { return (aligned(d) - dim(d)) * stride(d); }

  // Total size (in bytes) for tensor instance.
  size_t size() const { return size_; }

  // Number of elements in tensor.
  int elements() const { return shape_.elements(); }

  // Value for global tensor. Return null for parameters.
  const char *data() const { return data_; }
  char *data() { return data_; }

  // Get data as scalar. Return false if types do not match.
  template <typename T> bool GetData(T *value) const {
    if (data_ == nullptr) return false;
    auto &traits = Traits<T>();
    if (type_ != traits.type() || size_ != traits.size()) return false;
    *value = *reinterpret_cast<const T *>(data_);
    return true;
  }

  // Get data as vector. Return false if types do not match.
  template <typename T> bool GetData(std::vector<T> *value) const {
    if (data_ == nullptr) return false;
    auto &traits = Traits<T>();
    if (type_ != traits.type()) return false;
    if (elements() * traits.size() != size_) return false;
    const T *array = reinterpret_cast<const T *>(data_);
    value->assign(array, array + elements());
    return true;
  }

  // Pointer to constant tensor on device.
  DevicePtr device_data() const { return device_data_; }

  // Size (in bytes) of elements in tensor.
  int element_size() const { return TypeTraits::of(type_).size(); }

  // Offset in data instance block. Return NOOFFSET for constants and tensors
  // that are not stored on the host.
  size_t offset() const { return offset_; }

  // Offset in device data instance block. Return NOOFFSET for tensors that are
  // not stored in the instance block on the device.
  size_t device_offset() const { return device_offset_; }

  // Number bytes allocated for tensor in instance. This takes references into
  // account so these only take up space for one pointer.
  size_t space() const { return space_; }

  // Byte offset of element in tensor.
  size_t offset(int r) const {
    return r * stride(0);
  }
  size_t offset(int r, int c) const {
    return r * stride(0) + c * stride(1);
  }
  size_t offset(int r, int c, int k) const {
    return r * stride(0) + c * stride(1) + k * stride(2);
  }
  size_t offset(int r, int c, int k, int l) const {
    return r * stride(0) + c * stride(1) + k * stride(2) + l * stride(3);
  }
  size_t offset(const std::vector<int> &indices) const {
    size_t n = 0;
    for (int d = 0; d < rank(); ++d) {
      n += indices[d] * stride(d);
    }
    return n;
  }

  // Byte offset of element in tensor by linear index between 0 and the number
  // of elements in tensor.
  size_t LinearOffset(int index) const {
    int n = index;
    size_t ofs = 0;
    for (int d = rank() - 1; d >= 0; --d) {
      ofs += (n % dim(d)) * stride(d);
      n = n / dim(d);
    }
    return ofs;
  }

  // Index of element in tensor.
  int index(int r) const {
    return offset(r) / element_size();
  }
  int index(int r, int c) const {
    return offset(r, c) / element_size();
  }
  int index(int r, int c, int k) const {
    return offset(r, c, k) / element_size();
  }
  int index(int r, int c, int k, int l) const {
    return offset(r, c, k, l) / element_size();
  }
  int index(const std::vector<int> &indices) const {
    return offset(indices) / element_size();
  }

  // Check if tensor is a constant.
  bool constant() const { return constant_; }

  // Check if tensor is learnable.
  bool learnable() const { return !local_ && !constant_; }

  // Local variables are allocated in the instance block.
  bool IsLocal() const { return local_; }
  bool IsGlobal() const { return !local_; }

  // Return tensor placement.
  Placement placement() const { return placement_; }

  // Return tensor reference placement.
  Placement ref_placement() const { return ref_placement_; }

  // Add location for placement.
  void AddPlace(Placement place) {
    placement_ = static_cast<Placement>(placement_ | place);
  }

  // Add new location for current placement.
  void AddNewPlace(Placement place) {
    current_placement_ = static_cast<Placement>(current_placement_ | place);
  }

  // Add new location for referenced data.
  void AddRefPlace(Placement place) {
    ref_placement_ = static_cast<Placement>(ref_placement_ | place);
  }

  // Return the task index for consumers of this tensor or -1 if tensor is
  // consumed by operations in multiple tasks.
  int ConsumerTask() const;

  // Return scalar value.
  template<typename T> const T value() const {
    return *reinterpret_cast<const T *>(data_);
  }

  // Element order.
  Order order() const { return order_; }

  // Other tensor that this tensor shares storage with.
  Tensor *shared() const { return shared_; }
  void set_shared(Tensor *shared) { shared_ = shared; }

  // Check if tensor shares the underlying storage with another tensor.
  bool SharedWith(Tensor *other) const {
    return shared_ == other ||
           other->shared_ == this ||
           (shared_ != nullptr && shared_ == other->shared_);
  }

  // Circular list of tensors that tensor shares alignment requirements with.
  Tensor *prev_link() const { return prev_link_; }
  Tensor *next_link() const { return next_link_; }
  bool linked() const { return prev_link_ != this; }

  // Link other tensor for propagating alignment requirements.
  void Link(Tensor *link);

  // Step that produces tensor.
  Step *producer() const { return producer_; }

  // List of steps that uses tensor.
  const std::vector<Step *> &consumers() const { return consumers_; }

  // Return the number of usages of tensor.
  int usages() const { return consumers_.size(); }

  // Cell that tensor belongs to.
  Cell *cell() const { return cell_; }

  // Bitmap for sparse tensor.
  Tensor *sparse() const { return sparse_; }
  void set_sparse(Tensor *sparse) { sparse_ = sparse; }

  // Add sparsity map to make a sparse tensor. The sparsity map is over the
  // first dimension of the tensor.
  Tensor *MakeSparse(bool ref = false);

  // Input and output flags.
  bool in() const { return in_; }
  bool out() const { return out_; }

  // Live range for tensor.
  int first() const { return first_; }
  int last() const { return last_; }

  // Byte alignment.
  int byte_alignment() const { return byte_alignment_; }

  // Size of of channel elements based on this tensor.
  int ChannelElementSize() const;

  // Size of elements along an axis.
  int AxisSize(int axis) const;

  // Return corresponding gradient tensor.
  Tensor *Gradient() const;

  // Return tensor type as string.
  string TypeString() const;

  // Return contents of tensor as string.
  string ToString(const char *data, bool deref = true) const;
  string ToString() const { return ToString(data_); }

 private:
  // Offset in data instance block.
  size_t offset_ = NOOFFSET;

  // Offset in device data instance block.
  size_t device_offset_ = NOOFFSET;

  // Tensor name.
  string name_;

  // Element data type.
  Type type_ = DT_INVALID;

  // Tensor reference.
  bool ref_ = false;

  // Reference to dynamically sized tensor channel.
  bool dynamic_ = false;

  // Tensor shape.
  Shape shape_;

  // Minimum alignment requirement for each dimension.
  Shape minalign_;

  // Require dense encoding with no padding of dimensions.
  bool require_dense_ = false;

  // Tensor shape after alignment.
  Shape aligned_;

  // Size of each dimension after alignment.
  Shape stride_;

  // Total size (in bytes) for tensor instance.
  size_t size_ = 0;

  // Number of bytes allocated for tensor in instance.
  size_t space_ = 0;

  // Minimum alignment (in bytes) for tensor instance.
  int byte_alignment_ = 1;

  // Element order for data.
  Order order_ = ANY_ORDER;

  // Optional other tensor that this tensor shares storage with.
  Tensor *shared_ = nullptr;

  // Tensors that share alignment requirements are linked in a circular list.
  Tensor *prev_link_ = this;
  Tensor *next_link_ = this;

  // Value for global tensor (not owned).
  char *data_ = nullptr;

  // Pointer to global tensor data on device. This is only set for constant
  // or learnable tensors that need to be accessed from the device.
  DevicePtr device_data_ = DEVICE_NULL;

  // Constant tensors are global and cannot be modified.
  bool constant_ = false;

  // Initialization for tensor.
  Flow::Variable::Initialization init_ = Flow::Variable::INIT_ZERO;

  // Local tensors are allocated in the instance data block.
  bool local_ = true;

  // Cell that tensor is part of.
  Cell *cell_ = nullptr;

  // Step that produces tensor.
  Step *producer_ = nullptr;

  // Steps that consume tensor.
  std::vector<Step *> consumers_;

  // A sparse tensor has a bitmap tensor that keeps track of non-zero
  // sub-tensors.
  Tensor *sparse_ = nullptr;

  // Input and output flags.
  bool in_ = false;
  bool out_ = false;

  // Live range for tensor, i.e. index of first and last step using the tensor.
  int first_ = -1;
  int last_ = -1;

  // Placement of tensor.
  Placement placement_ = NOWHERE;

  // Current placement of tensor in compilation.
  Placement current_placement_ = NOWHERE;

  // Deferred placement for outputs from asynchronous steps.
  Placement deferred_placement_ = NOWHERE;

  // Placement for data referenced by a reference tensor.
  Placement ref_placement_ = NOWHERE;

  friend class Network;
  friend class InstanceAllocator;
};

// A step represents an operation that is part of a cell.
class Step : public Attributes {
 public:
  // Step name from flow operation.
  const string &name() const { return name_; }

  // Operation type for step.
  const string &type() const { return type_; }

  // Inputs to step.
  const std::vector<Tensor *> &inputs() const { return inputs_; }
  Tensor *input(int index) const { return inputs_[index]; }
  int indegree() const { return inputs_.size(); }

  // Outputs from step.
  const std::vector<Tensor *> &outputs() const { return outputs_; }
  Tensor *output(int index) const { return outputs_[index]; }
  int outdegree() const { return outputs_.size(); }

  // Kernel used for generating code for step.
  Kernel *kernel() const { return kernel_; }

  // Kernel variant.
  const string &variant() const { return variant_; }
  void set_variant(const string &variant) { variant_ = variant; }

  // Whether step does not do any computation.
  bool noop() const { return noop_; }

  // Return the complexity of the cell, i.e. number of numeric operations.
  int64 complexity() const { return noop_ ? 0 : kernel_->Complexity(this); }

  // Allocate auxiliary memory for kernel.
  char *AllocateKernelMemory(size_t size, int alignment);
  char *kernel_memory() const { return kernel_memory_; }

  // Cell that this step belongs to.
  Cell *cell() const { return cell_; }

  // Task index in cell for computing the step.
  int task_index() const { return task_index_; }

  // Device placement for kernel computation.
  Placement placement() const { return kernel_->Location(); }

  // Declare the number of general-purpose registers needed by step.
  void SetRegisterUsage(int regs);

  // Declare the number of preserved registers needed by step.
  void SetPreservedRegisterUsage(int regs);

  // Allow in-place operation between input and output. Return true if in-place
  // operation is supported, i.e. the operation must be the only consumer of
  // a non-preserved input.
  bool AllowInPlace(int input, int output, bool preserved = false);

  // A step in the main task that runs on the host but depends on inputs
  // produced on the device needs to be synchronized to ensure that the inputs
  // are ready before executing the task. This method checks if a step needs
  // to be synchronized before execution.
  bool NeedsSynchronization();

  // Get prototype variable for step. This is the biggest output from the step,
  // unless this is a scalar or the step does not have any outputs. In that
  // case, the biggest input is returned.
  Tensor *GetPrototype() const;

  // Get type signature for step.
  string Signature() const;

 private:
  // Step name from flow operation.
  string name_;

  // Operation type for step.
  string type_;

  // Cell that this step belongs to.
  Cell *cell_ = nullptr;

  // Task index in cell for computing the step.
  int task_index_ = -1;

  // Inputs to step.
  std::vector<Tensor *> inputs_;

  // Outputs from step.
  std::vector<Tensor *> outputs_;

  // Attributes.
  Attributes attributes_;

  // Kernel used for generating code for step (owned by library).
  Kernel *kernel_ = nullptr;

  // Auxiliary memory for kernel. This memory is owned by the memory pool for
  // the network.
  char *kernel_memory_ = nullptr;

  // Kernel variant. Only used for display purposes.
  string variant_;

  // Whether step is a no-op.
  bool noop_ = false;

  friend class Network;
};

// A channel is an array of tensors used for connecting cells in a network.
class Channel {
 public:
  // Initialize empty channel.
  Channel(const Tensor *format);

  // Delete channel.
  ~Channel();

  // Remove all elements from channel.
  void clear() { resize(0); }

  // Change size of channel.
  void resize(size_t n);

  // Change size of channel and clear all elements.
  void reset(size_t n);

  // Reserve space for channel elements.
  void reserve(size_t n);

  // Zero-fill element in channel.
  void zero(size_t n);

  // Return pointer to channel element.
  char *at(size_t index) const {
    return data_ + (index * element_size_);
  }

  // Add element to channel and return the last element.
  char *push() { resize(size_ + 1); return at(size_ - 1); }

  // Remove the last element from the channel.
  void pop() { resize(size_ - 1); }

  // Return the number of elements in the channel.
  size_t size() const { return size_; }

  // Return placement of channel.
  Placement placement() const {
    return format_->ref() ? format_->ref_placement() : format_->placement();
  }

  // Return runtime for channel.
  inline Runtime *runtime() const;

  // Return tensor format for channel elements.
  const Tensor *format() const { return format_; }

  // Return contents of channel as string.
  string ToString() const;

 private:
  // Data for the channel.
  char *data_ = nullptr;

  // Number of elements in channel.
  size_t size_ = 0;

  // Number of allocated elements.
  size_t capacity_ = 0;

  // A tensor describing the element type of the channel.
  const Tensor *format_;

  // Element size.
  int element_size_;

  // Byte alignment.
  int alignment_ = kMinDataAlignment;
};

// A tensor data object is a reference to a tensor value. It does not own the
// underlying storage for the tensor.
class TensorData {
 public:
  TensorData(const TensorData &other)
      : data_(other.data_), format_(other.format_) {}
  TensorData(char *data, const Tensor *format)
      : data_(data), format_(format) {}

  // Tensor element access.
  template<typename T> T &value() {
    DCHECK_EQ(Traits<T>().type(), type());
    return *reinterpret_cast<T *>(data_);
  }
  template<typename T> const T &value() const {
    DCHECK_EQ(Traits<T>().type(), type());
    return *reinterpret_cast<T *>(data_);
  }
  template<typename T> T &at(int r) {
    DCHECK_EQ(Traits<T>().type(), type());
    return *reinterpret_cast<T *>(data_ + format_->offset(r));
  }
  template<typename T> const T &at(int r) const {
    DCHECK_EQ(Traits<T>().type(), type());
    return *reinterpret_cast<const T *>(data_ + format_->offset(r));
  }
  template<typename T> T &at(int r, int c) {
    DCHECK_EQ(Traits<T>().type(), type());
    return *reinterpret_cast<T *>(data_ + format_->offset(r, c));
  }
  template<typename T> const T &at(int r, int c) const {
    DCHECK_EQ(Traits<T>().type(), type());
    return *reinterpret_cast<const T *>(data_ + format_->offset(r, c));
  }
  template<typename T> T &at(int r, int c, int k) {
    DCHECK_EQ(Traits<T>().type(), type());
    return *reinterpret_cast<T *>(data_ + format_->offset(r, c, k));
  }
  template<typename T> const T &at(int r, int c, int k) const {
    DCHECK_EQ(Traits<T>().type(), type());
    return *reinterpret_cast<const T *>(data_ + format_->offset(r, c, k));
  }
  template<typename T> T &at(int r, int c, int k, int l) {
    DCHECK_EQ(Traits<T>().type(), type());
    return *reinterpret_cast<T *>(data_ + format_->offset(r, c, k, l));
  }
  template<typename T> const T &at(int r, int c, int k, int l) const {
    DCHECK_EQ(Traits<T>().type(), type());
    return *reinterpret_cast<const T *>(data_ + format_->offset(r, c, k, l));
  }

  template<typename T> T &nth(int index) {
    DCHECK_EQ(Traits<T>().type(), type());
    return *reinterpret_cast<T *>(data_ + format_->LinearOffset(index));
  }
  template<typename T> const T &nth(int index) const {
    DCHECK_EQ(Traits<T>().type(), type());
    return *reinterpret_cast<const T *>(data_ + format_->LinearOffset(index));
  }

  // Return tensor type.
  Type type() const { return format_->type(); }

  // Return tensor shape.
  const Shape &shape() const { return format_->shape(); }

  // Return tensor rank.
  int rank() const { return format_->rank(); }

  // Return size of tensor dimension.
  int dim(int d) const { return format_->dim(d); }

  // Return tensor format.
  const Tensor &format() const { return *format_; }

  // Return tensor as string.
  string ToString() const { return format_->ToString(data_); }

 private:
  char *data_;            // data for tensor
  const Tensor *format_;  // tensor format
};

// A profile summary stores the profiling data for a cell and can be used for
// external profiling where the profiling data is collected from different
// instances. This requires that the network has been compiled for external
// profiling.
class ProfileSummary {
 public:
  ProfileSummary(Cell *cell);
  ~ProfileSummary();

  // Cell being profiled.
  Cell *cell() const { return cell_; }

  // Pointer to profile buffer.
  int64 *data() const { return data_; }

 private:
  Cell *cell_;              // cell being profiled
  int64 *data_ = nullptr;   // profile data
};

// An instance holds all the input, output, and intermediate parameters of a
// cell.
class Instance {
 public:
  // Create data instance.
  Instance() : data_(nullptr), cell_(nullptr) {}
  Instance(const Cell *cell);
  Instance(const Flow::Function *func) : Instance(func->cell) {}

  // Delete data instance.
  ~Instance();

  // Clear instance.
  void Clear();

  // Run cell computation on instance.
  inline void Compute();

  // Get raw pointer to location of parameter in instance memory.
  char *GetAddress(const Tensor *param) {
    DCHECK(param != nullptr);
    DCHECK(param->IsLocal()) << param->name();
    DCHECK(param->cell() == cell_) << param->name();
    return data_ + param->offset();
  }

  // Get pointer to location of parameter in instance memory.
  template<typename T> T *Get(const Tensor *param) {
    DCHECK(param != nullptr);
    DCHECK(param->IsLocal()) << param->name();
    DCHECK(!param->ref()) << param->name();
    DCHECK(param->cell() == cell_) << param->name();
    DCHECK_EQ(Traits<T>().type(), param->type()) << param->name();
    return reinterpret_cast<T *>(data_ + param->offset());
  }
  template<typename T> T *Get(const Flow::Variable *var) {
    DCHECK(var->tensor != nullptr) << var->name;
    return Get<T>(var->tensor);
  }

  // Get pointer to location of element of parameter in instance memory.
  template<typename T> T *Get(const Tensor *param, int r) {
    DCHECK(param != nullptr);
    DCHECK(param->IsLocal()) << param->name();
    DCHECK(!param->ref()) << param->name();
    DCHECK(param->cell() == cell_) << param->name();
    DCHECK_EQ(Traits<T>().type(), param->type()) << param->name();
    return reinterpret_cast<T *>(data_ + param->offset() + param->offset(r));
  }
  template<typename T> T *Get(const Tensor *param, int r, int c) {
    DCHECK(param != nullptr);
    DCHECK(param->IsLocal()) << param->name();
    DCHECK(!param->ref()) << param->name();
    DCHECK(param->cell() == cell_) << param->name();
    DCHECK_EQ(Traits<T>().type(), param->type()) << param->name();
    return reinterpret_cast<T *>(
        data_ + param->offset() + param->offset(r, c));
  }
  template<typename T> T *Get(const Flow::Variable *var, int r) {
    DCHECK(var->tensor != nullptr) << var->name;
    return Get<T>(var->tensor, r);
  }
  template<typename T> T *Get(const Flow::Variable *var, int r, int c) {
    DCHECK(var->tensor != nullptr) << var->name;
    return Get<T>(var->tensor, r, c);
  }

  // Set link to element in channel.
  void Set(const Tensor *param, Channel *channel, int index = 0) {
    DCHECK(param->ref()) << param->name();
    DCHECK(param->cell() == cell_) << param->name();
    DCHECK(param->IsAligned(channel->at(index))) << param->name();
    *reinterpret_cast<char **>(data_ + param->offset()) = channel->at(index);
  }

  // Set link to other instance.
  void Set(const Tensor *param, Instance *instance) {
    DCHECK(param->cell() == cell_) << param->name();
    *reinterpret_cast<char **>(data_ + param->offset()) = instance->data();
  }
  void Set(const Flow::Variable *var, Instance *instance) {
    DCHECK(var->tensor != nullptr) << var->name;
    return Set(var->tensor, instance);
  }

  // Sets a reference parameter to an address. Caller is responsible for
  // ensuring proper alignment and any other constraints.
  void SetReference(const Tensor *param, void *address) {
    DCHECK(param != nullptr);
    DCHECK(param->IsLocal()) << param->name();
    DCHECK(param->ref()) << param->name();
    DCHECK(param->cell() == cell_) << param->name();
    DCHECK(param->IsAligned(address)) << param->name();
    *reinterpret_cast<void **>(data_ + param->offset()) = address;
  }
  void SetReference(const Flow::Variable *var, void *address) {
    DCHECK(var->tensor != nullptr) << var->name;
    return SetReference(var->tensor, address);
  }

  // Sets a dynamic tensor to channel.
  void SetChannel(const Tensor *param, Channel *channel) {
    DCHECK(param != nullptr);
    DCHECK(param->IsLocal()) << param->name();
    DCHECK(param->dynamic()) << param->name();
    DCHECK(param->cell() == cell_) << param->name();
    *reinterpret_cast<Channel **>(data_ + param->offset()) = channel;
  }
  void SetChannel(const Flow::Variable *var, Channel *channel) {
    DCHECK(var->tensor != nullptr) << var->name;
    return SetChannel(var->tensor, channel);
  }

  // Clear instance tensor.
  void Clear(const Tensor *param) {
    memset(GetAddress(param), 0, param->space());
  }
  void Clear(const Flow::Variable *var) {
    DCHECK(var->tensor != nullptr) << var->name;
    Clear(var->tensor);
  }

  // Set profiling summary for collecting profiling data for instance.
  inline void set_profile(ProfileSummary *summary);

  // Return tensor data object for parameter in instance.
  TensorData operator[](const Tensor *param) {
    return TensorData(data_ + param->offset(), param);
  }
  TensorData operator[](const Flow::Variable *var) {
    DCHECK(var->tensor != nullptr) << var->name;
    return TensorData(data_ + var->tensor->offset(), var->tensor);
  }
  inline TensorData operator[](const string &name);

  // Return parameter as string.
  string ToString(const Tensor *param) const;

  // Return all parameters as string.
  string ToString() const;

  // Return pointer to data block for instance.
  char *data() const { return data_; }
  void set_data(char *data) { data_ = data; }

  // Return cell for instance.
  const Cell *cell() const { return cell_; }

  // Return runtime for cell.
  inline Runtime *runtime() const;

  // Number of auxiliary tasks used.
  inline int num_tasks() const;

  // Return task structure for task.
  inline Task *task(int index) const;

  // Return instance size.
  inline size_t size() const;

  // Return instance alignment.
  inline int alignment() const;

 private:
  // Aligned memory block with parameters.
  char *data_;

  // Cell for instance.
  const Cell *cell_;
};

// Resizable array of cell instances.
class InstanceArray {
 public:
  // Create empty array of cell instances.
  InstanceArray(Cell *cell);

  // Deallocate instance array.
  ~InstanceArray();

  // Index operator.
  Instance &operator[](size_t index) { return *(begin_ + index); }
  const Instance &operator[](size_t index) const { return *(begin_ + index); }

  // Size and capacity.
  size_t size() const { return end_ - begin_; }
  size_t capacity() const { return limit_ - begin_; }

  // Resize array. This will never shrink the capacity of the array.
  void Resize(size_t size);

  // Deallocate all the instances and reset the capacity to zero.
  void Clear();

 private:
  Cell *cell_;        // cell type for instances
  Instance *begin_;   // begining of instance array
  Instance *end_;     // end of used instances
  Instance *limit_;   // end of allocated instances
};

// A cell contains generated code for executing computation of a function.
class Cell {
 public:
  ~Cell() { delete profile_summary_; }

  // Cell name from flow function.
  const string &name() const { return name_; }

  // Cell computation steps.
  const std::vector<Step *> &steps() const { return steps_; }

  // Look up parameter and return null if it is not found.
  Tensor *LookupParameter(const string &name) const;

  // Get parameter.
  Tensor *GetParameter(const string &name) const;

  // Look up step and return null if it is not found.
  Step *LookupStep(const string &name) const;

  // Write code to file.
  void WriteCodeToFile(const string &filename) const;

  // Code object for compiled cell.
  const jit::Code &code() const { return code_; }

  // Network that cell is part of.
  Network *network() const { return network_; }

  // Runtime for cell.
  inline Runtime *runtime() const;

  // Size of data instance for cell.
  size_t instance_size() const { return instance_size_; }

  // Size of device data instance for cell.
  size_t device_instance_size() const { return device_instance_size_; }

  // Instance alignment.
  int instance_alignment() const { return instance_alignment_; }
  int device_instance_alignment() const { return device_instance_alignment_; }

  // Number of auxiliary tasks used by cell.
  int num_tasks() const { return tasks_.size(); }

  // Convert task index to task id.
  int task(int index) const { return tasks_[index].task; }

  // Get offset of task structure in instance data block.
  size_t task_offset(int index) const { return tasks_[index].offset; }

  // Start of data in instance block.
  size_t data_start() const { return data_start_; }

  // Tensor with profiling information.
  Tensor *profile() const { return profile_; }

  // Profile summary for global profiling.
  ProfileSummary *profile_summary() const { return profile_summary_; }

  // Return corresponding gradient cell.
  Cell *Gradient() const;

  // Return tensor for primal reference in gradient cell.
  Tensor *Primal() const;

  // Return cell in text format.
  string ToString() const;

 private:
  // Task state information.
  struct TaskInfo {
    TaskInfo(int task) : task(task) {}

    int task;                       // task id in flow
    TaskState state = PENDING;      // task state at current compilation point
    jit::Label entry;               // entry point for task function
    size_t offset = 0;              // instance offset for task structure
    Placement placement = NOWHERE;  // placement of task computation
  };

  // Network that cell is part of.
  Network *network_;

  // Cell name.
  string name_;

  // Steps for cell in order of execution (owned by network).
  std::vector<Step *> steps_;

  // Tasks for parallel execution of steps in cell computation.
  std::vector<TaskInfo> tasks_;

  // Number of general-purpose register needed by cell.
  int register_usage_ = 0;

  // Code for running the cell computation.
  jit::Code code_;

  // Size of data instance for cell.
  size_t instance_size_ = 0;

  // Size of device data instance for cell.
  size_t device_instance_size_ = 0;

  // Start of data in instance block.
  size_t data_start_ = 0;

  // Instance alignment.
  int instance_alignment_ = kMinDataAlignment;
  int device_instance_alignment_ = kMinDataAlignment;

  // Tensor with profiling information.
  Tensor *profile_ = nullptr;

  // Profile summary for global profiling.
  ProfileSummary *profile_summary_ = nullptr;

  friend class Network;
  friend class Step;
  friend class InstanceAllocator;
};

// Compiler options.
struct Options {
  Order parameter_element_order = ROW_MAJOR; // element order for parameters
  bool debug = false;                        // insert breakpoint in cell
  bool profiling = false;                    // enable profiling
  bool external_profiler = false;            // external profiling buffer
  bool global_profiler = false;              // global profiling buffer
  bool dynamic_allocation = false;           // dynamic instance allocation
  bool sync_steps = false;                   // synchronize all steps
  bool fast_math = false;                    // fast approximate math ops
  bool aot = false;                          // ahead-of-time compilation
  bool pic = false;                          // position-independent code
  int sparse_threshold = 64;                 // threshold for sparse update
  int64 *flops_address = nullptr;            // address of FLOPs counter

  bool ref_profiler() const { return external_profiler || global_profiler; }
};

// A network is a collection of cells and variables that are compiled as a unit.
class Network {
 public:
  // Resource that is managed by the network.
  class Resource {
   public:
    virtual ~Resource() = default;
  };

  Network();
  ~Network();

  // Compile network to generate code for all the cells.
  bool Compile(const Flow &flow, const Library &library);

  // Load flow from file and compile all the cells.
  bool Compile(const string &flowfile, const Library &library);

  // Bind artifacts in flow to tensors, steps and cells in network.
  void Bind(Flow *flow);

  // Look up cell returning null if it is not found.
  Cell *LookupCell(const string &name) const;

  // Get cell.
  Cell *GetCell(const string &name) const;

  // Look up up parameter tensor returning null if it is not found.
  Tensor *LookupParameter(const string &name) const;

  // Get parameter tensor.
  Tensor *GetParameter(const string &name) const;

  // Return tensor data object for global tensor.
  TensorData operator[](Tensor *global) {
    CHECK(global->IsGlobal());
    return TensorData(global->data(), global);
  }
  TensorData operator[](const Flow::Variable *var) {
    DCHECK(var->tensor != nullptr) << var->name;
    CHECK(var->tensor->IsGlobal());
    return TensorData(var->tensor->data(), var->tensor);
  }
  TensorData operator[](const string &name) {
    Tensor *global = GetParameter(name);
    DCHECK(global != nullptr) << "Unknown parameter: " << name;
    CHECK(global->IsGlobal());
    return TensorData(global->data(), global);
  }

  // Add tensor to network.
  void AddTensor(Tensor *tensor);

  // Allocate memory in memory pool.
  char *AllocateMemory(size_t size, int alignment);

  // Add resource to network. This is deleted together with the network.
  void AddResource(Resource *resource) { resources_.push_back(resource); }

  // Initialize model parameters.
  void InitModelParameters(int64 seed = 0);

  // Save weights after training. This copies the value of each learnable tensor
  // in the network to the corresponding variable in the flow. This clears the
  // learning flag for the variable and turns it into a constant.
  void SaveParameters(Flow *flow) const;

  // Copy weight from flow for learnable tensors.
  void LoadParameters(const Flow &flow);

  // Runtime support functions.
  Runtime *runtime() const { return runtime_; }
  void set_runtime(Runtime *runtime) { runtime_ = runtime; }

  // Linker.
  Linker *linker() const { return linker_; }
  void set_linker(Linker *linker) { linker_ = linker; }

  // Compiler options.
  Options &options() { return options_; }
  const Options &options() const { return options_; }

  // Set element order for parameters.
  void set_parameter_element_order(Order order) {
    options_.parameter_element_order = order;
  }

  // Enable debugging by inserting a break point in the generated code.
  void set_debug(bool debug) { options_.debug = debug; }

  // Enable profiling by instrumenting code with timestamp timing code.
  void set_profiling(bool profiling) { options_.profiling = profiling; }

  // Enable dynamic instance allocation which allows instance variables to
  // overlap in the instance data block.
  void set_dynamic_allocation(bool dynamic) {
    options_.dynamic_allocation = dynamic;
  }

  // Network cells.
  const std::vector<Cell *> cells() const { return cells_; }

  // Global tensors.
  const std::vector<Tensor *> globals() const { return globals_; }

  // Network parameters.
  const std::vector<Tensor *> parameters() const { return parameters_; }

  // Network steps.
  const std::vector<Step *> &steps() const { return steps_; }

 private:
  // Compute live ranges for all the variables.
  void ComputeLiveRanges();

  // Allocate aligned tensor from data in standard order.
  char *AllocateTensor(Tensor *tensor);

  // Network cells.
  std::vector<Cell *> cells_;

  // Global tensors in network, e.g. learnable or constant weight matrices and
  // scalars.
  std::vector<Tensor *> globals_;

  // Parameters in instance blocks (input, output, and intermediate values).
  std::vector<Tensor *> parameters_;

  // Steps for network computation in order of execution.
  std::vector<Step *> steps_;

  // Parameter names.
  std::unordered_map<string, Tensor *> names_;

  // Memory blocks owned by network.
  std::vector<char *> memory_;

  // Resources owned by network.
  std::vector<Resource *> resources_;

  // Runtime support.
  Runtime *runtime_;

  // Linker for linking code and data.
  Linker *linker_;

  // Compiler options.
  Options options_;

  friend class Instance;
};

// A custom kernel allows implementation of kernels in C++. The kernel function
// is called at runtime with the input and output parameters.
class CustomKernel : public Kernel {
 public:
  // Selection criterion function.
  typedef bool (*Criterion)(Step *step);

  // Input or output parameter constraints.
  struct Param {
    Type type = DT_INVALID;  // parameter type
    int rank = -1;           // parameter rank
  };

  // Create custom kernel.
  CustomKernel(const string &op, const string &name, void *func,
               int indegree, int outdegree);

  // Set type and rank for input.
  CustomKernel &Input(int index, Type type, int rank = -1);

  // Set type and rank for output.
  CustomKernel &Output(int index, Type type, int rank = -1);

  // Set selection criterion.
  CustomKernel &Select(Criterion criterion);

  // Kernel interface.
  string Name() override;
  string Operation() override;
  bool Supports(Step *step) override;
  void Generate(Step *step, MacroAssembler *masm) override;

 private:
  string op_;                   // operation supported by kernel
  string name_;                 // kernel name
  void *func_;                  // function implementing kernel operation
  std::vector<Param> inputs_;   // input parameter constraints
  std::vector<Param> outputs_;  // output parameter constraints
  Criterion criterion_;         // selection criterion
};

inline Runtime *Cell::runtime() const {
  return network_->runtime();
}

inline Runtime *Channel::runtime() const {
  return format_->cell()->runtime();
}

inline Runtime *Instance::runtime() const {
  return cell_->runtime();
}

inline void Instance::Compute() {
  cell_->code().Execute(data_);
}

inline int Instance::num_tasks() const {
  return cell_->num_tasks();
}

inline Task *Instance::task(int index) const {
  return reinterpret_cast<Task *>(data_ + cell_->task_offset(index));
}

inline size_t Instance::size() const {
  return cell_->instance_size();
}

inline int Instance::alignment() const {
  return cell_->instance_alignment();
}

inline void Instance::set_profile(ProfileSummary *summary) {
  DCHECK(cell_->profile() != nullptr);
  SetReference(cell_->profile(), summary->data());
}

inline TensorData Instance::operator[](const string &name) {
  Tensor *param = cell_->GetParameter(name);
  DCHECK(param != nullptr) << "Unknown parameter: " << name;
  return TensorData(data_ + param->offset(), param);
}

}  // namespace myelin
}  // namespace sling

#endif  // SLING_MYELIN_COMPUTE_H_

