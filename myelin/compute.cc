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

#include "myelin/compute.h"

#include <stdlib.h>
#include <algorithm>
#include <list>
#include <string>
#include <unordered_map>

#include "base/logging.h"
#include "base/types.h"
#include "file/file.h"
#include "myelin/macro-assembler.h"

namespace sling {
namespace myelin {

#if !defined(__x86_64__)
#error Myelin requires 64-bit x86
#endif

#define __ masm->

// Combined tensor order.
static const Order combined_order[4][4] = {
  {ANY_ORDER,         ROW_MAJOR,         COLUMN_MAJOR,      CONFLICTING_ORDER},
  {ROW_MAJOR,         ROW_MAJOR,         CONFLICTING_ORDER, CONFLICTING_ORDER},
  {COLUMN_MAJOR,      CONFLICTING_ORDER, COLUMN_MAJOR,      CONFLICTING_ORDER},
  {CONFLICTING_ORDER, CONFLICTING_ORDER, CONFLICTING_ORDER, CONFLICTING_ORDER},
};

// Placement names.
const char *placename[] = {"nowhere", "host", "device", "host and device"};

static int LeastCommonMultiple(int n, int m) {
  int a = n;
  int b = m;
  while (a != b) {
    if (a < b) {
      a += n;
    } else {
      b += m;
    }
  }
  return a;
}

static void EnsureAlignment(int *m, int n) {
  *m = LeastCommonMultiple(*m, n);
}

static bool IsPowerOfTwo32(int value) {
  return value && !(value & (value - 1));
}

static size_t Align(size_t n, int align) {
  DCHECK(IsPowerOfTwo32(align));
  return (n + align - 1) & ~(align - 1);
}

static char *MemAlloc(size_t size, int alignment) {
  DCHECK(IsPowerOfTwo32(alignment));
  DCHECK_GE(alignment, sizeof(void *));
  char *data;
  int rc = posix_memalign(reinterpret_cast<void **>(&data), alignment, size);
  CHECK_EQ(rc, 0) << "Cannot allocate memory, size: " << size
                  << " alignment: " << alignment;
  return data;
}

static void MemFree(char *data) {
  free(data);
}

// Basic runtime for serial execution of cells on a single CPU thread.
class BasicRuntime : public Runtime {
 public:
  void AllocateInstance(Instance *instance) override {
    char *data = MemAlloc(instance->size(), instance->alignment());
    memset(data, 0, instance->size());
    instance->set_data(data);
  }

  void FreeInstance(Instance *instance) override {
    MemFree(instance->data());
  }

  void ClearInstance(Instance *instance) override {
    memset(instance->data(), 0, instance->size());
  }

  bool SupportsAsync() override {
    return false;
  }

  TaskFunc StartTaskFunc() override {
    return StartTask;
  }

  TaskFunc WaitTaskFunc() override {
    return WaitTask;
  }

  static void StartTask(Task *task) {
    task->func(task->arg);
  }

  static void WaitTask(Task *task) {
  }
};

static BasicRuntime default_runtime;

// An instance allocator allocates space for variables in an instance data
// block. It keeps track of which parts of the block are in use and tries to
// allocate space by reusing free parts of the instance block that is no longer
// in use.
class InstanceAllocator {
 public:
  // Initialize instance allocator for cell and placement.
  InstanceAllocator(Cell *cell, Placement placement) : placement_(placement) {
    if (placement == HOST) {
      max_instance_size_ = &cell->instance_size_;
      instance_alignment_ = &cell->instance_alignment_;
    } else {
      max_instance_size_ = &cell->device_instance_size_;
      instance_alignment_ = &cell->device_instance_alignment_;
    }
    current_instance_size_ = *max_instance_size_;
  }

  // Allocate space for variable in instance data block.
  void Allocate(Tensor *var) {
    // Shared variables share offset.
    if (var->shared_ != nullptr) {
      if (placement_ == HOST) {
        CHECK(var->shared_->offset_ != -1)
            << var->name() << " " << var->shared_->name();
        var->offset_ = var->shared_->offset_;
      } else {
        CHECK(var->shared_->device_offset_ != -1)
            << var->name() << " " << var->shared_->name();
        var->device_offset_ = var->shared_->device_offset_;
      }
      return;
    }

    // Determine size and alignment.
    size_t size = var->space();
    int align = var->ref_ ? kMinDataAlignment : var->byte_alignment_;

    // Try to find free space in the instance block.
    size_t offset = -1;
    for (auto it = freelist_.begin(); it != freelist_.end(); ++it) {
      if (it->first + size > it->second) continue;
      int aligned = Align(it->first, align);
      if (aligned + size > it->second) continue;

      // Free block found.
      offset = aligned;

      // Remove block from free list and add back the alignment padding in front
      // and the excess data back to the free list.
      DCHECK(FreeListConsistent());
      int padding = aligned - it->first;
      int excess = it->second - aligned - size;
      if (padding == 0 && excess == 0) {
        freelist_.erase(it);
      } else if (padding == 0) {
        it->first = offset + size;
      } else if (excess == 0) {
        it->second = offset;
      } else {
        it->second = offset;
        Insert(offset + size, offset + size + excess);
      }
      DCHECK(FreeListConsistent());
      break;
    }

    if (offset == -1) {
      // No free space in instance block. Extend the instance block and add new
      // variable at the end.
      size_t end = current_instance_size_;
      size_t aligned = Align(end, align);
      offset = aligned;
      current_instance_size_ = aligned + size;
      if (current_instance_size_ > *max_instance_size_) {
        *max_instance_size_ = current_instance_size_;
      }
      if (aligned > end) {
        // Insert alignment padding in free list.
        Insert(end, aligned);
      }
    }

    // Ensure that instance has at least the same alignment as the tensor.
    if (var->byte_alignment_ > *instance_alignment_) {
      *instance_alignment_ = var->byte_alignment_;
    }

    // Assign offset to tensor.
    if (placement_ == HOST) {
      var->offset_ = offset;
    } else {
      var->device_offset_ = offset;
    }
  }

  // Release space used by variable in instance data block.
  void Release(Tensor *var) {
    // Shared variables are allocated together.
    if (var->shared_ != nullptr) return;

    // Get offset and size.
    size_t offset = placement_ == HOST  ? var->offset_ : var->device_offset_;
    size_t size = var->space();

    // Insert block in free list.
    Insert(offset, offset + size);
  }

  // Check consistency of free list.
  bool FreeListConsistent() const {
    size_t offset = 0;
    bool first = true;
    for (auto &e : freelist_) {
      const char *error = nullptr;
      if (e.second < e.first) {
        error = "Invalid free list entry";
      } else if (e.first == e.second) {
        error = "Zero-sized free list entry";
      } else if (e.first < offset) {
        error = "Free list entry out of order";
      } else if (!first && e.first == offset) {
        error = "Non-consolidated free list entry";
      }

      if (error != nullptr) {
        DumpFreeList();
        LOG(ERROR) << error << " at " << e.first << ", free list:";
        return false;
      }
      offset = e.second;
      first = false;
    }
    return true;
  }

  // Return size of free list.
  size_t Free() const {
    size_t free = 0;
    for (auto &e : freelist_) free+= e.second - e.first;
    return free;
  }

  // Dump free list to log.
  void DumpFreeList() const {
    size_t prev = 0;
    for (auto &e : freelist_) {
      size_t size = e.second - e.first;
      size_t gap = e.first - prev;
      prev = e.second;
      LOG(INFO) << "  at " << e.first
                << " end " <<  e.second
                << " size " << size
                << " gap " << gap;
      prev = e.second;
    }
  }

 private:
  // Insert element in free list.
  void Insert(size_t start, size_t end) {
    // Find first entry after the block or the first entry ending at the start
    // of the block.
    DCHECK_LT(start, end);
    DCHECK(FreeListConsistent());
    auto it = freelist_.begin();
    while (it != freelist_.end() &&
           it->second < start &&
           it->first < end) {
      ++it;
    }

    if (it == freelist_.end()) {
      // Add new free list entry at the end.
      freelist_.emplace_back(start, end);
    } else if (it->second == start) {
      // Append block to current entry.
      it->second = end;

      // Check if it can be merged with the next entry.
      auto prev = it;
      if (++it != freelist_.end() && it->first == end) {
        prev->second = it->second;
        freelist_.erase(it);
      }
    } else if (it->first == end) {
      // Prepend block to current entry.
      it->first = start;
    } else {
      // Insert new entry before the current entry.
      freelist_.emplace(it, start, end);
    }

    // Remove last free list entry if this extends to the end of the current
    // instance block.
    if (!freelist_.empty()) {
      auto &last = freelist_.back();
      if (last.second == current_instance_size_) {
        current_instance_size_ = last.first;
        freelist_.pop_back();
      }
    }

    DCHECK(FreeListConsistent());
  }

  // Instance data is allocated in either the host instance data block or the
  // device instance data block.
  Placement placement_;

  // Maximum size of instance.
  size_t *max_instance_size_;

  // Maximum instance alignment.
  int *instance_alignment_;

  // Current instance size.
  size_t current_instance_size_;

  // List of free blocks (start,end) in instance.
  std::list<std::pair<size_t, size_t>> freelist_;
};

Library::~Library() {
  if (owns_kernels_) {
    for (auto o : kernels_) {
      for (auto k : o.second) delete k;
    }
  }
}

void Library::Register(Kernel *kernel) {
  VLOG(7) << "Register " << kernel->Name() << " for " << kernel->Operation();
  kernels_[kernel->Operation()].push_back(kernel);
}

CustomKernel &Library::Register(const string &op, const string &name,
                                void (*func)(const TensorData &arg,
                                             TensorData *output)) {
  return RegisterCustomKernel(op, name,reinterpret_cast<void *>(func), 1, 1);
}

CustomKernel &Library::Register(const string &op, const string &name,
                                void (*func)(const TensorData &arg1,
                                             const TensorData &arg2,
                                             TensorData *output)) {
  return RegisterCustomKernel(op, name, reinterpret_cast<void *>(func), 2, 1);
}

CustomKernel &Library::Register(const string &op, const string &name,
                                void (*func)(const TensorData &arg1,
                                             const TensorData &arg2,
                                             const TensorData &arg3,
                                             TensorData *output)) {
  return RegisterCustomKernel(op, name, reinterpret_cast<void *>(func), 3, 1);
}

CustomKernel &Library::Register(const string &op, const string &name,
                                void (*func)(const TensorData &arg1,
                                             const TensorData &arg2,
                                             const TensorData &arg3,
                                             const TensorData &arg4,
                                             TensorData *output)) {
  return RegisterCustomKernel(op, name, reinterpret_cast<void *>(func), 4, 1);
}

CustomKernel &Library::RegisterCustomKernel(const string &op,
                                            const string &name,
                                            void *func,
                                            int indegree,
                                            int outdegree) {
  CustomKernel *kernel = new CustomKernel(op, name, func, indegree, outdegree);
  Register(kernel);
  return *kernel;
}

const Library::Kernels &Library::Lookup(const string &op) const {
  auto f = kernels_.find(op);
  if (f == kernels_.end()) return no_kernels_;
  return f->second;
}

bool Library::Singleton(const string &op,
                        const string &name,
                        Library *singleton) const {
  // Singleton library must be empty or already non-owning.
  CHECK(!singleton->owns_kernels_ || singleton->kernels_.empty());
  singleton->owns_kernels_ = false;

  // Find kernel.
  auto f = kernels_.find(op);
  if (f == kernels_.end()) return false;
  for (Kernel *kernel : f->second) {
    if (kernel->Name() == name) {
      singleton->kernels_[kernel->Operation()].push_back(kernel);
      return true;
    }
  }
  return false;
}

void Tensor::MinAlign(const Shape &align) {
  CHECK_LE(align.rank(), minalign_.rank());
  for (int d = 0; d < align.rank(); ++d) {
    minalign_.set(d, LeastCommonMultiple(minalign_.dim(d), align.dim(d)));
  }
}

void Tensor::MinAlignLast(int align) {
  if (rank() > 0) {
    int d = rank() - 1;
    minalign_.set(d, LeastCommonMultiple(minalign_.dim(d), align));
  }
}

void Tensor::SameAlign(Tensor *other) {
  MinAlign(other->minalign_);
  other->MinAlign(minalign_);
}

void Tensor::CompatibleAlign(Tensor *other) {
  int d1 = rank() - 1;
  int d2 = other->rank() - 1;
  while (d1 >= 0 && d2 >= 0) {
    int lcm = LeastCommonMultiple(minalign_.dim(d1), other->minalign_.dim(d2));
    minalign_.set(d1--, lcm);
    other->minalign_.set(d2--, lcm);
  }
}

bool Tensor::Compatible(const Tensor *other) const {
  int d1 = rank() - 1;
  int d2 = other->rank() - 1;
  while (d1 >= 0 && d2 >= 0) {
    int s1 = dim(d1--);
    int s2 = other->dim(d2--);
    if (s1 == -1 || s1 == 1) continue;
    if (s2 == -1 || d2 == 1) continue;
    if (s1 != s2) return false;
  }
  return true;
}

bool Tensor::SupportsAlignment(const Shape &align) const {
  if (align.rank() != rank()) return false;
  if (require_dense_) {
    for (int d = 0; d < align.rank(); ++d) {
      if (dim(d) % align.dim(d) != 0) return false;
    }
  }
  return true;
}

bool Tensor::SupportsOrder(Order order) {
  return combined_order[required_order_][order] != CONFLICTING_ORDER;
}

void Tensor::SetRequiredOrder(Order order) {
  required_order_ = combined_order[required_order_][order];
}

void Tensor::SetMiniumAlignment(int alignment) {
  EnsureAlignment(&byte_alignment_, alignment);
}

bool Tensor::HasSameShape(const Tensor *other) const {
  return shape() == other->shape();
}

int Tensor::ConsumerTask() const {
  int consumer_task = -2;
  for (Step *step : consumers_) {
    if (consumer_task == -2) {
      consumer_task = step->task_index();
    } else if (consumer_task != step->task_index()) {
      // Tensor is consumed by steps in different tasks.
      return -1;
    }
  }
  return consumer_task == -2 ? -1 : consumer_task;
}

string Tensor::TypeString() const {
  string str;
  if (ref_) str.append("&");
  str.append(TypeTraits::of(type_).name());
  if (!shape_.scalar()) {
    str.append("[");
    str.append(shape_.ToString());
    str.append("]");
  }
  return str;
}

Channel::~Channel() {
  MemFree(data_);
}

void Channel::resize(int n) {
  // Allocate more space if needed.
  if (n > capacity_) {
    int cap = capacity_ * 2;
    if (cap < n) cap = n;
    if (cap < 8) cap = 8;
    reserve(cap);
  }

  // Clear new elements.
  if (n > size_) {
    memset(at(size_), 0, (n - size_) * connector_->size());
  }

  // Change size.
  size_ = n;
}

void Channel::reserve(int n) {
  // Never remove any existing elements.
  if (n < size_) return;
  if (n == capacity_) return;

  // Allocate new data buffer.
  char *buffer = MemAlloc(n * connector_->size(), connector_->alignment());

  // Copy existing data to new buffer.
  if (data_ != nullptr) {
    memcpy(buffer, data_, size_ * connector_->size());
    MemFree(data_);
  }

  // Set new data buffer.
  data_ = buffer;
}

ProfileSummary::ProfileSummary(Cell *cell) : cell_(cell) {
  CHECK(cell->profile() != nullptr);
  int size = cell->profile()->elements();
  data_ = new int64[size];
  for (int i = 0; i < size; ++i) data_[i] = 0;
}

ProfileSummary::~ProfileSummary() {
  delete [] data_;
}

Instance::Instance(const Cell *cell) : cell_(cell) {
  cell_->runtime()->AllocateInstance(this);
}

Instance::~Instance() {
  cell_->runtime()->FreeInstance(this);
}

void Instance::Clear() {
  cell_->runtime()->ClearInstance(this);
}

string Instance::ToString(Tensor *param) const {
  // Locate parameter in instance.
  char *p  = data_ + param->offset();
  if (param->ref()) {
    p = *reinterpret_cast<char **>(p);
    if (p == nullptr) return "null";
  }
  if (!param->shape().defined()) return "*";

  // Get type traits for elements.
  const TypeTraits &traits = TypeTraits::of(param->type());

  // Output tensor as string.
  string str;
  if (param->rank() == 0) {
    // Scalar.
    str = traits.str(p);
  } else if (param->rank() == 1) {
    // Vector.
    str.append("[");
    for (int r = 0; r < param->dim(0); ++r) {
      if (r > 0) str.append(",");
      str.append(traits.str(p + param->offset(r)));
    }
    str.append("]");
  } else if (param->rank() == 2) {
    // Matrix.
    str.append("[");
    for (int r = 0; r < param->dim(0); ++r) {
      if (r > 0) str.append(",");
      str.append("[");
      for (int c = 0; c < param->dim(1); ++c) {
        if (c > 0) str.append(",");
        str.append(traits.str(p + param->offset(r, c)));
      }
      str.append("]");
    }
    str.append("]");
  } else {
    str = "<<" + std::to_string(param->rank()) + "D tensor>>";
  }

  return str;
}

string Instance::ToString() const {
  string str;
  for (Tensor *t : cell()->network()->parameters()) {
    if (t->cell() == cell() && t->shared() == nullptr) {
      str.append(t->name());
      str.append(" = ");
      str.append(ToString(t));
      str.append("\n");
    }
  }
  return str;
}

void Step::SetRegisterUsage(int regs) {
  if (cell_ != nullptr && cell_->register_usage_ < regs) {
    cell_->register_usage_ = regs;
  }
}

void Step::SetPreservedRegisterUsage(int regs) {
  // There are eight caller-saved registers.
  SetRegisterUsage(8 + regs);
}

bool Step::AllowInPlace(int input, int output, bool preserved) {
  // Get input and output that should be shared.
  DCHECK_GE(input, 0);
  DCHECK_LT(input, inputs_.size());
  DCHECK_GE(output, 0);
  DCHECK_LT(output, outputs_.size());
  Tensor *in = inputs_[input];
  Tensor *out = outputs_[output];

  // Check if input can be shared.
  Tensor *t = in;
  while (t != nullptr) {
    if (!preserved) {
      if (t->IsConstant()) return false;
      if (t->consumers().size() != 1) return false;
      if (t->out()) return false;
    }
    if (t->ref() != out->ref()) return false;
    in = t;
    t = t->shared();
  }

  // Check if output can be shared.
  if (out->shared()) return false;

  // Share input and output.
  out->set_shared(in);
  if (out->shape() == in->shape()) out->set_link(in);
  return true;
}

bool Step::NeedsSynchronization() {
  // Only steps running on the host need synchronization.
  if (placement() != HOST) return false;

  // Only steps running in the main task need synchronization.
  if (task_index_ != -1) return false;

  // Check if any of the inputs has been produced on the device.
  for (Tensor *input : inputs_) {
    Step *producer = input->producer();
    if (producer == nullptr) continue;
    if (producer->placement() == HOST) continue;
    if (producer->task_index_ != -1) continue;
    return true;
  }
  return false;
}

char *Step::AllocateKernelMemory(size_t size, int alignment) {
  CHECK(kernel_memory_ == nullptr);
  CHECK(cell_ != nullptr);
  kernel_memory_ = cell_->network()->AllocateMemory(size, alignment);
  return kernel_memory_;
}

Network::Network() {
  runtime_ = &default_runtime;
}

Network::~Network() {
  for (auto *m : memory_) MemFree(m);
  for (auto *t : parameters_) delete t;
  for (auto *t : constants_) {
    if (t->shared() == nullptr) {
      if (t->device_data_ != DEVICE_NULL) {
        runtime_->RemoveTensorFromDevice(t);
      }
      delete t;
    }
  }
  for (auto *c : cells_) delete c;
  for (auto *s : steps_) delete s;
  for (auto *c : connectors_) delete c;
}

Tensor *Network::GetParameter(const string &name) const {
  auto f = names_.find(name);
  return f == names_.end() ? nullptr : f->second;
}

char *Network::AllocateMemory(size_t size, int alignment) {
  char *data = MemAlloc(size, alignment);
  memory_.push_back(data);
  return data;
}

static bool CompareUsage(const std::pair<int, Tensor *> &a,
                         const std::pair<int, Tensor *> &b) {
  if (a.first == b.first) {
    // Inputs are sorted before outputs.
    Tensor *va = a.second;
    Tensor *vb = b.second;
    for (auto *op : va->consumers()) {
      if (op == vb->producer()) return true;
    }
    for (auto *op : vb->consumers()) {
      if (op == va->producer()) return false;
    }
  }
  return a.first < b.first;
}

bool Network::Compile(const Flow &flow, const Library &library) {
  // Fetch information about the CPU we are running on.
  jit::CPU::Probe();

  // Create tensors for all the variables (parameters and constants).
  std::unordered_map<void *, Tensor *> tensors;
  for (Flow::Variable *var : flow.vars()) {
    Tensor *tensor = new Tensor();
    tensors[var] = tensor;
    if (var->data != nullptr) {
      constants_.push_back(tensor);
      tensor->data_ = var->data;
    } else {
      parameters_.push_back(tensor);
      tensor->required_order_ = options_.parameter_element_order;
    }
    tensor->name_ = var->name;
    names_[var->name] = tensor;
    for (const string &alias : var->aliases) {
      names_[alias] = tensor;
    }
    tensor->type_ = var->type;
    tensor->ref_ = var->ref;
    tensor->shape_ = var->shape;
    tensor->aligned_ = var->shape;
    tensor->minalign_.fill(var->rank(), 1);
    tensor->stride_.fill(var->rank(), 0);
    tensor->byte_alignment_ = TypeTraits::of(var->type).size();

    // Input variables are initially placed in host memory.
    tensor->in_ = var->in;
    if (var->in) {
      tensor->current_placement_ = HOST;
      tensor->placement_ = HOST;
    }

    // Output variables must be available on the host after the computation.
    tensor->out_ = var->out;
    if (var->out) tensor->placement_ = HOST;
  }

  // Create connectors between variables.
  for (Flow::Connector *cnx : flow.cnxs()) {
    // Connectors must have at least one link.
    if (cnx->links.empty()) {
      LOG(WARNING) << "Skipping empty connector: " << cnx->name;
      continue;
    }

    // Create new connector.
    Connector *connector = new Connector();
    connectors_.push_back(connector);

    // Create tensor for connector.
    Tensor *t = new Tensor();
    t->name_ = cnx->name;
    t->required_order_ = ROW_MAJOR;
    connector->type_ = t;
    tensors[connector] = t;

    // Initialize connector tensor from first link.
    Tensor *prototype = tensors[cnx->links[0]];
    t->type_ = prototype->type_;
    t->shape_ = prototype->shape_;
    t->shape_.set(0, -1);
    t->minalign_ = prototype->minalign_;
    t->aligned_ = prototype->aligned_;
    t->stride_ = prototype->stride_;
    t->byte_alignment_ = prototype->byte_alignment_;

    // Link tensors to connector.
    for (Flow::Variable *link : cnx->links) {
      CHECK(link->ref);
      tensors[link]->link_ = t;
    }
  }

  // Find kernels for implementing each step.
  std::unordered_map<Flow::Function *, Cell *> cells;
  for (Flow::Operation *op : flow.ops()) {
    // Create step for operation.
    Step *step = new Step();
    steps_.push_back(step);
    step->name_ = op->name;
    step->type_ = op->type;
    step->attributes_ = op->attrs;

    // Set or create cell for step.
    Cell *cell = cells[op->func];
    if (cell == nullptr) {
      cell = new Cell();
      cell->network_ = this;
      cell->name_ = op->func->name;
      cells_.push_back(cell);
      cells[op->func] = cell;
    }
    cell->steps_.push_back(step);
    step->cell_ = cell;

    // Add inputs to step.
    for (Flow::Variable *input : op->inputs) {
      Tensor *tensor = tensors[input];
      CHECK(tensor != nullptr);
      step->inputs_.push_back(tensor);
      tensor->consumers_.push_back(step);

      // Assign input parameter to cell.
      if (step->cell_ != nullptr && !tensor->IsConstant()) {
        if (tensor->cell_ != nullptr && tensor->cell_ != step->cell_) {
          LOG(FATAL) << tensor->name_ << " belongs to both "
                     << tensor->cell_->name_ << " and "
                     << step->cell_->name_;
        }
        tensor->cell_ = step->cell_;
      }
    }

    // Add outputs to step.
    for (Flow::Variable *output : op->outputs) {
      Tensor *tensor = tensors[output];
      CHECK(tensor != nullptr);
      step->outputs_.push_back(tensor);
      CHECK(tensor->producer_ == nullptr);
      tensor->producer_ = step;

      // Assign output parameter to cell.
      if (step->cell_ != nullptr && !tensor->IsConstant()) {
        if (tensor->cell_ != nullptr && tensor->cell_ != step->cell_) {
          LOG(FATAL) << tensor->name_ << " belongs to both "
                     << tensor->cell_->name_ << " and "
                     << step->cell_->name_;
        }
        tensor->cell_ = step->cell_;
      }
    }

    // Assign task to step.
    if (runtime_->SupportsAsync() && op->task != 0) {
      // Add task to cell.
      int taskidx = -1;
      for (int i = 0; i < cell->tasks_.size(); ++i) {
        if (cell->tasks_[i].task == op->task) {
          taskidx = i;
          break;
        }
      }
      if (taskidx == -1) {
        // Add new task to cell.
        taskidx = cell->tasks_.size();
        cell->tasks_.emplace_back(op->task);
      }
      step->task_index_ = taskidx;
    }

    // Find kernel for implementing the operation.
    auto &kernels = library.Lookup(step->type());
    for (int k = kernels.size() - 1; k >= 0; --k) {
      Kernel *kernel = kernels[k];
      if (kernel->Supports(step)) {
        // Check that kernel location is compatible with task placement.
        bool compatible = true;
        if (step->task_index_ != -1) {
          auto &task = cell->tasks_[step->task_index_];
          if (task.placement == NOWHERE) {
            // Task has not been placed yet. Use the kernel location for
            // placement.
            task.placement = kernel->Location();
          } else if (task.placement != kernel->Location()) {
            // Kernel location is incompatible with task placement.
            VLOG(7) << kernel->Name() << " cannot run on "
                    << placename[task.placement];
            compatible = false;
          }
        }

        if (compatible) {
          step->kernel_ = kernel;
          break;
        }
      } else {
        VLOG(7) << kernel->Name() << " does not support " << step->name();
      }
    }
    if (step->kernel_ == nullptr) {
      LOG(ERROR) << "No kernel supports " << step->name()
                 << " of type " << step->type();
      return false;
    }
    VLOG(3) << "Step " << step->name() << " implemented by "
            << step->kernel_->Name();
  }

  // Add tensors for profiling.
  if (options_.profiling) {
    for (Cell *cell : cells_) {
      // Allocate tensor for storing profiling information. The tensor is an
      // int64 vector where the first element is the invocation counter followed
      // by one element for each step in the cell computation for storing the
      // cycle counts. If the cell has parallel tasks, two additional cycle
      // counters are stored for each task.
      size_t size = 1 + cell->steps_.size() + 2 * cell->tasks_.size();
      Tensor *profile = new Tensor();
      profile->name_ = "timing/" + cell->name_;
      profile->cell_ = cell;
      profile->type_ = DT_INT64;
      profile->shape_.assign(size);
      profile->size_ = profile->space_ = size * sizeof(int64);
      profile->ref_ = options_.external_profiler;
      profile->aligned_ = profile->shape_;
      profile->minalign_.assign(sizeof(int64));
      profile->stride_.assign(sizeof(int64));
      profile->placement_ = HOST;
      profile->current_placement_ = HOST;
      profile->in_ = true;
      profile->out_ = true;
      parameters_.push_back(profile);
      tensors[profile] = profile;
      cell->profile_ = profile;
    }
  }

  // Let kernels adjust the input and output data alignment requirements.
  for (Step *step : steps_) {
    step->kernel_->Adjust(step);
  }

  // Propagate alignment between linked tensors.
  bool again = true;
  while (again) {
    // Keep propagating alignment constraints until there are no more
    // constraints to propagate.
    again = false;
    for (auto it : tensors) {
      Tensor *t = it.second;
      Tensor *l = t->link_;
      if (l == nullptr) continue;

      // Check type compatibility between linked tensors.
      if (t->type_ != l->type_ || !t->Compatible(l)) {
        LOG(ERROR) << "Incompatible type for tensor " << t->name()
                   << " " << t->TypeString() << " linked to " << l->name()
                   << " " << l->TypeString();
        return false;
      }

      // Propagate alignment.
      Shape &mint = t->minalign_;
      Shape &minl = l->minalign_;
      int dt = t->rank() - 1;
      int dl = l->rank() - 1;
      while (dt >= 0 && dl >= 0) {
        if (t->dim(dt) != -1 && l->dim(dl) != -1) {
          // Propagate minimum alignment in both directions.
          int align = LeastCommonMultiple(mint.dim(dt), minl.dim(dl));
          if (mint.dim(dt) != align) {
            mint.set(dt, align);
            again = true;
          }
          if (minl.dim(dl) != align) {
            minl.set(dl, align);
            again = true;
          }
        }
        dt--;
        dl--;
      }

      // Propagate order constraints.
      if (t->required_order_ != l->required_order_) {
        Order c = combined_order[t->required_order_][l->required_order_];
        if (t->required_order_ != c || l->required_order_ != c) {
          t->required_order_ = c;
          l->required_order_ = c;
          again = true;
        }
      }

      // Propagate byte alignment.
      int align = LeastCommonMultiple(t->byte_alignment_, l->byte_alignment_);
      if (t->byte_alignment_ != align) {
        t->byte_alignment_ = align;
        again = true;
      }
      if (t->byte_alignment_ != align) {
        l->byte_alignment_ = align;
        again = true;
      }
    }
  }

  // Compute tensor sizes.
  for (auto it : tensors) {
    Tensor *tensor = it.second;

    // Determine element order.
    CHECK_EQ(tensor->order(), ROW_MAJOR);
    switch (tensor->required_order_) {
      case COLUMN_MAJOR:
        // Swap element order.
        tensor->order_ = COLUMN_MAJOR;
        break;
      case ANY_ORDER:
      case ROW_MAJOR:
        // Already in row-major order.
        break;
      case CONFLICTING_ORDER:
        LOG(ERROR) << "Conflicting order requirements for " << tensor->name();
        return false;
    }

    // Check for dense encoding conflicts.
    if (tensor->require_dense_) {
      for (int d = 0; d < tensor->rank(); ++d) {
        if (tensor->dim(d) % tensor->minalign(d) != 0) {
          LOG(ERROR) << "Conflicting dense encoding requirements for "
                     << tensor->name()
                     << " shape " << tensor->shape().ToString()
                     << " align " << tensor->minalign().ToString();
          return false;
        }
      }
    }

    // Compute stride size for each dimension.
    size_t size = TypeTraits::of(tensor->type()).size();
    if (tensor->order_ == ROW_MAJOR) {
      for (int d = tensor->rank() - 1; d >= 0; --d) {
        tensor->stride_.set(d, size);
        int dim = tensor->shape_.dim(d);
        if (dim == -1) dim = 1;
        int align = Align(dim, tensor->minalign_.dim(d));
        tensor->aligned_.set(d, align);
        size *= align;
      }
    } else {
      for (int d = 0; d < tensor->rank(); ++d) {
        tensor->stride_.set(d, size);
        int dim = tensor->shape_.dim(d);
        if (dim == -1) dim = 1;
        int align = Align(dim, tensor->minalign_.dim(d));
        tensor->aligned_.set(d, align);
        size *= align;
      }
    }
    tensor->size_ = size;
    tensor->space_ = tensor->ref() ? sizeof(void *) : size;

    // Determine placement for tensor based on producer and consumer locations.
    if (tensor->producer_ != nullptr) {
      // Tensor is available in the place it is produced.
      tensor->AddPlace(tensor->producer_->placement());
    }
    for (Step *consumer : tensor->consumers_) {
      // Tensor must be made available in the places it is consumed.
      tensor->AddPlace(consumer->placement());
    }

    VLOG(5) << "Tensor " << tensor->name_ << ": " << tensor->TypeString()
            << " align " << tensor->minalign_.ToString()
            << ":" << tensor->byte_alignment_
            << " aligned " << tensor->aligned_.ToString()
            << " size " << tensor->space_
            << " stride " << tensor->stride_.ToString()
            << " order " << tensor->order_
            << " on " << placename[tensor->placement_];
  }

  // Propagate size and alignment for shared tensors.
  for (auto it : tensors) {
    Tensor *tensor = it.second;
    Tensor *next = tensor->shared_;
    while (next != nullptr) {
      if (next->size_ < tensor->size_) {
        next->size_ = tensor->size_;
      }
      if (next->byte_alignment_ < tensor->byte_alignment_) {
        next->byte_alignment_ = tensor->byte_alignment_;
      }
      next = next->shared_;
    }
  }

  // Compute alignment for connectors.
  for (Connector *connector : connectors_) {
    Tensor *t = connector->type_;
    EnsureAlignment(&connector->alignment_, t->byte_alignment_);
    EnsureAlignment(&connector->alignment_, jit::CPU::CacheLineSize());

    VLOG(5) << "Connector " << connector->name() << ": " << t->TypeString()
            << " min " << t->minalign_.ToString()
            << ":" << connector->alignment_
            << " aligned " << t->aligned_.ToString()
            << " size " << t->size_
            << " stride " << t->stride_.ToString()
            << " order " << t->order_;
  }

  // Move all variables that are shared with a constant to the constant pool.
  for (auto it = parameters_.begin(); it != parameters_.end();) {
    // Check if tensor is shared with a constant.
    Tensor *t = *it;
    if (t->shared_ != nullptr && t->shared_->IsConstant()) {
      // Move variable to constant pool.
      VLOG(5) << "Convert " << t->name() << " to constant";
      it = parameters_.erase(it);
      constants_.push_back(t);
    } else {
      ++it;
    }
  }

  // Compute live ranges for all variables.
  ComputeLiveRanges();
  std::vector<std::pair<int, Tensor *>> enter;
  std::vector<std::pair<int, Tensor *>> leave;
  for (Tensor *var : parameters_) {
    if (var->first_ != -1) enter.emplace_back(var->first_, var);
    if (var->last_ != -1) leave.emplace_back(var->last_, var);
  }
  std::sort(enter.begin(), enter.end(), CompareUsage);
  std::sort(leave.begin(), leave.end(), CompareUsage);

  // Compute cell instance size and offset of each parameter.
  for (Cell *cell : cells_) {
    // Adjust cell instance to cache lines.
    EnsureAlignment(&cell->instance_alignment_, jit::CPU::CacheLineSize());

    // Allocate space for runtime data at the beginning of the instance block.
    cell->instance_size_ = runtime_->ExtraInstanceData(cell);

    // Allocate task structures in instance.
    for (auto &t : cell->tasks_) {
      t.offset = cell->instance_size_;
      cell->instance_size_ += sizeof(Task);
    }

    // Allocate space for variables in instance data blocks.
    cell->data_start_ = cell->instance_size_;
    InstanceAllocator host_allocator(cell, HOST);
    InstanceAllocator device_allocator(cell, DEVICE);
    int e = 0;
    int l = options_.dynamic_allocation ? 0 : leave.size();
    int s = 0;
    while (e < enter.size() || l < leave.size()) {
      // Allocate space for new variables produced by step.
      while (e < enter.size() && enter[e].first <= s) {
        Tensor *var = enter[e].second;
        if (var->cell_ == cell) {
          if (var->placement_ & HOST) host_allocator.Allocate(var);
          if (var->placement_ & DEVICE) device_allocator.Allocate(var);
        }
        e++;
      }

      // Release space for variables that are no longer needed after step.
      while (l < leave.size() && leave[l].first <= s) {
        Tensor *var = leave[l].second;
        if (var->cell_ == cell) {
          if (var->placement_ & HOST) host_allocator.Release(var);
          if (var->placement_ & DEVICE) device_allocator.Release(var);
        }
        l++;
      }
      s++;
    }
  }

  // Copy and align constants.
  for (Tensor *tensor : constants_) {
    if (tensor->shared_ != nullptr) {
      // Shared constant. Copy reference to data.
      CHECK(tensor->shared_->IsConstant());
      tensor->data_ = tensor->shared_->data_;
      tensor->AddNewPlace(HOST);
      if (tensor->placement_ & DEVICE) {
        tensor->device_data_ = tensor->shared_->device_data_;
        tensor->AddNewPlace(DEVICE);
      }
    } else {
      // Allocate aligned tensor and copy data.
      tensor->data_ = AllocateTensor(tensor);
      if (tensor->data_ == nullptr) return false;
      tensor->AddNewPlace(HOST);

      // Copy constant to device if needed.
      if (tensor->placement_ & DEVICE) {
        VLOG(5) << "Copy tensor " << tensor->name() << " to device";
        tensor->device_data_ = runtime_->CopyTensorToDevice(tensor);
        CHECK(tensor->device_data_ != DEVICE_NULL);
        tensor->AddNewPlace(DEVICE);
      }
    }
  }

  // Compile each cell computation.
  for (Cell *cell : cells_) {
    // Create macro assembler for code generation.
    MacroAssembler masm(nullptr, 0, options_);
    masm.set_runtime(runtime_);

    // Declare the number of registers needed by the cell.
    if (!masm.rr().usage(cell->register_usage_)) return false;

    // Insert break point in the beginning of the generated code in debug mode.
    if (options_.debug) masm.Breakpoint();

    // Generate prologue for main cell computation.
    masm.Prologue();
    runtime_->GeneratePrologue(cell, &masm);

    // Increment the invocation counter.
    if (options_.profiling) {
      // Invocation counter is the first element of the timing block.
      masm.IncrementInvocations(cell->profile()->offset());

      // Start runtime profiler.
      masm.CallInstanceFunction(runtime_->StartProfilerFunc());
    }

    // Copy input variables that do not have the placement required by the
    // consumers.
    bool sync = false;
    for (Tensor *tensor : parameters_) {
      if (tensor->cell_ != cell) continue;
      if (tensor->placement_ == EVERYWHERE) {
        int task = tensor->ConsumerTask();
        if (tensor->current_placement_ == HOST) {
          // Copy parameter tensor from host to device.
          runtime_->EmitCopyTensorToDevice(tensor, cell, task, &masm);
          tensor->AddNewPlace(DEVICE);
        } else if (tensor->current_placement_ == DEVICE) {
          // Copy parameter tensor from device to host.
          runtime_->EmitCopyTensorFromDevice(tensor, cell, task, &masm);
          tensor->AddNewPlace(HOST);
        }
        if (task == -1) sync = true;
      }
    }

    // Let kernels generate code for each step.
    int stepnum = 0;
    for (Step *step : cell->steps_) {
      if (step->task_index_ == -1) {
        // Wait for completion of all inputs.
        for (Tensor *input : step->inputs_) {
          // Check if input is produced by parallel task.
          if (input->producer() == nullptr) continue;
          int tidx = input->producer()->task_index();
          if (tidx == -1) continue;

          // Wait for producing task to complete.
          auto &t = cell->tasks_[tidx];
          CHECK(t.state != PENDING) << cell->name_ << " task " << t.task;
          if (t.state == ACTIVE) {
            // Wait for parallel task to complete.
            masm.WaitForTask(t.offset);
            t.state = COMPLETED;

            // Profile task wait.
            if (options_.profiling) {
              int timing = cell->profile()->offset();
              int slot = 1 + cell->steps_.size() + tidx * 2 + 1;
              masm.TimeStep(timing, slot * sizeof(int64));
            }
          }
        }

        // Synchronize main task if needed before executing step.
        if (sync && step->NeedsSynchronization()) {
          masm.CallInstanceFunction(runtime_->SyncMainFunc());
          sync = false;
        }

        // Generate code for step.
        auto pc = masm.pc_offset();
        VLOG(8) << step->name() << " @ " << reinterpret_cast<uint64 *>(pc);
        step->kernel_->Generate(step, &masm);
        if (masm.pc_offset() == pc) step->noop_ = true;

        // No registers are preserved between steps, so reset register
        // allocation.
        masm.ResetRegisterUsage();

        // Copy outputs that do not have the placement required by the
        // consumers.
        for (Tensor *output : step->outputs_) {
          output->AddNewPlace(step->placement());
          if (output->placement_ == EVERYWHERE) {
            int task = output->ConsumerTask();
            if (output->current_placement_ == HOST) {
              // Copy output from host to device.
              runtime_->EmitCopyTensorToDevice(output, cell, task, &masm);
              output->AddNewPlace(DEVICE);
            } else if (output->current_placement_ == DEVICE) {
              // Copy output from device to host.
              runtime_->EmitCopyTensorFromDevice(output, cell, task, &masm);
              output->AddNewPlace(HOST);
            }
            if (task == -1) sync = true;
          }
        }

        // Profile step.
        if (options_.profiling && !step->noop_) {
          int timing = cell->profile()->offset();
          masm.TimeStep(timing, (stepnum + 1) * sizeof(int64));
        }
      } else {
        // Parallel step.
        int tidx = step->task_index_;
        auto &t = cell->tasks_[tidx];
        CHECK(t.state != COMPLETED) << cell->name_ << " task " << t.task;
        if (t.state == PENDING) {
          // Flush asynchronous operations.
          if (sync) {
            masm.CallInstanceFunction(runtime_->SyncMainFunc());
            sync = false;
          }

          // Start parallel task.
          masm.StartTask(t.offset, t.task, step->task_index_, &t.entry);
          t.state = ACTIVE;

          // Profile task start.
          if (options_.profiling) {
            int timing = cell->profile()->offset();
            int slot = 1 + cell->steps_.size() + tidx * 2;
            masm.TimeStep(timing, slot * sizeof(int64));
          }
        }

        // Update output placements.
        for (Tensor *output : step->outputs_) {
          output->AddNewPlace(step->placement());
          if (output->placement_ == EVERYWHERE) {
            if (output->current_placement_ == HOST) {
              // Set deferred copy from host to device.
              output->deferred_placement_ = DEVICE;
              output->AddNewPlace(DEVICE);
            } else if (output->current_placement_ == DEVICE) {
              // Set deferred copy from device to host.
              output->deferred_placement_ = HOST;
              output->AddNewPlace(HOST);
            }
          }
        }
      }
      stepnum++;
    }

    // Make sure all tasks have completed.
    for (auto &task : cell->tasks_) {
      if (task.state == ACTIVE) {
        masm.WaitForTask(task.offset);
        task.state = COMPLETED;
      }
    }
    if (sync) {
      masm.CallInstanceFunction(runtime_->SyncMainFunc());
    }

    // Stop runtime profiler.
    if (options_.profiling) {
      masm.CallInstanceFunction(runtime_->StopProfilerFunc());
    }

    // Generate epilogue for main cell computation.
    runtime_->GenerateEpilogue(cell, &masm);
    masm.Epilogue();

    // Generate code for parallel tasks.
    int task_index = 0;
    for (auto &task : cell->tasks_) {
      // Set entry for task function.
      masm.bind(&task.entry);

      // Generate parallel task prologue.
      masm.Prologue();

      // Let kernels generate code for each step.
      int stepnum = 0;
      for (Step *step : cell->steps_) {
        if (step->task_index_ == task_index) {
          // Generate code for step.
          auto pc = masm.pc_offset();
          VLOG(8) << step->name() << " @ " << reinterpret_cast<uint64 *>(pc);
          step->kernel_->Generate(step, &masm);
          if (masm.pc_offset() == pc) step->noop_ = true;

          // No registers are preserved between steps, so reset register
          // allocation.
          masm.ResetRegisterUsage();

          // Copy outputs that do not have the placement required by the
          // consumers.
          for (Tensor *output : step->outputs_) {
            if (output->deferred_placement_ == DEVICE) {
              // Copy output from host to device.
              runtime_->EmitCopyTensorToDevice(
                  output, cell, task_index, &masm);
            } else if (output->deferred_placement_ == HOST) {
              // Copy output from device to host.
              runtime_->EmitCopyTensorFromDevice(
                  output, cell, task_index, &masm);
            }
          }

          // Profile step.
          if (options_.profiling && !step->noop_) {
            int timing = cell->profile()->offset();
            masm.TimeStep(timing, (stepnum + 1) * sizeof(int64));
          }
        }
        stepnum++;
      }

      // Generate parallel task epilogue.
      masm.Epilogue();

      task_index++;
    }

    // Generate static data blocks.
    masm.GenerateDataBlocks();

    // Allocate executable code object for generated code.
    cell->code_.Allocate(&masm);
    VLOG(5) << cell->name()
            << " entry address: " << cell->code_.entry()
            << " code size: " << cell->code_.size()
            << " data size: " << cell->instance_size();
  }

  return true;
}

bool Network::Compile(const string &flowfile, const Library &library) {
  // Load flow file.
  Flow flow;
  if (!flow.Load(flowfile).ok()) {
    LOG(ERROR) << "Error loading flow file " << flowfile;
    return false;
  }

  // Analyze flow graph.
  flow.Analyze(library);

  // Generate code for flow.
  return Compile(flow, library);
}

void Network::ComputeLiveRanges() {
  // Check that the network is not empty.
  if (steps_.empty()) return;

  // All inputs and outputs from the network must be alive before and after the
  // computation.
  for (Tensor *t : parameters_) {
    t->first_ = t->in_ ? 0 : -1;
    t->last_ = t->out_ ? steps_.size() - 1 : -1;
  }

  // Find first and last use of each variable.
  for (int i = 0; i < steps_.size(); ++i) {
    Step *step = steps_[i];
    for (Tensor *input : step->inputs_) {
      if (input->first_ == -1) input->first_ = i;
      if (!input->out_) input->last_ = i;
    }
    for (Tensor *output : step->outputs_) {
      if (output->first_ == -1) output->first_ = i;
      if (!output->out_) output->last_ = i;
    }
  }

  // Extend live range for all shared variables.
  for (Tensor *t : parameters_) {
    if (t->shared_ != nullptr) {
      if (t->first_ < t->shared_->first_) t->shared_->first_ = t->first_;
      if (t->last_ > t->shared_->last_) t->shared_->last_ = t->last_;
    }
  }
}

char *Network::AllocateTensor(Tensor *tensor) {
  // Determine alignment for tensor.
  int alignment = tensor->byte_alignment_;
  if (alignment < kMinDataAlignment) alignment = kMinDataAlignment;
  if (alignment < jit::CPU::CacheLineSize()) {
    alignment = jit::CPU::CacheLineSize();
  }

  // Allocate memory for tensor.
  char *data = AllocateMemory(tensor->size_, alignment);
  memset(data, 0, tensor->size_);

  // Copy data.
  if (tensor->rank() == 0 || tensor->rank() == 1) {
    // Vectors and scalars can just be copied regardless of alignment and
    // order.
    memcpy(data, tensor->data_, tensor->size_);
  } else if (tensor->rank() == 2) {
    // Copy matrix one element at a time.
    const char *src = tensor->data_;
    int element_size = tensor->element_size();
    for (int r = 0; r < tensor->dim(0); ++r) {
      for (int c = 0; c < tensor->dim(1); ++c) {
        memcpy(data + tensor->offset(r, c), src, element_size);
        src += element_size;
      }
    }
  } else if (tensor->rank() == 3) {
    const char *src = tensor->data_;
    int element_size = tensor->element_size();
    for (int r = 0; r < tensor->dim(0); ++r) {
      for (int c = 0; c < tensor->dim(1); ++c) {
        for (int k = 0; k < tensor->dim(2); ++k) {
          memcpy(data + tensor->offset(r, c, k), src, element_size);
          src += element_size;
        }
      }
    }
  } else if (tensor->rank() == 4) {
    const char *src = tensor->data_;
    int element_size = tensor->element_size();
    for (int r = 0; r < tensor->dim(0); ++r) {
      for (int c = 0; c < tensor->dim(1); ++c) {
        for (int k = 0; k < tensor->dim(2); ++k) {
          for (int l = 0; l < tensor->dim(3); ++l) {
            memcpy(data + tensor->offset(r, c, k, l), src, element_size);
            src += element_size;
          }
        }
      }
    }
  } else {
    LOG(ERROR) << tensor->rank() << "D tensor not supported: "
               << tensor->name();
    return nullptr;
  }

  return data;
}

Cell *Network::GetCell(const string &name) const {
  for (Cell *cell : cells_) {
    if (cell->name() == name) return cell;
  }
  return nullptr;
}

Connector *Network::GetConnector(const string &name) const {
  for (Connector *connector : connectors_) {
    if (connector->name() == name) return connector;
  }
  return nullptr;
}

Tensor *Cell::GetParameter(const string &name) const {
  return network_->GetParameter(name);
}

void Cell::WriteCodeToFile(const string &filename) const {
  CHECK(File::WriteContents(filename, code_.begin(), code_.size()));
}

static bool CompareOffset(Tensor *t1, Tensor *t2) {
  return t1->offset() < t2->offset();
}

static bool Contains(const std::vector<Tensor *> &v, Tensor *t) {
  return std::find(v.begin(), v.end(), t) != v.end();
}

string Cell::ToString() const {
  string str;
  StringAppendF(&str, "cell %s {  // size %lu\n", name_.c_str(), instance_size_);

  // Output instance data fields.
  std::vector<Tensor *> fields;
  for (Tensor *t : network_->parameters()) {
    if (t->cell() == this) fields.push_back(t);
  }
  std::sort(fields.begin(), fields.end(), CompareOffset);

  int prev_offset = -1;
  for (Tensor *t : fields) {
    if (t->placement() & HOST) {
      str.append("  ");
      if (t->offset() == prev_offset) {
        str.append("  union ");
      } else {
        if (t->in()) str.append("input ");
        if (t->out()) str.append("output ");
        str.append("var ");
      }
      StringAppendF(&str, "%s: %s  // offset %lu size %lu\n",
                    t->name().c_str(),
                    t->TypeString().c_str(),
                    t->offset(),
                    t->space());
      prev_offset = t->offset();
    }
  }

  prev_offset = -1;
  for (Tensor *t : fields) {
    if (t->placement() & DEVICE) {
      if (t->device_offset() == prev_offset) {
        str.append("  union ");
      } else {
        if (t->in()) str.append("input ");
        if (t->out()) str.append("output ");
        str.append("device var ");
      }
      StringAppendF(&str, "%s: %s  // offset %lu size %lu\n",
                    t->name().c_str(),
                    t->TypeString().c_str(),
                    t->device_offset(),
                    t->space());
      prev_offset = t->device_offset();
    }
  }

  // Output constants used by cell.
  std::vector<Tensor *> constants;
  for (Step *step : steps_) {
    for (Tensor *input : step->inputs()) {
      if (input->IsConstant() && !Contains(constants, input)) {
        constants.push_back(input);
      }
    }
  }
  if (!constants.empty()) {
    str.append("\n");
    for (Tensor *t : constants) {
      str.append("  ");
      if (t->placement() != HOST) {
        str.append(placename[t->placement()]);
        str.append(" ");
      }
      StringAppendF(&str, "const %s: %s   // size %lu\n",
                    t->name().c_str(),
                    t->TypeString().c_str(),
                    t->size());
    }
  }

  // Output cell steps.
  if (!steps_.empty()) {
    str.append("\n");
    for (Step *step : steps_) {
      if (step->noop()) continue;
      str.append("  ");

      if (!step->outputs().empty()) {
        bool first = true;
        for (Tensor *output : step->outputs()) {
          if (!first) str.append(", ");
          str.append(output->name());
          first = false;
        }
        str.append(" = ");
      }

      str.append(step->kernel()->Name());

      str.append("(");
      bool first = true;
      for (Tensor *input : step->inputs()) {
        if (!first) str.append(", ");
        str.append(input->name());
        first = false;
      }
      str.append(")");

      if (step->placement() & DEVICE) str.append(" on device");

      str.append("\n");
    }
  }

  str.append("}\n");
  return str;
}

static bool Always(Step *step) { return true; }

CustomKernel::CustomKernel(const string &op, const string &name, void *func,
                           int indegree, int outdegree)
    : op_(op), name_(name), func_(func), criterion_(Always) {
  inputs_.resize(indegree);
  outputs_.resize(outdegree);
}

CustomKernel &CustomKernel::Input(int index, Type type, int rank) {
  CHECK_GE(index, 0);
  CHECK_LT(index, inputs_.size());
  inputs_[index].type = type;
  inputs_[index].rank = rank;
  return *this;
}

CustomKernel &CustomKernel::Output(int index, Type type, int rank) {
  CHECK_GE(index, 0);
  CHECK_LT(index, outputs_.size());
  outputs_[index].type = type;
  outputs_[index].rank = rank;
  return *this;
}

CustomKernel &CustomKernel::Select(Criterion criterion) {
  criterion_ = criterion;
  return *this;
}

string CustomKernel::Name() {
  return name_;
}

string CustomKernel::Operation() {
  return op_;
}

bool CustomKernel::Supports(Step *step) {
  // Check that number of inputs and outputs matches.
  if (step->indegree() != inputs_.size()) return false;
  if (step->outdegree() != outputs_.size()) return false;

  // Check input type and rank constraints.
  for (int i = 0; i < inputs_.size(); ++i) {
    const Param &p = inputs_[i];
    if (p.type != DT_INVALID && p.type != step->input(i)->type()) return false;
    if (p.rank != -1 && p.rank != step->input(i)->rank()) return false;
  }

  // Check output type and rank constraints.
  for (int i = 0; i < outputs_.size(); ++i) {
    const Param &p = outputs_[i];
    if (p.type != DT_INVALID && p.type != step->output(i)->type()) return false;
    if (p.rank != -1 && p.rank != step->output(i)->rank()) return false;
  }

  // Check custom selection criterion.
  if (!criterion_(step)) return false;

  return true;
}

void CustomKernel::Generate(Step *step, MacroAssembler *masm) {
  using namespace jit;
  CHECK_EQ(sizeof(TensorData), sizeof(void *) * 2);

  // Allocate space on stack for tensor data objects.
  int args = step->indegree() + step->outdegree();
  __ subq(rsp, Immediate(args * sizeof(TensorData)));

  // Build tensor data structures on stack and set up arguments to kernel
  // function.
  Register tmp = masm->rr().alloc_temp();
  int offset = 0;
  int argnum = 1;
  for (int i = 0; i < step->inputs().size(); ++i) {
    // Build input tensor data structure on stack.
    __ LoadTensorAddress(tmp, step->input(i));
    __ movq(Operand(rsp, offset), tmp);
    __ movp(tmp, step->input(i));
    __ movq(Operand(rsp, offset + sizeof(void *)), tmp);

    // Put address of input tensor data structure into argument register.
    __ leaq(masm->rr().arg(argnum++), Operand(rsp, offset));
    offset += sizeof(TensorData);
  }
  for (int i = 0; i < step->outputs().size(); ++i) {
    // Build output tensor data structure on stack.
    __ LoadTensorAddress(tmp, step->output(i));
    __ movq(Operand(rsp, offset), tmp);
    __ movp(tmp, step->output(i));
    __ movq(Operand(rsp, offset + sizeof(void *)), tmp);

    // Put address of output tensor data structure into argument register.
    __ leaq(masm->rr().arg(argnum++), Operand(rsp, offset));
    offset += sizeof(TensorData);
  }

  // Call kernel function.
  __ movp(tmp, func_);
  __ call(tmp);

  // Remove kernel arguments from stack.
  __ addq(rsp, Immediate(args * sizeof(TensorData)));
}

}  // namespace myelin
}  // namespace sling

