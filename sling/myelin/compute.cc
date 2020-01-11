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

#include "sling/myelin/compute.h"

#include <stdlib.h>
#include <algorithm>
#include <list>
#include <random>
#include <string>
#include <unordered_map>

#include "sling/base/logging.h"
#include "sling/base/types.h"
#include "sling/file/file.h"
#include "sling/myelin/macro-assembler.h"
#include "sling/string/printf.h"

namespace sling {
namespace myelin {

#if !defined(__x86_64__)
#error Myelin requires 64-bit x86
#endif

#define __ masm->

// Element order names.
const char *ordername[] = {
  "unspecified",
  "row-major",
  "column-major",
  "row-major preferred",
  "column-major preferred",
  "conflicting"
};

// Placement names.
const char *placename[] = {
  "nowhere", "host", "device", "host and device"
};

static Order CombinedOrder(Order a, Order b) {
  if (a == ANY_ORDER) return b;
  if (b == ANY_ORDER) return a;

  if (a == CONFLICTING_ORDER) return CONFLICTING_ORDER;
  if (b == CONFLICTING_ORDER) return CONFLICTING_ORDER;

  if (a == ROW_MAJOR) {
    if (b == COLUMN_MAJOR) return CONFLICTING_ORDER;
    return ROW_MAJOR;
  }
  if (b == ROW_MAJOR) {
    if (a == COLUMN_MAJOR) return CONFLICTING_ORDER;
    return ROW_MAJOR;
  }

  if (a == COLUMN_MAJOR) {
    if (b == ROW_MAJOR) return CONFLICTING_ORDER;
    return COLUMN_MAJOR;
  }
  if (b == COLUMN_MAJOR) {
    if (a == ROW_MAJOR) return CONFLICTING_ORDER;
    return COLUMN_MAJOR;
  }

  if (a == COLUMN_MAJOR_PREFERRED && b == COLUMN_MAJOR_PREFERRED) {
    return COLUMN_MAJOR_PREFERRED;
  } else {
    return ROW_MAJOR_PREFERRED;
  }
}

static Order FinalOrder(Order order, Order preferred) {
  switch (order) {
    case ANY_ORDER: return preferred;
    case ROW_MAJOR: return ROW_MAJOR;
    case COLUMN_MAJOR: return COLUMN_MAJOR;
    case ROW_MAJOR_PREFERRED:  return ROW_MAJOR;
    case COLUMN_MAJOR_PREFERRED: return COLUMN_MAJOR;
    case CONFLICTING_ORDER: return CONFLICTING_ORDER;
  }
  return CONFLICTING_ORDER;
}

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

  char *AllocateChannel(char *data, size_t old_size, size_t new_size,
                        size_t alignment, Placement placement) override {
    DCHECK_EQ(placement, HOST);
    char *buffer = MemAlloc(new_size, alignment);
    if (data != nullptr) {
      memcpy(buffer, data, old_size);
      MemFree(data);
    }
    return buffer;
  }

  void ClearChannel(char *data, size_t pos, size_t size,
                    Placement placement) override {
    DCHECK_EQ(placement, HOST);
    memset(data + pos, 0, size);
  }

  void FreeChannel(char *data, Placement placement) override {
    DCHECK_EQ(placement, HOST);
    MemFree(data);
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

// Linker for JIT generation of code.
class JITLinker : public Linker {
 public:
  void EndCell(Cell *cell,
               jit::CodeGenerator *generator,
               jit::Code *code,
               int data_size) override {
    // Allocate executable code object in memory.
    code->Allocate(generator);
  }
};

static JITLinker jit_linker;

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
        CHECK(var->shared_->offset_ != NOOFFSET)
            << var->name() << " " << var->shared_->name();
        var->offset_ = var->shared_->offset_;
      } else {
        CHECK(var->shared_->device_offset_ != NOOFFSET)
            << var->name() << " " << var->shared_->name();
        var->device_offset_ = var->shared_->device_offset_;
      }
      return;
    }

    // Determine size and alignment.
    size_t size = var->space();
    int align = var->ref_ ? kMinDataAlignment : var->byte_alignment_;

    // Try to find free space in the instance block.
    size_t offset = NOOFFSET;
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

    if (offset == NOOFFSET) {
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
  for (auto o : kernels_) {
    for (auto k : o.second) delete k;
  }
}

void Library::Register(Kernel *kernel) {
  VLOG(12) << "Register " << kernel->Name() << " for " << kernel->Operation();
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

void Tensor::Link(Tensor *link) {
  next_link_->prev_link_ = link->prev_link_;
  link->prev_link_->next_link_ = next_link_;
  next_link_ = link;
  link->prev_link_ = this;
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
    if (s2 == -1 || s2 == 1) continue;
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

int Tensor::ChannelElementSize() const {
  return Align(size(), byte_alignment());
}

int Tensor::AxisSize(int axis) const {
  if (axis > 0) return stride(axis - 1);
  if (dynamic_) return ChannelElementSize();
  return size_;
}

bool Tensor::SupportsOrder(Order order) {
  return CombinedOrder(order_, order) != CONFLICTING_ORDER;
}

void Tensor::RequireOrder(Order order) {
  order_ = CombinedOrder(order_, order);
}

void Tensor::SetMiniumAlignment(int alignment) {
  EnsureAlignment(&byte_alignment_, alignment);
}

bool Tensor::HasSameShape(const Tensor *other) const {
  return shape() == other->shape();
}

bool Tensor::HasDenseLayout() const {
  bool singular = true;
  for (int d = 0; d < rank(); ++d) {
    if (dim(d) != 1) singular = false;
    if (dim(d) % minalign(d) != 0 && !singular) return false;
  }
  return true;
}

Tensor *Tensor::MakeSparse(bool ref) {
  if (sparse_ == nullptr) {
    // Make a bit map tensor over the fist dimension.
    CHECK_GT(rank(), 0);
    CHECK(IsLocal());
    int bits = dim(0);

    // The size of the bitmap is rounded up to a multiple of 64 bits.
    int words = (bits + 63) / 64;

    // Allocate bitmap tensor as int64 array.
    sparse_ = new Tensor();
    sparse_->name_ = name_ + "/sparse";
    sparse_->cell_ = cell_;
    sparse_->local_ = local_;
    sparse_->type_ = DT_INT64;
    sparse_->ref_ = ref;
    sparse_->shape_.assign(words);
    sparse_->aligned_ = sparse_->shape_;
    sparse_->minalign_.assign(sizeof(int64));
    sparse_->stride_.assign(sizeof(int64));
    sparse_->in_ = sparse_->out_ = true;
    sparse_->placement_ = HOST;
    sparse_->current_placement_ = HOST;
    cell_->network()->AddTensor(sparse_);
  }
  return sparse_;
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

Tensor *Tensor::Gradient() const {
  return cell()->network()->LookupParameter(GradientVarName(name_));
}

string Tensor::TypeString() const {
  string str;
  if (ref_) str.append("&");
  str.append(TypeTraits::of(type_).name());
  if (dynamic_) str.append("<>");
  if (!shape_.scalar()) {
    str.append("[");
    str.append(shape_.ToString());
    str.append("]");
  }
  return str;
}

string Tensor::ToString(const char *data, bool deref) const {
  // Resolve references.
  if (deref) {
    if (dynamic()) {
      data = *reinterpret_cast<const char * const *>(data);
      if (data == nullptr) return "null";
      data = *reinterpret_cast<const char * const *>(data);
    } else if (ref()) {
      data = *reinterpret_cast<const char * const *>(data);
    }
  }

  // Check for shape and null.
  if (!shape().defined()) return "*";
  if (data == nullptr) return "null";

  // Get type traits for elements.
  const TypeTraits &traits = TypeTraits::of(type());

  // Output tensor as string.
  string str;
  if (rank() == 0) {
    // Scalar.
    str = traits.str(data);
  } else if (rank() == 1) {
    // Vector.
    str.append("[");
    for (int r = 0; r < dim(0); ++r) {
      if (r > 0) str.append(",");
      str.append(traits.str(data + offset(r)));
    }
    str.append("]");
  } else if (rank() == 2) {
    // Matrix.
    str.append("[");
    for (int r = 0; r < dim(0); ++r) {
      if (r > 0) str.append(",");
      str.append("[");
      for (int c = 0; c < dim(1); ++c) {
        if (c > 0) str.append(",");
        str.append(traits.str(data + offset(r, c)));
      }
      str.append("]");
    }
    str.append("]");
  } else {
    str = "<<" + std::to_string(rank()) + "D tensor>>";
  }

  return str;
}

Channel::Channel(const Tensor *format) : format_(format) {
  if (format != nullptr) {
    // Align the element size to the byte alignment of the format tensor to
    // ensure proper alignment of the elements in the channel array.
    DCHECK(format->order() == ROW_MAJOR) << format->name();
    DCHECK_GE(format->rank(), 1) << format->name();
    DCHECK_EQ(format->dim(0), 1) << format->name();
    element_size_ = format->ChannelElementSize();

    // Channel are aligned to the element alignment and cache lines.
    EnsureAlignment(&alignment_, format->byte_alignment());
    EnsureAlignment(&alignment_, jit::CPU::CacheLineSize());
  }
}

Channel::~Channel() {
  if (data_ != nullptr) {
    runtime()->FreeChannel(data_, placement());
  }
}

void Channel::resize(size_t n) {
  // Allocate more space if needed.
  if (n > capacity_) {
    size_t cap = capacity_ * 2;
    if (cap < n) cap = n;
    if (cap < 8) cap = 8;
    reserve(cap);
  }

  // Clear new elements.
  if (n > size_) {
    size_t pos = size_ * element_size_;
    size_t bytes = (n - size_) * element_size_;
    runtime()->ClearChannel(data_, pos, bytes, placement());
  }

  // Change size.
  size_ = n;
}

void Channel::reset(size_t n) {
  // Allocate more space if needed.
  if (n > capacity_) {
    size_t cap = capacity_ * 2;
    if (cap < n) cap = n;
    if (cap < 8) cap = 8;
    reserve(cap);
  }

  // Clear all elements.
  runtime()->ClearChannel(data_, 0, n * element_size_, placement());

  // Change size.
  size_ = n;
}

void Channel::reserve(size_t n) {
  // Never remove any existing elements.
  if (n < size_) return;
  if (n == capacity_) return;

  // Allocate or reallocate data buffer.
  data_ = runtime()->AllocateChannel(data_,
                                     size_ * element_size_,
                                     n * element_size_,
                                     alignment_,
                                     placement());

  // Change capacity.
  capacity_ = n;
}

void Channel::zero(size_t n) {
  runtime()->ClearChannel(data_, n * element_size_, element_size_, placement());
}

string Channel::ToString() const {
  string str;
  for (int i = 0; i < size_; ++i) {
    str.append(std::to_string(i));
    str.append(": ");
    char *p = at(i);
    char *buffer = nullptr;
    if (placement() & DEVICE) {
      p = buffer = runtime()->FetchDataFromDevice(
        reinterpret_cast<DevicePtr>(p), element_size_);
    }
    str.append(format_->ToString(p, false));
    str.append("\n");
    free(buffer);
  }
  return str;
}

ProfileSummary::ProfileSummary(Cell *cell) : cell_(cell) {
  if (cell->profile()) {
    int size = cell->profile()->elements();
    data_ = new int64[size];
    for (int i = 0; i < size; ++i) data_[i] = 0;
  }
}

ProfileSummary::~ProfileSummary() {
  delete [] data_;
}

Instance::Instance(const Cell *cell) : cell_(cell) {
  if (cell_ != nullptr) {
    cell_->runtime()->AllocateInstance(this);
  } else {
    data_ = nullptr;
  }
}

Instance::~Instance() {
  if (cell_ != nullptr) {
    cell_->runtime()->FreeInstance(this);
  }
}

void Instance::Clear() {
  cell_->runtime()->ClearInstance(this);
}

string Instance::ToString(const Tensor *param) const {
  // Locate parameter in instance.
  char *p;
  char *buffer = nullptr;
  if (param->placement() == DEVICE) {
    p = buffer = runtime()->FetchTensorFromDevice(this, param);
  } else {
    p = data_ + param->offset();
  }

  // Dereference reference tensors.
  if (param->ref() && p != nullptr) {
    p = *reinterpret_cast<char **>(p);
    if (buffer) {
      free(buffer);
      buffer = nullptr;
    }
    if (param->ref_placement() == DEVICE && p != nullptr) {
      // Fetch referenced data from device.
      DevicePtr devptr = reinterpret_cast<DevicePtr>(p);
      p = buffer = runtime()->FetchDataFromDevice(devptr, param->size());
    }
  }

  // Convert tensor to string.
  string str = param->ToString(p, false);
  if (buffer) free(buffer);
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

InstanceArray::InstanceArray(Cell *cell)
  : cell_(cell), begin_(nullptr), end_(nullptr), limit_(nullptr) {}

InstanceArray::~InstanceArray() {
  // Destruct all elements.
  for (Instance *d = begin_; d < limit_; ++d) d->~Instance();

  // Free array.
  free(begin_);
}

void InstanceArray::Resize(size_t size) {
  int cap = capacity();
  if (size < cap) {
    end_ = begin_ + size;
  } else if (size > cap) {
    // This awkward way of assigning the new data buffer to begin_ is needed to
    // avoid getting a GCC 8+ class-memaccess warning.
    size_t bytes = size * sizeof(Instance);
    void **data = reinterpret_cast<void **>(&begin_);
    *data = realloc(*data, bytes);
    end_ = begin_ + cap;
    limit_ = begin_ + size;
    while (end_ < limit_) new (end_++) Instance(cell_);
  }
}

void InstanceArray::Clear() {
  for (Instance *d = begin_; d < limit_; ++d) d->~Instance();
  free(begin_);
  begin_ = end_ = limit_ = nullptr;
}

void Step::SetRegisterUsage(int regs) {
  if (cell_ != nullptr && cell_->register_usage_ < regs) {
    cell_->register_usage_ = regs;
  }
}

void Step::SetPreservedRegisterUsage(int regs) {
  // There are nine caller-saved registers.
  SetRegisterUsage(9 + regs);
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
      if (t->constant()) return false;
      if (t->consumers().size() != 1) return false;
      if (t->out()) return false;
    }
    if (t->ref() != out->ref()) return false;
    if (t->dynamic() != out->dynamic()) return false;
    in = t;
    t = t->shared();
  }

  // Check if output can be shared.
  if (out->shared()) return false;
  if (out->ref()) {
    if (preserved) {
      if (out->out() && in->in()) return false;
    } else {
      if (out->out() || in->in()) return false;
    }
  }

  // Share input and output.
  out->set_shared(in);
  if (out->shape() == in->shape()) out->Link(in);

  return true;
}

bool Step::NeedsSynchronization() {
  // Only steps running on the host need synchronization.
  if (placement() != HOST) return false;

  // Only steps running in the main task need synchronization.
  if (task_index_ != -1) return false;

  // Check if any of the inputs have been produced on the device.
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

Tensor *Step::GetPrototype() const {
  Tensor *prototype = nullptr;
  for (Tensor *output : outputs_) {
    if (prototype == nullptr || output->elements() > prototype->elements()) {
      prototype = output;
    }
  }
  if (prototype == nullptr || prototype->rank() == 0) {
    for (Tensor *input : inputs_) {
      if (prototype == nullptr || input->elements() > prototype->elements()) {
        prototype = input;
      }
    }
  }
  return prototype;
}

string Step::Signature() const {
  string str;
  if (!outputs_.empty()) {
    bool first = true;
    for (Tensor *output : outputs_) {
      if (!first) str.append(",");
      str.append(output->TypeString());
      first = false;
    }
    str.append("=");
  }
  str.append(type_);
  str.append("(");
  bool first = true;
  for (Tensor *input : inputs_) {
    if (!first) str.append(",");
    str.append(input->TypeString());
    first = false;
  }
  str.append(")");
  return str;
}

Network::Network() {
  runtime_ = &default_runtime;
  linker_ = &jit_linker;
}

Network::~Network() {
  for (auto *r : resources_) delete r;
  for (auto *m : memory_) MemFree(m);
  for (auto *t : parameters_) delete t;
  for (auto *t : globals_) {
    if (t->shared() == nullptr) {
      if (t->device_data_ != DEVICE_NULL) {
        runtime_->RemoveTensorFromDevice(t);
      }
      delete t;
    }
  }
  for (auto *c : cells_) delete c;
  for (auto *s : steps_) delete s;
}

// Orthogonalize a set of vectors stored as the columns of matrix A (m x n)
// using the Gram-Schmidt process.
static void OrthogonalizeColumns(float *A, int m, int n) {
  // Orthogonalize one column vector at a time.
  float *aj, *ak;
  for (int j = 0; j < n; ++j) {
    // To orthogonalize the vector in column j with respect to the previous
    // vectors, subtract from it its projection onto each of the previous
    // vectors.
    for (int k = 0; k < j; ++k) {
      // Compute dot product r = A_k * A_j.
      float r = 0.0;
      ak = A + k;
      aj = A + j;
      for (int i = 0; i < m; ++i, ak += n, aj += n) r += *ak * *aj;

      // Update A_j -= r * A_k.
      ak = A + k;
      aj = A + j;
      for (int i = 0; i < m; ++i, ak += n, aj += n) *aj -= r * *ak;
    }

    // Normalize A_j.
    aj = A + j;
    float sum = 0.0;
    for (int i = 0; i < m; ++i, aj += n) sum += *aj * *aj;
    float scaler = 1.0/ sqrt(sum);
    aj = A + j;
    for (int i = 0; i < m; ++i, aj += n) *aj *= scaler;
  }
}

// Orthogonalize a set of vectors stored as the rows of matrix A (m x n)
// using the Gram-Schmidt process.
static void OrthogonalizeRows(float *A, int m, int n) {
  // Orthogonalize one row vector at a time.
  float *aj, *ak;
  for (int j = 0; j < m; ++j) {
    // To orthogonalize the vector in row j with respect to the previous
    // vectors, subtract from it its projection onto each of the previous
    // vectors.
    aj = A + j * n;
    for (int k = 0; k < j; ++k) {
      // Compute dot product r = A_k * A_j.
      float r = 0.0;
      ak = A + k * n;
      for (int i = 0; i < n; ++i) r += ak[i] * aj[i];

      // Update A_j -= r * A_k.
      for (int i = 0; i < n; ++i) aj[i] -= r * ak[i];
    }

    // Normalize A_j.
    float sum = 0.0;
    for (int i = 0; i < n; ++i) sum += aj[i] * aj[i];
    float scaler = 1.0/ sqrt(sum);
    for (int i = 0; i < n; ++i) aj[i] *= scaler;
  }
}

void Network::InitModelParameters(int64 seed) {
  // Initialize random generator.
  std::mt19937_64 prng;
  prng.seed(seed);
  std::normal_distribution<float> normal(0, 1.0);
  std::uniform_real_distribution<float> uniform(-1.0, 1.0);

  // Initialize model parameters.
  for (auto *tensor : globals_) {
    if (tensor->type() != DT_FLOAT) continue;
    if (tensor->data() == nullptr) continue;

    float dim = tensor->elements();
    float scale = 1.0 / sqrt(dim);

    if (tensor->HasStandardLayout()) {
      float *data = reinterpret_cast<float *>(tensor->data());
      switch (tensor->init_) {
        case Flow::Variable::INIT_ZERO:
          // Variables are already zero-initialized.
          break;
        case Flow::Variable::INIT_UNIFORM: {
          for (int i = 0; i < tensor->elements(); ++i) {
            data[i] = uniform(prng) * scale;
          }
          break;
        }
        case Flow::Variable::INIT_NORMAL: {
          for (int i = 0; i < tensor->elements(); ++i) {
            data[i] = normal(prng) * scale;
          }
          break;
        }
        case Flow::Variable::INIT_ORTHO: {
          for (int i = 0; i < tensor->elements(); ++i) {
            data[i] = normal(prng);
          }

          if (tensor->rank() >= 2) {
            int m = tensor->dim(0);
            int n = tensor->elements() / m;
            if (n > m) {
              OrthogonalizeRows(data, m, n);
            } else {
              OrthogonalizeColumns(data, m, n);
            }
          }
          break;
        }
        default:
          LOG(WARNING) << "Unknown initialization for " << tensor->name();
      }
    } else {
      switch (tensor->init_) {
        case Flow::Variable::INIT_ZERO:
          // Variables are already zero-initialized.
          break;
        case Flow::Variable::INIT_UNIFORM: {
          for (int i = 0; i < tensor->elements(); ++i) {
            size_t offset = tensor->LinearOffset(i);
            float *p = reinterpret_cast<float *>(tensor->data() + offset);
            *p = uniform(prng) * scale;
          }
          break;
        }
        case Flow::Variable::INIT_NORMAL: {
          for (int i = 0; i < tensor->elements(); ++i) {
            size_t offset = tensor->LinearOffset(i);
            float *p = reinterpret_cast<float *>(tensor->data() + offset);
            *p = normal(prng) * scale;
          }
          break;
        }
        case Flow::Variable::INIT_ORTHO: {
          LOG(WARNING) << "Cannot initialize tensor with non-standard layout "
                       << "with orthogonal vectors: " << tensor->name();
          for (int i = 0; i < tensor->elements(); ++i) {
            size_t offset = tensor->LinearOffset(i);
            float *p = reinterpret_cast<float *>(tensor->data() + offset);
            *p = normal(prng) * scale;
          }
          break;
        }
        default:
          LOG(WARNING) << "Unknown initialization for " << tensor->name();
      }
    }
  }
}

void Network::SaveParameters(Flow *flow) const {
  // Find all learnable variables in flow.
  for (Flow::Variable *var : flow->vars()) {
    if (!var->learnable()) continue;

    // Find tensor for variable.
    Tensor *tensor = LookupParameter(var->name);
    if (tensor == nullptr) continue;

    // If tensor data has standard layout we can copy the data directly.
    // Otherwise, tensor data is copied element-by-element.
    if (tensor->HasStandardLayout()) {
      // Copy directly.
      var->size = tensor->size();
      var->data = flow->AllocateMemory(var->size);
      memcpy(var->data, tensor->data(), var->size);
    } else {
      // Allocate data.
      int elements = tensor->shape().elements();
      int element_size = tensor->element_size();
      var->size = elements * element_size;
      char *dst = flow->AllocateMemory(var->size);
      char *src = tensor->data();
      var->data = dst;

      // Copy elements one at a time.
      for (int i = 0; i < elements; ++i) {
        size_t offset = tensor->LinearOffset(i);
        memcpy(dst, src + offset, element_size);
        dst += element_size;
      }
    }
    var->set_learnable(false);
  }
}

void Network::LoadParameters(const Flow &flow) {
  // Find all learnable variables in flow.
  for (const Flow::Variable *var : flow.vars()) {
    // Find tensor for variable.
    Tensor *tensor = LookupParameter(var->name);
    if (tensor == nullptr) continue;
    if (!tensor->learnable()) continue;

    // Check that type and shape match.
    if (tensor->type() != var->type || tensor->shape() != var->shape) {
      LOG(WARNING) << "Tensor " << tensor->name() << " type mismatch: "
                   << tensor->TypeString() << " vs " << var->TypeString();
      continue;
    }

    // If tensor data has standard layout we can copy the data directly.
    // Otherwise, tensor data is copied element-by-element.
    if (tensor->HasStandardLayout()) {
      // Copy directly.
      memcpy(tensor->data(), var->data, var->size);
    } else {
      // Allocate data.
      int elements = tensor->shape().elements();
      int element_size = tensor->element_size();
      char *dst = tensor->data();
      char *src = var->data;

      // Copy elements one at a time.
      for (int i = 0; i < elements; ++i) {
        size_t offset = tensor->LinearOffset(i);
        memcpy(dst + offset, src, element_size);
        src += element_size;
      }
    }
  }
}

void Network::AddTensor(Tensor *tensor) {
  names_[tensor->name()] = tensor;
  if (tensor->IsLocal()) {
    parameters_.push_back(tensor);
  } else {
    globals_.push_back(tensor);
  }
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
    if (va->in() && !vb->in()) return true;
    if (vb->in() && !va->in()) return false;
  }
  return a.first < b.first;
}

bool Network::Compile(const Flow &flow, const Library &library) {
  // Fetch information about the CPU we are running on.
  jit::CPU::Probe();

  // Check compiler options.
  DCHECK(flow.IsConsistent());
  if (!options_.aot && options_.pic) {
    LOG(ERROR) << "Position-independent code generation not supported for JIT";
    return false;
  }

  // Create tensors for all the variables (parameters and constants).
  std::unordered_map<Flow::Variable *, Tensor *> varmap;
  for (Flow::Variable *var : flow.vars()) {
    // Allocate new tensor.
    Tensor *tensor = new Tensor();
    varmap[var] = tensor;
    tensor->constant_ = var->constant();
    tensor->local_ = !var->global();
    tensor->init_ = var->init;
    tensor->name_ = var->name;
    for (const string &alias : var->aliases) {
      names_[alias] = tensor;
    }
    tensor->type_ = var->type;
    tensor->ref_ = var->ref();
    tensor->dynamic_ = var->dynamic();
    tensor->shape_ = var->shape;
    tensor->aligned_ = var->shape;
    tensor->minalign_.fill(var->rank(), 1);
    tensor->stride_.fill(var->rank(), 0);
    tensor->byte_alignment_ = TypeTraits::of(var->type).size();

    // Add tensor to network.
    AddTensor(tensor);

    // Input variables are initially placed in host memory.
    tensor->in_ = var->in();
    if (var->in()) {
      tensor->current_placement_ = HOST;
      tensor->placement_ = HOST;
      if (tensor->local_) {
        tensor->order_ = options_.parameter_element_order;
      }
    }

    // Output variables must be available on the host after the computation.
    tensor->out_ = var->out();
    if (var->out()) {
      tensor->placement_ = HOST;
      if (tensor->local_) {
        tensor->order_ = options_.parameter_element_order;
      }
    }

    // Set order constraint from flow.
    if (var->is(Flow::Variable::ROW)) {
      tensor->order_ = ROW_MAJOR;
    } else if (var->is(Flow::Variable::COL)) {
      tensor->order_ = COLUMN_MAJOR;
    }

    // Initialize constant tensors with data from the flow variable so they can
    // be used before tensor data allocation.
    if (tensor->constant()) {
      size_t stride = TypeTraits::of(tensor->type()).size();
      for (int d = tensor->rank() - 1; d >= 0; --d) {
        tensor->stride_.set(d, stride);
        stride *= tensor->shape_.dim(d);
      }

      if (var->size < stride) {
        LOG(ERROR) << "Invalid data size for variable " << var->name << " "
                   << var->TypeString() << ", "
                   << var->size << " bytes, " << stride << " expected";
        return false;
      } else if (var->size > stride) {
        LOG(WARNING) << "Excess data for variable " << var->name << " "
                   << var->TypeString() << ", "
                   << var->size << " bytes, " << stride << " expected";
      }

      tensor->data_ = var->data;
      tensor->size_ = stride;
    }
  }

  // Link tensors in each connector.
  for (Flow::Connector *cnx : flow.cnxs()) {
    if (cnx->links.empty()) continue;
    Tensor *t = varmap[cnx->links[0]];
    for (int i = 1; i < cnx->links.size(); ++i) {
      Tensor *l = varmap[cnx->links[i]];
      l->Link(t);
    }
  }

  // Let linker configure network before compilation.
  linker_->BeginNetwork(this);

  // Create cells for all functions.
  std::unordered_map<Flow::Function *, Cell *> cells;
  for (Flow::Function *func : flow.funcs()) {
    // Set or create cell for step.
    Cell *cell = new Cell();
    cell->network_ = this;
    cell->name_ = func->name;
    cells_.push_back(cell);
    cells[func] = cell;

    // Add unused input variables to cell. Unused input variables still need to
    // be allocated in the instance block. For example, identity functions does
    // not have any operations, but the inputs and outputs are aliased.
    for (Flow::Variable *v : func->unused) {
      varmap[v]->cell_ = cell;
    }
  }

  // Create steps for all the operations.
  for (Flow::Function *func : flow.funcs()) {
    for (Flow::Operation *op : func->ops) {
      // Create step for operation.
      Step *step = new Step();
      steps_.push_back(step);
      step->name_ = op->name;
      step->type_ = op->type;
      step->CopyAttrsFrom(*op);

      // Set cell for step.
      Cell *cell = cells[op->func];
      cell->steps_.push_back(step);
      step->cell_ = cell;

      // Add inputs to step.
      for (Flow::Variable *input : op->inputs) {
        Tensor *tensor = varmap[input];
        CHECK(tensor != nullptr);
        step->inputs_.push_back(tensor);
        tensor->consumers_.push_back(step);

        // Assign input parameter to cell.
        if (step->cell_ != nullptr && tensor->IsLocal()) {
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
        Tensor *tensor = varmap[output];
        CHECK(tensor != nullptr);
        step->outputs_.push_back(tensor);
        CHECK(tensor->producer_ == nullptr);
        tensor->producer_ = step;

        // Assign output parameter to cell.
        if (step->cell_ != nullptr && tensor->IsLocal()) {
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
    }
  }

  // Find kernels for implementing each step.
  for (Step *step : steps_) {
    auto &kernels = library.Lookup(step->type());
    for (int k = kernels.size() - 1; k >= 0; --k) {
      Kernel *kernel = kernels[k];
      if (kernel->Supports(step, options_)) {
        // Check that kernel location is compatible with task placement.
        bool compatible = true;
        if (step->task_index_ != -1) {
          auto &task = step->cell()->tasks_[step->task_index_];
          Placement location = kernel->Location();
          if (task.placement == NOWHERE) {
            // Task has not been placed yet. Use the kernel location for
            // placement.
            task.placement = location;
          } else if (task.placement != location && location != NOWHERE) {
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
      LOG(ERROR) << "No kernel supports step " << step->name() << ": "
                 << step->Signature();
      return false;
    }
    VLOG(3) << "Step " << step->name() << " implemented by "
            << step->kernel_->Name();

    // Let kernel adjust the input and output data alignment requirements.
    step->kernel_->Adjust(step, options_);
  }

  // Add tensor for profiling.
  if (options_.profiling) {
    for (Cell *cell : cells_) {
      // Allocate tensor for storing profiling information. The tensor is an
      // int64 vector with the following layout:
      //   struct TaskTiming {
      //     int64 start;
      //     int64 wait;
      //   };
      //   struct CellTiming {
      //     int64 invocations;
      //     int64 overhead;
      //     int64 steptime[#steps];
      //     TaskTiming tasktime[#tasks];
      //   };
      size_t size = 2 + cell->steps_.size() + 2 * cell->tasks_.size();
      Tensor *profile = new Tensor();
      profile->name_ = "timing/" + cell->name_;
      profile->cell_ = cell;
      profile->type_ = DT_INT64;
      profile->shape_.assign(size);
      profile->size_ = profile->space_ = size * sizeof(int64);
      profile->ref_ = options_.ref_profiler();
      profile->aligned_ = profile->shape_;
      profile->minalign_.assign(sizeof(int64));
      profile->stride_.assign(sizeof(int64));
      profile->placement_ = HOST;
      profile->current_placement_ = HOST;
      profile->in_ = true;
      profile->out_ = true;
      cell->profile_ = profile;
      AddTensor(profile);
    }
  }

  // Collect all tensors (global and local).
  std::vector<Tensor *> tensors;
  for (Tensor *t : globals_) tensors.push_back(t);
  for (Tensor *t : parameters_) tensors.push_back(t);

  // Propagate constraints between linked tensors.
  bool again = true;
  while (again) {
    // Keep propagating alignment constraints until there are no more
    // constraints to propagate.
    again = false;
    for (Tensor *t : tensors) {
      Tensor *l = t->next_link_;
      if (l == t) continue;

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
      t->require_dense_ |= l->require_dense_;
      l->require_dense_ |= t->require_dense_;

      // Propagate order constraints.
      if (t->order_ != l->order_) {
        Order c = CombinedOrder(t->order_, l->order_);
        if (t->order_ != c || l->order_ != c) {
          t->order_ = c;
          l->order_ = c;
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
  for (Tensor *tensor : tensors) {
    // Determine final element order.
    tensor->order_ = FinalOrder(tensor->order_, ROW_MAJOR);
    if (tensor->order_ == CONFLICTING_ORDER) {
      LOG(ERROR) << "Conflicting order requirements for " << tensor->name();
      return false;
    }

    // Check for dense encoding conflicts.
    if (tensor->require_dense_ && !tensor->HasDenseLayout()) {
      LOG(ERROR) << "Conflicting dense encoding requirements for "
                 << tensor->name()
                 << " shape " << tensor->shape().ToString()
                 << " align " << tensor->minalign().ToString();
      return false;
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

    // Set tensor size.
    tensor->size_ = size;
    tensor->space_ = tensor->ref() || tensor->dynamic() ? sizeof(void *) : size;

    // Determine placement for tensor based on producer and consumer locations.
    if (tensor->producer_ != nullptr) {
      // Tensor is available in the place it is produced.
      tensor->AddPlace(tensor->producer_->placement());
      if (tensor->ref()) tensor->AddRefPlace(tensor->producer_->placement());
    }
    for (Step *consumer : tensor->consumers_) {
      // Tensor must be made available in the places it is consumed.
      tensor->AddPlace(consumer->placement());
      if (tensor->ref()) tensor->AddRefPlace(consumer->placement());
    }

    VLOG(5) << "Tensor " << tensor->name_ << ": " << tensor->TypeString()
            << " align " << tensor->minalign_.ToString()
            << ":" << tensor->byte_alignment_
            << " aligned " << tensor->aligned_.ToString()
            << " size " << tensor->space_
            << " stride " << tensor->stride_.ToString()
            << " order " << ordername[tensor->order_]
            << " on " << placename[tensor->placement_];
  }

  // Propagate size and alignment for shared tensors.
  for (Tensor *tensor : tensors) {
    Tensor *next = tensor->shared_;
    while (next != nullptr) {
      if (next->size_ < tensor->size_) {
        next->size_ = tensor->size_;
      }
      if (next->byte_alignment_ < tensor->byte_alignment_) {
        next->byte_alignment_ = tensor->byte_alignment_;
      }
      if (next->placement_ != tensor->placement_) {
        next->AddPlace(tensor->placement_);
        tensor->AddPlace(next->placement_);
      }
      next = next->shared_;
    }
  }

  // Move all variables that are shared with a global to the global pool.
  for (auto it = parameters_.begin(); it != parameters_.end();) {
    // Check if tensor is shared with a constant.
    Tensor *t = *it;
    if (t->shared_ != nullptr && t->shared_->data_ != nullptr) {
      // Move variable to global pool.
      VLOG(5) << "Convert " << t->name() << " to global";
      it = parameters_.erase(it);
      globals_.push_back(t);
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

  // Allocate globals.
  for (Tensor *tensor : globals_) {
    if (tensor->shared_ != nullptr) {
      // Shared tensor. Copy reference to data.
      tensor->data_ = tensor->shared_->data_;
      tensor->AddNewPlace(HOST);
      if (tensor->placement_ & DEVICE) {
        tensor->device_data_ = tensor->shared_->device_data_;
        tensor->AddNewPlace(DEVICE);
      }
    } else {
      // Allocate aligned tensor.
      char *data = AllocateTensor(tensor);
      if (data == nullptr) return false;

      // Add tensor data to linker.
      linker_->AddData(tensor);

      // Copy constant to device if needed.
      if (tensor->placement_ & DEVICE) {
        VLOG(5) << "Copy tensor " << tensor->name() << " to device";
        tensor->device_data_ = runtime_->CopyTensorToDevice(tensor);
        CHECK(tensor->device_data_ != DEVICE_NULL);
        tensor->AddNewPlace(DEVICE);
      }

      // Place constant on host if needed.
      if (tensor->placement_ & HOST) {
        tensor->data_ = data;
        tensor->AddNewPlace(HOST);
        memory_.push_back(data);
      } else {
        MemFree(data);
      }
    }
  }

  // Compile each cell computation.
  for (Cell *cell : cells_) {
    // Start code generation for cell.
    linker_->BeginCell(cell);

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

    // Generate profiling prologue.
    if (options_.profiling) {
      // Load global profile summary.
      if (options_.global_profiler) {
        cell->profile_summary_ = new ProfileSummary(cell);
        masm.load_extern(jit::rdi, cell->profile_summary_->data(),
                         cell->name_ + "_profile");
        masm.movq(jit::Operand(masm.instance(), cell->profile_->offset_),
                  jit::rdi);
      }

      // Increment the invocation counter.
      masm.IncrementInvocations(cell->profile()->offset());

      // Start runtime profiler.
      masm.CallInstanceFunction(runtime_->StartProfilerFunc(),
                                "myelin_start_profiler");
    }

    // Copy input variables that do not have the placement required by the
    // consumers.
    bool sync = false;
    bool main_pending = false;
    Transfers xfers;
    for (Tensor *tensor : parameters_) {
      if (tensor->cell_ != cell) continue;
      if (!tensor->in_) continue;
      if (tensor->placement_ != EVERYWHERE) continue;

      int task = tensor->ConsumerTask();
      if (tensor->current_placement_ == HOST) {
        // Copy parameter tensor from host to device.
        xfers.add_host_to_device(tensor, task);
        tensor->AddNewPlace(DEVICE);
      } else if (tensor->current_placement_ == DEVICE) {
        // Copy parameter tensor from device to host.
        xfers.add_device_to_host(tensor, task);
        tensor->AddNewPlace(HOST);
      }
      if (task == -1) {
        sync = true;
        main_pending = true;
      }
    }
    runtime_->EmitTensorTransfers(xfers, cell, &masm);

    // Profile entry overhead.
    if (options_.profiling) {
      int timing = cell->profile()->offset();
      masm.TimeStep(timing, 1 * sizeof(int64));
    }

    // Let kernels generate code for each step.
    int stepnum = 0;
    int64 flops = 0;
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
              int slot = 2 + cell->steps_.size() + tidx * 2 + 1;
              masm.TimeStep(timing, slot * sizeof(int64));
            }
          }
        }

        // Synchronize main task if needed before executing step.
        if (options_.sync_steps || (sync && step->NeedsSynchronization())) {
          VLOG(8) << "Sync main task";
          masm.WaitForMainTask();
          sync = false;
          main_pending = false;
        }

        // Generate code for step.
        auto pc = masm.pc_offset();
        VLOG(8) << "Generate " << step->name() << " @ "
                << reinterpret_cast<uint64 *>(pc)
                << " with " << step->kernel_->Name()
                << " on " << placename[step->placement()];
        linker_->BeginStep(step, pc);
        step->kernel_->Generate(step, &masm);
        if (masm.pc_offset() == pc) step->noop_ = true;
        linker_->EndStep(step, pc);
        if (step->placement() == DEVICE) main_pending = true;

        // No registers are preserved between steps, so reset register
        // allocation.
        masm.ResetRegisterUsage();

        // Copy outputs that do not have the placement required by the
        // consumers.
        Transfers xfers;
        for (Tensor *output : step->outputs_) {
          output->AddNewPlace(step->placement());
          if (output->placement_ == EVERYWHERE) {
            int task = output->ConsumerTask();
            if (output->current_placement_ == HOST) {
              // Copy output from host to device.
              xfers.add_host_to_device(output, task);
              output->AddNewPlace(DEVICE);
            } else if (output->current_placement_ == DEVICE) {
              // Copy output from device to host.
              xfers.add_device_to_host(output, task);
              output->AddNewPlace(HOST);
              if (task == -1) sync = true;
            }
          }
        }
        runtime_->EmitTensorTransfers(xfers, cell, &masm);

        // Profile step.
        if (options_.profiling && !step->noop_) {
          int timing = cell->profile()->offset();
          masm.TimeStep(timing, (stepnum + 2) * sizeof(int64));
        }
      } else {
        // Parallel step.
        int tidx = step->task_index_;
        auto &t = cell->tasks_[tidx];
        CHECK(t.state != COMPLETED) << cell->name_ << " task " << t.task;
        if (t.state == PENDING) {
          // Flush asynchronous operations.
          if (sync) {
            masm.WaitForMainTask();
            sync = false;
            main_pending = false;
          }

          // Start parallel task.
          masm.StartTask(t.offset, t.task, step->task_index_, &t.entry);
          t.state = ACTIVE;

          // Profile task start.
          if (options_.profiling) {
            int timing = cell->profile()->offset();
            int slot = 2 + cell->steps_.size() + tidx * 2;
            masm.TimeStep(timing, slot * sizeof(int64));
          }
        }

        // Update output placements.
        for (Tensor *output : step->outputs_) {
          output->AddNewPlace(step->placement());
          if (output->placement_ == EVERYWHERE) {
            if (output->current_placement_ == HOST) {
              // Set deferred copy from host to device.
              VLOG(8) << "Deferred transfer " << output->name()
                      << " from host to device";
              output->deferred_placement_ = DEVICE;
              output->AddNewPlace(DEVICE);
            } else if (output->current_placement_ == DEVICE) {
              // Set deferred copy from device to host.
              VLOG(8) << "Deferred transfer " << output->name()
                      << " from device to host";
              output->deferred_placement_ = HOST;
              output->AddNewPlace(HOST);
            }
          }
        }
      }

      // Sum complexity for cell.
      if (options_.flops_address) {
        int64 complexity = step->complexity();
        if (complexity > 0) flops += complexity;
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

    // Synchronize main task.
    if (main_pending) masm.WaitForMainTask();

    // Stop runtime profiler.
    if (options_.profiling) {
      masm.CallInstanceFunction(runtime_->StopProfilerFunc(),
                                "myelin_stop_profiler");
    }

    // Profile exit overhead.
    if (options_.profiling) {
      int timing = cell->profile()->offset();
      masm.TimeStep(timing, 1 * sizeof(int64));
    }

    // Update global FLOPs counter for performance measurements.
    if (options_.flops_address) {
      masm.UpdateCounter(options_.flops_address, flops);
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
          Transfers xfers;
          for (Tensor *output : step->outputs_) {
            if (output->deferred_placement_ == DEVICE) {
              // Copy output from host to device.
              xfers.add_host_to_device(output, task_index);
            } else if (output->deferred_placement_ == HOST) {
              // Copy output from device to host.
              xfers.add_device_to_host(output, task_index);
            }
          }
          runtime_->EmitTensorTransfers(xfers, cell, &masm);

          // Profile step.
          if (options_.profiling && !step->noop_) {
            int timing = cell->profile()->offset();
            masm.TimeStep(timing, (stepnum + 2) * sizeof(int64));
          }
        }
        stepnum++;
      }

      // Generate parallel task epilogue.
      masm.Epilogue();

      task_index++;
    }

    // Generate static data blocks.
    auto code_size = masm.pc_offset();
    masm.GenerateDataBlocks();

    // Add generated code to linker.
    linker_->EndCell(cell, &masm, &cell->code_, masm.pc_offset() - code_size);
    VLOG(5) << cell->name()
            << " entry address: " << cell->code_.entry()
            << " code size: " << cell->code_.size()
            << " data size: " << cell->instance_size();
  }

  // Notify linker that compilation of network has completed.
  linker_->EndNetwork(this);

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

void Network::Bind(Flow *flow) {
  // Bind functions to cells and operations to steps.
  for (auto *func : flow->funcs()) {
    func->cell = LookupCell(func->name);
    if (func->cell != nullptr) {
      for (auto *op : func->ops) {
        op->step = func->cell->LookupStep(op->name);
      }
    }
  }

  // Bind variables to tensors.
  for (auto *var : flow->vars()) {
    var->tensor = LookupParameter(var->name);
  }
}

void Network::ComputeLiveRanges() {
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

  // Extend live range for all shared variables and sparsity vectors.
  for (Tensor *t : parameters_) {
    if (t->shared_ != nullptr) {
      if (t->first_ < t->shared_->first_) t->shared_->first_ = t->first_;
      if (t->last_ > t->shared_->last_) t->shared_->last_ = t->last_;
    }
    if (t->sparse_ != nullptr) {
      if (t->first_ < t->sparse_->first_) t->sparse_->first_ = t->first_;
      if (t->last_ > t->sparse_->last_) t->sparse_->last_ = t->last_;
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
  char *data = MemAlloc(tensor->size_, alignment);
  memset(data, 0, tensor->size_);

  // Copy data.
  if (tensor->constant()) {
    if (tensor->HasStandardLayout()) {
      // Tensors with standard layout can be copied directly.
      memcpy(data, tensor->data_, tensor->size_);
    } else {
      // Copy tensor one element at a time.
      const char *src = tensor->data_;
      int element_size = tensor->element_size();
      if (tensor->rank() == 2) {
        for (int r = 0; r < tensor->dim(0); ++r) {
          for (int c = 0; c < tensor->dim(1); ++c) {
            memcpy(data + tensor->offset(r, c), src, element_size);
            src += element_size;
          }
        }
      } else if (tensor->rank() == 3) {
        for (int r = 0; r < tensor->dim(0); ++r) {
          for (int c = 0; c < tensor->dim(1); ++c) {
            for (int k = 0; k < tensor->dim(2); ++k) {
              memcpy(data + tensor->offset(r, c, k), src, element_size);
              src += element_size;
            }
          }
        }
      } else {
        for (int i = 0; i < tensor->elements(); ++i) {
          memcpy(data + tensor->LinearOffset(i), src, element_size);
          src += element_size;
        }
      }
    }
  }

  return data;
}

Cell *Network::LookupCell(const string &name) const {
  for (Cell *cell : cells_) {
    if (cell->name() == name) return cell;
  }
  return nullptr;
}

Cell *Network::GetCell(const string &name) const {
  Cell *cell = LookupCell(name);
  CHECK(cell != nullptr) << "Unknown cell: " << name;
  return cell;
}

Tensor *Network::LookupParameter(const string &name) const {
  auto f = names_.find(name);
  return f == names_.end() ? nullptr : f->second;
}

Tensor *Network::GetParameter(const string &name) const {
  Tensor *tensor = LookupParameter(name);
  CHECK(tensor != nullptr) << "Unknown parameter: " << name;
  return tensor;
}

Tensor *Cell::LookupParameter(const string &name) const {
  return network_->LookupParameter(name);
}

Tensor *Cell::GetParameter(const string &name) const {
  return network_->GetParameter(name);
}

Step *Cell::LookupStep(const string &name) const {
  for (Step *step : steps_) {
    if (step->name() == name) return step;
  }
  return nullptr;
}

Cell *Cell::Gradient() const {
  return network()->LookupCell(GradientFuncName(name_));
}

Tensor *Cell::Primal() const {
  return network()->LookupParameter(PrimalVarName(name_));
}

void Cell::WriteCodeToFile(const string &filename) const {
  CHECK(File::WriteContents(filename, code_.begin(), code_.size()));
}

static bool CompareOffset(Tensor *t1, Tensor *t2) {
  if (t1->offset() != t2->offset()) {
    return t1->offset() < t2->offset();
  } else {
    return t1->device_offset() < t2->device_offset();
  }
}

static bool Contains(const std::vector<Tensor *> &v, Tensor *t) {
  return std::find(v.begin(), v.end(), t) != v.end();
}

string Cell::ToString() const {
  string str;
  StringAppendF(&str, "cell %s {  // size %lu", name_.c_str(), instance_size_);
  if (device_instance_size_ > 0) {
    StringAppendF(&str, ", device size %lu", device_instance_size_);
  }
  str.append("\n");

  // Output instance data fields.
  std::vector<Tensor *> fields;
  for (Tensor *t : network_->parameters()) {
    if (t->cell() == this) fields.push_back(t);
  }
  std::sort(fields.begin(), fields.end(), CompareOffset);

  size_t prev_offset = NOOFFSET;
  bool on_device = false;
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
      StringAppendF(&str, "%s: %s  // offset %lu size %lu alignment %d",
                    t->name().c_str(),
                    t->TypeString().c_str(),
                    t->offset(),
                    t->space(), t->byte_alignment());
      StringAppendF(&str, " %s", ordername[t->order()]);
      if (t->ref_placement() != NOWHERE) {
        StringAppendF(&str, " %s ref", placename[t->ref_placement()]);
      }
      if (t->linked()) {
        StringAppendF(&str, " linked to %s", t->next_link()->name().c_str());
      }
      if (t->sparse()) {
        StringAppendF(&str, " sparsity %s", t->sparse()->name().c_str());
      }
      str.append("\n");
      prev_offset = t->offset();
    }
    if (t->placement() & DEVICE) on_device = true;
  }

  if (on_device) {
    str.append("\n");
    prev_offset = NOOFFSET;
    for (Tensor *t : fields) {
      if (t->placement() & DEVICE) {
        str.append("  ");
        if (t->device_offset() == prev_offset) {
          str.append("  union ");
        } else {
          if (t->in()) str.append("input ");
          if (t->out()) str.append("output ");
          str.append("device var ");
        }
        StringAppendF(&str, "%s: %s  // offset %lu size %lu alignment %d",
                      t->name().c_str(),
                      t->TypeString().c_str(),
                      t->device_offset(),
                      t->space(), t->byte_alignment());
        StringAppendF(&str, " %s", ordername[t->order()]);
        if (t->ref_placement() != NOWHERE) {
          StringAppendF(&str, " %s ref", placename[t->ref_placement()]);
        }
        if (t->linked()) {
          StringAppendF(&str, " linked to %s", t->next_link()->name().c_str());
        }
        str.append("\n");
        prev_offset = t->device_offset();
      }
    }
  }

  // Output globals used by cell.
  std::vector<Tensor *> globals;
  for (Step *step : steps_) {
    for (Tensor *input : step->inputs()) {
      if (input->IsGlobal() && !Contains(globals, input)) {
        globals.push_back(input);
      }
    }
    for (Tensor *output : step->outputs()) {
      if (output->IsGlobal() && !Contains(globals, output)) {
        globals.push_back(output);
      }
    }
  }
  if (!globals.empty()) {
    str.append("\n");
    for (Tensor *t : globals) {
      str.append("  ");
      if (t->placement() != HOST) {
        str.append(placename[t->placement()]);
        str.append(" ");
      }
      StringAppendF(&str, "%s %s: %s   // size %lu alignment %d",
                    t->constant() ? "const" : "global",
                    t->name().c_str(),
                    t->TypeString().c_str(),
                    t->size(), t->byte_alignment());
      StringAppendF(&str, " %s", ordername[t->order()]);
      if (t->linked()) {
        StringAppendF(&str, " linked to %s", t->next_link()->name().c_str());
      }
      str.append("\n");
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
      if (!step->variant().empty()) {
        str.append("[");
        str.append(step->variant());
        str.append("]");
      }

      str.append("(");
      bool first = true;
      for (Tensor *input : step->inputs()) {
        if (!first) str.append(", ");
        str.append(input->name());
        first = false;
      }
      str.append(")");

      string expr = step->GetAttr("expr");
      if (!expr.empty()) {
        str.append(" ");
        str.append(expr);
      }

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
    __ load_extern(tmp, step->input(i),
            StringPrintf("%s_input_%d", step->name().c_str(), i));
    __ movq(Operand(rsp, offset + sizeof(void *)), tmp);

    // Put address of input tensor data structure into argument register.
    __ leaq(masm->rr().arg(argnum++), Operand(rsp, offset));
    offset += sizeof(TensorData);
  }
  for (int i = 0; i < step->outputs().size(); ++i) {
    // Build output tensor data structure on stack.
    __ LoadTensorAddress(tmp, step->output(i));
    __ movq(Operand(rsp, offset), tmp);
    __ load_extern(tmp, step->output(i),
                   StringPrintf("%s_output_%d", step->name().c_str(), i));

    __ movq(Operand(rsp, offset + sizeof(void *)), tmp);

    // Put address of output tensor data structure into argument register.
    __ leaq(masm->rr().arg(argnum++), Operand(rsp, offset));
    offset += sizeof(TensorData);
  }

  // Call kernel function.
  __ call_extern(func_, Name());

  // Remove kernel arguments from stack.
  __ addq(rsp, Immediate(args * sizeof(TensorData)));
}

}  // namespace myelin
}  // namespace sling

