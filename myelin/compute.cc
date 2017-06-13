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
#include <unordered_map>
#include <string>

#include "base/logging.h"
#include "base/types.h"
#include "file/file.h"
#include "myelin/macro-assembler.h"

namespace sling {
namespace myelin {

#if !defined(__x86_64__)
#error Myelin requires 64-bit x86
#endif

// Combined tensor order.
static const Order combined_order[4][4] = {
  {ANY_ORDER,         ROW_MAJOR,         COLUMN_MAJOR,      CONFLICTING_ORDER},
  {ROW_MAJOR,         ROW_MAJOR,         CONFLICTING_ORDER, CONFLICTING_ORDER},
  {COLUMN_MAJOR,      CONFLICTING_ORDER, COLUMN_MAJOR,      CONFLICTING_ORDER},
  {CONFLICTING_ORDER, CONFLICTING_ORDER, CONFLICTING_ORDER, CONFLICTING_ORDER},
};

// Placement names.
const char *placename[] = {"nowhere", "host", "device", "host and device"};

static bool IsPowerOfTwo32(int value) {
  return value && !(value & (value - 1));
}

static int Align(int n, int align) {
  DCHECK(IsPowerOfTwo32(align));
  return (n + align - 1) & ~(align - 1);
}

static char *AllocateMemory(int size, int alignment) {
  DCHECK(IsPowerOfTwo32(alignment));
  DCHECK_GE(alignment, sizeof(void *));
  char *data;
  int rc = posix_memalign(reinterpret_cast<void **>(&data), alignment, size);
  CHECK_EQ(rc, 0) << "Cannot allocate memory, size: " << size
                  << " alignment: " << alignment;
  return data;
}

static void FreeMemory(char *data) {
  free(data);
}

// Basic runtime for serial execution of cells on a single CPU thread.
class BasicRuntime : public Runtime {
 public:
  void AllocateInstance(Instance *instance) override {
    char *data = AllocateMemory(instance->size(), instance->alignment());
    memset(data, 0, instance->size());
    instance->set_data(data);
  }

  void FreeInstance(Instance *instance) override {
    FreeMemory(instance->data());
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

Library::~Library() {
  if (owns_kernels_) {
    for (auto o : kernels_) {
      for (auto k : o.second) delete k;
    }
  }
}

void Library::Register(Kernel *kernel) {
  VLOG(7) << "Add " << kernel->Name() << " for " << kernel->Operation();
  kernels_[kernel->Operation()].push_back(kernel);
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

void Tensor::Align(const Shape &align) {
  CHECK_LE(align.rank(), alignment_.rank());
  for (int d = 0; d < align.rank(); ++d) {
    if (align.dim(d) > alignment_.dim(d)) alignment_.set(d, align.dim(d));
  }
}

void Tensor::AlignLast(int align) {
  if (align > alignment_.dim(rank() - 1)) {
    alignment_.set(rank() - 1, align);
  }
}

void Tensor::SameAlign(Tensor *other) {
  Align(other->alignment_);
  other->Align(alignment_);
}

void Tensor::CompatibleAlign(Tensor *other) {
  int d1 = rank() - 1;
  int d2 = other->rank() - 1;
  while (d1 >= 0 && d2 >= 0) {
    int align = std::max(alignment_.dim(d1), other->alignment_.dim(d2));
    alignment_.set(d1--, align);
    other->alignment_.set(d2--, align);
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

bool Tensor::SupportsOrder(Order order) {
  return combined_order[required_order_][order] != CONFLICTING_ORDER;
}

void Tensor::SetRequiredOrder(Order order) {
  required_order_ = combined_order[required_order_][order];
}

void Tensor::SetMiniumAlignment(int alignment) {
  if (alignment > byte_alignment_) byte_alignment_ = alignment;
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
  FreeMemory(data_);
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
  char *buffer =
    AllocateMemory(n * connector_->size(), connector_->alignment());

  // Copy existing data to new buffer.
  if (data_ != nullptr) {
    memcpy(buffer, data_, size_ * connector_->size());
    FreeMemory(data_);
  }

  // Set new data buffer.
  data_ = buffer;
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
    if (p == nullptr) return "null";
    p = *reinterpret_cast<char **>(p);
  }
  if (param->shape().partial()) return "*";

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

bool Step::AllowInPlace(int input, int output) {
  // Get input and output that should be shared.
  DCHECK_GE(input, 0);
  DCHECK_LT(input, inputs_.size());
  DCHECK_GE(output, 0);
  DCHECK_LT(output, outputs_.size());
  Tensor *in = inputs_[input];
  Tensor *out = outputs_[output];
  if (in->consumers().size() != 1) return false;
  if (in->ref() != out->ref()) return false;
  if (out->shared()) return false;
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

Network::Network() {
  runtime_ = &default_runtime;
}

Network::~Network() {
  for (auto *m : memory_) FreeMemory(m);
  for (auto *t : parameters_) delete t;
  for (auto *t : constants_) {
    if (t->device_data_ != DEVICE_NULL) {
      runtime_->RemoveTensorFromDevice(t);
    }
    delete t;
  }
  for (auto *c : cells_) delete c;
  for (auto *s : steps_) delete s;
  for (auto *c : connectors_) delete c;
}

Tensor *Network::GetParameter(const string &name) const {
  auto f = names_.find(name);
  return f == names_.end() ? nullptr : f->second;
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
      tensor->required_order_ = parameter_element_order_;
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
    tensor->alignment_.fill(var->rank(), 1);
    tensor->stride_.fill(var->rank(), 0);
    tensor->byte_alignment_ = TypeTraits::of(var->type).size();

    // Input variables are initially placed in host memory.
    if (var->in) {
      tensor->current_placement_ = HOST;
      tensor->placement_ = HOST;
    }

    // Output variables must be available on the host after the computation.
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
    t->alignment_ = prototype->alignment_;
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
    for (Kernel *kernel : library.Lookup(step->type())) {
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
  if (profiling_) {
    for (Cell *cell : cells_) {
      // Allocate tensor for storing profiling information. The tensor is an
      // int64 vector where the first element is the invocation counter followed
      // by one element for each step in the cell computation for storing the
      // cycle counts. If the cell has parallel tasks, two additional cycle
      // counters are stored for each task.
      int size = 1 + cell->steps_.size() + 2 * cell->tasks_.size();
      Tensor *profile = new Tensor();
      profile->name_ = "timing/" + cell->name_;
      profile->cell_ = cell;
      profile->type_ = DT_INT64;
      profile->shape_.assign(size);
      profile->size_ = profile->space_ = size * sizeof(int64);
      profile->aligned_ = profile->shape_;
      profile->alignment_.assign(sizeof(int64));
      profile->stride_.fill(sizeof(int64));
      profile->placement_ = HOST;
      profile->current_placement_ = HOST;
      parameters_.push_back(profile);
      cell->profile_ = profile;
    }
  }

  // Let kernels adjust the input and output data alignment requirements.
  for (Step *step : steps_) {
    step->kernel_->Adjust(step);
  }

  // Propagate alignment for shared tensors.
  for (auto it : tensors) {
    Tensor *tensor = it.second;
    Tensor *next = tensor->shared_;
    while (next != nullptr) {
      if (next->byte_alignment_ < tensor->byte_alignment_) {
        next->byte_alignment_ = tensor->byte_alignment_;
      }
      next = next->shared_;
    }
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
      Shape &at = t->alignment_;
      Shape &al = l->alignment_;
      int dt = t->rank() - 1;
      int dl = l->rank() - 1;
      while (dt >= 0 && dl >= 0) {
        if (t->dim(dt) != -1 && l->dim(dl) != -1) {
          if (at.dim(dt) > al.dim(dl)) {
            al.set(dl, at.dim(dt));
            again = true;
          } else if (at.dim(dt) < al.dim(dl)) {
            at.set(dt, al.dim(dl));
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
      if (t->byte_alignment_ < l->byte_alignment_) {
        t->byte_alignment_ = l->byte_alignment_;
        again = true;
      } else if (t->byte_alignment_ > l->byte_alignment_) {
        l->byte_alignment_ = t->byte_alignment_;
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

    // Compute stride size for each dimension.
    int size = TypeTraits::of(tensor->type()).size();
    if (tensor->order_ == ROW_MAJOR) {
      for (int d = tensor->rank() - 1; d >= 0; --d) {
        tensor->stride_.set(d, size);
        int dim = tensor->shape_.dim(d);
        if (dim == -1) dim = 1;
        int align = Align(dim, tensor->alignment_.dim(d));
        tensor->aligned_.set(d, align);
        size *= align;
      }
    } else {
      for (int d = 0; d < tensor->rank(); ++d) {
        tensor->stride_.set(d, size);
        int dim = tensor->shape_.dim(d);
        if (dim == -1) dim = 1;
        int align = Align(dim, tensor->alignment_.dim(d));
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
            << " alignment " << tensor->alignment_.ToString()
            << ":" << tensor->byte_alignment_
            << " aligned " << tensor->aligned_.ToString()
            << " size " << tensor->space_
            << " stride " << tensor->stride_.ToString()
            << " order " << tensor->order_
            << " placement " << placename[tensor->placement_];
  }

  // Compute size and alignment for connectors.
  for (Connector *connector : connectors_) {
    Tensor *t = connector->type_;

    if (connector->alignment_ < t->byte_alignment_) {
      connector->alignment_ = t->byte_alignment_;
    }

    if (connector->alignment_ < jit::CPU::CacheLineSize()) {
      connector->alignment_ = jit::CPU::CacheLineSize();
    }

    VLOG(5) << "Connector " << connector->name() << ": " << t->TypeString()
            << " alignment " << t->alignment_.ToString()
            << ":" << connector->alignment_
            << " aligned " << t->aligned_.ToString()
            << " size " << t->size_
            << " stride " << t->stride_.ToString()
            << " order " << t->order_;
  }

  // Initialize instance blocks.
  for (Cell *cell : cells_) {
    // Adjust instance to cache lines.
    if (cell->instance_alignment_ < jit::CPU::CacheLineSize()) {
      cell->instance_alignment_ = jit::CPU::CacheLineSize();
    }

    // Allocate space for runtime data at the beginning of the instance block.
    cell->instance_size_ = runtime_->ExtraInstanceData(cell);

    // Allocate task structures in instance.
    for (auto &t : cell->tasks_) {
      t.offset = cell->instance_size_;
      cell->instance_size_ += sizeof(Task);
    }
  }

  // Compute cell instance size and offset of each parameter.
  for (Tensor *tensor : parameters_) {
    if (tensor->cell_ != nullptr) {
      Cell *c = tensor->cell_;
      int align = tensor->ref_ ? kMinDataAlignment : tensor->byte_alignment_;

      // Set offset for tensor in instance.
      CHECK_GE(tensor->space(), 0) << tensor->name() << " " << tensor->space();
      if (tensor->placement_ & HOST) {
        if (tensor->shared_ != nullptr) {
          tensor->offset_ = tensor->shared_->offset_;
          VLOG(5) << "Share " << tensor->name() << " with "
                  << tensor->shared()->name();
        } else {
          // Ensure alignment of tensor in instance.
          c->instance_size_ = Align(c->instance_size_, align);

          // Assign offset to tensor and update instance size.
          tensor->offset_ = c->instance_size_;
          c->instance_size_ += tensor->space();

          // Ensure that instance has at least the same aligment as the tensor.
          if (tensor->byte_alignment_ > c->instance_alignment_) {
            c->instance_alignment_ = tensor->byte_alignment_;
          }
        }
      }

      // Set offset for tensor in device instance.
      if (tensor->placement_ & DEVICE) {
        if (tensor->shared_ != nullptr) {
          tensor->device_offset_ = tensor->shared_->device_offset_;
          VLOG(5) << "Share " << tensor->name() << " with "
                  << tensor->shared()->name() << " on device";
        } else {
          // Ensure alignment of tensor in device instance.
          c->device_instance_size_ = Align(c->device_instance_size_, align);

          // Assign offset to tensor and update device instance size.
          tensor->device_offset_ = c->device_instance_size_;
          c->device_instance_size_ += tensor->space();

          // Ensure that instance has at least the same aligment as the tensor.
          if (tensor->byte_alignment_ > c->device_instance_alignment_) {
            c->device_instance_alignment_ = tensor->byte_alignment_;
          }
        }
      }
    }
  }

  // Copy and align constants.
  for (Tensor *tensor : constants_) {
    // Determine alignment for tensor.
    int alignment = tensor->byte_alignment_;
    if (alignment < kMinDataAlignment) alignment = kMinDataAlignment;
    if (alignment < jit::CPU::CacheLineSize()) {
      alignment = jit::CPU::CacheLineSize();
    }

    // Allocate memory for tensor.
    char *data = AllocateMemory(tensor->size_, alignment);
    memory_.push_back(data);
    memset(data, 0, tensor->size_);

    // Copy data.
    if (tensor->rank() == 0 || tensor->rank() == 1) {
      // Vectors and scalars can just be copied regardless of alignment and
      // order.
      memcpy(data, tensor->data_, tensor->size_);
    } else if (tensor->rank() == 2) {
      // Copy matrix one element at a time.
      char *src = tensor->data_;
      int element_size = tensor->element_size();
      for (int r = 0; r < tensor->dim(0); ++r) {
        for (int c = 0; c < tensor->dim(1); ++c) {
          memcpy(data + tensor->offset(r, c), src, element_size);
          src += element_size;
        }
      }
    } else {
      LOG(ERROR) << tensor->rank() << "D tensor not supported: "
                 << tensor->name();
      return false;
    }
    tensor->data_ = data;
    tensor->AddNewPlace(HOST);

    // Copy constant to device if needed.
    if (tensor->placement_ & DEVICE) {
      VLOG(5) << "Copy tensor " << tensor->name() << " to device";
      tensor->device_data_ = runtime_->CopyTensorToDevice(tensor);
      CHECK(tensor->device_data_ != DEVICE_NULL);
      tensor->AddNewPlace(DEVICE);
    }
  }

  // Compile each cell computation.
  for (Cell *cell : cells_) {
    // Create macro assembler for code generation.
    MacroAssembler masm(nullptr, 0);
    masm.set_runtime(runtime_);

    // Declare the number of registers needed by the cell.
    masm.rr().usage(cell->register_usage_);

    // Enable timing measurement instrumentation if profiling is active.
    if (profiling_) masm.set_timing(true);

    // Insert break point in the beginning of the generated code in debug mode.
    if (debug_) masm.Breakpoint();

    // Generate prolog for main cell computation.
    masm.Prolog();

    // Increment the invocation counter.
    if (profiling_) {
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
            if (profiling_) {
              int timing = cell->profile()->offset();
              int slot = 1 + cell->steps_.size() + tidx * 2 + 1;
              masm.TimeStep(timing + slot * sizeof(int64));
            }
          }
        }

        // Synchronize main task if needed before executing step.
        if (sync && step->NeedsSynchronization()) {
          masm.CallInstanceFunction(runtime_->SyncMainFunc());
          sync = false;
        }

        // Generate code for step.
        step->kernel_->Generate(step, &masm);

        // No registers are preserved between steps, so reset register
        // allocation.
        masm.rr().reset();
        masm.mm().reset();

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
        if (profiling_) {
          int timing = cell->profile()->offset();
          masm.TimeStep(timing + (stepnum + 1) * sizeof(int64));
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
          if (profiling_) {
            int timing = cell->profile()->offset();
            int slot = 1 + cell->steps_.size() + tidx * 2;
            masm.TimeStep(timing + slot * sizeof(int64));
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
    if (profiling_) {
      masm.CallInstanceFunction(runtime_->StopProfilerFunc());
    }

    // Generate epilog for main cell computation.
    masm.Epilog();

    // Generate code for parallel tasks.
    int task_index = 0;
    for (auto &task : cell->tasks_) {
      // Set entry for task function.
      masm.bind(&task.entry);

      // Generate parallel task prolog.
      masm.Prolog();

      // Let kernels generate code for each step.
      int stepnum = 0;
      for (Step *step : cell->steps_) {
        if (step->task_index_ == task_index) {
          // Generate code for step.
          step->kernel_->Generate(step, &masm);

          // No registers are preserved between steps, so reset register
          // allocation.
          masm.rr().reset();
          masm.mm().reset();

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
          if (profiling_) {
            int timing = cell->profile()->offset();
            masm.TimeStep(timing + (stepnum + 1) * sizeof(int64));
          }
        }
        stepnum++;
      }

      // Generate parallel task epilog.
      masm.Epilog();

      task_index++;
    }

    // Allocate executable code object for generated code.
    cell->code_.Allocate(&masm);
    VLOG(7) << cell->name()
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
  StringAppendF(&str, "cell %s {  // size %d\n", name_.c_str(), instance_size_);

  // Output instance data fields.
  std::vector<Tensor *> fields;
  for (Tensor *t : network_->parameters()) {
    if (t->cell() == this) fields.push_back(t);
  }
  std::sort(fields.begin(), fields.end(), CompareOffset);

  int prev_offset = -1;
  for (Tensor *t : fields) {
    if (t->placement() & HOST) {
      if (t->offset() == prev_offset) {
        str.append("    union ");
      } else {
        str.append("  var ");
      }
      StringAppendF(&str, "%s: %s  // offset %d size %d\n",
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
        str.append("    union ");
      } else {
        str.append("  device var ");
      }
      StringAppendF(&str, "%s: %s  // offset %d size %d\n",
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
      StringAppendF(&str, "const %s: %s   // size %d\n",
                    t->name().c_str(),
                    t->TypeString().c_str(),
                    t->size());
    }
  }

  // Output cell steps.
  if (!steps_.empty()) {
    str.append("\n");
    for (Step *step : steps_) {
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

}  // namespace myelin
}  // namespace sling

