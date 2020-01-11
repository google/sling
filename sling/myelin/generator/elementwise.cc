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

#include <map>

#include "sling/myelin/generator/elementwise.h"

#define __ masm->

namespace sling {
namespace myelin {

using namespace jit;

ElementwiseIndexGenerator::ElementwiseIndexGenerator(
    const Step *step, MacroAssembler *masm) : IndexGenerator(masm) {
  // Get size from first output.
  assign_ = step->type() == "Assign";
  Tensor *prototype = step->GetPrototype();
  type_ = prototype->type();
  shape_ = prototype->shape();

  // Allocate locators for all inputs and outputs.
  input_.resize(step->indegree());
  for (int i = 0; i < step->indegree(); ++i) {
    CHECK(step->input(i)->type() == type_);
    input_[i] = GetLocator(step->input(i));
  }
  if (assign_) {
    // Add assigment locator.
    output_.resize(1);
    output_[0] = GetLocator(step->input(0));

    // Optionally output reference to assignment target.
    if (step->outdegree() == 1) {
      output_ref_ = step->output(0);
    }
  } else {
    output_.resize(step->outdegree());
    for (int i = 0; i < step->outdegree(); ++i) {
      CHECK(step->output(i)->type() == type_);
      CHECK(step->output(i)->shape() == shape_ || step->output(i)->rank() == 0);
      output_[i] = GetLocator(step->output(i));
    }
  }
}

ElementwiseIndexGenerator::~ElementwiseIndexGenerator() {
  for (auto *i : iterators_) delete i;
  for (auto *l : locators_) delete l;
}

ElementwiseIndexGenerator::Iterator *ElementwiseIndexGenerator::GetIterator(
    IteratorType type,
    size_t size) {
  // Try to find existing iterator.
  for (Iterator *it : iterators_) {
    if (type == it->type && size == it->size) {
      return it;
    }
  }

  // Create new iterator.
  Iterator *it = new Iterator(type, size);
  iterators_.push_back(it);
  return it;
}

ElementwiseIndexGenerator::Locator *ElementwiseIndexGenerator::GetLocator(
    Tensor *var) {
  // Try to find existing locator.
  for (Locator *loc : locators_) {
    if (var == loc->var) return loc;
  }

  // Create new locator.
  Locator *loc = new Locator(var);
  locators_.push_back(loc);

  // Determine iterator type for variable.
  if (var->elements() == 1) {
    // Variable only has one element; use a scalar/const iterator.
    loc->iterator = GetIterator(var->constant() ? CONST : SCALAR, 1);
  } else if (var->shape() == shape_) {
    // Variable has same shape as output; use simple iterator.
    loc->iterator = GetIterator(SIMPLE, shape_.elements());
  } else if (var->rank() <= shape_.rank()) {
    // Find common suffix between variable and output.
    int n = 1;
    int d1 = var->rank() - 1;
    int d2 = shape_.rank() - 1;
    while (d1 >= 0) {
      int n1 = var->dim(d1);
      int n2 = shape_.dim(d2);
      if (n1 != n2) break;
      n *= n1;
      d1--;
      d2--;
    }

    if (n == var->elements()) {
      if (var->elements() == shape_.elements()) {
        // The variable shape prefix is a one vector so use a simple iterator.
        loc->iterator = GetIterator(SIMPLE, shape_.elements());
      } else {
        // Variable shape is a suffix of the output shape; use a repeated
        // iterator.
        DCHECK(shape_.elements() % n == 0);
        loc->iterator = GetIterator(REPEAT, n);
      }
    } else if (d1 >= 0 && d2 >= 0 && var->dim(d1) == 1 &&
               var->elements() * shape_.dim(d2) == shape_.elements()) {
      // Create broadcast iterator over one (singular) dimension.
      loc->iterator = GetIterator(n == 1 ? SINGLE : BROADCAST, n);
      loc->broadcast = shape_.dim(d2);
    } else {
      LOG(FATAL) << "Unsupported broadcast: " << var->name()
                 << " input: " << var->shape().ToString()
                 << " output: " << shape_.ToString();
    }
  } else if (var->shape().outer(shape_.rank()) == 1) {
    // The variable shape prefix is a one vector so use a simple iterator.
    loc->iterator = GetIterator(SIMPLE, var->elements());
  } else {
    LOG(FATAL) << "Unsupported iterator: " << var->name() << " with shape "
               << var->shape().ToString()
               << " to output shape " << shape_.ToString();
  }

  return loc;
}

void ElementwiseIndexGenerator::Initialize(size_t vecsize) {
  vecsize_ = vecsize;
  single_ = shape_.elements() * element_size() == vecsize_;
}

bool ElementwiseIndexGenerator::EnableSparse(Tensor *sparse) {
  // Check if sparse iteration can be used.
  if (single_) return false;
  Iterator *inner = nullptr;
  for (Iterator *it : iterators_) {
    if (it->type == SIMPLE) {
      // Only one simple iterator allowed for sparse iteration.
      if (inner != nullptr) return false;
      inner = it;
    } else if (it->type != SCALAR) {
      // Only simple and scalar iterators can be combined with sparse iteration.
      return false;
    }
  }
  if (inner == nullptr) return false;

  // Check that the vector size is compatible the inner block size.
  if (shape_.rank() < 1) return false;
  size_t block_size = shape_.inner(1) * element_size();
  if (block_size % vecsize_ != 0) return false;

  sparse_ = sparse;
  return true;
}

bool ElementwiseIndexGenerator::AllocateRegisters() {
  // Allocate temp vars.
  if (!IndexGenerator::AllocateRegisters()) return false;

  // Allocate register for output offset.
  Registers &rr = masm_->rr();
  if (!single_) {
    offset_ = rr.try_alloc();
    if (!offset_.is_valid()) return false;
  }

  // Allocate registers for sparse iterator.
  if (sparse_) {
    bitmap_ = rr.try_alloc();
    if (!bitmap_.is_valid()) return false;
    bits_ = rr.try_alloc();
    if (!bits_.is_valid()) return false;
    mask_ = rr.try_alloc();
    if (!mask_.is_valid()) return false;
    iend_ = rr.try_alloc();
    if (!iend_.is_valid()) return false;
  }

  // Assignment target needs a base register.
  if (output_ref_ != nullptr) {
    input_[0]->base = rr.try_alloc();
    if (!input_[0]->base.is_valid()) return false;
  }

  // Allocate registers for iterators.
  for (auto *it : iterators_) {
    if (it->type == REPEAT || it->type == BROADCAST) {
      // Allocate index register.
      it->offset = rr.try_alloc();
      if (!it->offset.is_valid()) return false;
    }
  }

  // Allocate registers for locators.
  std::vector<Locator *> simple_locators;
  for (auto *loc : locators_) {
    switch (loc->iterator->type) {
      case SIMPLE:
      case SCALAR:
        // Base register only needed for non-instance variables. Allocation of
        // a register is deferred until registers for other locators have been
        // allocated.
        if (loc->var->IsGlobal() || loc->var->ref()) {
          if (!loc->base.is_valid()) simple_locators.push_back(loc);
        }
        break;
      case CONST:
        // Constants use pc-relative addressing, so no extra registers are
        // needed.
        break;
      case REPEAT:
        // Allocate base register for non-instance variables.
        if (loc->var->IsGlobal() || loc->var->ref()) {
          if (!loc->base.is_valid()) loc->base = rr.try_alloc();
          if (!loc->base.is_valid()) return false;
        }
        break;
      case SINGLE:
      case BROADCAST:
        // Allocate base and broadcast registers.
        if (!loc->base.is_valid()) loc->base = rr.try_alloc();
        if (!loc->base.is_valid()) return false;
        loc->repeat = rr.try_alloc();
        if (!loc->repeat.is_valid()) return false;
        break;
      default:
        return false;
    };
  }

  // Allocate registers for simple locators. These locators are loop-invariant
  // and can either be initialized before the loop or on demand inside the
  // loop depending on how many registers are available.
  jit::Register scratch = jit::no_reg;
  bool ondemand = rr.num_free() < simple_locators.size();
  if (ondemand) {
    scratch = rr.try_alloc();
    if (!scratch.is_valid()) return false;
  }
  for (auto *loc : simple_locators) {
    loc->base = rr.try_alloc();
    if (!loc->base.is_valid()) {
      loc->base = scratch;
      loc->ondemand = true;
    }
  }

  // Try to allocate extra base registers as an optimization. The base registers
  // can be shared between local tensors with the same location.
  if (!single_) {
    std::map<int, Register> base_regs;
    for (auto *loc : locators_) {
      // Do not allocate register if locator already has a base register.
      if (loc->base.is_valid()) continue;

      // Only simple and repeat iterators can use extra base registers.
      if (loc->iterator->type == SIMPLE || loc->iterator->type == REPEAT) {
        if (loc->var->offset() != -1) {
          // Try to find existing base register for offset in instance.
          auto f = base_regs.find(loc->var->offset());
          if (f != base_regs.end()) {
            // Use shared base register.
            loc->base = f->second;
            loc->shared = true;
          } else {
            // Try to allocate new base register.
            loc->base = rr.try_alloc();
            if (loc->base.is_valid()) {
              base_regs[loc->var->offset()] = loc->base;
            }
          }
        } else if (loc->var->IsGlobal()) {
          loc->base = rr.try_alloc();
        }
      }
    }
  }

  return true;
}

void ElementwiseIndexGenerator::GenerateInit() {
  // Load tensor addresses and initialize index registers.
  MacroAssembler *masm = masm_;
  for (auto *loc : locators_) {
    if (loc->base.is_valid() && !loc->shared && !loc->ondemand) {
      __ LoadTensorAddress(loc->base, loc->var);
    }
    if (loc->repeat.is_valid()) {
      __ xorq(loc->repeat, loc->repeat);
    }
  }
  for (auto *it : iterators_) {
    if (!single_ && it->offset.is_valid()) {
      __ xorq(it->offset, it->offset);
    }
  }

  // Load address of sparsity bitmap.
  if (sparse_) {
    __ LoadTensorAddress(bitmap_, sparse_);
  }
}

void ElementwiseIndexGenerator::GenerateLoopBegin() {
  // Generate loop start, unless there is only one iteration.
  MacroAssembler *masm = masm_;
  if (sparse_) {
    // Use the sparsity bitmap to loop over all the non-zero inner blocks.
    size_t size = shape_.elements() * element_size();
    size_t block_size = shape_.inner(1) * element_size();
    __ xorq(offset_, offset_);

    // Get next 64-bit word from bitmap.
    __ bind(&begin_);
    __ movq(bits_, Operand(bitmap_));
    __ addq(bitmap_, Immediate(sizeof(uint64)));

    // If all bits are zero, skip all 64 inner blocks.
    __ testq(bits_, bits_);
    __ j(not_zero, &spl1_);
    __ addq(offset_, Immediate(64 * block_size));
    __ cmpq(offset_, Immediate(size));
    __ j(less, &begin_);

    // Initialize mask.
    __ bind(&spl1_);
    __ movq(mask_, Immediate(1));

    // If current bit is zero, skip inner block.
    __ bind(&spl2_);
    __ testq(bits_, mask_);
    __ j(not_zero, &spl3_);
    __ addq(offset_, Immediate(block_size));
    __ jmp(&spl5_);

    // Loop over inner block.
    __ bind(&spl3_);
    if (block_size != vecsize_) {
      __ leaq(iend_, Operand(offset_, block_size));
      __ bind(&spl4_);
    }
  } else if (!single_) {
    __ xorq(offset_, offset_);
    __ bind(&begin_);
  }
}

void ElementwiseIndexGenerator::GenerateLoopEnd() {
  MacroAssembler *masm = masm_;
  if (sparse_) {
    // Move to next output element.
    size_t size = shape_.elements() * element_size();
    size_t block_size = shape_.inner(1) * element_size();
    __ addq(offset_, Immediate(vecsize_));
    if (block_size != vecsize_) {
      __ cmpq(offset_, iend_);
      __ j(less, &spl4_);
    }

    // Next bit, loop until all 64 bits have been tested.
    __ bind(&spl5_);
    __ shlq(mask_, Immediate(1));
    __ j(not_zero, &spl2_);

    // Check if we have reached the end of the output.
    __ cmpq(offset_, Immediate(size));
    __ j(less, &begin_);
  } else if (!single_) {
    // Move to next output element.
    __ addq(offset_, Immediate(vecsize_));

    // Update iterators.
    for (Iterator *it : iterators_) {
      if (it->type == REPEAT) {
        size_t repeat_size = element_size() * it->size;
        __ addq(it->offset, Immediate(vecsize_));
        if ((repeat_size & (repeat_size - 1)) == 0) {
          // The repeat block size is a power of two, so the index can be
          // computed using masking.
          __ andq(it->offset, Immediate(repeat_size - 1));
        } else {
          // Increment offset and reset at end of repeat.
          Label l;
          __ cmpq(it->offset, Immediate(repeat_size));
          __ j(less, &l);
          __ xorq(it->offset, it->offset);
          __ bind(&l);
        }
      } else if (it->type == SINGLE) {
        for (Locator *loc : locators_) {
          if (loc->iterator != it) continue;
          int stride = vecsize_ / element_size();
          if (stride == 1) {
            __ incq(loc->repeat);
          } else {
            __ addq(loc->repeat, Immediate(stride));
          }
          Label l2;
          if ((loc->broadcast & (loc->broadcast - 1)) == 0) {
            __ andq(loc->repeat, Immediate(loc->broadcast - 1));
            __ j(not_zero, &l2);
          } else {
            __ cmpq(loc->repeat, Immediate(loc->broadcast));
            __ j(less, &l2);
            __ xorq(loc->repeat, loc->repeat);
          }
          __ addq(loc->base, Immediate(element_size()));
          __ bind(&l2);
        }
      } else if (it->type == BROADCAST) {
        size_t block_size = element_size() * it->size;
        Label l1;
        // Move to next inner element block.
        __ addq(it->offset, Immediate(vecsize_));
        __ cmpq(it->offset, Immediate(block_size));
        __ j(less, &l1);

        // Next repetition of block.
        __ xorq(it->offset, it->offset);
        for (Locator *loc : locators_) {
          if (loc->iterator != it) continue;
          Label l2;
          __ incq(loc->repeat);
          __ cmpq(loc->repeat, Immediate(loc->broadcast));
          __ j(less, &l2);
          __ xorq(loc->repeat, loc->repeat);
          __ addq(loc->base, Immediate(block_size));
          __ bind(&l2);
        }
        __ bind(&l1);
      }
    }

    // Check if we have reached the end of the output.
    size_t size = element_size() * shape_.elements();
    __ cmpq(offset_, Immediate(size));
    __ j(less, &begin_);
  }

  // Optionally output reference to assignment target.
  if (output_ref_ != nullptr) {
    CHECK(output_ref_->IsLocal());
    CHECK(output_ref_->ref());
    CHECK(input_[0]->base.is_valid());
    __ movq(Operand(masm->instance(), output_ref_->offset()), input_[0]->base);
  }
}

bool ElementwiseIndexGenerator::NeedsBroadcast(Express::Var *var) {
  // Constants do not need broadcasting.
  if (var->type == Express::NUMBER || var->type == Express::CONST) {
    return false;
  }

  // Get locator.
  CHECK(Valid(var));
  Locator *loc = LookupLocator(var);

  // Single broadcasts need broadcast.
  if (loc->iterator->type == SINGLE) return true;

  // Memory variable needs broadcast if it is a scalar value and the vector size
  // of the generator is more than one element.
  return vecsize_ > element_size() && loc->var->elements() == 1;
}

Operand ElementwiseIndexGenerator::addr(Express::Var *var) {
  if (var->type == Express::NUMBER) {
    // System-defined constant.
    switch (type_) {
      case DT_FLOAT: {
        float number = Express::NumericFlt32(var->id);
        int repeat = vecsize_ / sizeof(float);
        return masm_->GetConstant(number, repeat)->address();
      }
      case DT_DOUBLE: {
        double number = Express::NumericFlt64(var->id);
        int repeat = vecsize_ / sizeof(double);
        return masm_->GetConstant(number, repeat)->address();
      }
      case DT_INT8: {
        int8 number = Express::NumericFlt32(var->id);
        int repeat = vecsize_ / sizeof(int8);
        return masm_->GetConstant(number, repeat)->address();
      }
      case DT_INT16: {
        int16 number = Express::NumericFlt32(var->id);
        int repeat = vecsize_ / sizeof(int16);
        return masm_->GetConstant(number, repeat)->address();
      }
      case DT_INT32: {
        int32 number = Express::NumericFlt32(var->id);
        int repeat = vecsize_ / sizeof(int32);
        return masm_->GetConstant(number, repeat)->address();
      }
      case DT_INT64: {
        int64 number = Express::NumericFlt32(var->id);
        int repeat = vecsize_ / sizeof(int64);
        return masm_->GetConstant(number, repeat)->address();
      }
      default:
        LOG(FATAL) << "Unsupported constant type";
        return Operand(no_reg);
    }
  } else {
    // Get locator.
    CHECK(Valid(var));
    Locator *loc = LookupLocator(var);

    // Load base address on demand if needed.
    if (loc->ondemand) {
      masm_->LoadTensorAddress(loc->base, loc->var);
    }

    // Return operand for accessing variable.
    switch (loc->iterator->type) {
      case SIMPLE:
        if (single_) {
          // Single iteration.
          if (loc->base.is_valid()) {
            // Index single element using base register.
            return Operand(loc->base);
          } else {
            // Index single element using offset in instance.
            return Operand(masm_->instance(), loc->var->offset());
          }
        } else {
          // Multiple iterations.
          if (loc->base.is_valid()) {
            // Index element using base register and index.
            return Operand(loc->base, offset_);
          } else {
            // Index element using offset in instance and index.
            return Operand(masm_->instance(), offset_, times_1,
                           loc->var->offset());
          }
        }
      case SCALAR:
        if (loc->base.is_valid()) {
          // Index scalar using base register.
          return Operand(loc->base);
        } else {
          // Index scalar using offset in instance.
          return Operand(masm_->instance(), loc->var->offset());
        }
      case CONST: {
        // Scalar constant in code block, vectorized if needed.
        DCHECK(loc->var->constant());
        int size = loc->var->element_size();
        int repeat = vecsize_ / size;
        DCHECK_EQ(repeat * size, vecsize_);
        return masm_->GetData(loc->var->data(), size, repeat)->address();
      }
      case REPEAT:
        if (single_) {
          // Single iteration.
          if (loc->base.is_valid()) {
            // Index single element using base register.
            return Operand(loc->base);
          } else {
            // Index single element using offset in instance.
            return Operand(masm_->instance(), loc->var->offset());
          }
        } else {
          // Multiple iterations.
          if (loc->base.is_valid()) {
            // Index element using base register and index.
            return Operand(loc->base, loc->iterator->offset);
          } else {
            // Index element using offset in instance and index.
            return Operand(masm_->instance(), loc->iterator->offset, times_1,
                           loc->var->offset());
          }
        }
      case SINGLE:
        // Return block base.
        return Operand(loc->base);
      case BROADCAST:
        // Return block base plus block offset.
        return Operand(loc->base, loc->iterator->offset);
      default:
        LOG(FATAL) << "Unsupported iterator type";
        return Operand(no_reg);
    }
  }
}

const void *ElementwiseIndexGenerator::data(Express::Var *var) {
  DCHECK_EQ(var->type, Express::CONST);
  Locator *loc = LookupLocator(var);
  DCHECK(loc->var->IsGlobal());
  return loc->var->data();
}

bool ElementwiseIndexGenerator::Valid(Express::Var *var) const {
  if (var->type == Express::OUTPUT) {
    return var->id >= 0 && var->id < output_.size();
  } else {
    return var->id >= 0 && var->id < input_.size();
  }
}

}  // namespace myelin
}  // namespace sling

