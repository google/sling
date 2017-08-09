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

#include "myelin/generator/elementwise.h"

#define __ masm->

namespace sling {
namespace myelin {

using namespace jit;

ElementwiseIndexGenerator::ElementwiseIndexGenerator(
    const Step *step, MacroAssembler *masm) : IndexGenerator(masm) {
  // Get size from first output.
  CHECK_GE(step->outdegree(), 1);
  type_ = step->output(0)->type();
  shape_ = step->output(0)->shape();

  // Allocate locators for all inputs and outputs.
  input_.resize(step->indegree());
  for (int i = 0; i < step->indegree(); ++i) {
    CHECK(step->input(i)->type() == type_);
    CHECK(InitializeLocator(step->input(i), &input_[i]));
  }
  output_.resize(step->outdegree());
  for (int i = 0; i < step->outdegree(); ++i) {
    CHECK(step->output(i)->type() == type_);
    CHECK(step->output(i)->shape() == step->output(0)->shape());
    CHECK(InitializeLocator(step->output(i), &output_[i]));
  }
}

ElementwiseIndexGenerator::~ElementwiseIndexGenerator() {
  for (auto *i : iterators_) delete i;
}

bool ElementwiseIndexGenerator::InitializeLocator(Tensor *var, Locator *loc) {
  // Set variable for locator.
  loc->var = var;

  // Determine iterator type for variable.
  if (var->elements() == 1) {
    // Variable only has one element; use a scalar/const iterator.
    loc->iterator = NewIterator(var->IsConstant() ? CONST : SCALAR);
  } else if (var->shape() == shape_) {
    // Variable has same shape as output; use simple iterator.
    loc->iterator = NewIterator(SIMPLE);
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
        loc->iterator = NewIterator(SIMPLE);
      } else {
        // Variable shape is a suffix of the output shape; use a repeated
        // iterator.
        DCHECK(shape_.elements() % n == 0);
        loc->iterator = NewIterator(REPEAT);
        loc->iterator->size = n;
      }
    } else if (d1 >= 0 && d2 >= 0 && var->dim(d1) == 1 &&
               var->elements() * shape_.dim(d2) == shape_.elements()) {
      // Create broadcast iterator over one dimension.
      loc->iterator = NewIterator(BROADCAST);
      loc->iterator->size = n;
      loc->iterator->broadcast = shape_.dim(d2);
    } else {
      LOG(WARNING) << "Unsupported broadcast: " << var->name()
                   << " input: " << var->shape().ToString()
                   << " output: " << shape_.ToString();
      return false;
    }
  } else {
    return false;
  }

  return true;
}

void ElementwiseIndexGenerator::Initialize(size_t vecsize) {
  vecsize_ = vecsize;
  single_ = shape_.elements() * element_size() == vecsize_;
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

  // Allocate registers for locators.
  for (auto &loc : input_) {
    if (!AllocateLocatorRegisters(&loc)) return false;
  }
  for (auto &loc : output_) {
    if (!AllocateLocatorRegisters(&loc)) return false;
  }

  // Try to allocate extra base registers as an optimization.
  if (!single_) {
    for (auto &loc : input_) {
      if (loc.base.is_valid()) continue;
      if (loc.iterator->type == SIMPLE || loc.iterator->type == REPEAT) {
        loc.base = rr.try_alloc();
      }
    }
    for (auto &loc : output_) {
      if (loc.base.is_valid()) continue;
      if (loc.iterator->type == SIMPLE) {
        loc.base = rr.try_alloc();
      }
    }
  }

  return true;
}

bool ElementwiseIndexGenerator::AllocateLocatorRegisters(Locator *loc) {
  Registers &rr = masm_->rr();
  switch (loc->iterator->type) {
    case SIMPLE:
    case SCALAR:
      // Allocate base register for non-instance variables.
      if (loc->var->offset() == -1 || loc->var->ref()) {
        loc->base = rr.try_alloc();
        if (!loc->base.is_valid()) return false;
      }
      break;
    case CONST:
      // Constants use pc-relative addressing, so no extra registers are needed.
      break;
    case REPEAT:
      // Allocate base register for non-instance variables.
      if (loc->var->offset() == -1 || loc->var->ref()) {
        loc->base = rr.try_alloc();
        if (!loc->base.is_valid()) return false;
      }

      // Allocate index register.
      loc->iterator->offset = rr.try_alloc();
      if (!loc->iterator->offset.is_valid()) return false;
      break;
    case BROADCAST:
      // Allocate block, index, and broadcast registers.
      loc->iterator->block = rr.try_alloc();
      if (!loc->iterator->block.is_valid()) return false;
      loc->iterator->offset = rr.try_alloc();
      if (!loc->iterator->offset.is_valid()) return false;
      loc->iterator->repeat = rr.try_alloc();
      if (!loc->iterator->repeat.is_valid()) return false;
      break;
    default:
      return false;
  };

  return true;
}

void ElementwiseIndexGenerator::BeginLoop() {
  // Load tensor addresses and initialize index registers.
  MacroAssembler *masm = masm_;
  for (auto &loc : input_) {
    if (loc.base.is_valid()) {
      __ LoadTensorAddress(loc.base, loc.var);
    }
    if (loc.iterator->block.is_valid()) {
      __ LoadTensorAddress(loc.iterator->block, loc.var);
    }
    if (!single_ && loc.iterator->offset.is_valid()) {
      __ xorq(loc.iterator->offset, loc.iterator->offset);
    }
    if (loc.iterator->repeat.is_valid()) {
      __ xorq(loc.iterator->repeat, loc.iterator->repeat);
    }
  }
  for (auto &loc : output_) {
    if (loc.base.is_valid()) {
      __ LoadTensorAddress(loc.base, loc.var);
    }
  }

  // Generate loop start, unless there is only one iteration.
  if (!single_) {
    __ xorq(offset_, offset_);
    __ bind(&begin_);
  }
}

void ElementwiseIndexGenerator::EndLoop() {
  MacroAssembler *masm = masm_;
  if (!single_) {
    // Move to next output element.
    __ addq(offset_, Immediate(vecsize_));

    // Update iterators.
    for (Iterator *it : iterators_) {
      if (it->type == REPEAT) {
        size_t repeat_size = element_size() * it->size;
        if ((repeat_size & (repeat_size - 1)) == 0) {
          // The repeat block size is a power of two, so the index can be
          // computed using masking.
          __ movq(it->offset, offset_);
          __ andq(it->offset, Immediate(repeat_size - 1));
        } else {
          // Increment offset and reset at end of repeat.
          Label l;
          __ addq(it->offset, Immediate(vecsize_));
          __ cmpq(it->offset, Immediate(repeat_size));
          __ j(less, &l);
          __ xorq(it->offset, it->offset);
          __ bind(&l);
        }
      } else if (it->type == BROADCAST) {
        size_t block_size = element_size() * it->size;
        Label l;
        // Move to next inner element block.
        __ addq(it->offset, Immediate(vecsize_));
        __ cmpq(it->offset, Immediate(block_size));
        __ j(less, &l);

        // Next repetition of block.
        __ xorq(it->offset, it->offset);   // at end of block, reset index
        __ incq(it->repeat);               // increment block repeat
        __ cmpq(it->repeat, Immediate(it->broadcast));
        __ j(less, &l);

        // Move to next block.
        __ xorq(it->repeat, it->repeat);  // move to next block
        __ addq(it->block, Immediate(block_size));
        __ bind(&l);
      }
    }

    // Check if we have reached the end of the output.
    size_t size = element_size() * shape_.elements();
    __ cmpq(offset_, Immediate(size));
    __ j(less, &begin_);
  }
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
        return Operand(rbp);
    }
  } else {
    // Get locator.
    CHECK(Valid(var));
    Locator *loc = GetLocator(var);

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
        DCHECK(loc->var->IsConstant());
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
      case BROADCAST:
        // Return block base plus block offset.
        return Operand(loc->iterator->block, loc->iterator->offset);
      default:
        LOG(FATAL) << "Unsupported iterator type";
        return Operand(rbp);
    }
  }
}

const void *ElementwiseIndexGenerator::data(Express::Var *var) {
  DCHECK_EQ(var->type, Express::CONST);
  Locator *loc = GetLocator(var);
  DCHECK(loc->var->IsConstant());
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

