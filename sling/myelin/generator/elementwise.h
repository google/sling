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

#ifndef SLING_MYELIN_GENERATOR_ELEMENTWISE_H_
#define SLING_MYELIN_GENERATOR_ELEMENTWISE_H_

#include <vector>

#include "sling/myelin/generator/index.h"

namespace sling {
namespace myelin {

// Index generator for element-wise operations. It supports operations over
// multiple inputs and outputs. All the outputs must have the same size and
// type. The inputs must be broadcast compatible with the output. This index
// generator has optimized versions for different kinds of input:
//  - SIMPLE (iterator with the same size as the output)
//  - SCALAR (scalar input broadcast over all the elemements)
//  - CONST (scalar constant input broadcast over all the elemements)
//  - REPEAT (iterator repeated over all the elements)
//  - BROADCAST (general broadcast iterator with one broadcast dimension)
class ElementwiseIndexGenerator : public IndexGenerator {
 public:
  // Create element-wise index generator for step.
  ElementwiseIndexGenerator(const Step *step, MacroAssembler *masm);
  ~ElementwiseIndexGenerator();

  // Initialize index generator for vector size.
  void Initialize(size_t vecsize) override;

  // Allocate registers. Return false in case of register overflow.
  bool AllocateRegisters() override;

  // Return operand for accessing memory variable.
  jit::Operand addr(Express::Var *var) override;

  // Return pointer to constant data.
  const void *data(Express::Var *var) override;

  // Generate start and end of loop.
  void BeginLoop();
  void EndLoop();

  // Whether only one iteration is needed.
  bool single() const { return single_; }

 private:
  enum IteratorType {SIMPLE, SCALAR, CONST, REPEAT, BROADCAST};
  struct Locator;
  struct Iterator;

  // Initialize locator for variable.
  bool InitializeLocator(Tensor *var, Locator *loc);

  // Allocate registers for locator.
  bool AllocateLocatorRegisters(Locator *loc);

  // Get locator for variable.
  Locator *GetLocator(Express::Var *var) {
    return var->type == Express::OUTPUT ? &output_[var->id] : &input_[var->id];
  }

  // Check if variable is a valid index.
  bool Valid(Express::Var *var) const;

  // Return the output element size.
  size_t element_size() const {
    return TypeTraits::of(type_).size();
  }

  // Create new iterator.
  Iterator *NewIterator(IteratorType type) {
    Iterator *it = new Iterator(type);
    iterators_.push_back(it);
    return it;
  }

  // Iterator for looping over (vector) elements in tensor.
  struct Iterator {
    Iterator(IteratorType type) : type(type) {}

    IteratorType type;                   // iterator type
    size_t size = 0;                     // number of elements to iterate over
    size_t broadcast = 0;                // broadcast iterations
    jit::Register block = jit::no_reg;   // block base
    jit::Register offset = jit::no_reg;  // offset from base
    jit::Register repeat = jit::no_reg;  // broadcast counter
  };

  // Locator for generating address operands for variables.
  struct Locator {
    Tensor *var;                        // variable to iterate over
    jit::Register base = jit::no_reg;   // base address register
    Iterator *iterator = nullptr;       // iterator for iterating over elements
  };

  // Output type.
  Type type_;

  // Output shape.
  Shape shape_;

  // Vector size.
  size_t vecsize_ = 1;

  // Loop begin label.
  jit::Label begin_;

  // Output offset register.
  jit::Register offset_;

  // Whether only one iteration is needed.
  bool single_ = false;

  // Input and output locators.
  std::vector<Locator> input_;
  std::vector<Locator> output_;

  // Iterators.
  std::vector<Iterator *> iterators_;
};

}  // namespace myelin
}  // namespace sling

#endif  // SLING_MYELIN_GENERATOR_INDEX_H_

