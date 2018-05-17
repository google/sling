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

#ifndef SLING_MYELIN_GENERATOR_INDEX_H_
#define SLING_MYELIN_GENERATOR_INDEX_H_

#include <vector>

#include "sling/myelin/compute.h"
#include "sling/myelin/express.h"
#include "sling/myelin/macro-assembler.h"

namespace sling {
namespace myelin {

// An index generator implements a loop structure with indexing of the
// input and output variables in an expression. It also handles register
// allocation for temporary and auxiliary variables.
class IndexGenerator {
 public:
  IndexGenerator(MacroAssembler *masm) : masm_(masm) {}
  virtual ~IndexGenerator() = default;

  // Extended register set.
  bool extended_regs() const { return extended_regs_; }
  void set_extended_regs(bool enable) { extended_regs_ = enable; }

  // Initialize index generator.
  virtual void Initialize(size_t vecsize) = 0;

  // Allocate registers. Return false in case of register overflow.
  virtual bool AllocateRegisters();

  // Return operand for accessing memory variable.
  virtual jit::Operand addr(Express::Var *var) = 0;

  // Check if variable needs to be broadcast to whole vector after loading.
  virtual bool NeedsBroadcast(Express::Var *var) { return false; }

  // Return pointer to constant data.
  virtual const void *data(Express::Var *var) = 0;

  // Return register for accessing temporary variable.
  jit::Register reg(int idx) { return regs_[idx]; }
  jit::XMMRegister xmm(int idx) {
    return jit::XMMRegister::from_code(mmregs_[idx]);
  }
  jit::YMMRegister ymm(int idx) {
    return jit::YMMRegister::from_code(mmregs_[idx]);
  }
  jit::ZMMRegister zmm(int idx) {
    return jit::ZMMRegister::from_code(mmregs_[idx]);
  }

  // Return auxiliary register.
  jit::Register aux(int idx) { return aux_[idx]; }
  jit::XMMRegister xmmaux(int idx) {
    return jit::XMMRegister::from_code(mmaux_[idx]);
  }
  jit::YMMRegister ymmaux(int idx) {
    return jit::YMMRegister::from_code(mmaux_[idx]);
  }
  jit::ZMMRegister zmmaux(int idx) {
    return jit::ZMMRegister::from_code(mmaux_[idx]);
  }

  // Reserve fixed register for generating instructions that operate on
  // special registers.
  void ReserveFixedRegister(jit::Register reg);

  // Reserve registers used for holding intermediate values in expressions.
  void ReserveRegisters(int count);
  void ReserveXMMRegisters(int count);
  void ReserveYMMRegisters(int count);
  void ReserveZMMRegisters(int count);

  // Reserve auxiliary registers for expression generators that need extra
  // registers for compiling expression operations.
  void ReserveAuxRegisters(int count);
  void ReserveAuxXMMRegisters(int count);
  void ReserveAuxYMMRegisters(int count);
  void ReserveAuxZMMRegisters(int count);

 protected:
  MacroAssembler *masm_;              // macro assembler for code generation

 private:
  bool extended_regs_ = false;        // use extended register set
  std::vector<jit::Register> fixed_;  // reserved fixed registers
  std::vector<jit::Register> regs_;   // reserved temporary registers
  std::vector<int> mmregs_;           // reserved SIMD registers (xmm/ymm)
  std::vector<jit::Register> aux_;    // reserved auxiliary registers
  std::vector<int> mmaux_;            // reserved auxiliary SIMD registers
};

}  // namespace myelin
}  // namespace sling

#endif  // SLING_MYELIN_GENERATOR_INDEX_H_

