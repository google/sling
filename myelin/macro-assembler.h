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

#ifndef MYELIN_MACRO_ASSEMBLER_H_
#define MYELIN_MACRO_ASSEMBLER_H_

#include "myelin/compute.h"
#include "third_party/jit/assembler.h"

namespace sling {
namespace myelin {

// Register allocation.
class Registers {
 public:
  // An x64 CPU has 16 general 64-bit registers.
  static const int kNumRegisters = 16;

  // Allocate register.
  jit::Register alloc();

  // Allocate preserved register.
  jit::Register alloc_preserved();

  // Allocate register with preference.
  jit::Register alloc_preferred(jit::Register r);

  // Allocate fixed register.
  jit::Register alloc_fixed(jit::Register r);

  // Mark register as being in use.
  void use(int r) { used_regs_ |= (1 << r); }
  void use(jit::Register r) { use(r.code()); }

  // Mark register as being free.
  void release(int r) { used_regs_ &= ~(1 << r); }
  void release(jit::Register r) { release(r.code()); }

  // Check if register is used.
  bool used(int r) const { return ((1 << r) & used_regs_) != 0; }
  bool used(jit::Register r) { return used(r.code()); }

  // Reset allocated registers.
  void reset() { used_regs_ = kPreservedRegisters & ~saved_regs_; }

  // Reserve callee-saved register for use.
  void reserve(int r);
  void reserve(jit::Register r) { reserve(r.code()); }

  // Free callee-saved register after it has been restored.
  void free(int r);
  void free(jit::Register r) { free(r.code()); }

  // Declare the number of registers needed. If more than eight registers are
  // needed, an additional five callee-saved registers can be reserved.
  bool usage(int n);

  // Check if register should be saved.
  bool saved(int r) const { return ((1 << r) & saved_regs_) != 0; }
  bool saved(jit::Register r) { return saved(r.code()); }

  // Check if register is a callee-saved register.
  static bool preserved(int r) { return ((1 << r) & kPreservedRegisters) != 0; }
  static bool preserved(jit::Register r) { return preserved(r.code()); }

 private:
  // Preserved registers.
  static const int kPreservedRegisters =
    1 << jit::Register::kCode_rbx |
    1 << jit::Register::kCode_rsp |
    1 << jit::Register::kCode_rbp |
    1 << jit::Register::kCode_r12 |
    1 << jit::Register::kCode_r13 |
    1 << jit::Register::kCode_r14 |
    1 << jit::Register::kCode_r15;

  // Bit mask of register that are in use.
  int used_regs_ = kPreservedRegisters;

  // Bit mask of registers that should be saved by callee.
  int saved_regs_ = 0;
};

// SIMD register allocation.
class SIMDRegisters {
 public:
  // An x64 CPU has up to 16 SIMD registers.
  static const int kNumRegisters = 16;

  // Allocate 128-bit XMM register.
  jit::XMMRegister allocx() { return jit::XMMRegister::from_code(alloc()); }

  // Allocate 256-bit YMM register.
  jit::YMMRegister allocy() { return jit::YMMRegister::from_code(alloc()); }

  // Allocate SIMD register.
  int alloc();

  // Mark register as being in use.
  void use(int r) { used_regs_ |= (1 << r); }
  void use(jit::XMMRegister r) { use(r.code()); }
  void use(jit::YMMRegister r) { use(r.code()); }

  // Mark register as being free.
  void release(int r) { used_regs_ &= ~(1 << r); }
  void release(jit::XMMRegister r) { release(r.code()); }
  void release(jit::YMMRegister r) { release(r.code()); }

  // Check if register is used.
  bool used(int r) const { return ((1 << r) & used_regs_) != 0; }
  bool used(jit::XMMRegister r) { return used(r.code()); }
  bool used(jit::YMMRegister r) { return used(r.code()); }

  // Reset allocated registers.
  void reset() { used_regs_ = 0; }

 private:
  // Bit mask of register that are in use.
  int used_regs_ = 0;
};

// Macro assembler for generating code for computations.
class MacroAssembler : public jit::Assembler {
 public:
  MacroAssembler(void *buffer, int buffer_size);

  // Generate function prolog.
  void Prolog();

  // Generate function prolog.
  void Epilog();

  // Load address of tensor.
  void LoadTensorAddress(jit::Register dst, Tensor *tensor);

  // Emit breakpoint.
  void Breakpoint() { int3(); }

  // Copy memory.
  void Copy(jit::Register dst, int ddisp,
            jit::Register src, int sdisp,
            int size);

  // Load integer from array into 64-bit register.
  void LoadInteger(jit::Register dst, jit::Register base, jit::Register index,
                   Type type);

  // Store integer into array from 64-bit register.
  void StoreInteger(jit::Register base, jit::Register index, jit::Register src,
                    Type type);

  // Multiply register with constant.
  void Multiply(jit::Register reg, int64 scalar);

  // Start of loop. Align code and bind label.
  void LoopStart(jit::Label *label);

  // Increment invocation counter.
  void IncrementInvocations(int offset);

  // Generate timing for step and update instance block.
  void TimeStep(int offset);

  // Start task.
  void StartTask(int offset, int32 id, int32 index, jit::Label *entry);

  // Wait for task to complete.
  void WaitForTask(int offset);

  // General purpose register allocation.
  Registers &rr() { return rr_; }

  // SIMD register allocation.
  SIMDRegisters &mm() { return mm_; }

  // Returns the instance data register.
  jit::Register instance() const;

  // Timing measurement instrumentation.
  bool timing() const { return timing_; }
  void set_timing(bool timing) { timing_ = timing; }

  // Runtime support functions.
  Runtime *runtime() const { return runtime_; }
  void set_runtime(Runtime *runtime) { runtime_ = runtime; }

 private:
  // Register allocation.
  Registers rr_;
  SIMDRegisters mm_;

  // Timing measurements using timestamp counter.
  bool timing_ = false;

  // Runtime support functions.
  Runtime *runtime_ = nullptr;
};

}  // namespace myelin
}  // namespace sling

#endif  // MYELIN_MACRO_ASSEMBLER_H_

