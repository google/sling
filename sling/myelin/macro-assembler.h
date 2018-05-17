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

#ifndef SLING_MYELIN_MACRO_ASSEMBLER_H_
#define SLING_MYELIN_MACRO_ASSEMBLER_H_

#include "sling/myelin/compute.h"
#include "third_party/jit/assembler.h"

namespace sling {
namespace myelin {

// Register allocation.
class Registers {
 public:
  // An x64 CPU has 16 general 64-bit registers.
  static const int kNumRegisters = 16;

  // Initialize registers.
  Registers()
      : used_regs_(kPreservedRegisters), saved_regs_(0) {}
  Registers(const Registers &rr)
      : used_regs_(rr.used_regs_), saved_regs_(rr.saved_regs_) {}
  Registers &operator=(const Registers &rr) {
    used_regs_ = rr.used_regs_;
    saved_regs_ = rr.saved_regs_;
    return *this;
  }

  // Allocate register.
  jit::Register try_alloc();
  jit::Register alloc();

  // Allocate preserved register.
  jit::Register try_alloc_preserved();
  jit::Register alloc_preserved();

  // Allocate register with preference.
  jit::Register alloc_preferred(jit::Register r);

  // Allocate fixed register.
  jit::Register alloc_fixed(jit::Register r);

  // Allocate temporary register that is neither preserved or used as an
  // argument register.
  jit::Register alloc_temp();

  // Allocate argument register (1-6) or return register (0).
  jit::Register arg(int n);

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

  // Return the number of free registers.
  int num_free() const;

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

  // Bit mask of registers that are in use.
  int used_regs_;

  // Bit mask of registers that should be saved by callee.
  int saved_regs_;
};

// SIMD register allocation.
class SIMDRegisters {
 public:
  // An x64 CPU has up to 16 SIMD registers (or 32 in AVX512 mode).
  static const int kNumXRegisters = 16;
  static const int kNumZRegisters = 32;

  // Initialize SIMD registers.
  SIMDRegisters() : used_regs_(0) {}
  SIMDRegisters(const SIMDRegisters &mm) : used_regs_(mm.used_regs_) {}
  SIMDRegisters &operator=(const SIMDRegisters &mm) {
    used_regs_ = mm.used_regs_;
    return *this;
  }

  // Allocate 128-bit XMM register.
  jit::XMMRegister allocx() { return jit::XMMRegister::from_code(alloc()); }
  jit::XMMRegister try_allocx() {
    return jit::XMMRegister::from_code(try_alloc());
  }

  // Allocate 256-bit YMM register.
  jit::YMMRegister allocy() { return jit::YMMRegister::from_code(alloc()); }
  jit::YMMRegister try_allocy() {
    return jit::YMMRegister::from_code(try_alloc());
  }

  // Allocate 512-bit ZMM register.
  jit::ZMMRegister allocz(bool extended = true) {
    return jit::ZMMRegister::from_code(alloc(extended));
  }
  jit::ZMMRegister try_allocz(bool extended = true) {
    return jit::ZMMRegister::from_code(try_alloc(extended));
  }

  // Allocate SIMD register.
  int try_alloc(bool extended = false);
  int alloc(bool extended = false);

  // Mark register as being in use.
  void use(int r) { used_regs_ |= (1 << r); }
  void use(jit::XMMRegister r) { use(r.code()); }
  void use(jit::YMMRegister r) { use(r.code()); }
  void use(jit::ZMMRegister r) { use(r.code()); }

  // Mark register as being free.
  void release(int r) { used_regs_ &= ~(1 << r); }
  void release(jit::XMMRegister r) { release(r.code()); }
  void release(jit::YMMRegister r) { release(r.code()); }
  void release(jit::ZMMRegister r) { release(r.code()); }

  // Check if register is used.
  bool used(int r) const { return ((1 << r) & used_regs_) != 0; }
  bool used(jit::XMMRegister r) { return used(r.code()); }
  bool used(jit::YMMRegister r) { return used(r.code()); }
  bool used(jit::ZMMRegister r) { return used(r.code()); }

  // Reset allocated registers.
  void reset() { used_regs_ = 0; }

 private:
  // Bit mask of registers that are in use.
  uint32 used_regs_;
};

// Opmask register allocation.
class OpmaskRegisters {
 public:
  // There are 8 opmask registers (k0 to k7) where k0 is a constant register.
  static const int kNumRegisters = 8;

  // Initialize opmask registers.
  OpmaskRegisters() : used_regs_(kSpecialRegisters) {}
  OpmaskRegisters(const OpmaskRegisters &kk) : used_regs_(kk.used_regs_) {}
  OpmaskRegisters &operator=(const OpmaskRegisters &kk) {
    used_regs_ = kk.used_regs_;
    return *this;
  }

  // Allocate opmask register.
  jit::OpmaskRegister alloc();
  jit::OpmaskRegister try_alloc();

  // Mark register as being in use.
  void use(jit::OpmaskRegister k)  { used_regs_ |= (1 << k.code()); }

  // Mark register as being free.
  void release(jit::OpmaskRegister k)  { used_regs_ &= ~(1 << k.code()); }

  // Check if register is used.
  bool used(jit::OpmaskRegister k) const {
    return ((1 << k.code()) & used_regs_) != 0;
  }

  // Reset allocated registers.
  void reset() { used_regs_ = kSpecialRegisters; }

 private:
  // The k0 register is a constant registers.
  static const int kSpecialRegisters = 1 << jit::OpmaskRegister::kCode_k0;

  // Bit mask of register that are in use.
  uint32 used_regs_;
};

// Static data blocks are generated at the end of the code block. The location
// label can be used for referencing the data.
class StaticData {
 public:
  // Create new static data block.
  StaticData(int alignment = 1) : alignment_(alignment), address_(&location_) {}

  // Add data to data block.
  void AddData(const void *buffer, int size, int repeat = 1);
  template<typename T> void Add(T value, int repeat = 1) {
    AddData(&value, sizeof(T), repeat);
  }

  // Check if data block is equal to (repeated) constant.
  bool Equals(const void *data, int size, int repeat) const;

  // Generate data blocks and fix up references to it.
  void Generate(MacroAssembler *masm);

  // Location of data block.
  jit::Label *location() { return &location_; }

  // Address of data block as operand.
  const jit::Operand address() const { return address_; }

 private:
  int alignment_;            // required alignment for data
  std::vector<uint8> data_;  // data in data block
  jit::Label location_;      // location of data in generated code block
  jit::Operand address_;     // pc-relative address of data in code block
};

// Macro assembler for generating code for computations.
class MacroAssembler : public jit::Assembler {
 public:
  MacroAssembler(void *buffer, int buffer_size, const Options &options);
  ~MacroAssembler();

  // Generate function prologue.
  void Prologue();

  // Generate function epilogue.
  void Epilogue();

  // Create new static data block.
  StaticData *CreateDataBlock(int alignment = 1);

  // Find existing static data block.
  StaticData *FindDataBlock(const void *data, int size, int repeat);

  // Create new static data block with (repeated) constant.
  template<typename T> StaticData *Constant(T value, int repeat = 1) {
    StaticData *data = CreateDataBlock(repeat * sizeof(T));
    data->Add(value, repeat);
    return data;
  }

  // Find existing static data block with repeated constant or create a new one.
  template<typename T> StaticData *GetConstant(T value, int repeat = 1) {
    StaticData *data = FindDataBlock(&value, sizeof(T), repeat);
    if (data == nullptr) {
      data = CreateDataBlock(repeat * sizeof(T));
      data->Add(value, repeat);
    }
    return data;
  }

  // Get static data block for value.
  StaticData *GetData(const void *value, int size, int repeat = 1) {
    StaticData *data = FindDataBlock(value, size, repeat);
    if (data == nullptr) {
      data = CreateDataBlock(size * repeat);
      data->AddData(value, size, repeat);
    }
    return data;
  }

  // Generate static data blocks in the code buffer.
  void GenerateDataBlocks();

  // Load address of tensor.
  void LoadTensorAddress(jit::Register dst, Tensor *tensor);

  // Load address of element in tensor.
  void LoadTensorAddress(jit::Register dst, Tensor *tensor, Tensor *indices);

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

  // Load mask into opmask register with the lower n bits set. Allocates new
  // opmask register if no mask register is provided as input.
  jit::OpmaskRegister LoadMask(int n,
                               jit::OpmaskRegister k = jit::no_opmask_reg);

  // Add value to global counter.
  void UpdateCounter(int64 *counter, int64 value);

  // Start of loop. Align code and bind label.
  void LoopStart(jit::Label *label);

  // Call function with instance as argument.
  void CallInstanceFunction(void (*func)(void *), const string &symbol);

  // Increment invocation counter.
  void IncrementInvocations(int offset);

  // Generate timing for step and update instance block.
  void TimeStep(int offset, int disp);

  // Start task.
  void StartTask(int offset, int32 id, int32 index, jit::Label *entry);

  // Wait for task to complete.
  void WaitForTask(int offset);

  // Reset register usage.
  void ResetRegisterUsage();

  // General purpose register allocation.
  Registers &rr() { return rr_; }

  // SIMD register allocation.
  SIMDRegisters &mm() { return mm_; }

  // Opmask register allocation.
  OpmaskRegisters &kk() { return kk_; }

  // Returns the instance data register.
  jit::Register instance() const;

  // Compiler options.
  const Options &options() const { return options_; }

  // Runtime support functions.
  Runtime *runtime() const { return runtime_; }
  void set_runtime(Runtime *runtime) { runtime_ = runtime; }

 private:
  // Register allocation.
  Registers rr_;
  SIMDRegisters mm_;
  OpmaskRegisters kk_;

  // Static data blocks.
  std::vector<StaticData *> data_blocks_;

  // Compiler options.
  const Options &options_;

  // Runtime support functions.
  Runtime *runtime_ = nullptr;
};

}  // namespace myelin
}  // namespace sling

#endif  // SLING_MYELIN_MACRO_ASSEMBLER_H_

