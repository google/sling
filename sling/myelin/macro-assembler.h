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

#include <limits>

#include "sling/myelin/compute.h"
#include "third_party/jit/assembler.h"

namespace sling {
namespace myelin {

// Reduction operations.
enum Reduction {
  REDUCE_ADD,
  REDUCE_MUL,
  REDUCE_MIN,
  REDUCE_MAX,
  REDUCE_AND,
  REDUCE_OR
};

// Register allocation.
class Registers {
 public:
  typedef jit::Register Register;

  // An x64 CPU has 16 general 64-bit registers.
  static const int NUM_REGISTERS = 16;

  // Initialize registers.
  Registers()
      : used_regs_(PRESERVED_REGISTERS), saved_regs_(0) {}
  Registers(const Registers &rr)
      : used_regs_(rr.used_regs_), saved_regs_(rr.saved_regs_) {}
  Registers &operator=(const Registers &rr) {
    used_regs_ = rr.used_regs_;
    saved_regs_ = rr.saved_regs_;
    return *this;
  }

  // Allocate register.
  Register try_alloc();
  Register alloc();

  // Allocate preserved register.
  Register try_alloc_preserved();
  Register alloc_preserved();

  // Allocate register with preference.
  Register alloc_preferred(Register r);

  // Allocate fixed register.
  Register alloc_fixed(Register r);

  // Allocate temporary register that is neither preserved nor used as an
  // argument register.
  Register alloc_temp();

  // Allocate argument register (1-6) or return register (0).
  Register arg(int n);

  // Allocate extra preserved register. This needs to be restored and freed.
  Register alloc_extra();

  // Mark register as being in use.
  void use(int r) { used_regs_ |= (1 << r); }
  void use(Register r) { use(r.code()); }

  // Mark register as being free.
  void release(int r) { used_regs_ &= ~(1 << r); }
  void release(Register r) { release(r.code()); }

  // Check if register is used.
  bool used(int r) const { return ((1 << r) & used_regs_) != 0; }
  bool used(Register r) { return used(r.code()); }

  // Reset allocated registers.
  void reset() { used_regs_ = PRESERVED_REGISTERS & ~saved_regs_; }

  // Reserve callee-saved register for use.
  void reserve(int r);
  void reserve(Register r) { reserve(r.code()); }
  void reserve_all();

  // Free callee-saved register after it has been restored.
  void free(int r);
  void free(Register r) { free(r.code()); }

  // Declare the number of registers needed. If more than eight registers are
  // needed, an additional five callee-saved registers can be reserved.
  bool usage(int n);

  // Check if register should be saved.
  bool saved(int r) const { return ((1 << r) & saved_regs_) != 0; }
  bool saved(Register r) { return saved(r.code()); }

  // Check if register is a callee-saved register.
  static bool preserved(int r) { return ((1 << r) & PRESERVED_REGISTERS) != 0; }
  static bool preserved(Register r) { return preserved(r.code()); }

  // Check if register is an extra callee-saved register.
  static bool extra(int r) { return ((1 << r) & EXTRA_REGISTERS) != 0; }
  static bool extra(Register r) { return extra(r.code()); }

  // Return the number of free registers.
  int num_free() const;

 private:
  // Preserved registers.
  static const int PRESERVED_REGISTERS =
    1 << Register::kCode_rbx |
    1 << Register::kCode_rsp |
    1 << Register::kCode_rbp |
    1 << Register::kCode_r12 |
    1 << Register::kCode_r13 |
    1 << Register::kCode_r14 |
    1 << Register::kCode_r15;

  // Extra callee-saved registers.
  static const int EXTRA_REGISTERS =
    1 << Register::kCode_rbx |
    1 << Register::kCode_r12 |
    1 << Register::kCode_r13 |
    1 << Register::kCode_r14 |
    1 << Register::kCode_r15;

  // Bit mask of registers that are in use.
  int used_regs_;

  // Bit mask of registers that should be saved by callee.
  int saved_regs_;
};

// SIMD register allocation.
class SIMDRegisters {
 public:
  typedef jit::XMMRegister XMMRegister;
  typedef jit::YMMRegister YMMRegister;
  typedef jit::ZMMRegister ZMMRegister;

  // An x64 CPU has up to 16 SIMD registers (or 32 in AVX512 mode).
  static const int NUM_X_REGISTERS = 16;
  static const int NUM_Z_REGISTERS = 32;

  // Initialize SIMD registers.
  SIMDRegisters() : used_regs_(0) {}
  SIMDRegisters(const SIMDRegisters &mm) : used_regs_(mm.used_regs_) {}
  SIMDRegisters &operator=(const SIMDRegisters &mm) {
    used_regs_ = mm.used_regs_;
    next_ = mm.next_;
    return *this;
  }

  // Allocate 128-bit XMM register.
  XMMRegister allocx() { return XMMRegister::from_code(alloc()); }
  XMMRegister try_allocx() { return XMMRegister::from_code(try_alloc()); }

  // Allocate 256-bit YMM register.
  YMMRegister allocy() { return YMMRegister::from_code(alloc()); }
  YMMRegister try_allocy() { return YMMRegister::from_code(try_alloc()); }

  // Allocate 512-bit ZMM register.
  ZMMRegister allocz(bool extended = true) {
    return ZMMRegister::from_code(alloc(extended));
  }
  ZMMRegister try_allocz(bool extended = true) {
    return ZMMRegister::from_code(try_alloc(extended));
  }

  // Allocate SIMD register.
  int try_alloc(bool extended = false);
  int alloc(bool extended = false);

  // Mark register as being in use.
  void use(int r) { used_regs_ |= (1 << r); }
  void use(XMMRegister r) { use(r.code()); }
  void use(YMMRegister r) { use(r.code()); }
  void use(ZMMRegister r) { use(r.code()); }

  // Mark register as being free.
  void release(int r) { used_regs_ &= ~(1 << r); }
  void release(XMMRegister r) { release(r.code()); }
  void release(YMMRegister r) { release(r.code()); }
  void release(ZMMRegister r) { release(r.code()); }

  // Check if register is used.
  bool used(int r) const { return ((1 << r) & used_regs_) != 0; }
  bool used(XMMRegister r) { return used(r.code()); }
  bool used(YMMRegister r) { return used(r.code()); }
  bool used(ZMMRegister r) { return used(r.code()); }

  // Reset allocated registers.
  void reset() { used_regs_ = 0; next_ = 0; }

 private:
  // Bit mask of registers that are in use.
  uint32 used_regs_;

  // Next register to allocate in rotation.
  int next_ = 0;
};

// Opmask register allocation.
class OpmaskRegisters {
 public:
  typedef jit::OpmaskRegister OpmaskRegister;

  // There are 8 opmask registers (k0 to k7) where k0 is a constant register.
  static const int NUM_REGISTERS = 8;

  // Initialize opmask registers.
  OpmaskRegisters() : used_regs_(kSpecialRegisters) {}
  OpmaskRegisters(const OpmaskRegisters &kk) : used_regs_(kk.used_regs_) {}
  OpmaskRegisters &operator=(const OpmaskRegisters &kk) {
    used_regs_ = kk.used_regs_;
    return *this;
  }

  // Allocate opmask register.
  OpmaskRegister alloc();
  OpmaskRegister try_alloc();

  // Mark register as being in use.
  void use(OpmaskRegister k)  { used_regs_ |= (1 << k.code()); }

  // Mark register as being free.
  void release(OpmaskRegister k)  { used_regs_ &= ~(1 << k.code()); }

  // Check if register is used.
  bool used(OpmaskRegister k) const {
    return ((1 << k.code()) & used_regs_) != 0;
  }

  // Reset allocated registers.
  void reset() { used_regs_ = kSpecialRegisters; }

 private:
  // The k0 register is a constant registers.
  static const int kSpecialRegisters = 1 << OpmaskRegister::kCode_k0;

  // Bit mask of register that are in use.
  uint32 used_regs_;
};

// Static data blocks are generated at the end of the code block. The location
// label can be used for referencing the data.
class StaticData {
 public:
  typedef jit::Label Label;
  typedef jit::Operand Operand;

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
  Label *location() { return &location_; }

  // Address of data block as operand.
  const Operand &address() const { return address_; }

  // External symbol name.
  const string &symbol() const { return symbol_; }
  void set_symbol(const string &symbol) { symbol_ = symbol; }

 private:
  int alignment_;            // required alignment for data
  std::vector<uint8> data_;  // data in data block
  Label location_;           // location of data in generated code block
  Operand address_;          // pc-relative address of data in code block
  string symbol_;            // external symbol name
};

// Macro assembler for generating code for computations.
class MacroAssembler : public jit::Assembler {
 public:
  typedef jit::Register Register;
  typedef jit::XMMRegister XMMRegister;
  typedef jit::YMMRegister YMMRegister;
  typedef jit::ZMMRegister ZMMRegister;
  typedef jit::OpmaskRegister OpmaskRegister;
  typedef jit::Label Label;
  typedef jit::Operand Operand;

  MacroAssembler(void *buffer, int buffer_size, const Options &options);
  ~MacroAssembler();

  // Allocate registers for function prologue/epilogue.
  void AllocateFunctionRegisters();

  // Generate function prologue.
  void Prologue();

  // Generate function epilogue.
  void Epilogue();

  // Create new static data block.
  StaticData *CreateDataBlock(int alignment = 1);

  // Find existing static data block.
  StaticData *FindDataBlock(const void *data, int size, int repeat = 1);
  StaticData *FindDataBlock(const void *data, int size, const string &symbol);

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

  // Get static data block for external reference.
  StaticData *GetExtern(const string &symbol, const void *address) {
    int size = sizeof(void *);
    StaticData *data = FindDataBlock(&address, size, symbol);
    if (data == nullptr) {
      data = CreateDataBlock(size);
      data->AddData(&address, size);
      data->set_symbol(symbol);
    }
    return data;
  }

  // Get data block for minimum value for type.
  template<typename T> StaticData *MinVal(int repeat = 1) {
    return GetConstant<T>(std::numeric_limits<T>::lowest(), repeat);
  }

  // Get data block for maximum value for type.
  template<typename T> StaticData *MaxVal(int repeat = 1) {
    return GetConstant<T>(std::numeric_limits<T>::max(), repeat);
  }

  // Generate static data blocks in the code buffer.
  void GenerateDataBlocks();

  // Load address of tensor.
  void LoadTensorAddress(Register dst, Tensor *tensor);

  // Load address of element in tensor.
  void LoadTensorAddress(Register dst, Tensor *tensor, Tensor *indices);

  // Load address of tensor on device.
  void LoadTensorDeviceAddress(Register dst, Tensor *tensor);

  // Load size of dynamic tensor (e.g. channel) and multiply with scalar.
  void LoadDynamicSize(Register dst, Tensor *tensor, int scalar = 1);

  // Emit breakpoint.
  void Breakpoint() { int3(); }

  // Copy memory.
  void Copy(Register dst, int ddisp,
            Register src, int sdisp,
            int size);

  // Load integer from array into 64-bit register.
  void LoadInteger(Register dst, Register base, Register index, Type type);

  // Store integer into array from 64-bit register.
  void StoreInteger(Register base, Register index, Register src, Type type);

  // Multiply register with constant.
  void Multiply(Register reg, int64 scalar);

  // Load mask into opmask register with the lower n bits set. Allocates new
  // opmask register if no mask register is provided as input.
  OpmaskRegister LoadMask(int n, OpmaskRegister k = jit::no_opmask_reg);

  // Combine pairwise elements with reduction operator, i.e. acc = acc <op> r.
  void Accumulate(Reduction op, Type type, XMMRegister acc, XMMRegister r);
  void Accumulate(Reduction op, Type type, YMMRegister acc, YMMRegister r);
  void Accumulate(Reduction op, Type type, ZMMRegister acc, ZMMRegister r);
  void Accumulate(Reduction op, Type type, XMMRegister acc, const Operand &src);
  void Accumulate(Reduction op, Type type, YMMRegister acc, const Operand &src);
  void Accumulate(Reduction op, Type type, ZMMRegister acc, const Operand &src,
                  OpmaskRegister k = jit::no_opmask_reg);

  // Reduction operation over all elements in an accumulator register using an
  // auxiliary register.
  void Reduce(Reduction op, Type type, XMMRegister acc, XMMRegister aux);
  void Reduce(Reduction op, Type type, YMMRegister acc, YMMRegister aux);
  void Reduce(Reduction op, Type type, ZMMRegister acc, ZMMRegister aux);

  // Add value to global counter.
  void UpdateCounter(int64 *counter, int64 value);

  // Start of loop. Align code and bind label.
  void LoopStart(Label *label);

  // Call function with instance as argument.
  void CallInstanceFunction(void (*func)(void *), const string &symbol);

  // Increment invocation counter.
  void IncrementInvocations(int offset);

  // Generate timing for step and update instance block.
  void TimeStep(int offset, int disp);

  // Start task.
  void StartTask(int offset, int32 id, int32 index, Label *entry);

  // Wait for task to complete.
  void WaitForTask(int offset);

  // Wait for main task to complete.
  void WaitForMainTask();

  // Reset register usage.
  void ResetRegisterUsage();

  // Call to external function.
  void call_extern(const void *func, const string &symbol) {
    if (options_.pic) {
      call(func, symbol);
    } else {
      call(GetExtern(symbol, func)->address());
    }
  }

  // Type-dependent instructions.
  void vpermil(Type type, XMMRegister dst, XMMRegister src, int8_t imm8);
  void vpermil(Type type, YMMRegister dst, YMMRegister src, int8_t imm8);
  void vpermil(Type type, ZMMRegister dst, ZMMRegister src, int8_t imm8);

  // General purpose register allocation.
  Registers &rr() { return rr_; }

  // SIMD register allocation.
  SIMDRegisters &mm() { return mm_; }

  // Opmask register allocation.
  OpmaskRegisters &kk() { return kk_; }

  // Returns the instance data register.
  Register instance() const;

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

