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

#ifndef SLING_MYELIN_SIMD_ASSEMBLER_H_
#define SLING_MYELIN_SIMD_ASSEMBLER_H_

#include "sling/myelin/macro-assembler.h"

namespace sling {
namespace myelin {

// Code generator for SIMD vector instructions.
class SIMDGenerator {
 public:
  SIMDGenerator(MacroAssembler *masm, bool aligned)
      : masm_(masm), aligned_(aligned) {}
  virtual ~SIMDGenerator() = default;

  // Number of bytes per vector register.
  virtual int VectorBytes() = 0;

  // Number of elements per vector register.
  virtual int VectorSize() = 0;

  // Returns false if computations should not be unrolled.
  virtual bool SupportsUnroll();

  // Allocate SIMD register.
  virtual int Alloc() = 0;

  // Move value from one regiser to another.
  virtual void Move(int dst, int src) = 0;

  // Load memory into register.
  virtual void Load(int dst, const jit::Operand &src) = 0;

  // Store register into memory.
  virtual void Store(const jit::Operand &dst, int src) = 0;

  // Broadcast value to all elements of register.
  virtual void Broadcast(int dst, int src);
  virtual void Broadcast(int dst, const jit::Operand &src);

  // Clear register.
  virtual void Zero(int r) = 0;

  // Load neutral value for op into register.
  virtual void LoadNeutral(Reduction op, int r);

  // Add src1 and src2 and store it in dst.
  virtual void Add(int dst, int src1, int src2) = 0;
  virtual void Add(int dst, int src1, const jit::Operand &src2) = 0;

  // Multiply src1 and src2 and store it in dst.
  virtual void Mul(int dst, int src1, const jit::Operand &src2) = 0;

  // Multiply src1 and src2 and add it to dst. If the retain flag is false the
  // contents of src1 can possibly be destroyed.
  virtual void MulAdd(int dst, int src1, const jit::Operand &src2,
                      bool retain) = 0;

  // Accumulate value in src into acc.
  virtual void Accumulate(Reduction op, int acc, int src);
  virtual void Accumulate(Reduction op, int acc, const jit::Operand &src);

  // Horizontal sum of all elements in register.
  virtual void Sum(int r);

  // Horizontal reduction of all elements in register.
  virtual void Reduce(Reduction op, int r);

  // Some vector instructions support masking (e.g. AVX512) that allow loading
  // and storing partial results.
  virtual bool SupportsMasking();
  virtual void SetMask(int bits);
  virtual void MaskedLoad(int dst, const jit::Operand &src);
  virtual void MaskedStore(const jit::Operand &dst, int src);
  virtual void MaskedAdd(int dst, int src1, const jit::Operand &src2);
  virtual void MaskedMul(int dst, int src1, const jit::Operand &src2);
  virtual void MaskedMulAdd(int dst, int src1, const jit::Operand &src2);
  virtual void MaskedAccumulate(Reduction op, int acc, const jit::Operand &src);

 protected:
  // Get neutral element for operation and type. Returns null for zero.
  StaticData *NeutralElement(Reduction op, Type type, int repeat = 1);

  // Get register from register code.
  jit::Register reg(int r) { return jit::Register::from_code(r); }
  jit::XMMRegister xmm(int r) { return jit::XMMRegister::from_code(r); }
  jit::YMMRegister ymm(int r) { return jit::YMMRegister::from_code(r); }
  jit::ZMMRegister zmm(int r) { return jit::ZMMRegister::from_code(r); }

  MacroAssembler *masm_;  // assembler for code generation
  bool aligned_;          // aligned load/store allowed
};

// Assembler for SIMD vector code generation. The main generator is used for
// the (unrolled) bulk of the vector operation. The generator cascade is
// used for successively smaller vector registers for handling the remaining
// elements ending with a handler for scalars.
class SIMDAssembler {
 public:
  SIMDAssembler(MacroAssembler *masm, Type type, bool aligned);
  ~SIMDAssembler();

  // The first generator is the main generator.
  SIMDGenerator *main() const { return cascade_.front(); }

  // The last generator is for scalars.
  SIMDGenerator *scalar() const { return cascade_.back(); }

  // Generator cascade from bigger to smaller vector registers.
  const std::vector<SIMDGenerator *> cascade() const { return cascade_; }

  // Allocate SIMD register(s).
  int alloc() { return main()->Alloc(); }
  std::vector<int> alloc(int n) {
    std::vector<int> regs(n);
    for (auto &r : regs) r = alloc();
    return regs;
  }

  // Vertical sum of list of registers. Result is in the first register.
  void Sum(const std::vector<int> &regs);

  // Vertical reduction of list of registers. Result is in the first register.
  void Reduce(Reduction op, const std::vector<int> &regs);

  // Check if type is supported.
  static bool Supports(Type type);

  // Returns the number of regular register used by SIMD generators.
  static int RegisterUsage(Type type);

  // Return biggest vector size in bytes.
  static int VectorBytes(Type type);

  const string &name() const { return name_; }

 private:
  // Add generator to cascade.
  void add(SIMDGenerator *generator) { cascade_.push_back(generator); }

  // Generator name.
  string name_;

  // Code generator cascade.
  std::vector<SIMDGenerator *> cascade_;
};

// A SIMD strategy breaks down operations on general vectors into a number
// of phases. Each phase is a repeated, unrolled/masked operation on parts
// of the vector.
class SIMDStrategy {
 public:
  // Maximum number of loop unrolls.
  static const int kMaxUnrolls = 4;

  // A phase is a (repeated) (unrolled) (masked) operation on parts of a vector.
  struct Phase {
    Phase(SIMDGenerator *generator) : generator(generator) {}

    int offset = 0;             // start offset in vector
    int unrolls = 1;            // number of unrolls of operation
    int repeat = 1;             // number of repeats of unrolled operation
    int masked = 0;             // number of elements for masked operation
    int regs = 1;               // number of registers used for operation
    SIMDGenerator *generator;   // code generator for phase
  };

  // Compute a strategy for processing a vector of a certain size.
  SIMDStrategy(SIMDAssembler *sasm, int size);

  // Maximum number of unrolls.
  int MaxUnrolls();

  // Pre-load masks for all masked phases.
  void PreloadMasks();

  const std::vector<Phase> &phases() const { return phases_; }

 private:
  // Vector processing phases.
  std::vector<Phase> phases_;
};

}  // namespace myelin
}  // namespace sling

#endif  // SLING_MYELIN_SIMD_ASSEMBLER_H_
