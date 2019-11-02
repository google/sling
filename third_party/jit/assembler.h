// Copyright (c) 1994-2006 Sun Microsystems Inc.
// All Rights Reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// - Redistributions of source code must retain the above copyright notice,
// this list of conditions and the following disclaimer.
//
// - Redistribution in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// - Neither the name of Sun Microsystems or the names of contributors may
// be used to endorse or promote products derived from this software without
// specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
// IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
// THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

// The original source code covered by the above license above has been
// modified significantly by Google Inc.
// Copyright 2012 the V8 project authors. All rights reserved.
// Copyright 2017 Google Inc. All rights reserved.

#ifndef JIT_ASSEMBLER_H_
#define JIT_ASSEMBLER_H_

#include "sling/base/logging.h"
#include "third_party/jit/code.h"
#include "third_party/jit/cpu.h"
#include "third_party/jit/instructions.h"
#include "third_party/jit/memory.h"
#include "third_party/jit/registers.h"
#include "third_party/jit/types.h"

namespace sling {
namespace jit {

class Assembler;

// Machine instruction immediates.
class Immediate {
 public:
  explicit Immediate(int32_t value) : value_(value) {}

 private:
  int32_t value_;

  friend class Assembler;
};

// Machine instruction operands.
enum ScaleFactor {
  times_1 = 0,
  times_2 = 1,
  times_4 = 2,
  times_8 = 3,
  times_int_size = times_4,
  times_pointer_size = (kPointerSize == 8) ? times_8 : times_4
};

enum LoadMode {
  full = 0,
  broadcast = 1,
};

class Operand {
 public:
  // [base + disp/r]
  explicit Operand(Register base, int32_t disp = 0, LoadMode load = full);

  // [base + index*scale + disp/r]
  Operand(Register base,
          Register index,
          ScaleFactor scale = times_1,
          int32_t disp = 0,
          LoadMode load = full);

  // [index*scale + disp/r]
  Operand(Register index,
          ScaleFactor scale,
          int32_t disp = 0,
          LoadMode load = full);

  // [rip + disp/r]
  explicit Operand(Label *label, LoadMode load = full);

  // Whether the generated instruction will have a REX prefix.
  bool requires_rex() const { return rex_ != 0; }

  // Size of the ModR/M, SIB and displacement parts of the generated
  // instruction.
  int operand_size() const { return len_; }

  // Whather the operand should be broadcast to all destination vector elements.
  LoadMode load() const { return load_; }

 private:
  byte rex_;       // register extension
  byte buf_[9];    // operand encoding
  byte len_;       // operand encoding size
  LoadMode load_;  // broadcast operand (avx512)

  // Set the ModR/M byte without an encoded 'reg' register. The
  // register is encoded later as part of the emit_operand operation.
  // set_modrm can be called before or after set_sib and set_disp*.
  void set_modrm(int mod, Register rm_reg) {
    DCHECK(is_uint2(mod));
    buf_[0] = mod << 6 | rm_reg.low_bits();
    // Set REX.B to the high bit of rm.code().
    rex_ |= rm_reg.high_bit();
  }

  // Set the SIB byte if one is needed. Sets the length to 2 rather than 1.
  void set_sib(ScaleFactor scale, Register index, Register base) {
    DCHECK(len_ == 1);
    DCHECK(is_uint2(scale));
    // Use SIB with no index register only for base rsp or r12. Otherwise we
    // would skip the SIB byte entirely.
    DCHECK(!index.is(rsp) || base.is(rsp) || base.is(r12));
    buf_[1] = (scale << 6) | (index.low_bits() << 3) | base.low_bits();
    rex_ |= index.high_bit() << 1 | base.high_bit();
    len_ = 2;
  }

  // Adds operand displacement fields (offsets added to the memory address).
  // Needs to be called after set_sib, not before it.
  void set_disp8(int disp) {
    DCHECK(is_int8(disp));
    DCHECK(len_ == 1 || len_ == 2);
    int8_t *p = reinterpret_cast<int8_t *>(&buf_[len_]);
    *p = disp;
    len_ += sizeof(int8_t);
  }

  void set_disp32(int disp) {
    DCHECK(len_ == 1 || len_ == 2);
    int32_t *p = reinterpret_cast<int32_t *>(&buf_[len_]);
    *p = disp;
    len_ += sizeof(int32_t);
  }

  void set_disp64(int64_t disp) {
    DCHECK_EQ(1, len_);
    int64_t *p = reinterpret_cast<int64_t *>(&buf_[len_]);
    *p = disp;
    len_ += sizeof(disp);
  }

  friend class Assembler;
};

// Operation masking.
enum MaskOp {
  merging = 0,
  zeroing = 1,
};

class Mask {
 public:
  Mask(OpmaskRegister reg, MaskOp op) : reg_(reg), op_(op) {}

  OpmaskRegister reg() const { return reg_; }
  MaskOp op() const { return op_; }

 private:
  // Mask register.
  OpmaskRegister reg_;

  // Masking operation.
  MaskOp op_;
};

const Mask nomask = Mask(k0, merging);

// Assembler for generating Intel x86-64 machine code.
class Assembler : public CodeGenerator {
 public:
  // Create an assembler. Instructions are emitted into a buffer, with the
  // instructions starting from the beginning.
  Assembler(void *buffer, int buffer_size);

  // Check if CPU feature is enabled by assembler.
  bool Enabled(CpuFeature f) {
    return (cpu_features_ & (1u << f)) != 0;
  }

  // Enable CPU feature.
  void Enable(CpuFeature f) {
    cpu_features_ |= (1u << f);
  }

  // Disable CPU feature.
  void Disable(CpuFeature f) {
    cpu_features_ &= ~(1u << f);
  }

  // One byte prefix for a short conditional jump.
  static const byte kJccShortPrefix = 0x70;
  static const byte kJncShortOpcode = kJccShortPrefix | not_carry;
  static const byte kJcShortOpcode = kJccShortPrefix | carry;
  static const byte kJnzShortOpcode = kJccShortPrefix | not_zero;
  static const byte kJzShortOpcode = kJccShortPrefix | zero;

  // VEX prefix encodings.
  enum SIMDPrefix { kNone = 0x0, k66 = 0x1, kF3 = 0x2, kF2 = 0x3 };
  enum VectorLength { kL128 = 0x0, kL256 = 0x4, kLIG = kL128, kLZ = kL128 };
  enum VexW { kW0 = 0x0, kW1 = 0x80, kWIG = kW0 };
  enum LeadingOpcode { k0F = 0x1, k0F38 = 0x2, k0F3A = 0x3 };

  // EVEX prefix encodings.
  enum EvexFlags {
    EVEX_ENDS   = (1 << 0),   // non-destructive source
    EVEX_ENDD   = (1 << 1),   // non-destructive destination
    EVEX_EDDS   = (1 << 2),   // destructive destination and source

    EVEX_LIG    = (1 << 3),   // EVEX.LL ignored
    EVEX_L128   = (1 << 4),   // EVEX.LL 128-bit operands
    EVEX_L256   = (1 << 5),   // EVEX.LL 256-bit operands
    EVEX_L512   = (1 << 6),   // EVEX.LL 512-bit operands

    EVEX_P66    = (1 << 7),   // EVEX.PP 0x66 prefix
    EVEX_PF2    = (1 << 8),   // EVEX.PP 0xF2 prefix
    EVEX_PF3    = (1 << 9),   // EVEX.PP 0xF3 prefix

    EVEX_M0F    = (1 << 10),  // VEX.MM 0x0F leading opcode
    EVEX_M0F38  = (1 << 11),  // VEX.MM 0x0F38 leading opcode
    EVEX_M0F3A  = (1 << 12),  // VEX.MM 0x0F3A leading opcode

    EVEX_W0     = (1 << 13),  // EVEX.W=0
    EVEX_W1     = (1 << 14),  // EVEX.W=1
    EVEX_WIG    = (1 << 15),  // EVEX.W ignored

    EVEX_BCST   = (1 << 16),  // EVEX.B broadcast
    EVEX_ER     = (1 << 17),  // static-rounding
    EVEX_SAE    = (1 << 18),  // suppress all exceptions

    EVEX_R0     = (1 << 19),   // rounding mode bit 0
    EVEX_R1     = (1 << 20),   // rounding mode bit 1

    EVEX_IMM    = (1 << 21),  // immediate operand

    EVEX_DT1    = (1 << 22),   // 1-byte data type
    EVEX_DT2    = (1 << 23),   // 2-byte data type
    EVEX_DT4    = (1 << 24),   // 4-byte data type
    EVEX_DT8    = (1 << 25),   // 8-byte data type
    EVEX_DT16   = (1 << 26),   // 16-byte data type
    EVEX_DT32   = (1 << 27),   // 32-byte data type
    EVEX_DT64   = (1 << 28),   // 64-byte data type

    EVEX_BT4    = (1 << 29),   // 4-byte broadcast data type
    EVEX_BT8    = (1 << 30),   // 8-byte broadcast data type
  };

  // Code generation
  //
  // Function names correspond one-to-one to x64 instruction mnemonics.
  // Unless specified otherwise, instructions operate on 64-bit operands.
  //
  // If we need versions of an assembly instruction that operate on different
  // width arguments, we add a single-letter suffix specifying the width.
  // This is done for the following instructions: mov, cmp, inc, dec,
  // add, sub, and test.
  // There are no versions of these instructions without the suffix.
  // - Instructions on 8-bit (byte) operands/registers have a trailing 'b'.
  // - Instructions on 16-bit (word) operands/registers have a trailing 'w'.
  // - Instructions on 32-bit (doubleword) operands/registers use 'l'.
  // - Instructions on 64-bit (quadword) operands/registers use 'q'.

  static_assert(kPointerSize == kInt64Size || kPointerSize == kInt32Size,
                "unsupported pointer size");

#define DECLARE_INSTRUCTION(instruction)                \
  template<class P1>                                    \
  void instruction##p(P1 p1) {                          \
    emit_##instruction(p1, kPointerSize);               \
  }                                                     \
                                                        \
  template<class P1>                                    \
  void instruction##l(P1 p1) {                          \
    emit_##instruction(p1, kInt32Size);                 \
  }                                                     \
                                                        \
  template<class P1>                                    \
  void instruction##q(P1 p1) {                          \
    emit_##instruction(p1, kInt64Size);                 \
  }                                                     \
                                                        \
  template<class P1, class P2>                          \
  void instruction##l(P1 p1, P2 p2) {                   \
    emit_##instruction(p1, p2, kInt32Size);             \
  }                                                     \
                                                        \
  template<class P1, class P2>                          \
  void instruction##q(P1 p1, P2 p2) {                   \
    emit_##instruction(p1, p2, kInt64Size);             \
  }                                                     \
                                                        \
  template<class P1, class P2, class P3>                \
  void instruction##p(P1 p1, P2 p2, P3 p3) {            \
    emit_##instruction(p1, p2, p3, kPointerSize);       \
  }                                                     \
                                                        \
  template<class P1, class P2, class P3>                \
  void instruction##l(P1 p1, P2 p2, P3 p3) {            \
    emit_##instruction(p1, p2, p3, kInt32Size);         \
  }                                                     \
                                                        \
  template<class P1, class P2, class P3>                \
  void instruction##q(P1 p1, P2 p2, P3 p3) {            \
    emit_##instruction(p1, p2, p3, kInt64Size);         \
  }
  ASSEMBLER_INSTRUCTION_LIST(DECLARE_INSTRUCTION)
#undef DECLARE_INSTRUCTION

  // Insert the smallest number of nop instructions
  // possible to align the pc offset to a multiple
  // of m, where m must be a power of 2.
  void Align(int m);

  // Insert the smallest number of zero bytes possible to align the pc offset
  // to a mulitple of m. m must be a power of 2 (>= 2).
  void DataAlign(int m);
  void Nop(int bytes = 1);

  // Aligns code to something that's optimal for a jump target for the platform.
  void CodeTargetAlign();

  // Stack
  void pushfq();
  void popfq();
  void pushq(Immediate value);

  // Push a 32 bit integer, and guarantee that it is actually pushed as a
  // 32 bit value, the normal push will optimize the 8 bit case.
  void pushq_imm32(int32_t imm32);
  void pushq(Register src);
  void pushq(const Operand &src);

  void popq(Register dst);
  void popq(const Operand &dst);

  void enter(Immediate size);
  void leave();

  // Moves
  void movb(Register dst, const Operand &src);
  void movb(Register dst, Immediate imm);
  void movb(const Operand &dst, Register src);
  void movb(const Operand &dst, Immediate imm);

  // Move the low 16 bits of a 64-bit register value to a 16-bit
  // memory location.
  void movw(Register dst, const Operand &src);
  void movw(Register dst, Immediate imm);
  void movw(const Operand &dst, Register src);
  void movw(const Operand &dst, Immediate imm);

  // Move the offset of the label location relative to the current
  // position (after the move) to the destination.
  void movl(const Operand &dst, Label *src);

  // Loads a pointer into a register.
  void movp(Register dst, const void *ptr);

  // Loads a 64-bit immediate into a register.
  void movq(Register dst, int64_t value);
  void movq(Register dst, uint64_t value);

  void movsxbl(Register dst, Register src);
  void movsxbl(Register dst, const Operand &src);
  void movsxbq(Register dst, Register src);
  void movsxbq(Register dst, const Operand &src);
  void movsxwl(Register dst, Register src);
  void movsxwl(Register dst, const Operand &src);
  void movsxwq(Register dst, Register src);
  void movsxwq(Register dst, const Operand &src);
  void movsxlq(Register dst, Register src);
  void movsxlq(Register dst, const Operand &src);

  // Repeated moves.
  void repmovsb();
  void repmovsw();
  void repmovsp() { emit_repmovs(kPointerSize); }
  void repmovsl() { emit_repmovs(kInt32Size); }
  void repmovsq() { emit_repmovs(kInt64Size); }

  // Repeated stores.
  void repstosb();
  void repstosw();
  void repstosp() { emit_repstos(kPointerSize); }
  void repstosl() { emit_repstos(kInt32Size); }
  void repstosq() { emit_repstos(kInt64Size); }

  // Loads an external reference into a register.
  void load_extern(Register dst, const void *ptr, const string &symbol,
                   bool pic = false);

  // Instruction to load from an immediate 64-bit pointer into RAX.
  void load_rax(const void *ptr);

  // Conditional moves.
  void cmovq(Condition cc, Register dst, Register src);
  void cmovq(Condition cc, Register dst, const Operand &src);
  void cmovl(Condition cc, Register dst, Register src);
  void cmovl(Condition cc, Register dst, const Operand &src);

  void cmpb(Register dst, Immediate src) {
    immediate_arithmetic_op_8(0x7, dst, src);
  }

  void cmpb_al(Immediate src);

  void cmpb(Register dst, Register src) {
    arithmetic_op_8(0x3A, dst, src);
  }
  void cmpb(Register dst, const Operand &src) {
    arithmetic_op_8(0x3A, dst, src);
  }
  void cmpb(const Operand &dst, Register src) {
    arithmetic_op_8(0x38, src, dst);
  }
  void cmpb(const Operand &dst, Immediate src) {
    immediate_arithmetic_op_8(0x7, dst, src);
  }

  void cmpw(const Operand &dst, Immediate src) {
    immediate_arithmetic_op_16(0x7, dst, src);
  }
  void cmpw(Register dst, Immediate src) {
    immediate_arithmetic_op_16(0x7, dst, src);
  }
  void cmpw(Register dst, const Operand &src) {
    arithmetic_op_16(0x3B, dst, src);
  }
  void cmpw(Register dst, Register src) {
    arithmetic_op_16(0x3B, dst, src);
  }
  void cmpw(const Operand &dst, Register src) {
    arithmetic_op_16(0x39, src, dst);
  }

  void testb(Register reg, const Operand &op) { testb(op, reg); }

  void testw(Register reg, const Operand &op) { testw(op, reg); }

  void andb(Register dst, Immediate src) {
    immediate_arithmetic_op_8(0x4, dst, src);
  }

  // Lock prefix.
  void lock();

  void xchgb(Register reg, const Operand &op);
  void xchgw(Register reg, const Operand &op);

  void cmpxchgb(const Operand &dst, Register src);
  void cmpxchgw(const Operand &dst, Register src);

  // Sign-extends rax into rdx:rax.
  void cqo();

  // Sign-extends eax into edx:eax.
  void cdq();

  // Sign-extends ax into dx:ax.
  void cbw();

  // Multiply eax by src, put the result in edx:eax.
  void mull(Register src);
  void mull(const Operand &src);

  // Multiply rax by src, put the result in rdx:rax.
  void mulq(Register src);

#define DECLARE_SHIFT_INSTRUCTION(instruction, subcode)                       \
  void instruction##p(Register dst, Immediate imm8) {                         \
    shift(dst, imm8, subcode, kPointerSize);                                  \
  }                                                                           \
                                                                              \
  void instruction##l(Register dst, Immediate imm8) {                         \
    shift(dst, imm8, subcode, kInt32Size);                                    \
  }                                                                           \
                                                                              \
  void instruction##q(Register dst, Immediate imm8) {                         \
    shift(dst, imm8, subcode, kInt64Size);                                    \
  }                                                                           \
                                                                              \
  void instruction##p(Operand dst, Immediate imm8) {                          \
    shift(dst, imm8, subcode, kPointerSize);                                  \
  }                                                                           \
                                                                              \
  void instruction##l(Operand dst, Immediate imm8) {                          \
    shift(dst, imm8, subcode, kInt32Size);                                    \
  }                                                                           \
                                                                              \
  void instruction##q(Operand dst, Immediate imm8) {                          \
    shift(dst, imm8, subcode, kInt64Size);                                    \
  }                                                                           \
                                                                              \
  void instruction##p_cl(Register dst) { shift(dst, subcode, kPointerSize); } \
                                                                              \
  void instruction##l_cl(Register dst) { shift(dst, subcode, kInt32Size); }   \
                                                                              \
  void instruction##q_cl(Register dst) { shift(dst, subcode, kInt64Size); }   \
                                                                              \
  void instruction##p_cl(Operand dst) { shift(dst, subcode, kPointerSize); }  \
                                                                              \
  void instruction##l_cl(Operand dst) { shift(dst, subcode, kInt32Size); }    \
                                                                              \
  void instruction##q_cl(Operand dst) { shift(dst, subcode, kInt64Size); }
  SHIFT_INSTRUCTION_LIST(DECLARE_SHIFT_INSTRUCTION)
#undef DECLARE_SHIFT_INSTRUCTION

  // Shifts dst:src left by cl bits, affecting only dst.
  void shld(Register dst, Register src);

  // Shifts src:dst right by cl bits, affecting only dst.
  void shrd(Register dst, Register src);

  void store_rax(const void *dst);

  void addb(Register dst, Register src) { emit_add(dst, src, 1); }
  void addb(Register dst, const Operand &src) { emit_add(dst, src, 1); }
  void addb(const Operand &dst, Register src) { emit_add(dst, src, 1); }
  void addb(const Operand &dst, Immediate src) { emit_add(dst, src, 1); }
  void addb(Register dst, Immediate src) { emit_add(dst, src, 1); }

  void addw(Register dst, Register src) { emit_add(dst, src, 2); }
  void addw(Register dst, const Operand &src) { emit_add(dst, src, 2); }
  void addw(const Operand &dst, Register src) { emit_add(dst, src, 2); }
  void addw(const Operand &dst, Immediate src) { emit_add(dst, src, 2); }
  void addw(Register dst, Immediate src) { emit_add(dst, src, 2); }

  void subb(Register dst, Register src) { emit_sub(dst, src, 1); }
  void subb(Register dst, const Operand &src) { emit_sub(dst, src, 1); }
  void subb(const Operand &dst, Register src) { emit_sub(dst, src, 1); }
  void subb(const Operand &dst, Immediate src) { emit_sub(dst, src, 1); }
  void subb(Register dst, Immediate src) { emit_sub(dst, src, 1); }

  void subw(Register dst, Register src) { emit_sub(dst, src, 2); }
  void subw(Register dst, const Operand &src) { emit_sub(dst, src, 2); }
  void subw(const Operand &dst, Register src) { emit_sub(dst, src, 2); }
  void subw(const Operand &dst, Immediate src) { emit_sub(dst, src, 2); }
  void subw(Register dst, Immediate src) { emit_sub(dst, src, 2); }

  void imulb(Register src) { emit_imul(src, 1); }
  void imulb(const Operand &src) { emit_imul(src, 1); }
  void imulw(Register src) { emit_imul(src, 2); }
  void imulw(const Operand &src) { emit_imul(src, 2); }
  void imulw(Register dst, Register src) { emit_imul(dst, src, 2); }
  void imulw(Register dst, const Operand &src) { emit_imul(dst, src, 2); }

  void imulw(Register dst, Register src, Immediate imm) {
    emit_imul(dst, src, imm, 2);
  }
  void imulw(Register dst, const Operand &src, Immediate imm) {
    emit_imul(dst, src, imm, 2);
  }
  void imull(Register dst, Register src, Immediate imm) {
    emit_imul(dst, src, imm, 4);
  }
  void imull(Register dst, const Operand &src, Immediate imm) {
    emit_imul(dst, src, imm, 4);
  }
  void imulq(Register dst, Register src, Immediate imm) {
    emit_imul(dst, src, imm, 8);
  }
  void imulq(Register dst, const Operand &src, Immediate imm) {
    emit_imul(dst, src, imm, 8);
  }

  void idivb(Register src) { emit_idiv(src, 1); }
  void idivb(const Operand &src) { emit_idiv(src, 1); }
  void idivw(Register src) { emit_idiv(src, 2); }
  void idivw(const Operand &src) { emit_idiv(src, 2); }

  void incb(Register dst) { emit_inc(dst, 1); }
  void incb(const Operand &dst) { emit_inc(dst, 1); }
  void incw(Register dst) { emit_inc(dst, 2); }
  void incw(const Operand &dst) { emit_inc(dst, 2); }

  void decb(Register dst);
  void decb(const Operand &dst);
  void decw(Register dst) { emit_dec(dst, 2); }
  void decw(const Operand &dst) { emit_dec(dst, 2); }

  void shlb(Register dst, Immediate imm8) { shift(dst, imm8, 0x4, 1); }
  void shlb(const Operand &dst, Immediate imm8) { shift(dst, imm8, 0x4, 1); }
  void shlw(Register dst, Immediate imm8) { shift(dst, imm8, 0x4, 2); }
  void shlw(const Operand &dst, Immediate imm8) { shift(dst, imm8, 0x4, 2); }

  void shrb(Register dst, Immediate imm8) { shift(dst, imm8, 0x5, 1); }
  void shrb(const Operand &dst, Immediate imm8) { shift(dst, imm8, 0x5, 1); }
  void shrw(Register dst, Immediate imm8) { shift(dst, imm8, 0x5, 2); }
  void shrw(const Operand &dst, Immediate imm8) { shift(dst, imm8, 0x5, 2); }

  void salb(Register dst, Immediate imm8) { shift(dst, imm8, 0x4, 1); }
  void salb(const Operand &dst, Immediate imm8) { shift(dst, imm8, 0x4, 1); }
  void salw(Register dst, Immediate imm8) { shift(dst, imm8, 0x4, 2); }
  void salw(const Operand &dst, Immediate imm8) { shift(dst, imm8, 0x4, 2); }

  void sarb(Register dst, Immediate imm8) { shift(dst, imm8, 0x7, 1); }
  void sarb(const Operand &dst, Immediate imm8) { shift(dst, imm8, 0x7, 1); }
  void sarw(Register dst, Immediate imm8) { shift(dst, imm8, 0x7, 2); }
  void sarw(const Operand &dst, Immediate imm8) { shift(dst, imm8, 0x7, 2); }

  void testb(Register dst, Register src);
  void testb(Register reg, Immediate mask);
  void testb(const Operand &op, Immediate mask);
  void testb(const Operand &op, Register reg);

  void testw(Register dst, Register src);
  void testw(Register reg, Immediate mask);
  void testw(const Operand &op, Immediate mask);
  void testw(const Operand &op, Register reg);

  // Bit operations.
  void bt(const Operand &dst, Register src);
  void bts(const Operand &dst, Register src);
  void bsrq(Register dst, Register src);
  void bsrq(Register dst, const Operand &src);
  void bsrl(Register dst, Register src);
  void bsrl(Register dst, const Operand &src);
  void bsfq(Register dst, Register src);
  void bsfq(Register dst, const Operand &src);
  void bsfl(Register dst, Register src);
  void bsfl(Register dst, const Operand &src);

  // Miscellaneous
  void clc();
  void cld();
  void cpuid();
  void hlt();
  void int3();
  void nop();
  void ret(int imm16);
  void ud2();
  void setcc(Condition cc, Register reg);
  void rdtsc();

  void prefetcht0(const Operand &src) { emit_prefetch(src, 1); }
  void prefetcht1(const Operand &src) { emit_prefetch(src, 2); }
  void prefetcht2(const Operand &src) { emit_prefetch(src, 3); }
  void prefetchnta(const Operand &src) { emit_prefetch(src, 0); }

  // Label operations & relative jumps (PPUM Appendix D)
  //
  // Takes a branch opcode (cc) and a label (L) and generates
  // either a backward branch or a forward branch and links it
  // to the label fixup chain. Usage:
  //
  // Label l;    // unbound label
  // j(cc, &l);  // forward branch to unbound label
  // bind(&l);   // bind label to the current pc
  // j(cc, &l);  // backward branch to bound label
  // bind(&l);   // illegal: a label may be bound only once
  //
  // Note: The same Label can be used for forward and backward branches
  // but it may be bound only once.

  // Calls
  // Call near relative 32-bit displacement, relative to next instruction.
  void call(Label *l);

  // Calls directly to the given address using a relative offset.
  // Should only ever be used in Code objects for calls within the
  // same Code object. Should not be used when generating new code (use labels),
  // but only when patching existing code.
  void call(Address target);

  // Call near absolute indirect, address in register
  void call(Register adr);

  // Call external using pc-relative relocation.
  void call(const void *target, const string &symbol);

  // Jumps
  // Jump short or near relative.
  // Use a 32-bit signed displacement.
  // Unconditional jump to L
  void jmp(Label *l, Label::Distance distance = Label::kFar);

  // Jump near absolute indirect (r64)
  void jmp(Register adr);
  void jmp(const Operand &src);

  // Conditional jumps
  void j(Condition cc,
         Label *l,
         Label::Distance distance = Label::kFar);

  // Floating-point operations
  void fld(int i);

  void fld1();
  void fldz();
  void fldpi();
  void fldln2();

  void fld_s(const Operand &adr);
  void fld_d(const Operand &adr);

  void fstp_s(const Operand &adr);
  void fstp_d(const Operand &adr);
  void fstp(int index);

  void fild_s(const Operand &adr);
  void fild_d(const Operand &adr);

  void fist_s(const Operand &adr);

  void fistp_s(const Operand &adr);
  void fistp_d(const Operand &adr);

  void fisttp_s(const Operand &adr);
  void fisttp_d(const Operand &adr);

  void fabs();
  void fchs();

  void fadd(int i);
  void fsub(int i);
  void fmul(int i);
  void fdiv(int i);

  void fisub_s(const Operand &adr);

  void faddp(int i = 1);
  void fsubp(int i = 1);
  void fsubrp(int i = 1);
  void fmulp(int i = 1);
  void fdivp(int i = 1);
  void fprem();
  void fprem1();

  void fxch(int i = 1);
  void fincstp();
  void ffree(int i = 0);

  void ftst();
  void fucomp(int i);
  void fucompp();
  void fucomi(int i);
  void fucomip();

  void fcompp();
  void fnstsw_ax();
  void fwait();
  void fnclex();

  void fsin();
  void fcos();
  void fptan();
  void fyl2x();
  void f2xm1();
  void fscale();
  void fninit();

  void frndint();

  void sahf();

  // SSE instructions.
  void addss(XMMRegister dst, XMMRegister src);
  void addss(XMMRegister dst, const Operand &src);
  void subss(XMMRegister dst, XMMRegister src);
  void subss(XMMRegister dst, const Operand &src);
  void mulss(XMMRegister dst, XMMRegister src);
  void mulss(XMMRegister dst, const Operand &src);
  void divss(XMMRegister dst, XMMRegister src);
  void divss(XMMRegister dst, const Operand &src);

  void maxss(XMMRegister dst, XMMRegister src);
  void maxss(XMMRegister dst, const Operand &src);
  void minss(XMMRegister dst, XMMRegister src);
  void minss(XMMRegister dst, const Operand &src);

  void cmpss(XMMRegister dst, XMMRegister src, int8_t cmp);
  void cmpss(XMMRegister dst, const Operand &src, int8_t cmp);

  void sqrtss(XMMRegister dst, XMMRegister src);
  void sqrtss(XMMRegister dst, const Operand &src);
  void rsqrtss(XMMRegister dst, XMMRegister src);
  void rsqrtss(XMMRegister dst, const Operand &src);

  void ucomiss(XMMRegister dst, XMMRegister src);
  void ucomiss(XMMRegister dst, const Operand &src);

  void movaps(XMMRegister dst, XMMRegister src);
  void movaps(XMMRegister dst, const Operand &src);
  void movaps(const Operand &dst, XMMRegister src);

  // Don't use this unless it's important to keep the
  // top half of the destination register unchanged.
  // Use movaps when moving float values and movd for integer
  // values in xmm registers.
  void movss(XMMRegister dst, XMMRegister src);
  void movss(XMMRegister dst, const Operand &src);
  void movss(const Operand &dst, XMMRegister src);

  void shufps(XMMRegister dst, XMMRegister src, byte imm8);
  void shufps(XMMRegister dst, const Operand &src, byte imm8);

  void shufpd(XMMRegister dst, XMMRegister src, byte imm8);
  void shufpd(XMMRegister dst, const Operand &src, byte imm8);

  void cvttss2si(Register dst, const Operand &src);
  void cvttss2si(Register dst, XMMRegister src);
  void cvtlsi2ss(XMMRegister dst, const Operand &src);
  void cvtlsi2ss(XMMRegister dst, Register src);

  void cvttps2dq(XMMRegister dst, XMMRegister src);
  void cvttps2dq(XMMRegister dst, const Operand &src);
  void cvttpd2dq(XMMRegister dst, XMMRegister src);
  void cvttpd2dq(XMMRegister dst, const Operand &src);

  void andps(XMMRegister dst, XMMRegister src);
  void andps(XMMRegister dst, const Operand &src);
  void orps(XMMRegister dst, XMMRegister src);
  void orps(XMMRegister dst, const Operand &src);
  void xorps(XMMRegister dst, XMMRegister src);
  void xorps(XMMRegister dst, const Operand &src);
  void andnps(XMMRegister dst, XMMRegister src);
  void andnps(XMMRegister dst, const Operand &src);

  void addps(XMMRegister dst, XMMRegister src);
  void addps(XMMRegister dst, const Operand &src);
  void subps(XMMRegister dst, XMMRegister src);
  void subps(XMMRegister dst, const Operand &src);
  void mulps(XMMRegister dst, XMMRegister src);
  void mulps(XMMRegister dst, const Operand &src);
  void divps(XMMRegister dst, XMMRegister src);
  void divps(XMMRegister dst, const Operand &src);
  void minps(XMMRegister dst, XMMRegister src);
  void minps(XMMRegister dst, const Operand &src);
  void maxps(XMMRegister dst, XMMRegister src);
  void maxps(XMMRegister dst, const Operand &src);

  void addpd(XMMRegister dst, XMMRegister src);
  void addpd(XMMRegister dst, const Operand &src);
  void subpd(XMMRegister dst, XMMRegister src);
  void subpd(XMMRegister dst, const Operand &src);
  void mulpd(XMMRegister dst, XMMRegister src);
  void mulpd(XMMRegister dst, const Operand &src);
  void divpd(XMMRegister dst, XMMRegister src);
  void divpd(XMMRegister dst, const Operand &src);
  void minpd(XMMRegister dst, XMMRegister src);
  void minpd(XMMRegister dst, const Operand &src);
  void maxpd(XMMRegister dst, XMMRegister src);
  void maxpd(XMMRegister dst, const Operand &src);

  void movmskps(Register dst, XMMRegister src);
  void movhlps(XMMRegister dst, XMMRegister src);

  // SSE2 instructions.
#define DECLARE_SSE2_INSTRUCTION(instruction, prefix, escape, opcode) \
  void instruction(XMMRegister dst, XMMRegister src) {                \
    sse2_instr(dst, src, 0x##prefix, 0x##escape, 0x##opcode);         \
  }                                                                   \
  void instruction(XMMRegister dst, const Operand &src) {             \
    sse2_instr(dst, src, 0x##prefix, 0x##escape, 0x##opcode);         \
  }

  SSE2_INSTRUCTION_LIST(DECLARE_SSE2_INSTRUCTION)
#undef DECLARE_SSE2_INSTRUCTION

  // SSSE3 instructions.
#define DECLARE_SSSE3_INSTRUCTION(instruction, prefix, escape1, escape2,     \
                                  opcode)                                    \
  void instruction(XMMRegister dst, XMMRegister src) {                       \
    ssse3_instr(dst, src, 0x##prefix, 0x##escape1, 0x##escape2, 0x##opcode); \
  }                                                                          \
  void instruction(XMMRegister dst, const Operand &src) {                    \
    ssse3_instr(dst, src, 0x##prefix, 0x##escape1, 0x##escape2, 0x##opcode); \
  }

  SSSE3_INSTRUCTION_LIST(DECLARE_SSSE3_INSTRUCTION)
#undef DECLARE_SSSE3_INSTRUCTION

  void lddqu(XMMRegister dst, const Operand &src);

  void haddps(XMMRegister dst, XMMRegister src);
  void haddps(XMMRegister dst, const Operand &src);

  void movshdup(XMMRegister dst, XMMRegister src);
  void movshdup(XMMRegister dst, const Operand &src);

  // SSE4 instructions.
#define DECLARE_SSE4_INSTRUCTION(instruction, prefix, escape1, escape2,     \
                                 opcode)                                    \
  void instruction(XMMRegister dst, XMMRegister src) {                      \
    sse4_instr(dst, src, 0x##prefix, 0x##escape1, 0x##escape2, 0x##opcode); \
  }                                                                         \
  void instruction(XMMRegister dst, const Operand &src) {                   \
    sse4_instr(dst, src, 0x##prefix, 0x##escape1, 0x##escape2, 0x##opcode); \
  }

  SSE4_INSTRUCTION_LIST(DECLARE_SSE4_INSTRUCTION)
#undef DECLARE_SSE4_INSTRUCTION

  // SSE 4.1 instructions.
  void insertps(XMMRegister dst, XMMRegister src, byte imm8);
  void extractps(Register dst, XMMRegister src, byte imm8);

  void pextrb(Register dst, XMMRegister src, int8_t imm8);
  void pextrb(const Operand &dst, XMMRegister src, int8_t imm8);
  void pextrw(Register dst, XMMRegister src, int8_t imm8);
  void pextrw(const Operand &dst, XMMRegister src, int8_t imm8);
  void pextrd(Register dst, XMMRegister src, int8_t imm8);
  void pextrd(const Operand &dst, XMMRegister src, int8_t imm8);
  void pextrq(Register dst, XMMRegister src, int8_t imm8);
  void pextrq(const Operand &dst, XMMRegister src, int8_t imm8);
  void pinsrb(XMMRegister dst, Register src, int8_t imm8);
  void pinsrb(XMMRegister dst, const Operand &src, int8_t imm8);
  void pinsrw(XMMRegister dst, Register src, int8_t imm8);
  void pinsrw(XMMRegister dst, const Operand &src, int8_t imm8);
  void pinsrd(XMMRegister dst, Register src, int8_t imm8);
  void pinsrd(XMMRegister dst, const Operand &src, int8_t imm8);
  void pinsrq(XMMRegister dst, Register src, int8_t imm8);
  void pinsrq(XMMRegister dst, const Operand &src, int8_t imm8);

  void blendps(XMMRegister dst, const Operand &src, int8_t mask);
  void blendps(XMMRegister dst, XMMRegister src, int8_t mask);
  void blendpd(XMMRegister dst, XMMRegister src, int8_t mask);
  void blendpd(XMMRegister dst, const Operand &src, int8_t mask);

  void roundss(XMMRegister dst, const Operand &src, int8_t mode);
  void roundss(XMMRegister dst, XMMRegister src, int8_t mode);
  void roundsd(XMMRegister dst, XMMRegister src, int8_t mode);
  void roundsd(XMMRegister dst, const Operand &src, int8_t mode);

  void roundps(XMMRegister dst, const Operand &src, int8_t mode);
  void roundps(XMMRegister dst, XMMRegister src, int8_t mode);
  void roundpd(XMMRegister dst, XMMRegister src, int8_t mode);
  void roundpd(XMMRegister dst, const Operand &src, int8_t mode);

  void cmpps(XMMRegister dst, XMMRegister src, int8_t cmp);
  void cmpps(XMMRegister dst, const Operand &src, int8_t cmp);
  void cmppd(XMMRegister dst, XMMRegister src, int8_t cmp);
  void cmppd(XMMRegister dst, const Operand &src, int8_t cmp);

  void ptest(XMMRegister dst, XMMRegister src) {
    sse4_instr(dst, src, 0x66, 0x0F, 0x38, 0x17);
  }
  void ptest(XMMRegister dst, const Operand &src) {
    sse4_instr(dst, src, 0x66, 0x0F, 0x38, 0x17);
  }

  void pmovsxdq(XMMRegister dst, XMMRegister src) {
    sse4_instr(dst, src, 0x66, 0x0F, 0x38, 0x25);
  }

#define SSE_CMP_P(instr, imm8)                                                \
  void instr##ps(XMMRegister dst, XMMRegister src) {                          \
    cmpps(dst, src, imm8);                                                    \
  }                                                                           \
  void instr##ps(XMMRegister dst, const Operand &src) {                       \
    cmpps(dst, src, imm8);                                                    \
  }                                                                           \
  void instr##pd(XMMRegister dst, XMMRegister src) {                          \
    cmppd(dst, src, imm8);                                                    \
  }                                                                           \
  void instr##pd(XMMRegister dst, const Operand &src) {                       \
    cmppd(dst, src, imm8);                                                    \
  }

  SSE_CMP_P(cmpeq, 0x0);
  SSE_CMP_P(cmplt, 0x1);
  SSE_CMP_P(cmple, 0x2);
  SSE_CMP_P(cmpneq, 0x4);
  SSE_CMP_P(cmpnlt, 0x5);
  SSE_CMP_P(cmpnle, 0x6);

#undef SSE_CMP_P

  void rcpss(XMMRegister dst, XMMRegister src);
  void rcpss(XMMRegister dst, const Operand &src);
  void rcpps(XMMRegister dst, XMMRegister src);
  void rcpps(XMMRegister dst, const Operand &src);

  void sqrtps(XMMRegister dst, XMMRegister src);
  void sqrtps(XMMRegister dst, const Operand &src);
  void sqrtpd(XMMRegister dst, XMMRegister src);
  void sqrtpd(XMMRegister dst, const Operand &src);

  void rsqrtps(XMMRegister dst, XMMRegister src);
  void rsqrtps(XMMRegister dst, const Operand &src);

  void movups(XMMRegister dst, XMMRegister src);
  void movups(XMMRegister dst, const Operand &src);
  void movups(const Operand &dst, XMMRegister src);
  void psrldq(XMMRegister dst, uint8_t shift);
  void pshufd(XMMRegister dst, XMMRegister src, uint8_t shuffle);
  void pshufd(XMMRegister dst, const Operand &src, uint8_t shuffle);
  void pshuflw(XMMRegister dst, XMMRegister src, uint8_t shuffle);
  void pshuflw(XMMRegister dst, const Operand &src, uint8_t shuffle);
  void cvtdq2ps(XMMRegister dst, XMMRegister src);
  void cvtdq2ps(XMMRegister dst, const Operand &src);
  void cvtdq2pd(XMMRegister dst, XMMRegister src);
  void cvtdq2pd(XMMRegister dst, const Operand &src);

  // AVX instructions.
#define DECLARE_SSE2_AVX_INSTRUCTION(instruction, prefix, escape, opcode)    \
  void v##instruction(XMMRegister dst, XMMRegister src1, XMMRegister src2) { \
    vinstr(0x##opcode, dst, src1, src2, k##prefix, k##escape, kW0);          \
  }                                                                          \
  void v##instruction(XMMRegister dst, XMMRegister src1,                     \
                      const Operand &src2) {                                 \
    vinstr(0x##opcode, dst, src1, src2, k##prefix, k##escape, kW0);          \
  }                                                                          \
  void v##instruction(YMMRegister dst, YMMRegister src1, YMMRegister src2) { \
    vinstr(0x##opcode, dst, src1, src2, k##prefix, k##escape, kW0);          \
  }                                                                          \
  void v##instruction(YMMRegister dst, YMMRegister src1,                     \
                      const Operand &src2) {                                 \
    vinstr(0x##opcode, dst, src1, src2, k##prefix, k##escape, kW0);          \
  }

  SSE2_INSTRUCTION_LIST(DECLARE_SSE2_AVX_INSTRUCTION)
#undef DECLARE_SSE2_AVX_INSTRUCTION

#define DECLARE_SSE34_AVX_INSTRUCTION(instruction, prefix, escape1, escape2,  \
                                      opcode)                                 \
  void v##instruction(XMMRegister dst, XMMRegister src1, XMMRegister src2) {  \
    vinstr(0x##opcode, dst, src1, src2, k##prefix, k##escape1##escape2, kW0); \
  }                                                                           \
  void v##instruction(XMMRegister dst, XMMRegister src1,                      \
                      const Operand &src2) {                                  \
    vinstr(0x##opcode, dst, src1, src2, k##prefix, k##escape1##escape2, kW0); \
  }                                                                           \
  void v##instruction(YMMRegister dst, YMMRegister src1, YMMRegister src2) {  \
    vinstr(0x##opcode, dst, src1, src2, k##prefix, k##escape1##escape2, kW0); \
  }                                                                           \
  void v##instruction(YMMRegister dst, YMMRegister src1,                      \
                      const Operand &src2) {                                  \
    vinstr(0x##opcode, dst, src1, src2, k##prefix, k##escape1##escape2, kW0); \
  }

  SSSE3_INSTRUCTION_LIST(DECLARE_SSE34_AVX_INSTRUCTION)
  SSE4_INSTRUCTION_LIST(DECLARE_SSE34_AVX_INSTRUCTION)
#undef DECLARE_SSE34_AVX_INSTRUCTION

  void movd(XMMRegister dst, Register src);
  void movd(XMMRegister dst, const Operand &src);
  void movd(Register dst, XMMRegister src);
  void movd(const Operand &dst, XMMRegister src);
  void movq(XMMRegister dst, Register src);
  void movq(Register dst, XMMRegister src);
  void movq(XMMRegister dst, XMMRegister src);
  void movq(const Operand &dst, XMMRegister src);

  // Don't use this unless it's important to keep the
  // top half of the destination register unchanged.
  // Use movapd when moving double values and movq for integer
  // values in xmm registers.
  void movsd(XMMRegister dst, XMMRegister src);

  void movsd(const Operand &dst, XMMRegister src);
  void movsd(XMMRegister dst, const Operand &src);

  void movdqa(XMMRegister dst, XMMRegister src);
  void movdqa(const Operand &dst, XMMRegister src);
  void movdqa(XMMRegister dst, const Operand &src);

  void movdqu(XMMRegister dst, XMMRegister src);
  void movdqu(const Operand &dst, XMMRegister src);
  void movdqu(XMMRegister dst, const Operand &src);

  void movapd(XMMRegister dst, XMMRegister src);
  void movapd(XMMRegister dst, const Operand &src);
  void movapd(const Operand &dst, XMMRegister src);

  void movupd(XMMRegister dst, XMMRegister src);
  void movupd(XMMRegister dst, const Operand &src);
  void movupd(const Operand &dst, XMMRegister src);

  void psllq(XMMRegister reg, byte imm8);
  void psrlq(XMMRegister reg, byte imm8);
  void psllw(XMMRegister reg, byte imm8);
  void pslld(XMMRegister reg, byte imm8);
  void psrlw(XMMRegister reg, byte imm8);
  void psrld(XMMRegister reg, byte imm8);
  void psraw(XMMRegister reg, byte imm8);
  void psrad(XMMRegister reg, byte imm8);

  void cvttsd2si(Register dst, const Operand &src);
  void cvttsd2si(Register dst, XMMRegister src);
  void cvttss2siq(Register dst, XMMRegister src);
  void cvttss2siq(Register dst, const Operand &src);
  void cvttsd2siq(Register dst, XMMRegister src);
  void cvttsd2siq(Register dst, const Operand &src);

  void cvtlsi2sd(XMMRegister dst, const Operand &src);
  void cvtlsi2sd(XMMRegister dst, Register src);

  void cvtqsi2ss(XMMRegister dst, const Operand &src);
  void cvtqsi2ss(XMMRegister dst, Register src);

  void cvtqsi2sd(XMMRegister dst, const Operand &src);
  void cvtqsi2sd(XMMRegister dst, Register src);

  void cvtss2sd(XMMRegister dst, XMMRegister src);
  void cvtss2sd(XMMRegister dst, const Operand &src);
  void cvtsd2ss(XMMRegister dst, XMMRegister src);
  void cvtsd2ss(XMMRegister dst, const Operand &src);

  void cvtsd2si(Register dst, XMMRegister src);
  void cvtsd2siq(Register dst, XMMRegister src);

  void addsd(XMMRegister dst, XMMRegister src);
  void addsd(XMMRegister dst, const Operand &src);
  void subsd(XMMRegister dst, XMMRegister src);
  void subsd(XMMRegister dst, const Operand &src);
  void mulsd(XMMRegister dst, XMMRegister src);
  void mulsd(XMMRegister dst, const Operand &src);
  void divsd(XMMRegister dst, XMMRegister src);
  void divsd(XMMRegister dst, const Operand &src);

  void cmpsd(XMMRegister dst, XMMRegister src, int8_t cmp);
  void cmpsd(XMMRegister dst, const Operand &src, int8_t cmp);

  void maxsd(XMMRegister dst, XMMRegister src);
  void maxsd(XMMRegister dst, const Operand &src);
  void minsd(XMMRegister dst, XMMRegister src);
  void minsd(XMMRegister dst, const Operand &src);

  void andpd(XMMRegister dst, XMMRegister src);
  void andpd(XMMRegister dst, const Operand &src);
  void orpd(XMMRegister dst, XMMRegister src);
  void orpd(XMMRegister dst, const Operand &src);
  void xorpd(XMMRegister dst, XMMRegister src);
  void xorpd(XMMRegister dst, const Operand &src);
  void andnpd(XMMRegister dst, XMMRegister src);
  void andnpd(XMMRegister dst, const Operand &src);

  void sqrtsd(XMMRegister dst, XMMRegister src);
  void sqrtsd(XMMRegister dst, const Operand &src);

  void ucomisd(XMMRegister dst, XMMRegister src);
  void ucomisd(XMMRegister dst, const Operand &src);
  void cmpltsd(XMMRegister dst, XMMRegister src);

  void movmskpd(Register dst, XMMRegister src);

  void punpckldq(XMMRegister dst, XMMRegister src);
  void punpckldq(XMMRegister dst, const Operand &src);
  void punpckhdq(XMMRegister dst, XMMRegister src);

  void vmovd(XMMRegister dst, Register src);
  void vmovd(XMMRegister dst, const Operand &src);
  void vmovd(Register dst, XMMRegister src);
  void vmovd(const Operand &dst, XMMRegister src);
  void vmovq(XMMRegister dst, Register src);
  void vmovq(XMMRegister dst, const Operand &src);
  void vmovq(Register dst, XMMRegister src);
  void vmovq(const Operand &dst, XMMRegister src);

  void vmovsd(XMMRegister dst, XMMRegister src1, XMMRegister src2) {
    vsd(0x10, dst, src1, src2);
  }
  void vmovsd(XMMRegister dst, const Operand &src) {
    vsd(0x10, dst, xmm0, src);
  }
  void vmovsd(const Operand &dst, XMMRegister src) {
    vsd(0x11, src, xmm0, dst);
  }

#define AVX_SP_3(instr, opcode) \
  AVX_S_3(instr, opcode)        \
  AVX_P_3(instr, opcode)

#define AVX_S_3(instr, opcode)  \
  AVX_3(instr##ss, opcode, vss) \
  AVX_3(instr##sd, opcode, vsd)

#define AVX_P_3(instr, opcode)  \
  AVX_3(instr##ps, opcode, vps) \
  AVX_3(instr##pd, opcode, vpd)

#define AVX_3(instr, opcode, impl)                                     \
  void instr(XMMRegister dst, XMMRegister src1, XMMRegister src2) {    \
    impl(opcode, dst, src1, src2);                                     \
  }                                                                    \
  void instr(XMMRegister dst, XMMRegister src1, const Operand &src2) { \
    impl(opcode, dst, src1, src2);                                     \
  }                                                                    \
  void instr(YMMRegister dst, YMMRegister src1, YMMRegister src2) {    \
    impl(opcode, dst, src1, src2);                                     \
  }                                                                    \
  void instr(YMMRegister dst, YMMRegister src1, const Operand &src2) { \
    impl(opcode, dst, src1, src2);                                     \
  }

  AVX_SP_3(vadd, 0x58);
  AVX_SP_3(vsub, 0x5c);
  AVX_SP_3(vmul, 0x59);
  AVX_SP_3(vdiv, 0x5e);
  AVX_SP_3(vmin, 0x5d);
  AVX_SP_3(vmax, 0x5f);
  AVX_P_3(vand, 0x54);
  AVX_P_3(vandn, 0x55);
  AVX_P_3(vor, 0x56);
  AVX_P_3(vxor, 0x57);
  AVX_3(vcvtsd2ss, 0x5a, vsd);

#undef AVX_3
#undef AVX_S_3
#undef AVX_P_3
#undef AVX_SP_3

  void vpsrlq(XMMRegister dst, XMMRegister src, byte imm8) {
    XMMRegister iop = {2};
    vpd(0x73, iop, dst, src);
    emit(imm8);
  }
  void vpsllq(XMMRegister dst, XMMRegister src, byte imm8) {
    XMMRegister iop = {6};
    vpd(0x73, iop, dst, src);
    emit(imm8);
  }
  void vpsrlq(YMMRegister dst, YMMRegister src, byte imm8) {
    YMMRegister iop = {2};
    vpd(0x73, iop, dst, src);
    emit(imm8);
  }
  void vpsllq(YMMRegister dst, YMMRegister src, byte imm8) {
    YMMRegister iop = {6};
    vpd(0x73, iop, dst, src);
    emit(imm8);
  }

  void vcvtss2sd(XMMRegister dst, XMMRegister src1, XMMRegister src2) {
    vinstr(0x5a, dst, src1, src2, kF3, k0F, kWIG);
  }
  void vcvtss2sd(XMMRegister dst, XMMRegister src1, const Operand &src2) {
    vinstr(0x5a, dst, src1, src2, kF3, k0F, kWIG);
  }

  void vcvtlsi2sd(XMMRegister dst, XMMRegister src1, Register src2) {
    XMMRegister isrc2 = {src2.code()};
    vinstr(0x2a, dst, src1, isrc2, kF2, k0F, kW0);
  }
  void vcvtlsi2sd(XMMRegister dst, XMMRegister src1, const Operand &src2) {
    vinstr(0x2a, dst, src1, src2, kF2, k0F, kW0);
  }

  void vcvtlsi2ss(XMMRegister dst, XMMRegister src1, Register src2) {
    XMMRegister isrc2 = {src2.code()};
    vinstr(0x2a, dst, src1, isrc2, kF3, k0F, kW0);
  }
  void vcvtlsi2ss(XMMRegister dst, XMMRegister src1, const Operand &src2) {
    vinstr(0x2a, dst, src1, src2, kF3, k0F, kW0);
  }

  void vcvtqsi2ss(XMMRegister dst, XMMRegister src1, Register src2) {
    XMMRegister isrc2 = {src2.code()};
    vinstr(0x2a, dst, src1, isrc2, kF3, k0F, kW1);
  }
  void vcvtqsi2ss(XMMRegister dst, XMMRegister src1, const Operand &src2) {
    vinstr(0x2a, dst, src1, src2, kF3, k0F, kW1);
  }

  void vcvtqsi2sd(XMMRegister dst, XMMRegister src1, Register src2) {
    XMMRegister isrc2 = {src2.code()};
    vinstr(0x2a, dst, src1, isrc2, kF2, k0F, kW1);
  }
  void vcvtqsi2sd(XMMRegister dst, XMMRegister src1, const Operand &src2) {
    vinstr(0x2a, dst, src1, src2, kF2, k0F, kW1);
  }

  void vcvttss2si(Register dst, XMMRegister src) {
    XMMRegister idst = {dst.code()};
    vinstr(0x2c, idst, xmm0, src, kF3, k0F, kW0);
  }
  void vcvttss2si(Register dst, const Operand &src) {
    XMMRegister idst = {dst.code()};
    vinstr(0x2c, idst, xmm0, src, kF3, k0F, kW0);
  }

  void vcvttsd2si(Register dst, XMMRegister src) {
    XMMRegister idst = {dst.code()};
    vinstr(0x2c, idst, xmm0, src, kF2, k0F, kW0);
  }
  void vcvttsd2si(Register dst, const Operand &src) {
    XMMRegister idst = {dst.code()};
    vinstr(0x2c, idst, xmm0, src, kF2, k0F, kW0);
  }

  void vcvttss2siq(Register dst, XMMRegister src) {
    XMMRegister idst = {dst.code()};
    vinstr(0x2c, idst, xmm0, src, kF3, k0F, kW1);
  }
  void vcvttss2siq(Register dst, const Operand &src) {
    XMMRegister idst = {dst.code()};
    vinstr(0x2c, idst, xmm0, src, kF3, k0F, kW1);
  }

  void vcvttsd2siq(Register dst, XMMRegister src) {
    XMMRegister idst = {dst.code()};
    vinstr(0x2c, idst, xmm0, src, kF2, k0F, kW1);
  }
  void vcvttsd2siq(Register dst, const Operand &src) {
    XMMRegister idst = {dst.code()};
    vinstr(0x2c, idst, xmm0, src, kF2, k0F, kW1);
  }

  void vcvtsd2si(Register dst, XMMRegister src) {
    XMMRegister idst = {dst.code()};
    vinstr(0x2d, idst, xmm0, src, kF2, k0F, kW0);
  }

  void vucomisd(XMMRegister dst, XMMRegister src) {
    vinstr(0x2e, dst, xmm0, src, k66, k0F, kWIG);
  }
  void vucomisd(XMMRegister dst, const Operand &src) {
    vinstr(0x2e, dst, xmm0, src, k66, k0F, kWIG);
  }

  void vroundss(XMMRegister dst, XMMRegister src1, XMMRegister src2,
                int8_t mode) {
    vinstr(0x0a, dst, src1, src2, k66, k0F3A, kWIG);
    emit(mode | 0x8);  // mask precision exception
  }
  void vroundss(XMMRegister dst, XMMRegister src1, const Operand &src2,
                int8_t mode) {
    vinstr(0x0a, dst, src1, src2, k66, k0F3A, kWIG, 1);
    emit(mode | 0x8);  // mask precision exception
  }

  void vroundsd(XMMRegister dst, XMMRegister src1, XMMRegister src2,
                int8_t mode) {
    vinstr(0x0b, dst, src1, src2, k66, k0F3A, kWIG);
    emit(mode | 0x8);  // mask precision exception
  }
  void vroundsd(XMMRegister dst, XMMRegister src1, const Operand &src2,
                int8_t mode) {
    vinstr(0x0b, dst, src1, src2, k66, k0F3A, kWIG, 1);
    emit(mode | 0x8);  // mask precision exception
  }

  void vroundps(XMMRegister dst, XMMRegister src, int8_t mode) {
    vinstr(0x08, dst, xmm0, src, k66, k0F3A, kWIG);
    emit(mode);
  }
  void vroundps(XMMRegister dst, const Operand &src, int8_t mode) {
    vinstr(0x08, dst, xmm0, src, k66, k0F3A, kWIG, 1);
    emit(mode);
  }
  void vroundps(YMMRegister dst, YMMRegister src, int8_t mode) {
    vinstr(0x08, dst, ymm0, src, k66, k0F3A, kWIG);
    emit(mode);
  }
  void vroundps(YMMRegister dst, const Operand &src, int8_t mode) {
    vinstr(0x08, dst, ymm0, src, k66, k0F3A, kWIG, 1);
    emit(mode);
  }

  void vroundpd(XMMRegister dst, XMMRegister src, int8_t mode) {
    vinstr(0x09, dst, xmm0, src, k66, k0F3A, kWIG);
    emit(mode);
  }
  void vroundpd(XMMRegister dst, const Operand &src, int8_t mode) {
    vinstr(0x09, dst, xmm0, src, k66, k0F3A, kWIG, 1);
    emit(mode);
  }
  void vroundpd(YMMRegister dst, YMMRegister src, int8_t mode) {
    vinstr(0x09, dst, ymm0, src, k66, k0F3A, kWIG);
    emit(mode);
  }
  void vroundpd(YMMRegister dst, const Operand &src, int8_t mode) {
    vinstr(0x09, dst, ymm0, src, k66, k0F3A, kWIG, 1);
    emit(mode);
  }

  void vblendps(XMMRegister dst, XMMRegister src1, XMMRegister src2,
                int8_t mask) {
    vinstr(0x0C, dst, src1, src2, k66, k0F3A, kWIG);
    emit(mask);
  }
  void vblendps(XMMRegister dst, XMMRegister src1, const Operand &src2,
                int8_t mask) {
    vinstr(0x0C, dst, src1, src2, k66, k0F3A, kWIG, 1);
    emit(mask);
  }
  void vblendps(YMMRegister dst, YMMRegister src1, YMMRegister src2,
                int8_t mask) {
    vinstr(0x0C, dst, src1, src2, k66, k0F3A, kWIG);
    emit(mask);
  }
  void vblendps(YMMRegister dst, YMMRegister src1, const Operand &src2,
                int8_t mask) {
    vinstr(0x0C, dst, src1, src2, k66, k0F3A, kWIG, 1);
    emit(mask);
  }

  void vblendpd(XMMRegister dst, XMMRegister src1, XMMRegister src2,
                int8_t mask) {
    vinstr(0x0D, dst, src1, src2, k66, k0F3A, kWIG);
    emit(mask);
  }
  void vblendpd(XMMRegister dst, XMMRegister src1, const Operand &src2,
                int8_t mask) {
    vinstr(0x0D, dst, src1, src2, k66, k0F3A, kWIG, 1);
    emit(mask);
  }
  void vblendpd(YMMRegister dst, YMMRegister src1, YMMRegister src2,
                int8_t mask) {
    vinstr(0x0D, dst, src1, src2, k66, k0F3A, kWIG);
    emit(mask);
  }
  void vblendpd(YMMRegister dst, YMMRegister src1, const Operand &src2,
                int8_t mask) {
    vinstr(0x0D, dst, src1, src2, k66, k0F3A, kWIG, 1);
    emit(mask);
  }

  void vblendvps(XMMRegister dst, XMMRegister src1, XMMRegister src2,
                 XMMRegister mask) {
    vinstr(0x4A, dst, src1, src2, k66, k0F3A, kW0);
    emit(mask.code() << 4);
  }
  void vblendvps(XMMRegister dst, XMMRegister src1, const Operand &src2,
                 XMMRegister mask) {
    vinstr(0x4A, dst, src1, src2, k66, k0F3A, kW0, 1);
    emit(mask.code() << 4);
  }
  void vblendvps(YMMRegister dst, YMMRegister src1, YMMRegister src2,
                 YMMRegister mask) {
    vinstr(0x4A, dst, src1, src2, k66, k0F3A, kW0);
    emit(mask.code() << 4);
  }
  void vblendvps(YMMRegister dst, YMMRegister src1, const Operand &src2,
                 YMMRegister mask) {
    vinstr(0x4A, dst, src1, src2, k66, k0F3A, kW0, 1);
    emit(mask.code() << 4);
  }

  void vblendvpd(XMMRegister dst, XMMRegister src1, XMMRegister src2,
                 XMMRegister mask) {
    vinstr(0x4B, dst, src1, src2, k66, k0F3A, kW0);
    emit(mask.code() << 4);
  }
  void vblendvpd(XMMRegister dst, XMMRegister src1, const Operand &src2,
                 XMMRegister mask) {
    vinstr(0x4B, dst, src1, src2, k66, k0F3A, kW0, 1);
    emit(mask.code() << 4);
  }
  void vblendvpd(YMMRegister dst, YMMRegister src1, YMMRegister src2,
                 YMMRegister mask) {
    vinstr(0x4B, dst, src1, src2, k66, k0F3A, kW0);
    emit(mask.code() << 4);
  }
  void vblendvpd(YMMRegister dst, YMMRegister src1, const Operand &src2,
                 YMMRegister mask) {
    vinstr(0x4B, dst, src1, src2, k66, k0F3A, kW0, 1);
    emit(mask.code() << 4);
  }

  void vmovss(XMMRegister dst, XMMRegister src1, XMMRegister src2) {
    vss(0x10, dst, src1, src2);
  }
  void vmovss(XMMRegister dst, const Operand &src) {
    vss(0x10, dst, xmm0, src);
  }
  void vmovss(const Operand &dst, XMMRegister src) {
    vss(0x11, src, xmm0, dst);
  }
  void vmovss(YMMRegister dst, YMMRegister src1, YMMRegister src2) {
    vss(0x10, dst, src1, src2);
  }
  void vmovss(YMMRegister dst, const Operand &src) {
    vss(0x10, dst, ymm0, src);
  }
  void vmovss(const Operand &dst, YMMRegister src) {
    vss(0x11, src, ymm0, dst);
  }

  void vucomiss(XMMRegister dst, XMMRegister src);
  void vucomiss(XMMRegister dst, const Operand &src);

  void vmovdqa(XMMRegister dst, XMMRegister src) {
    vinstr(0x6f, dst, xmm0, src, k66, k0F, kWIG);
  }
  void vmovdqa(XMMRegister dst, const Operand &src) {
    vinstr(0x6f, dst, xmm0, src, k66, k0F, kWIG);
  }
  void vmovdqa(const Operand &dst, XMMRegister src) {
    vinstr(0x7f, src, xmm0, dst, k66, k0F, kWIG);
  }
  void vmovdqa(YMMRegister dst, YMMRegister src) {
    vinstr(0x6f, dst, ymm0, src, k66, k0F, kWIG);
  }
  void vmovdqa(YMMRegister dst, const Operand &src) {
    vinstr(0x6f, dst, ymm0, src, k66, k0F, kWIG);
  }
  void vmovdqa(const Operand &dst, YMMRegister src) {
    vinstr(0x7f, src, ymm0, dst, k66, k0F, kWIG);
  }

  void vmovdqu(XMMRegister dst, XMMRegister src) {
    vinstr(0x6f, dst, xmm0, src, kF3, k0F, kWIG);
  }
  void vmovdqu(XMMRegister dst, const Operand &src) {
    vinstr(0x6f, dst, xmm0, src, kF3, k0F, kWIG);
  }
  void vmovdqu(const Operand &dst, XMMRegister src) {
    vinstr(0x7f, src, xmm0, dst, kF3, k0F, kWIG);
  }
  void vmovdqu(YMMRegister dst, YMMRegister src) {
    vinstr(0x6f, dst, ymm0, src, kF3, k0F, kWIG);
  }
  void vmovdqu(YMMRegister dst, const Operand &src) {
    vinstr(0x6f, dst, ymm0, src, kF3, k0F, kWIG);
  }
  void vmovdqu(const Operand &dst, YMMRegister src) {
    vinstr(0x7f, src, ymm0, dst, kF3, k0F, kWIG);
  }

  void vmovaps(XMMRegister dst, XMMRegister src) {
    vps(0x28, dst, xmm0, src);
  }
  void vmovaps(XMMRegister dst, const Operand &src) {
    vps(0x28, dst, xmm0, src);
  }
  void vmovaps(const Operand &dst, XMMRegister src) {
    vps(0x29, src, xmm0, dst);
  }
  void vmovaps(YMMRegister dst, YMMRegister src) {
    vps(0x28, dst, ymm0, src);
  }
  void vmovaps(YMMRegister dst, const Operand &src) {
    vps(0x28, dst, ymm0, src);
  }
  void vmovaps(const Operand &dst, YMMRegister src) {
    vps(0x29, src, ymm0, dst);
  }

  void vmovups(XMMRegister dst, XMMRegister src) {
    vps(0x10, dst, xmm0, src);
  }
  void vmovups(XMMRegister dst, const Operand &src) {
    vps(0x10, dst, xmm0, src);
  }
  void vmovups(const Operand &dst, XMMRegister src) {
    vps(0x11, src, xmm0, dst);
  }
  void vmovups(YMMRegister dst, YMMRegister src) {
    vps(0x10, dst, ymm0, src);
  }
  void vmovups(YMMRegister dst, const Operand &src) {
    vps(0x10, dst, ymm0, src);
  }
  void vmovups(const Operand &dst, YMMRegister src) {
    vps(0x11, src, ymm0, dst);
  }

  void vmovapd(XMMRegister dst, XMMRegister src) {
    vpd(0x28, dst, xmm0, src);
  }
  void vmovapd(XMMRegister dst, const Operand &src) {
    vpd(0x28, dst, xmm0, src);
  }
  void vmovapd(const Operand &dst, XMMRegister src) {
    vpd(0x29, src, xmm0, dst);
  }
  void vmovapd(YMMRegister dst, YMMRegister src) {
    vpd(0x28, dst, ymm0, src);
  }
  void vmovapd(YMMRegister dst, const Operand &src) {
    vpd(0x28, dst, ymm0, src);
  }
  void vmovapd(const Operand &dst, YMMRegister src) {
    vpd(0x29, src, ymm0, dst);
  }

  void vmovupd(XMMRegister dst, XMMRegister src) {
    vpd(0x10, dst, xmm0, src);
  }
  void vmovupd(XMMRegister dst, const Operand &src) {
    vpd(0x10, dst, xmm0, src);
  }
  void vmovupd(const Operand &dst, XMMRegister src) {
    vpd(0x11, src, xmm0, dst);
  }
  void vmovupd(YMMRegister dst, YMMRegister src) {
    vpd(0x10, dst, ymm0, src);
  }
  void vmovupd(YMMRegister dst, const Operand &src) {
    vpd(0x10, dst, ymm0, src);
  }
  void vmovupd(const Operand &dst, YMMRegister src) {
    vpd(0x11, src, ymm0, dst);
  }

  void vmovmskps(Register dst, XMMRegister src) {
    XMMRegister idst = {dst.code()};
    vps(0x50, idst, xmm0, src);
  }
  void vmovmskps(Register dst, YMMRegister src) {
    YMMRegister idst = {dst.code()};
    vps(0x50, idst, ymm0, src);
  }

  void vmovmskpd(Register dst, XMMRegister src) {
    XMMRegister idst = {dst.code()};
    vpd(0x50, idst, xmm0, src);
  }
  void vmovmskpd(Register dst, YMMRegister src) {
    YMMRegister idst = {dst.code()};
    vpd(0x50, idst, ymm0, src);
  }

  void vpmovsxdq(XMMRegister dst, XMMRegister src) {
    vinstr(0x25, dst, xmm0, src, k66, k0F38, kWIG);
  }
  void vpmovsxdq(YMMRegister dst, XMMRegister src) {
    YMMRegister isrc = {src.code()};
    vinstr(0x25, dst, ymm0, isrc, k66, k0F38, kWIG);
  }

  void vcmpss(XMMRegister dst, XMMRegister src1, XMMRegister src2, int8_t cmp) {
    vss(0xC2, dst, src1, src2);
    emit(cmp);
  }
  void vcmpss(XMMRegister dst, XMMRegister src1, const Operand &src2,
              int8_t cmp) {
    vss(0xC2, dst, src1, src2, 1);
    emit(cmp);
  }

  void vcmpsd(XMMRegister dst, XMMRegister src1, XMMRegister src2, int8_t cmp) {
    vsd(0xC2, dst, src1, src2);
    emit(cmp);
  }
  void vcmpsd(XMMRegister dst, XMMRegister src1, const Operand &src2,
              int8_t cmp) {
    vsd(0xC2, dst, src1, src2, 1);
    emit(cmp);
  }

  void vcmpps(XMMRegister dst, XMMRegister src1, XMMRegister src2, int8_t cmp) {
    vps(0xC2, dst, src1, src2);
    emit(cmp);
  }
  void vcmpps(XMMRegister dst, XMMRegister src1, const Operand &src2,
              int8_t cmp) {
    vps(0xC2, dst, src1, src2, 1);
    emit(cmp);
  }
  void vcmpps(YMMRegister dst, YMMRegister src1, YMMRegister src2, int8_t cmp) {
    vps(0xC2, dst, src1, src2);
    emit(cmp);
  }
  void vcmpps(YMMRegister dst, YMMRegister src1, const Operand &src2,
              int8_t cmp) {
    vps(0xC2, dst, src1, src2, 1);
    emit(cmp);
  }

  void vcmppd(XMMRegister dst, XMMRegister src1, XMMRegister src2, int8_t cmp) {
    vpd(0xC2, dst, src1, src2);
    emit(cmp);
  }
  void vcmppd(XMMRegister dst, XMMRegister src1, const Operand &src2,
              int8_t cmp) {
    vpd(0xC2, dst, src1, src2, 1);
    emit(cmp);
  }
  void vcmppd(YMMRegister dst, YMMRegister src1, YMMRegister src2, int8_t cmp) {
    vpd(0xC2, dst, src1, src2);
    emit(cmp);
  }
  void vcmppd(YMMRegister dst, YMMRegister src1, const Operand &src2,
              int8_t cmp) {
    vpd(0xC2, dst, src1, src2, 1);
    emit(cmp);
  }

  void vtestps(XMMRegister dst, XMMRegister src) {
    vinstr(0x0e, dst, xmm0, src, k66, k0F38, kW0);
  }
  void vtestps(XMMRegister dst, const Operand &src) {
    vinstr(0x0e, dst, xmm0, src, k66, k0F38, kW0);
  }
  void vtestps(YMMRegister dst, YMMRegister src) {
    vinstr(0x0e, dst, ymm0, src, k66, k0F38, kW0);
  }
  void vtestps(YMMRegister dst, const Operand &src) {
    vinstr(0x0e, dst, ymm0, src, k66, k0F38, kW0);
  }

  void vtestpd(XMMRegister dst, XMMRegister src) {
    vinstr(0x0f, dst, xmm0, src, k66, k0F38, kW0);
  }
  void vtestpd(XMMRegister dst, const Operand &src) {
    vinstr(0x0f, dst, xmm0, src, k66, k0F38, kW0);
  }
  void vtestpd(YMMRegister dst, YMMRegister src) {
    vinstr(0x0f, dst, ymm0, src, k66, k0F38, kW0);
  }
  void vtestpd(YMMRegister dst, const Operand &src) {
    vinstr(0x0f, dst, ymm0, src, k66, k0F38, kW0);
  }

  void vptest(XMMRegister dst, XMMRegister src) {
    vinstr(0x17, dst, xmm0, src, k66, k0F38, kWIG);
  }
  void vtest(XMMRegister dst, const Operand &src) {
    vinstr(0x17, dst, xmm0, src, k66, k0F38, kWIG);
  }
  void vptest(YMMRegister dst, YMMRegister src) {
    vinstr(0x17, dst, ymm0, src, k66, k0F38, kWIG);
  }
  void vtest(YMMRegister dst, const Operand &src) {
    vinstr(0x17, dst, ymm0, src, k66, k0F38, kWIG);
  }

#define AVX_CMP_P(instr, imm8)                                             \
  void instr##ps(XMMRegister dst, XMMRegister src1, XMMRegister src2) {    \
    vcmpps(dst, src1, src2, imm8);                                         \
  }                                                                        \
  void instr##ps(XMMRegister dst, XMMRegister src1, const Operand &src2) { \
    vcmpps(dst, src1, src2, imm8);                                         \
  }                                                                        \
  void instr##pd(XMMRegister dst, XMMRegister src1, XMMRegister src2) {    \
    vcmppd(dst, src1, src2, imm8);                                         \
  }                                                                        \
  void instr##pd(XMMRegister dst, XMMRegister src1, const Operand &src2) { \
    vcmppd(dst, src1, src2, imm8);                                         \
  }                                                                        \
  void instr##ps(YMMRegister dst, YMMRegister src1, YMMRegister src2) {    \
    vcmpps(dst, src1, src2, imm8);                                         \
  }                                                                        \
  void instr##ps(YMMRegister dst, YMMRegister src1, const Operand &src2) { \
    vcmpps(dst, src1, src2, imm8);                                         \
  }                                                                        \
  void instr##pd(YMMRegister dst, YMMRegister src1, YMMRegister src2) {    \
    vcmppd(dst, src1, src2, imm8);                                         \
  }                                                                        \
  void instr##pd(YMMRegister dst, YMMRegister src1, const Operand &src2) { \
    vcmppd(dst, src1, src2, imm8);                                         \
  }

  AVX_CMP_P(vcmpeq, 0x0);
  AVX_CMP_P(vcmplt, 0x1);
  AVX_CMP_P(vcmple, 0x2);
  AVX_CMP_P(vcmpneq, 0x4);
  AVX_CMP_P(vcmpnlt, 0x5);
  AVX_CMP_P(vcmpnle, 0x6);

#undef AVX_CMP_P

  void vlddqu(XMMRegister dst, const Operand &src) {
    vinstr(0xF0, dst, xmm0, src, kF2, k0F, kWIG);
  }
  void vlddqu(YMMRegister dst, const Operand &src) {
    vinstr(0xF0, dst, ymm0, src, kF2, k0F, kWIG);
  }

  void vpsllw(XMMRegister dst, XMMRegister src, int8_t imm8) {
    XMMRegister iop = {6};
    vinstr(0x71, iop, dst, src, k66, k0F, kWIG);
    emit(imm8);
  }
  void vpsllw(YMMRegister dst, YMMRegister src, int8_t imm8) {
    DCHECK(Enabled(AVX2));
    YMMRegister iop = {6};
    vinstr(0x71, iop, dst, src, k66, k0F, kWIG);
    emit(imm8);
  }

  void vpsrlw(XMMRegister dst, XMMRegister src, int8_t imm8) {
    XMMRegister iop = {2};
    vinstr(0x71, iop, dst, src, k66, k0F, kWIG);
    emit(imm8);
  }
  void vpsrlw(YMMRegister dst, YMMRegister src, int8_t imm8) {
    DCHECK(Enabled(AVX2));
    YMMRegister iop = {2};
    vinstr(0x71, iop, dst, src, k66, k0F, kWIG);
    emit(imm8);
  }

  void vpsraw(XMMRegister dst, XMMRegister src, int8_t imm8) {
    XMMRegister iop = {4};
    vinstr(0x71, iop, dst, src, k66, k0F, kWIG);
    emit(imm8);
  }
  void vpsraw(YMMRegister dst, YMMRegister src, int8_t imm8) {
    DCHECK(Enabled(AVX2));
    YMMRegister iop = {4};
    vinstr(0x71, iop, dst, src, k66, k0F, kWIG);
    emit(imm8);
  }

  void vpslld(XMMRegister dst, XMMRegister src, int8_t imm8) {
    XMMRegister iop = {6};
    vinstr(0x72, iop, dst, src, k66, k0F, kWIG);
    emit(imm8);
  }
  void vpslld(YMMRegister dst, YMMRegister src, int8_t imm8) {
    DCHECK(Enabled(AVX2));
    YMMRegister iop = {6};
    vinstr(0x72, iop, dst, src, k66, k0F, kWIG);
    emit(imm8);
  }

  void vpsrld(XMMRegister dst, XMMRegister src, int8_t imm8) {
    XMMRegister iop = {2};
    vinstr(0x72, iop, dst, src, k66, k0F, kWIG);
    emit(imm8);
  }
  void vpsrld(YMMRegister dst, YMMRegister src, int8_t imm8) {
    DCHECK(Enabled(AVX2));
    YMMRegister iop = {2};
    vinstr(0x72, iop, dst, src, k66, k0F, kWIG);
    emit(imm8);
  }

  void vpsrad(XMMRegister dst, XMMRegister src, int8_t imm8) {
    XMMRegister iop = {4};
    vinstr(0x72, iop, dst, src, k66, k0F, kWIG);
    emit(imm8);
  }
  void vpsrad(YMMRegister dst, YMMRegister src, int8_t imm8) {
    DCHECK(Enabled(AVX2));
    YMMRegister iop = {4};
    vinstr(0x72, iop, dst, src, k66, k0F, kWIG);
    emit(imm8);
  }

  void vpextrb(Register dst, XMMRegister src, int8_t imm8) {
    XMMRegister idst = {dst.code()};
    vinstr(0x14, src, xmm0, idst, k66, k0F3A, kW0);
    emit(imm8);
  }
  void vpextrb(const Operand &dst, XMMRegister src, int8_t imm8) {
    vinstr(0x14, src, xmm0, dst, k66, k0F3A, kW0, 1);
    emit(imm8);
  }

  void vpextrw(Register dst, XMMRegister src, int8_t imm8) {
    XMMRegister idst = {dst.code()};
    vinstr(0xc5, idst, xmm0, src, k66, k0F, kW0);
    emit(imm8);
  }
  void vpextrw(const Operand &dst, XMMRegister src, int8_t imm8) {
    vinstr(0x15, src, xmm0, dst, k66, k0F3A, kW0, 1);
    emit(imm8);
  }

  void vpextrd(Register dst, XMMRegister src, int8_t imm8) {
    XMMRegister idst = {dst.code()};
    vinstr(0x16, src, xmm0, idst, k66, k0F3A, kW0);
    emit(imm8);
  }
  void vpextrd(const Operand &dst, XMMRegister src, int8_t imm8) {
    vinstr(0x16, src, xmm0, dst, k66, k0F3A, kW0, 1);
    emit(imm8);
  }

  void vpextrq(Register dst, XMMRegister src, int8_t imm8) {
    XMMRegister idst = {dst.code()};
    vinstr(0x16, src, xmm0, idst, k66, k0F3A, kW1);
    emit(imm8);
  }
  void vpextrq(const Operand &dst, XMMRegister src, int8_t imm8) {
    vinstr(0x16, src, xmm0, dst, k66, k0F3A, kW1, 1);
    emit(imm8);
  }

  void vpinsrb(XMMRegister dst, XMMRegister src1, Register src2, int8_t imm8) {
    XMMRegister isrc = {src2.code()};
    vinstr(0x20, dst, src1, isrc, k66, k0F3A, kW0);
    emit(imm8);
  }
  void vpinsrb(XMMRegister dst, XMMRegister src1, const Operand &src2,
               int8_t imm8) {
    vinstr(0x20, dst, src1, src2, k66, k0F3A, kW0, 1);
    emit(imm8);
  }

  void vpinsrw(XMMRegister dst, XMMRegister src1, Register src2, int8_t imm8) {
    XMMRegister isrc = {src2.code()};
    vinstr(0xc4, dst, src1, isrc, k66, k0F, kW0);
    emit(imm8);
  }
  void vpinsrw(XMMRegister dst, XMMRegister src1, const Operand &src2,
               int8_t imm8) {
    vinstr(0xc4, dst, src1, src2, k66, k0F, kW0, 1);
    emit(imm8);
  }

  void vpinsrd(XMMRegister dst, XMMRegister src1, Register src2, int8_t imm8) {
    XMMRegister isrc = {src2.code()};
    vinstr(0x22, dst, src1, isrc, k66, k0F3A, kW0);
    emit(imm8);
  }
  void vpinsrd(XMMRegister dst, XMMRegister src1, const Operand &src2,
               int8_t imm8) {
    vinstr(0x22, dst, src1, src2, k66, k0F3A, kW0, 1);
    emit(imm8);
  }

  void vpinsrq(XMMRegister dst, XMMRegister src1, Register src2, int8_t imm8) {
    XMMRegister isrc = {src2.code()};
    vinstr(0x22, dst, src1, isrc, k66, k0F3A, kW1);
    emit(imm8);
  }
  void vpinsrq(XMMRegister dst, XMMRegister src1, const Operand &src2,
               int8_t imm8) {
    vinstr(0x22, dst, src1, src2, k66, k0F3A, kW1, 1);
    emit(imm8);
  }

  void vpshufd(XMMRegister dst, XMMRegister src, int8_t imm8) {
    vinstr(0x70, dst, xmm0, src, k66, k0F, kWIG);
    emit(imm8);
  }
  void vpshufd(XMMRegister dst, Operand &src, int8_t imm8) {
    vinstr(0x70, dst, xmm0, src, k66, k0F, kWIG, 1);
    emit(imm8);
  }
  void vpshufd(YMMRegister dst, YMMRegister src, int8_t imm8) {
    vinstr(0x70, dst, ymm0, src, k66, k0F, kWIG);
    emit(imm8);
  }
  void vpshufd(YMMRegister dst, Operand &src, int8_t imm8) {
    vinstr(0x70, dst, ymm0, src, k66, k0F, kWIG, 1);
    emit(imm8);
  }

  void vshufps(XMMRegister dst, XMMRegister src1, XMMRegister src2,
               int8_t imm8) {
    vinstr(0xc6, dst, src1, src2, kNone, k0F, kWIG);
    emit(imm8);
  }
  void vshufps(XMMRegister dst, XMMRegister src1, const Operand &src2,
               int8_t imm8) {
    vinstr(0xc6, dst, src1, src2, kNone, k0F, kWIG, 1);
    emit(imm8);
  }
  void vshufps(YMMRegister dst, YMMRegister src1, YMMRegister src2,
               int8_t imm8) {
    vinstr(0xc6, dst, src1, src2, kNone, k0F, kWIG);
    emit(imm8);
  }
  void vshufps(YMMRegister dst, YMMRegister src1, const Operand &src2,
               int8_t imm8) {
    vinstr(0xc6, dst, src1, src2, kNone, k0F, kWIG, 1);
    emit(imm8);
  }

  void vshufpd(XMMRegister dst, XMMRegister src1, XMMRegister src2,
               int8_t imm8) {
    vinstr(0xc6, dst, src1, src2, k66, k0F, kWIG);
    emit(imm8);
  }
  void vshufpd(XMMRegister dst, XMMRegister src1, const Operand &src2,
               int8_t imm8) {
    vinstr(0xc6, dst, src1, src2, k66, k0F, kWIG, 1);
    emit(imm8);
  }
  void vshufpd(YMMRegister dst, YMMRegister src1, YMMRegister src2,
               int8_t imm8) {
    vinstr(0xc6, dst, src1, src2, k66, k0F, kWIG);
    emit(imm8);
  }
  void vshufpd(YMMRegister dst, YMMRegister src1, const Operand &src2,
               int8_t imm8) {
    vinstr(0xc6, dst, src1, src2, k66, k0F, kWIG, 1);
    emit(imm8);
  }

  void vpshuflw(XMMRegister dst, XMMRegister src, int8_t imm8) {
    vinstr(0x70, dst, xmm0, src, kF2, k0F, kWIG);
    emit(imm8);
  }
  void vpshuflw(XMMRegister dst, const Operand &src, int8_t imm8) {
    vinstr(0x70, dst, xmm0, src, kF2, k0F, kWIG, 1);
    emit(imm8);
  }
  void vpshuflw(YMMRegister dst, YMMRegister src, int8_t imm8) {
    vinstr(0x70, dst, ymm0, src, kF2, k0F, kWIG);
    emit(imm8);
  }
  void vpshuflw(YMMRegister dst, const Operand &src, int8_t imm8) {
    vinstr(0x70, dst, ymm0, src, kF2, k0F, kWIG, 1);
    emit(imm8);
  }


  void vpermq(YMMRegister dst, YMMRegister src, int8_t imm8) {
    vinstr(0x00, dst, ymm0, src, k66, k0F3A, kW1);
    emit(imm8);
  }
  void vpermq(YMMRegister dst, const Operand &src, int8_t imm8) {
    vinstr(0x00, dst, ymm0, src, k66, k0F3A, kW1, 1);
    emit(imm8);
  }

  void vbroadcastss(XMMRegister dst, XMMRegister src) {
    vinstr(0x18, dst, xmm0, src, k66, k0F38, kW0);
  }
  void vbroadcastss(XMMRegister dst, const Operand &src) {
    vinstr(0x18, dst, xmm0, src, k66, k0F38, kW0);
  }
  void vbroadcastss(YMMRegister dst, YMMRegister src) {
    vinstr(0x18, dst, ymm0, src, k66, k0F38, kW0);
  }
  void vbroadcastss(YMMRegister dst, const Operand &src) {
    vinstr(0x18, dst, ymm0, src, k66, k0F38, kW0);
  }

  void vbroadcastsd(YMMRegister dst, YMMRegister src) {
    vinstr(0x19, dst, ymm0, src, k66, k0F38, kW0);
  }
  void vbroadcastsd(YMMRegister dst, const Operand &src) {
    vinstr(0x19, dst, ymm0, src, k66, k0F38, kW0);
  }

  void vbroadcastf128(YMMRegister dst, YMMRegister src) {
    vinstr(0x1a, dst, ymm0, src, k66, k0F38, kW0);
  }
  void vbroadcastf128(YMMRegister dst, const Operand &src) {
    vinstr(0x1a, dst, ymm0, src, k66, k0F38, kW0);
  }

  void vpbroadcastb(XMMRegister dst, XMMRegister src) {
    vinstr(0x78, dst, xmm0, src, k66, k0F38, kW0);
  }
  void vpbroadcastb(XMMRegister dst, const Operand &src) {
    vinstr(0x78, dst, xmm0, src, k66, k0F38, kW0);
  }
  void vpbroadcastb(YMMRegister dst, YMMRegister src) {
    vinstr(0x78, dst, ymm0, src, k66, k0F38, kW0);
  }
  void vpbroadcastb(YMMRegister dst, const Operand &src) {
    vinstr(0x78, dst, ymm0, src, k66, k0F38, kW0);
  }

  void vpbroadcastw(XMMRegister dst, XMMRegister src) {
    vinstr(0x79, dst, xmm0, src, k66, k0F38, kW0);
  }
  void vpbroadcastw(XMMRegister dst, const Operand &src) {
    vinstr(0x79, dst, xmm0, src, k66, k0F38, kW0);
  }
  void vpbroadcastw(YMMRegister dst, YMMRegister src) {
    vinstr(0x79, dst, ymm0, src, k66, k0F38, kW0);
  }
  void vpbroadcastw(YMMRegister dst, const Operand &src) {
    vinstr(0x79, dst, ymm0, src, k66, k0F38, kW0);
  }

  void vpbroadcastd(XMMRegister dst, XMMRegister src) {
    vinstr(0x58, dst, xmm0, src, k66, k0F38, kW0);
  }
  void vpbroadcastd(XMMRegister dst, const Operand &src) {
    vinstr(0x58, dst, xmm0, src, k66, k0F38, kW0);
  }
  void vpbroadcastd(YMMRegister dst, YMMRegister src) {
    vinstr(0x58, dst, ymm0, src, k66, k0F38, kW0);
  }
  void vpbroadcastd(YMMRegister dst, const Operand &src) {
    vinstr(0x58, dst, ymm0, src, k66, k0F38, kW0);
  }

  void vpbroadcastq(XMMRegister dst, XMMRegister src) {
    vinstr(0x59, dst, xmm0, src, k66, k0F38, kW0);
  }
  void vpbroadcastq(XMMRegister dst, const Operand &src) {
    vinstr(0x59, dst, xmm0, src, k66, k0F38, kW0);
  }
  void vpbroadcastq(YMMRegister dst, YMMRegister src) {
    vinstr(0x59, dst, ymm0, src, k66, k0F38, kW0);
  }
  void vpbroadcastq(YMMRegister dst, const Operand &src) {
    vinstr(0x59, dst, ymm0, src, k66, k0F38, kW0);
  }

  void vinsertf128(YMMRegister dst, YMMRegister src1, XMMRegister src2,
                   int8_t imm8) {
    YMMRegister isrc = {src2.code()};
    vinstr(0x18, dst, src1, isrc, k66, k0F3A, kW0);
    emit(imm8);
  }
  void vinsertf128(YMMRegister dst, YMMRegister src1, const Operand &src2,
                   int8_t imm8) {
    vinstr(0x18, dst, src1, src2, k66, k0F3A, kW0, 1);
    emit(imm8);
  }

  void vextractf128(XMMRegister dst, YMMRegister src, int8_t imm8) {
    YMMRegister idst = {dst.code()};
    vinstr(0x19, src, ymm0, idst, k66, k0F3A, kW0);
    emit(imm8);
  }
  void vextractf128(const Operand &dst, YMMRegister &src, int8_t imm8) {
    vinstr(0x19, src, ymm0, dst, k66, k0F3A, kW0, 1);
    emit(imm8);
  }

  void vmaskmovps(XMMRegister dst, XMMRegister src1, const Operand &src2) {
    vinstr(0x2c, dst, src1, src2, k66, k0F38, kW0);
  }
  void vmaskmovps(const Operand &dst, XMMRegister src1, XMMRegister src2) {
    vinstr(0x2e, src2, src1, dst, k66, k0F38, kW0);
  }
  void vmaskmovps(YMMRegister dst, YMMRegister src1, const Operand &src2) {
    vinstr(0x2c, dst, src1, src2, k66, k0F38, kW0);
  }
  void vmaskmovps(const Operand &dst, YMMRegister src1, YMMRegister src2) {
    vinstr(0x2e, src2, src1, dst, k66, k0F38, kW0);
  }

  void vmaskmovpd(XMMRegister dst, XMMRegister src1, const Operand &src2) {
    vinstr(0x2d, dst, src1, src2, k66, k0F38, kW0);
  }
  void vmaskmovpd(const Operand &dst, XMMRegister src1, XMMRegister src2) {
    vinstr(0x2f, src2, src1, dst, k66, k0F38, kW0);
  }
  void vmaskmovpd(YMMRegister dst, YMMRegister src1, const Operand &src2) {
    vinstr(0x2d, dst, src1, src2, k66, k0F38, kW0);
  }
  void vmaskmovpd(const Operand &dst, YMMRegister src1, YMMRegister src2) {
    vinstr(0x2f, src2, src1, dst, k66, k0F38, kW0);
  }

  void vpermilps(XMMRegister dst, XMMRegister src1, XMMRegister src2) {
    vinstr(0x0c, dst, src1, src2, k66, k0F38, kW0);
  }
  void vpermilps(XMMRegister dst, XMMRegister src1, const Operand &src2) {
    vinstr(0x0c, dst, src1, src2, k66, k0F38, kW0);
  }
  void vpermilps(XMMRegister dst, XMMRegister src, int8_t imm8) {
    vinstr(0x04, dst, xmm0, src, k66, k0F3A, kW0);
    emit(imm8);
  }
  void vpermilps(XMMRegister dst, const Operand &src, int8_t imm8) {
    vinstr(0x04, dst, xmm0, src, k66, k0F3A, kW0, 1);
    emit(imm8);
  }
  void vpermilps(YMMRegister dst, YMMRegister src1, YMMRegister src2) {
    vinstr(0x0c, dst, src1, src2, k66, k0F38, kW0);
  }
  void vpermilps(YMMRegister dst, YMMRegister src1, const Operand &src2) {
    vinstr(0x0c, dst, src1, src2, k66, k0F38, kW0);
  }
  void vpermilps(YMMRegister dst, YMMRegister src, int8_t imm8) {
    vinstr(0x04, dst, ymm0, src, k66, k0F3A, kW0);
    emit(imm8);
  }
  void vpermilps(YMMRegister dst, const Operand &src, int8_t imm8) {
    vinstr(0x04, dst, ymm0, src, k66, k0F3A, kW0, 1);
    emit(imm8);
  }

  void vpermilpd(XMMRegister dst, XMMRegister src1, XMMRegister src2) {
    vinstr(0x0d, dst, src1, src2, k66, k0F38, kW0);
  }
  void vpermilpd(XMMRegister dst, XMMRegister src1, const Operand &src2) {
    vinstr(0x0d, dst, src1, src2, k66, k0F38, kW0);
  }
  void vpermilpd(XMMRegister dst, XMMRegister src, int8_t imm8) {
    vinstr(0x05, dst, xmm0, src, k66, k0F3A, kW0);
    emit(imm8);
  }
  void vpermilpd(XMMRegister dst, const Operand &src, int8_t imm8) {
    vinstr(0x05, dst, xmm0, src, k66, k0F3A, kW0, 1);
    emit(imm8);
  }
  void vpermilpd(YMMRegister dst, YMMRegister src1, YMMRegister src2) {
    vinstr(0x0d, dst, src1, src2, k66, k0F38, kW0);
  }
  void vpermilpd(YMMRegister dst, YMMRegister src1, const Operand &src2) {
    vinstr(0x0d, dst, src1, src2, k66, k0F38, kW0);
  }
  void vpermilpd(YMMRegister dst, YMMRegister src, int8_t imm8) {
    vinstr(0x05, dst, ymm0, src, k66, k0F3A, kW0);
    emit(imm8);
  }
  void vpermilpd(YMMRegister dst, const Operand &src, int8_t imm8) {
    vinstr(0x05, dst, ymm0, src, k66, k0F3A, kW0, 1);
    emit(imm8);
  }

  void vperm2f128(YMMRegister dst, YMMRegister src1, YMMRegister src2,
                  int8_t imm8) {
    vinstr(0x06, dst, src1, src2, k66, k0F3A, kW0);
    emit(imm8);
  }
  void vperm2f128(YMMRegister dst, YMMRegister src1, const Operand &src2,
                  int8_t imm8) {
    vinstr(0x06, dst, src1, src2, k66, k0F3A, kW0, 1);
    emit(imm8);
  }

  void vperm2i128(YMMRegister dst, YMMRegister src1, YMMRegister src2,
                  int8_t imm8) {
    vinstr(0x46, dst, src1, src2, k66, k0F3A, kW0);
    emit(imm8);
  }
  void vperm2i128(YMMRegister dst, YMMRegister src1, const Operand &src2,
                  int8_t imm8) {
    vinstr(0x46, dst, src1, src2, k66, k0F3A, kW0, 1);
    emit(imm8);
  }

  void vhaddpd(XMMRegister dst, XMMRegister src1, XMMRegister src2) {
    vinstr(0x7c, dst, src1, src2, k66, k0F, kWIG);
  }
  void vhaddpd(XMMRegister dst, XMMRegister src1, const Operand &src2) {
    vinstr(0x7c, dst, src1, src2, k66, k0F, kWIG);
  }
  void vhaddpd(YMMRegister dst, YMMRegister src1, YMMRegister src2) {
    vinstr(0x7c, dst, src1, src2, k66, k0F, kWIG);
  }
  void vhaddpd(YMMRegister dst, YMMRegister src1, const Operand &src2) {
    vinstr(0x7c, dst, src1, src2, k66, k0F, kWIG);
  }

  void vhaddps(XMMRegister dst, XMMRegister src1, XMMRegister src2) {
    vinstr(0x7c, dst, src1, src2, kF2, k0F, kW0);
  }
  void vhaddps(XMMRegister dst, XMMRegister src1, const Operand &src2) {
    vinstr(0x7c, dst, src1, src2, kF2, k0F, kW0);
  }
  void vhaddps(YMMRegister dst, YMMRegister src1, YMMRegister src2) {
    vinstr(0x7c, dst, src1, src2, kF2, k0F, kW0);
  }
  void vhaddps(YMMRegister dst, YMMRegister src1, const Operand &src2) {
    vinstr(0x7c, dst, src1, src2, kF2, k0F, kW0);
  }

  void vcvttpd2dq(XMMRegister dst, XMMRegister src) {
    vinstr(0xe6, dst, xmm0, src, k66, k0F, kWIG);
  }
  void vcvttpd2dq(XMMRegister dst, const Operand &src) {
    vinstr(0xe6, dst, xmm0, src, k66, k0F, kWIG);
  }
  void vcvttpd2dq(YMMRegister dst, YMMRegister src) {
    vinstr(0xe6, dst, ymm0, src, k66, k0F, kWIG);
  }
  void vcvttpd2dq(YMMRegister dst, const Operand &src) {
    vinstr(0xe6, dst, ymm0, src, k66, k0F, kWIG);
  }

  void vcvttps2dq(XMMRegister dst, XMMRegister src) {
    vinstr(0x5b, dst, xmm0, src, kF3, k0F, kWIG);
  }
  void vcvttps2dq(XMMRegister dst, const Operand &src) {
    vinstr(0x5b, dst, xmm0, src, kF3, k0F, kWIG);
  }
  void vcvttps2dq(YMMRegister dst, YMMRegister src) {
    vinstr(0x5b, dst, ymm0, src, kF3, k0F, kWIG);
  }
  void vcvttps2dq(YMMRegister dst, const Operand &src) {
    vinstr(0x5b, dst, ymm0, src, kF3, k0F, kWIG);
  }

  void vcvtdq2pd(XMMRegister dst, XMMRegister src) {
    vinstr(0xe6, dst, xmm0, src, kF3, k0F, kWIG);
  }
  void vcvtdq2pd(XMMRegister dst, const Operand &src) {
    vinstr(0xe6, dst, xmm0, src, kF3, k0F, kWIG);
  }
  void vcvtdq2pd(YMMRegister dst, YMMRegister src) {
    vinstr(0xe6, dst, ymm0, src, kF3, k0F, kWIG);
  }
  void vcvtdq2pd(YMMRegister dst, const Operand &src) {
    vinstr(0xe6, dst, ymm0, src, kF3, k0F, kWIG);
  }

  void vcvtdq2ps(XMMRegister dst, XMMRegister src) {
    vinstr(0x5b, dst, xmm0, src, kNone, k0F, kWIG);
  }
  void vcvtdq2ps(XMMRegister dst, const Operand &src) {
    vinstr(0x5b, dst, xmm0, src, kNone, k0F, kWIG);
  }
  void vcvtdq2ps(YMMRegister dst, YMMRegister src) {
    vinstr(0x5b, dst, ymm0, src, kNone, k0F, kWIG);
  }
  void vcvtdq2ps(YMMRegister dst, const Operand &src) {
    vinstr(0x5b, dst, ymm0, src, kNone, k0F, kWIG);
  }

  void vrcpss(XMMRegister dst, XMMRegister src1, XMMRegister src2) {
    vinstr(0x53, dst, src1, src2, kF3, k0F, kWIG);
  }
  void vrcpss(XMMRegister dst, XMMRegister src1, const Operand &src2) {
    vinstr(0x53, dst, src1, src2, kF3, k0F, kWIG);
  }

  void vrcpps(XMMRegister dst, XMMRegister src) {
    vinstr(0x53, dst, xmm0, src, kNone, k0F, kWIG);
  }
  void vrcpps(XMMRegister dst, const Operand &src) {
    vinstr(0x53, dst, xmm0, src, kNone, k0F, kWIG);
  }
  void vrcpps(YMMRegister dst, YMMRegister src) {
    vinstr(0x53, dst, ymm0, src, kNone, k0F, kWIG);
  }
  void vrcpps(YMMRegister dst, const Operand &src) {
    vinstr(0x53, dst, ymm0, src, kNone, k0F, kWIG);
  }

  void vsqrtss(XMMRegister dst, XMMRegister src1, XMMRegister src2) {
    vinstr(0x51, dst, src1, src2, kF3, k0F, kWIG);
  }
  void vsqrtss(XMMRegister dst, XMMRegister src1, const Operand &src2) {
    vinstr(0x51, dst, src1, src2, kF3, k0F, kWIG);
  }
  void vsqrtsd(XMMRegister dst, XMMRegister src1, XMMRegister src2) {
    vinstr(0x51, dst, src1, src2, kF2, k0F, kWIG);
  }
  void vsqrtsd(XMMRegister dst, XMMRegister src1, const Operand &src2) {
    vinstr(0x51, dst, src1, src2, kF2, k0F, kWIG);
  }

  void vsqrtps(XMMRegister dst, XMMRegister src) {
    vinstr(0x51, dst, xmm0, src, kNone, k0F, kWIG);
  }
  void vsqrtps(XMMRegister dst, const Operand &src) {
    vinstr(0x51, dst, xmm0, src, kNone, k0F, kWIG);
  }
  void vsqrtps(YMMRegister dst, YMMRegister src) {
    vinstr(0x51, dst, ymm0, src, kNone, k0F, kWIG);
  }
  void vsqrtps(YMMRegister dst, const Operand &src) {
    vinstr(0x51, dst, ymm0, src, kNone, k0F, kWIG);
  }
  void vsqrtpd(XMMRegister dst, XMMRegister src) {
    vinstr(0x51, dst, xmm0, src, k66, k0F, kWIG);
  }
  void vsqrtpd(XMMRegister dst, const Operand &src) {
    vinstr(0x51, dst, xmm0, src, k66, k0F, kWIG);
  }
  void vsqrtpd(YMMRegister dst, YMMRegister src) {
    vinstr(0x51, dst, ymm0, src, k66, k0F, kWIG);
  }
  void vsqrtpd(YMMRegister dst, const Operand &src) {
    vinstr(0x51, dst, ymm0, src, k66, k0F, kWIG);
  }

  void vrsqrtss(XMMRegister dst, XMMRegister src1, XMMRegister src2) {
    vinstr(0x52, dst, src1, src2, kF3, k0F, kWIG);
  }
  void vrsqrtss(XMMRegister dst, XMMRegister src1, const Operand &src2) {
    vinstr(0x52, dst, src1, src2, kF3, k0F, kWIG);
  }

  void vrsqrtps(XMMRegister dst, XMMRegister src) {
    vinstr(0x52, dst, xmm0, src, kNone, k0F, kWIG);
  }
  void vrsqrtps(XMMRegister dst, const Operand &src) {
    vinstr(0x52, dst, xmm0, src, kNone, k0F, kWIG);
  }
  void vrsqrtps(YMMRegister dst, YMMRegister src) {
    vinstr(0x52, dst, ymm0, src, kNone, k0F, kWIG);
  }
  void vrsqrtps(YMMRegister dst, const Operand &src) {
    vinstr(0x52, dst, ymm0, src, kNone, k0F, kWIG);
  }

  void vzeroall();
  void vzeroupper();

  // Scalar single XMM FMA instructions.
  void vfmadd132ss(XMMRegister dst, XMMRegister src1, XMMRegister src2) {
    vfmas(0x99, dst, src1, src2);
  }
  void vfmadd213ss(XMMRegister dst, XMMRegister src1, XMMRegister src2) {
    vfmas(0xa9, dst, src1, src2);
  }
  void vfmadd231ss(XMMRegister dst, XMMRegister src1, XMMRegister src2) {
    vfmas(0xb9, dst, src1, src2);
  }
  void vfmadd132ss(XMMRegister dst, XMMRegister src1, const Operand &src2) {
    vfmas(0x99, dst, src1, src2);
  }
  void vfmadd213ss(XMMRegister dst, XMMRegister src1, const Operand &src2) {
    vfmas(0xa9, dst, src1, src2);
  }
  void vfmadd231ss(XMMRegister dst, XMMRegister src1, const Operand &src2) {
    vfmas(0xb9, dst, src1, src2);
  }
  void vfmsub132ss(XMMRegister dst, XMMRegister src1, XMMRegister src2) {
    vfmas(0x9b, dst, src1, src2);
  }
  void vfmsub213ss(XMMRegister dst, XMMRegister src1, XMMRegister src2) {
    vfmas(0xab, dst, src1, src2);
  }
  void vfmsub231ss(XMMRegister dst, XMMRegister src1, XMMRegister src2) {
    vfmas(0xbb, dst, src1, src2);
  }
  void vfmsub132ss(XMMRegister dst, XMMRegister src1, const Operand &src2) {
    vfmas(0x9b, dst, src1, src2);
  }
  void vfmsub213ss(XMMRegister dst, XMMRegister src1, const Operand &src2) {
    vfmas(0xab, dst, src1, src2);
  }
  void vfmsub231ss(XMMRegister dst, XMMRegister src1, const Operand &src2) {
    vfmas(0xbb, dst, src1, src2);
  }
  void vfnmadd132ss(XMMRegister dst, XMMRegister src1, XMMRegister src2) {
    vfmas(0x9d, dst, src1, src2);
  }
  void vfnmadd213ss(XMMRegister dst, XMMRegister src1, XMMRegister src2) {
    vfmas(0xad, dst, src1, src2);
  }
  void vfnmadd231ss(XMMRegister dst, XMMRegister src1, XMMRegister src2) {
    vfmas(0xbd, dst, src1, src2);
  }
  void vfnmadd132ss(XMMRegister dst, XMMRegister src1, const Operand &src2) {
    vfmas(0x9d, dst, src1, src2);
  }
  void vfnmadd213ss(XMMRegister dst, XMMRegister src1, const Operand &src2) {
    vfmas(0xad, dst, src1, src2);
  }
  void vfnmadd231ss(XMMRegister dst, XMMRegister src1, const Operand &src2) {
    vfmas(0xbd, dst, src1, src2);
  }
  void vfnmsub132ss(XMMRegister dst, XMMRegister src1, XMMRegister src2) {
    vfmas(0x9f, dst, src1, src2);
  }
  void vfnmsub213ss(XMMRegister dst, XMMRegister src1, XMMRegister src2) {
    vfmas(0xaf, dst, src1, src2);
  }
  void vfnmsub231ss(XMMRegister dst, XMMRegister src1, XMMRegister src2) {
    vfmas(0xbf, dst, src1, src2);
  }
  void vfnmsub132ss(XMMRegister dst, XMMRegister src1, const Operand &src2) {
    vfmas(0x9f, dst, src1, src2);
  }
  void vfnmsub213ss(XMMRegister dst, XMMRegister src1, const Operand &src2) {
    vfmas(0xaf, dst, src1, src2);
  }
  void vfnmsub231ss(XMMRegister dst, XMMRegister src1, const Operand &src2) {
    vfmas(0xbf, dst, src1, src2);
  }

  // Scalar double XMM FMA instructions.
  void vfmadd132sd(XMMRegister dst, XMMRegister src1, XMMRegister src2) {
    vfmad(0x99, dst, src1, src2);
  }
  void vfmadd213sd(XMMRegister dst, XMMRegister src1, XMMRegister src2) {
    vfmad(0xa9, dst, src1, src2);
  }
  void vfmadd231sd(XMMRegister dst, XMMRegister src1, XMMRegister src2) {
    vfmad(0xb9, dst, src1, src2);
  }
  void vfmadd132sd(XMMRegister dst, XMMRegister src1, const Operand &src2) {
    vfmad(0x99, dst, src1, src2);
  }
  void vfmadd213sd(XMMRegister dst, XMMRegister src1, const Operand &src2) {
    vfmad(0xa9, dst, src1, src2);
  }
  void vfmadd231sd(XMMRegister dst, XMMRegister src1, const Operand &src2) {
    vfmad(0xb9, dst, src1, src2);
  }
  void vfmsub132sd(XMMRegister dst, XMMRegister src1, XMMRegister src2) {
    vfmad(0x9b, dst, src1, src2);
  }
  void vfmsub213sd(XMMRegister dst, XMMRegister src1, XMMRegister src2) {
    vfmad(0xab, dst, src1, src2);
  }
  void vfmsub231sd(XMMRegister dst, XMMRegister src1, XMMRegister src2) {
    vfmad(0xbb, dst, src1, src2);
  }
  void vfmsub132sd(XMMRegister dst, XMMRegister src1, const Operand &src2) {
    vfmad(0x9b, dst, src1, src2);
  }
  void vfmsub213sd(XMMRegister dst, XMMRegister src1, const Operand &src2) {
    vfmad(0xab, dst, src1, src2);
  }
  void vfmsub231sd(XMMRegister dst, XMMRegister src1, const Operand &src2) {
    vfmad(0xbb, dst, src1, src2);
  }
  void vfnmadd132sd(XMMRegister dst, XMMRegister src1, XMMRegister src2) {
    vfmad(0x9d, dst, src1, src2);
  }
  void vfnmadd213sd(XMMRegister dst, XMMRegister src1, XMMRegister src2) {
    vfmad(0xad, dst, src1, src2);
  }
  void vfnmadd231sd(XMMRegister dst, XMMRegister src1, XMMRegister src2) {
    vfmad(0xbd, dst, src1, src2);
  }
  void vfnmadd132sd(XMMRegister dst, XMMRegister src1, const Operand &src2) {
    vfmad(0x9d, dst, src1, src2);
  }
  void vfnmadd213sd(XMMRegister dst, XMMRegister src1, const Operand &src2) {
    vfmad(0xad, dst, src1, src2);
  }
  void vfnmadd231sd(XMMRegister dst, XMMRegister src1, const Operand &src2) {
    vfmad(0xbd, dst, src1, src2);
  }
  void vfnmsub132sd(XMMRegister dst, XMMRegister src1, XMMRegister src2) {
    vfmad(0x9f, dst, src1, src2);
  }
  void vfnmsub213sd(XMMRegister dst, XMMRegister src1, XMMRegister src2) {
    vfmad(0xaf, dst, src1, src2);
  }
  void vfnmsub231sd(XMMRegister dst, XMMRegister src1, XMMRegister src2) {
    vfmad(0xbf, dst, src1, src2);
  }
  void vfnmsub132sd(XMMRegister dst, XMMRegister src1, const Operand &src2) {
    vfmad(0x9f, dst, src1, src2);
  }
  void vfnmsub213sd(XMMRegister dst, XMMRegister src1, const Operand &src2) {
    vfmad(0xaf, dst, src1, src2);
  }
  void vfnmsub231sd(XMMRegister dst, XMMRegister src1, const Operand &src2) {
    vfmad(0xbf, dst, src1, src2);
  }

  // Vector single XMM FMA instructions.
  void vfmadd132ps(XMMRegister dst, XMMRegister src1, XMMRegister src2) {
    vfmas(0x98, dst, src1, src2);
  }
  void vfmadd132ps(XMMRegister dst, XMMRegister src1, const Operand &src2) {
    vfmas(0x98, dst, src1, src2);
  }
  void vfmadd213ps(XMMRegister dst, XMMRegister src1, XMMRegister src2) {
    vfmas(0xa8, dst, src1, src2);
  }
  void vfmadd213ps(XMMRegister dst, XMMRegister src1, const Operand &src2) {
    vfmas(0xa8, dst, src1, src2);
  }
  void vfmadd231ps(XMMRegister dst, XMMRegister src1, XMMRegister src2) {
    vfmas(0xb8, dst, src1, src2);
  }
  void vfmadd231ps(XMMRegister dst, XMMRegister src1, const Operand &src2) {
    vfmas(0xb8, dst, src1, src2);
  }
  void vfmsub132ps(XMMRegister dst, XMMRegister src1, XMMRegister src2) {
    vfmas(0x9a, dst, src1, src2);
  }
  void vfmsub132ps(XMMRegister dst, XMMRegister src1, const Operand &src2) {
    vfmas(0x9a, dst, src1, src2);
  }
  void vfmsub213ps(XMMRegister dst, XMMRegister src1, XMMRegister src2) {
    vfmas(0xaa, dst, src1, src2);
  }
  void vfmsub213ps(XMMRegister dst, XMMRegister src1, const Operand &src2) {
    vfmas(0xaa, dst, src1, src2);
  }
  void vfmsub231ps(XMMRegister dst, XMMRegister src1, XMMRegister src2) {
    vfmas(0xba, dst, src1, src2);
  }
  void vfmsub231ps(XMMRegister dst, XMMRegister src1, const Operand &src2) {
    vfmas(0xba, dst, src1, src2);
  }

  // Vector double XMM FMA instructions.
  void vfmadd132pd(XMMRegister dst, XMMRegister src1, XMMRegister src2) {
    vfmad(0x98, dst, src1, src2);
  }
  void vfmadd132pd(XMMRegister dst, XMMRegister src1, const Operand &src2) {
    vfmad(0x98, dst, src1, src2);
  }
  void vfmadd213pd(XMMRegister dst, XMMRegister src1, XMMRegister src2) {
    vfmad(0xa8, dst, src1, src2);
  }
  void vfmadd213pd(XMMRegister dst, XMMRegister src1, const Operand &src2) {
    vfmad(0xa8, dst, src1, src2);
  }
  void vfmadd231pd(XMMRegister dst, XMMRegister src1, XMMRegister src2) {
    vfmad(0xb8, dst, src1, src2);
  }
  void vfmadd231pd(XMMRegister dst, XMMRegister src1, const Operand &src2) {
    vfmad(0xb8, dst, src1, src2);
  }
  void vfmsub132pd(XMMRegister dst, XMMRegister src1, XMMRegister src2) {
    vfmad(0x9a, dst, src1, src2);
  }
  void vfmsub132pd(XMMRegister dst, XMMRegister src1, const Operand &src2) {
    vfmad(0x9a, dst, src1, src2);
  }
  void vfmsub213pd(XMMRegister dst, XMMRegister src1, XMMRegister src2) {
    vfmad(0xaa, dst, src1, src2);
  }
  void vfmsub213pd(XMMRegister dst, XMMRegister src1, const Operand &src2) {
    vfmad(0xaa, dst, src1, src2);
  }
  void vfmsub231pd(XMMRegister dst, XMMRegister src1, XMMRegister src2) {
    vfmad(0xba, dst, src1, src2);
  }
  void vfmsub231pd(XMMRegister dst, XMMRegister src1, const Operand &src2) {
    vfmad(0xba, dst, src1, src2);
  }

  // Vector single YMM FMA instructions.
  void vfmadd132ps(YMMRegister dst, YMMRegister src1, YMMRegister src2) {
    vfmas(0x98, dst, src1, src2);
  }
  void vfmadd132ps(YMMRegister dst, YMMRegister src1, const Operand &src2) {
    vfmas(0x98, dst, src1, src2);
  }
  void vfmadd213ps(YMMRegister dst, YMMRegister src1, YMMRegister src2) {
    vfmas(0xa8, dst, src1, src2);
  }
  void vfmadd213ps(YMMRegister dst, YMMRegister src1, const Operand &src2) {
    vfmas(0xa8, dst, src1, src2);
  }
  void vfmadd231ps(YMMRegister dst, YMMRegister src1, YMMRegister src2) {
    vfmas(0xb8, dst, src1, src2);
  }
  void vfmadd231ps(YMMRegister dst, YMMRegister src1, const Operand &src2) {
    vfmas(0xb8, dst, src1, src2);
  }
  void vfmsub132ps(YMMRegister dst, YMMRegister src1, YMMRegister src2) {
    vfmas(0x9a, dst, src1, src2);
  }
  void vfmsub132ps(YMMRegister dst, YMMRegister src1, const Operand &src2) {
    vfmas(0x9a, dst, src1, src2);
  }
  void vfmsub213ps(YMMRegister dst, YMMRegister src1, YMMRegister src2) {
    vfmas(0xaa, dst, src1, src2);
  }
  void vfmsub213ps(YMMRegister dst, YMMRegister src1, const Operand &src2) {
    vfmas(0xaa, dst, src1, src2);
  }
  void vfmsub231ps(YMMRegister dst, YMMRegister src1, YMMRegister src2) {
    vfmas(0xba, dst, src1, src2);
  }
  void vfmsub231ps(YMMRegister dst, YMMRegister src1, const Operand &src2) {
    vfmas(0xba, dst, src1, src2);
  }

  // Vector double YMM FMA instructions.
  void vfmadd132pd(YMMRegister dst, YMMRegister src1, YMMRegister src2) {
    vfmad(0x98, dst, src1, src2);
  }
  void vfmadd132pd(YMMRegister dst, YMMRegister src1, const Operand &src2) {
    vfmad(0x98, dst, src1, src2);
  }
  void vfmadd213pd(YMMRegister dst, YMMRegister src1, YMMRegister src2) {
    vfmad(0xa8, dst, src1, src2);
  }
  void vfmadd213pd(YMMRegister dst, YMMRegister src1, const Operand &src2) {
    vfmad(0xa8, dst, src1, src2);
  }
  void vfmadd231pd(YMMRegister dst, YMMRegister src1, YMMRegister src2) {
    vfmad(0xb8, dst, src1, src2);
  }
  void vfmadd231pd(YMMRegister dst, YMMRegister src1, const Operand &src2) {
    vfmad(0xb8, dst, src1, src2);
  }
  void vfmsub132pd(YMMRegister dst, YMMRegister src1, YMMRegister src2) {
    vfmad(0x9a, dst, src1, src2);
  }
  void vfmsub132pd(YMMRegister dst, YMMRegister src1, const Operand &src2) {
    vfmad(0x9a, dst, src1, src2);
  }
  void vfmsub213pd(YMMRegister dst, YMMRegister src1, YMMRegister src2) {
    vfmad(0xaa, dst, src1, src2);
  }
  void vfmsub213pd(YMMRegister dst, YMMRegister src1, const Operand &src2) {
    vfmad(0xaa, dst, src1, src2);
  }
  void vfmsub231pd(YMMRegister dst, YMMRegister src1, YMMRegister src2) {
    vfmad(0xba, dst, src1, src2);
  }
  void vfmsub231pd(YMMRegister dst, YMMRegister src1, const Operand &src2) {
    vfmad(0xba, dst, src1, src2);
  }

  // AVX-512 opmask register instructions.
  void kmovb(OpmaskRegister k1, OpmaskRegister k2) {
    kinstr(0x90, k1, k2, k66, k0F, kW0);
  }
  void kmovb(OpmaskRegister k1, const Operand &src) {
    kinstr(0x90, k1, src, k66, k0F, kW0);
  }
  void kmovb(const Operand &dst, OpmaskRegister k1) {
    kinstr(0x91, dst, k1, k66, k0F, kW0);
  }
  void kmovb(OpmaskRegister k1, Register src) {
    kinstr(0x92, k1, src, k66, k0F, kW0);
  }
  void kmovb(Register dst,  OpmaskRegister k1) {
    kinstr(0x93, dst, k1, k66, k0F, kW0);
  }

  void kmovw(OpmaskRegister k1, OpmaskRegister k2) {
    kinstr(0x90, k1, k2, kNone, k0F, kW0);
  }
  void kmovw(OpmaskRegister k1, const Operand &src) {
    kinstr(0x90, k1, src, kNone, k0F, kW0);
  }
  void kmovw(const Operand &dst, OpmaskRegister k1) {
    kinstr(0x91, dst, k1, kNone, k0F, kW0);
  }
  void kmovw(OpmaskRegister k1, Register src) {
    kinstr(0x92, k1, src, kNone, k0F, kW0);
  }
  void kmovw(Register dst,  OpmaskRegister k1) {
    kinstr(0x93, dst, k1, kNone, k0F, kW0);
  }

  void kmovd(OpmaskRegister k1, OpmaskRegister k2) {
    kinstr(0x90, k1, k2, k66, k0F, kW1);
  }
  void kmovd(OpmaskRegister k1, const Operand &src) {
    kinstr(0x90, k1, src, k66, k0F, kW1);
  }
  void kmovd(const Operand &dst, OpmaskRegister k1) {
    kinstr(0x91, dst, k1, k66, k0F, kW1);
  }
  void kmovd(OpmaskRegister k1, Register src) {
    kinstr(0x92, k1, src, kF2, k0F, kW0);
  }
  void kmovd(Register dst,  OpmaskRegister k1) {
    kinstr(0x93, dst, k1, kF2, k0F, kW0);
  }

  void kmovq(OpmaskRegister k1, OpmaskRegister k2) {
    kinstr(0x90, k1, k2, kNone, k0F, kW1);
  }
  void kmovq(OpmaskRegister k1, const Operand &src) {
    kinstr(0x90, k1, src, kNone, k0F, kW1);
  }
  void kmovq(const Operand &dst, OpmaskRegister k1) {
    kinstr(0x91, dst, k1, kNone, k0F, kW1);
  }
  void kmovq(OpmaskRegister k1, Register src) {
    kinstr(0x92, k1, src, kF2, k0F, kW1);
  }
  void kmovq(Register dst,  OpmaskRegister k1) {
    kinstr(0x93, dst, k1, kF2, k0F, kW1);
  }

  void kandb(OpmaskRegister k1, OpmaskRegister k2, OpmaskRegister k3) {
    kinstr(0x41, k1, k2, k3, k66, k0F, kW0);
  }
  void kandw(OpmaskRegister k1, OpmaskRegister k2, OpmaskRegister k3) {
    kinstr(0x41, k1, k2, k3, kNone, k0F, kW0);
  }
  void kandd(OpmaskRegister k1, OpmaskRegister k2, OpmaskRegister k3) {
    kinstr(0x41, k1, k2, k3, k66, k0F, kW1);
  }
  void kandq(OpmaskRegister k1, OpmaskRegister k2, OpmaskRegister k3) {
    kinstr(0x41, k1, k2, k3, kNone, k0F, kW1);
  }

  void korb(OpmaskRegister k1, OpmaskRegister k2, OpmaskRegister k3) {
    kinstr(0x45, k1, k2, k3, k66, k0F, kW0);
  }
  void korw(OpmaskRegister k1, OpmaskRegister k2, OpmaskRegister k3) {
    kinstr(0x45, k1, k2, k3, kNone, k0F, kW0);
  }
  void kord(OpmaskRegister k1, OpmaskRegister k2, OpmaskRegister k3) {
    kinstr(0x45, k1, k2, k3, k66, k0F, kW1);
  }
  void korq(OpmaskRegister k1, OpmaskRegister k2, OpmaskRegister k3) {
    kinstr(0x45, k1, k2, k3, kNone, k0F, kW1);
  }

  void knotb(OpmaskRegister k1, OpmaskRegister k2) {
    kinstr(0x44, k1, k2, k66, k0F, kW0);
  }
  void knotw(OpmaskRegister k1, OpmaskRegister k2) {
    kinstr(0x44, k1, k2, kNone, k0F, kW0);
  }
  void knotd(OpmaskRegister k1, OpmaskRegister k2) {
    kinstr(0x44, k1, k2, k66, k0F, kW1);
  }
  void knotq(OpmaskRegister k1, OpmaskRegister k2) {
    kinstr(0x44, k1, k2, kNone, k0F, kW1);
  }

  void kxorb(OpmaskRegister k1, OpmaskRegister k2, OpmaskRegister k3) {
    kinstr(0x47, k1, k2, k3, k66, k0F, kW0);
  }
  void kxorw(OpmaskRegister k1, OpmaskRegister k2, OpmaskRegister k3) {
    kinstr(0x47, k1, k2, k3, kNone, k0F, kW0);
  }
  void kxord(OpmaskRegister k1, OpmaskRegister k2, OpmaskRegister k3) {
    kinstr(0x47, k1, k2, k3, k66, k0F, kW1);
  }
  void kxorq(OpmaskRegister k1, OpmaskRegister k2, OpmaskRegister k3) {
    kinstr(0x47, k1, k2, k3, kNone, k0F, kW1);
  }

  void kandnb(OpmaskRegister k1, OpmaskRegister k2, OpmaskRegister k3) {
    kinstr(0x42, k1, k2, k3, k66, k0F, kW0);
  }
  void kandnw(OpmaskRegister k1, OpmaskRegister k2, OpmaskRegister k3) {
    kinstr(0x42, k1, k2, k3, kNone, k0F, kW0);
  }
  void kandnd(OpmaskRegister k1, OpmaskRegister k2, OpmaskRegister k3) {
    kinstr(0x42, k1, k2, k3, k66, k0F, kW1);
  }
  void kandnq(OpmaskRegister k1, OpmaskRegister k2, OpmaskRegister k3) {
    kinstr(0x42, k1, k2, k3, kNone, k0F, kW1);
  }

  void kxnorb(OpmaskRegister k1, OpmaskRegister k2, OpmaskRegister k3) {
    kinstr(0x46, k1, k2, k3, k66, k0F, kW0);
  }
  void kxnorw(OpmaskRegister k1, OpmaskRegister k2, OpmaskRegister k3) {
    kinstr(0x46, k1, k2, k3, kNone, k0F, kW0);
  }
  void kxnord(OpmaskRegister k1, OpmaskRegister k2, OpmaskRegister k3) {
    kinstr(0x46, k1, k2, k3, k66, k0F, kW1);
  }
  void kxnorq(OpmaskRegister k1, OpmaskRegister k2, OpmaskRegister k3) {
    kinstr(0x46, k1, k2, k3, kNone, k0F, kW1);
  }

  void kshiftlb(OpmaskRegister k1, OpmaskRegister k2, int8_t imm8) {
    kinstr(0x32, k1, k2, imm8, k66, k0F3A, kW0);
  }
  void kshiftlw(OpmaskRegister k1, OpmaskRegister k2, int8_t imm8) {
    kinstr(0x32, k1, k2, imm8, k66, k0F3A, kW1);
  }
  void kshiftld(OpmaskRegister k1, OpmaskRegister k2, int8_t imm8) {
    kinstr(0x33, k1, k2, imm8, k66, k0F3A, kW0);
  }
  void kshiftlq(OpmaskRegister k1, OpmaskRegister k2, int8_t imm8) {
    kinstr(0x33, k1, k2, imm8, k66, k0F3A, kW1);
  }

  void kshiftrb(OpmaskRegister k1, OpmaskRegister k2, int8_t imm8) {
    kinstr(0x30, k1, k2, imm8, k66, k0F3A, kW0);
  }
  void kshiftrw(OpmaskRegister k1, OpmaskRegister k2, int8_t imm8) {
    kinstr(0x30, k1, k2, imm8, k66, k0F3A, kW1);
  }
  void kshiftrd(OpmaskRegister k1, OpmaskRegister k2, int8_t imm8) {
    kinstr(0x31, k1, k2, imm8, k66, k0F3A, kW0);
  }
  void kshiftrq(OpmaskRegister k1, OpmaskRegister k2, int8_t imm8) {
    kinstr(0x31, k1, k2, imm8, k66, k0F3A, kW1);
  }

  void ktestb(OpmaskRegister k1, OpmaskRegister k2) {
    kinstr(0x99, k1, k2, k66, k0F, kW0);
  }
  void ktestw(OpmaskRegister k1, OpmaskRegister k2) {
    kinstr(0x99, k1, k2, kNone, k0F, kW0);
  }
  void ktestd(OpmaskRegister k1, OpmaskRegister k2) {
    kinstr(0x99, k1, k2, k66, k0F, kW1);
  }
  void ktestq(OpmaskRegister k1, OpmaskRegister k2) {
    kinstr(0x99, k1, k2, kNone, k0F, kW1);
  }

  void kortestb(OpmaskRegister k1, OpmaskRegister k2) {
    kinstr(0x98, k1, k2, k66, k0F, kW0);
  }
  void kortestw(OpmaskRegister k1, OpmaskRegister k2) {
    kinstr(0x98, k1, k2, kNone, k0F, kW0);
  }
  void kortestd(OpmaskRegister k1, OpmaskRegister k2) {
    kinstr(0x98, k1, k2, k66, k0F, kW1);
  }
  void kortestq(OpmaskRegister k1, OpmaskRegister k2) {
    kinstr(0x98, k1, k2, kNone, k0F, kW1);
  }

  void kunpckbw(OpmaskRegister k1, OpmaskRegister k2, OpmaskRegister k3) {
    kinstr(0x4B, k1, k2, k3, k66, k0F, kW0);
  }
  void kunpckwd(OpmaskRegister k1, OpmaskRegister k2, OpmaskRegister k3) {
    kinstr(0x4B, k1, k2, k3, kNone, k0F, kW0);
  }
  void kunpckdq(OpmaskRegister k1, OpmaskRegister k2, OpmaskRegister k3) {
    kinstr(0x4B, k1, k2, k3, kNone, k0F, kW1);
  }

  void kaddb(OpmaskRegister k1, OpmaskRegister k2, OpmaskRegister k3) {
    kinstr(0x4A, k1, k2, k3, k66, k0F, kW0);
  }
  void kaddw(OpmaskRegister k1, OpmaskRegister k2, OpmaskRegister k3) {
    kinstr(0x4A, k1, k2, k3, kNone, k0F, kW0);
  }
  void kaddd(OpmaskRegister k1, OpmaskRegister k2, OpmaskRegister k3) {
    kinstr(0x4A, k1, k2, k3, k66, k0F, kW1);
  }
  void kaddq(OpmaskRegister k1, OpmaskRegister k2, OpmaskRegister k3) {
    kinstr(0x4A, k1, k2, k3, kNone, k0F, kW1);
  }

  // AVX-512F instructions.
  #include "third_party/jit/avx512.inc"

  // BMI instructions.
  void andnq(Register dst, Register src1, Register src2) {
    bmi1q(0xf2, dst, src1, src2);
  }
  void andnq(Register dst, Register src1, const Operand &src2) {
    bmi1q(0xf2, dst, src1, src2);
  }
  void andnl(Register dst, Register src1, Register src2) {
    bmi1l(0xf2, dst, src1, src2);
  }
  void andnl(Register dst, Register src1, const Operand &src2) {
    bmi1l(0xf2, dst, src1, src2);
  }
  void bextrq(Register dst, Register src1, Register src2) {
    bmi1q(0xf7, dst, src2, src1);
  }
  void bextrq(Register dst, const Operand &src1, Register src2) {
    bmi1q(0xf7, dst, src2, src1);
  }
  void bextrl(Register dst, Register src1, Register src2) {
    bmi1l(0xf7, dst, src2, src1);
  }
  void bextrl(Register dst, const Operand &src1, Register src2) {
    bmi1l(0xf7, dst, src2, src1);
  }
  void blsiq(Register dst, Register src) {
    Register ireg = {3};
    bmi1q(0xf3, ireg, dst, src);
  }
  void blsiq(Register dst, const Operand &src) {
    Register ireg = {3};
    bmi1q(0xf3, ireg, dst, src);
  }
  void blsil(Register dst, Register src) {
    Register ireg = {3};
    bmi1l(0xf3, ireg, dst, src);
  }
  void blsil(Register dst, const Operand &src) {
    Register ireg = {3};
    bmi1l(0xf3, ireg, dst, src);
  }
  void blsmskq(Register dst, Register src) {
    Register ireg = {2};
    bmi1q(0xf3, ireg, dst, src);
  }
  void blsmskq(Register dst, const Operand &src) {
    Register ireg = {2};
    bmi1q(0xf3, ireg, dst, src);
  }
  void blsmskl(Register dst, Register src) {
    Register ireg = {2};
    bmi1l(0xf3, ireg, dst, src);
  }
  void blsmskl(Register dst, const Operand &src) {
    Register ireg = {2};
    bmi1l(0xf3, ireg, dst, src);
  }
  void blsrq(Register dst, Register src) {
    Register ireg = {1};
    bmi1q(0xf3, ireg, dst, src);
  }
  void blsrq(Register dst, const Operand &src) {
    Register ireg = {1};
    bmi1q(0xf3, ireg, dst, src);
  }
  void blsrl(Register dst, Register src) {
    Register ireg = {1};
    bmi1l(0xf3, ireg, dst, src);
  }
  void blsrl(Register dst, const Operand &src) {
    Register ireg = {1};
    bmi1l(0xf3, ireg, dst, src);
  }
  void tzcntq(Register dst, Register src);
  void tzcntq(Register dst, const Operand &src);
  void tzcntl(Register dst, Register src);
  void tzcntl(Register dst, const Operand &src);

  void lzcntq(Register dst, Register src);
  void lzcntq(Register dst, const Operand &src);
  void lzcntl(Register dst, Register src);
  void lzcntl(Register dst, const Operand &src);

  void popcntq(Register dst, Register src);
  void popcntq(Register dst, const Operand &src);
  void popcntl(Register dst, Register src);
  void popcntl(Register dst, const Operand &src);

  void bzhiq(Register dst, Register src1, Register src2) {
    bmi2q(kNone, 0xf5, dst, src2, src1);
  }
  void bzhiq(Register dst, const Operand &src1, Register src2) {
    bmi2q(kNone, 0xf5, dst, src2, src1);
  }
  void bzhil(Register dst, Register src1, Register src2) {
    bmi2l(kNone, 0xf5, dst, src2, src1);
  }
  void bzhil(Register dst, const Operand &src1, Register src2) {
    bmi2l(kNone, 0xf5, dst, src2, src1);
  }
  void mulxq(Register dst1, Register dst2, Register src) {
    bmi2q(kF2, 0xf6, dst1, dst2, src);
  }
  void mulxq(Register dst1, Register dst2, const Operand &src) {
    bmi2q(kF2, 0xf6, dst1, dst2, src);
  }
  void mulxl(Register dst1, Register dst2, Register src) {
    bmi2l(kF2, 0xf6, dst1, dst2, src);
  }
  void mulxl(Register dst1, Register dst2, const Operand &src) {
    bmi2l(kF2, 0xf6, dst1, dst2, src);
  }
  void pdepq(Register dst, Register src1, Register src2) {
    bmi2q(kF2, 0xf5, dst, src1, src2);
  }
  void pdepq(Register dst, Register src1, const Operand &src2) {
    bmi2q(kF2, 0xf5, dst, src1, src2);
  }
  void pdepl(Register dst, Register src1, Register src2) {
    bmi2l(kF2, 0xf5, dst, src1, src2);
  }
  void pdepl(Register dst, Register src1, const Operand &src2) {
    bmi2l(kF2, 0xf5, dst, src1, src2);
  }
  void pextq(Register dst, Register src1, Register src2) {
    bmi2q(kF3, 0xf5, dst, src1, src2);
  }
  void pextq(Register dst, Register src1, const Operand &src2) {
    bmi2q(kF3, 0xf5, dst, src1, src2);
  }
  void pextl(Register dst, Register src1, Register src2) {
    bmi2l(kF3, 0xf5, dst, src1, src2);
  }
  void pextl(Register dst, Register src1, const Operand &src2) {
    bmi2l(kF3, 0xf5, dst, src1, src2);
  }
  void sarxq(Register dst, Register src1, Register src2) {
    bmi2q(kF3, 0xf7, dst, src2, src1);
  }
  void sarxq(Register dst, const Operand &src1, Register src2) {
    bmi2q(kF3, 0xf7, dst, src2, src1);
  }
  void sarxl(Register dst, Register src1, Register src2) {
    bmi2l(kF3, 0xf7, dst, src2, src1);
  }
  void sarxl(Register dst, const Operand &src1, Register src2) {
    bmi2l(kF3, 0xf7, dst, src2, src1);
  }
  void shlxq(Register dst, Register src1, Register src2) {
    bmi2q(k66, 0xf7, dst, src2, src1);
  }
  void shlxq(Register dst, const Operand &src1, Register src2) {
    bmi2q(k66, 0xf7, dst, src2, src1);
  }
  void shlxl(Register dst, Register src1, Register src2) {
    bmi2l(k66, 0xf7, dst, src2, src1);
  }
  void shlxl(Register dst, const Operand &src1, Register src2) {
    bmi2l(k66, 0xf7, dst, src2, src1);
  }
  void shrxq(Register dst, Register src1, Register src2) {
    bmi2q(kF2, 0xf7, dst, src2, src1);
  }
  void shrxq(Register dst, const Operand &src1, Register src2) {
    bmi2q(kF2, 0xf7, dst, src2, src1);
  }
  void shrxl(Register dst, Register src1, Register src2) {
    bmi2l(kF2, 0xf7, dst, src2, src1);
  }
  void shrxl(Register dst, const Operand &src1, Register src2) {
    bmi2l(kF2, 0xf7, dst, src2, src1);
  }
  void rorxq(Register dst, Register src, byte imm8);
  void rorxq(Register dst, const Operand &src, byte imm8);
  void rorxl(Register dst, Register src, byte imm8);
  void rorxl(Register dst, const Operand &src, byte imm8);

  // Writes a single word of data in the code stream.
  // Used for inline tables, e.g., jump-tables.
  void db(uint8_t data);
  void dd(uint32_t data);
  void dq(uint64_t data);
  void dp(uintptr_t data) { dq(data); }
  void dq(Label *label);

  // Call near indirect
  void call(const Operand &operand);

 private:
  // Code emission.
  void emit(byte x) { *pc_++ = x; }

  void emitl(uint32_t x) {
    Memory::uint32_at(pc_) = x;
    pc_ += sizeof(uint32_t);
  }

  void emitp(const void *x) {
    uintptr_t value = reinterpret_cast<uintptr_t>(x);
    Memory::uintptr_at(pc_) = value;
    pc_ += sizeof(uintptr_t);
  }

  void emitq(uint64_t x) {
    Memory::uint64_at(pc_) = x;
    pc_ += sizeof(uint64_t);
  }

  void emitw(uint16_t x) {
    Memory::uint16_at(pc_) = x;
    pc_ += sizeof(uint16_t);
  }

  void emit(Immediate x) {
    emitl(x.value_);
  }

  // Emits a REX prefix that encodes a 64-bit operand size and
  // the top bit of both register codes.
  // High bit of reg goes to REX.R, high bit of rm_reg goes to REX.B.
  // REX.W is set.
  void emit_rex_64(Register reg, Register rm_reg) {
    emit(0x48 | reg.high_bit() << 2 | rm_reg.high_bit());
  }
  void emit_rex_64(XMMRegister reg, Register rm_reg) {
    emit(0x48 | (reg.code() & 0x8) >> 1 | rm_reg.code() >> 3);
  }
  void emit_rex_64(Register reg, XMMRegister rm_reg) {
    emit(0x48 | (reg.code() & 0x8) >> 1 | rm_reg.code() >> 3);
  }

  // Emits a REX prefix that encodes a 64-bit operand size and
  // the top bit of the destination, index, and base register codes.
  // The high bit of reg is used for REX.R, the high bit of op's base
  // register is used for REX.B, and the high bit of op's index register
  // is used for REX.X.  REX.W is set.
  void emit_rex_64(Register reg, const Operand &op) {
    emit(0x48 | reg.high_bit() << 2 | op.rex_);
  }
  void emit_rex_64(XMMRegister reg, const Operand &op) {
    emit(0x48 | (reg.code() & 0x8) >> 1 | op.rex_);
  }

  // Emits a REX prefix that encodes a 64-bit operand size and
  // the top bit of the register code.
  // The high bit of register is used for REX.B.
  // REX.W is set and REX.R and REX.X are clear.
  void emit_rex_64(Register rm_reg) {
    DCHECK_EQ(rm_reg.code() & 0xf, rm_reg.code());
    emit(0x48 | rm_reg.high_bit());
  }

  // Emits a REX prefix that encodes a 64-bit operand size and
  // the top bit of the index and base register codes.
  // The high bit of op's base register is used for REX.B, and the high
  // bit of op's index register is used for REX.X.
  // REX.W is set and REX.R clear.
  void emit_rex_64(const Operand &op) {
    emit(0x48 | op.rex_);
  }

  // Emit a REX prefix that only sets REX.W to choose a 64-bit operand size.
  void emit_rex_64() { emit(0x48); }

  // High bit of reg goes to REX.R, high bit of rm_reg goes to REX.B.
  // REX.W is clear.
  void emit_rex_32(Register reg, Register rm_reg) {
    emit(0x40 | reg.high_bit() << 2 | rm_reg.high_bit());
  }

  // The high bit of reg is used for REX.R, the high bit of op's base
  // register is used for REX.B, and the high bit of op's index register
  // is used for REX.X.  REX.W is cleared.
  void emit_rex_32(Register reg, const Operand &op) {
    emit(0x40 | reg.high_bit() << 2  | op.rex_);
  }

  // High bit of rm_reg goes to REX.B.
  // REX.W, REX.R and REX.X are clear.
  void emit_rex_32(Register rm_reg) {
    emit(0x40 | rm_reg.high_bit());
  }

  // High bit of base goes to REX.B and high bit of index to REX.X.
  // REX.W and REX.R are clear.
  void emit_rex_32(const Operand &op) {
    emit(0x40 | op.rex_);
  }

  // High bit of reg goes to REX.R, high bit of rm_reg goes to REX.B.
  // REX.W is cleared.  If no REX bits are set, no byte is emitted.
  void emit_optional_rex_32(Register reg, Register rm_reg) {
    byte rex_bits = reg.high_bit() << 2 | rm_reg.high_bit();
    if (rex_bits != 0) emit(0x40 | rex_bits);
  }

  // The high bit of reg is used for REX.R, the high bit of op's base
  // register is used for REX.B, and the high bit of op's index register
  // is used for REX.X.  REX.W is cleared.  If no REX bits are set, nothing
  // is emitted.
  void emit_optional_rex_32(Register reg, const Operand &op) {
    byte rex_bits =  reg.high_bit() << 2 | op.rex_;
    if (rex_bits != 0) emit(0x40 | rex_bits);
  }

  // As for emit_optional_rex_32(Register, Register), except that
  // the registers are XMM registers.
  void emit_optional_rex_32(XMMRegister reg, Register base) {
    byte rex_bits =  (reg.code() & 0x8) >> 1 | (base.code() & 0x8) >> 3;
    if (rex_bits != 0) emit(0x40 | rex_bits);
  }

  // As for emit_optional_rex_32(Register, Register), except that
  // one of the registers is an XMM registers.
  void emit_optional_rex_32(XMMRegister reg, XMMRegister base) {
    byte rex_bits =  (reg.code() & 0x8) >> 1 | (base.code() & 0x8) >> 3;
    if (rex_bits != 0) emit(0x40 | rex_bits);
  }

  // As for emit_optional_rex_32(Register, Register), except that
  // one of the registers is an XMM registers.
  void emit_optional_rex_32(Register reg, XMMRegister base) {
    byte rex_bits =  (reg.code() & 0x8) >> 1 | (base.code() & 0x8) >> 3;
    if (rex_bits != 0) emit(0x40 | rex_bits);
  }

  // As for emit_optional_rex_32(Register, const Operand&), except that
  // the register is an XMM register.
  void emit_optional_rex_32(XMMRegister reg, const Operand &op) {
    byte rex_bits =  (reg.code() & 0x8) >> 1 | op.rex_;
    if (rex_bits != 0) emit(0x40 | rex_bits);
  }

  // Optionally do as emit_rex_32(Register) if the register number has
  // the high bit set.
  void emit_optional_rex_32(Register rm_reg) {
    if (rm_reg.high_bit()) emit(0x41);
  }
  void emit_optional_rex_32(XMMRegister rm_reg) {
    if (rm_reg.high_bit()) emit(0x41);
  }

  // Optionally do as emit_rex_32(const Operand&) if the operand register
  // numbers have a high bit set.
  void emit_optional_rex_32(const Operand &op) {
    if (op.rex_ != 0) emit(0x40 | op.rex_);
  }

  void emit_rex(int size) {
    if (size == kInt64Size) {
      emit_rex_64();
    }
  }

  template<class P1>
  void emit_rex(P1 p1, int size) {
    if (size == kInt64Size) {
      emit_rex_64(p1);
    } else {
      emit_optional_rex_32(p1);
    }
  }

  template<class P1, class P2>
  void emit_rex(P1 p1, P2 p2, int size) {
    if (size == kInt64Size) {
      emit_rex_64(p1, p2);
    } else {
      emit_optional_rex_32(p1, p2);
    }
  }

  // Emit VEX prefix.
  void emit_vex2_byte0() { emit(0xc5); }

  void emit_vex2_byte1(XMMRegister reg, XMMRegister v, VectorLength l,
                       SIMDPrefix pp) {
    byte rv = ~((reg.high_bit() << 4) | v.code()) << 3;
    emit(rv | l | pp);
  }

  void emit_vex3_byte0() { emit(0xc4); }

  void emit_vex3_byte1(XMMRegister reg, XMMRegister rm, LeadingOpcode m) {
    byte rxb = ~((reg.high_bit() << 2) | rm.high_bit()) << 5;
    emit(rxb | m);
  }

  void emit_vex3_byte1(XMMRegister reg, const Operand &rm, LeadingOpcode m) {
    byte rxb = ~((reg.high_bit() << 2) | rm.rex_) << 5;
    emit(rxb | m);
  }

  void emit_vex3_byte2(VexW w, XMMRegister v, VectorLength l, SIMDPrefix pp) {
    emit(w | ((~v.code() & 0xf) << 3) | l | pp);
  }

  void emit_vex_prefix(XMMRegister reg, XMMRegister vreg, XMMRegister rm,
                       VectorLength l, SIMDPrefix pp, LeadingOpcode mm,
                       VexW w) {
    if (rm.high_bit() || mm != k0F || w != kW0) {
      emit_vex3_byte0();
      emit_vex3_byte1(reg, rm, mm);
      emit_vex3_byte2(w, vreg, l, pp);
    } else {
      emit_vex2_byte0();
      emit_vex2_byte1(reg, vreg, l, pp);
    }
  }

  void emit_vex_prefix(Register reg, Register vreg, Register rm, VectorLength l,
                       SIMDPrefix pp, LeadingOpcode mm, VexW w) {
    XMMRegister ireg = {reg.code()};
    XMMRegister ivreg = {vreg.code()};
    XMMRegister irm = {rm.code()};
    emit_vex_prefix(ireg, ivreg, irm, l, pp, mm, w);
  }

  void emit_vex_prefix(XMMRegister reg, XMMRegister vreg, const Operand &rm,
                       VectorLength l, SIMDPrefix pp, LeadingOpcode mm,
                       VexW w) {
    if (rm.rex_ || mm != k0F || w != kW0) {
      emit_vex3_byte0();
      emit_vex3_byte1(reg, rm, mm);
      emit_vex3_byte2(w, vreg, l, pp);
    } else {
      emit_vex2_byte0();
      emit_vex2_byte1(reg, vreg, l, pp);
    }
  }

  void emit_vex_prefix(Register reg, Register vreg, const Operand &rm,
                       VectorLength l, SIMDPrefix pp, LeadingOpcode mm,
                       VexW w) {
    XMMRegister ireg = {reg.code()};
    XMMRegister ivreg = {vreg.code()};
    emit_vex_prefix(ireg, ivreg, rm, l, pp, mm, w);
  }

  // Emit EVEX prefix.
  void emit_evex_prefix(ZMMRegister reg, ZMMRegister vreg, ZMMRegister rm,
                        Mask mask, int flags);
  void emit_evex_prefix(ZMMRegister reg, ZMMRegister vreg, const Operand &rm,
                        Mask mask, int flags);

  // Emit the ModR/M byte, and optionally the SIB byte and
  // 1- or 4-byte offset for a memory operand.  Also encodes
  // the second operand of the operation, a register or operation
  // subcode, into the reg field of the ModR/M byte.
  void emit_operand(Register reg, const Operand &adr, int sl = 0, int tl = 0) {
    emit_operand(reg.low_bits(), adr, sl);
  }

  // Emit the ModR/M byte, and optionally the SIB byte and
  // 1- or 4-byte offset for a memory operand.  Also used to encode
  // a three-bit opcode extension into the ModR/M byte.
  // The sl parameter encodes the instruction suffix length, i.e. the number
  // of bytes in the instruction after the operand. Currently only suffix
  // lengths of 0 and 1 are supported.
  // The ts parameter encodes the tuple size used for EVEX disp8*N compression.
  void emit_operand(int code, const Operand &adr, int sl = 0, int ts = 0);

  // Emit a ModR/M byte with registers coded in the reg and rm_reg fields.
  void emit_modrm(Register reg, Register rm_reg) {
    emit(0xC0 | reg.low_bits() << 3 | rm_reg.low_bits());
  }

  // Emit a ModR/M byte with an operation subcode in the reg field and
  // a register in the rm_reg field.
  void emit_modrm(int code, Register rm_reg) {
    DCHECK(is_uint3(code));
    emit(0xC0 | code << 3 | rm_reg.low_bits());
  }

  // The first argument is the reg field, the second argument is the r/m field.
  void emit_sse_operand(XMMRegister dst, XMMRegister src);
  void emit_sse_operand(XMMRegister reg, const Operand &adr, int sl = 0);
  void emit_sse_operand(Register reg, const Operand &adr, int sl = 0);
  void emit_sse_operand(XMMRegister dst, Register src);
  void emit_sse_operand(Register dst, XMMRegister src);
  void emit_sse_operand(XMMRegister dst);

  void emit_sse_operand(ZMMRegister dst, ZMMRegister src);
  void emit_sse_operand(ZMMRegister reg, const Operand &adr, int flags);
  void emit_sse_operand(ZMMRegister dst, Register src);
  void emit_sse_operand(Register dst, ZMMRegister src);

  // Emit machine code for one of the operations ADD, ADC, SUB, SBC,
  // AND, OR, XOR, or CMP.  The encodings of these operations are all
  // similar, differing just in the opcode or in the reg field of the
  // ModR/M byte.
  void arithmetic_op_8(byte opcode, Register reg, Register rm_reg);
  void arithmetic_op_8(byte opcode, Register reg, const Operand &rm_reg);
  void arithmetic_op_16(byte opcode, Register reg, Register rm_reg);
  void arithmetic_op_16(byte opcode, Register reg, const Operand &rm_reg);
  // Operate on operands/registers with pointer size, 32-bit or 64-bit size.
  void arithmetic_op(byte opcode, Register reg, Register rm_reg, int size);
  void arithmetic_op(byte opcode,
                     Register reg,
                     const Operand &rm_reg,
                     int size);

  // Operate on a byte in memory or register.
  void immediate_arithmetic_op_8(byte subcode,
                                 Register dst,
                                 Immediate src);
  void immediate_arithmetic_op_8(byte subcode,
                                 const Operand &dst,
                                 Immediate src);

  // Operate on a word in memory or register.
  void immediate_arithmetic_op_16(byte subcode,
                                  Register dst,
                                  Immediate src);
  void immediate_arithmetic_op_16(byte subcode,
                                  const Operand &dst,
                                  Immediate src);

  // Operate on operands/registers with any size, 8-bit, 16-bit, 32-bit or
  // 64-bit size.
  void immediate_arithmetic_op(byte subcode,
                               Register dst,
                               Immediate src,
                               int size);
  void immediate_arithmetic_op(byte subcode,
                               const Operand &dst,
                               Immediate src,
                               int size);

  // Emit machine code for a shift operation.
  void shift(Operand dst, Immediate shift_amount, int subcode, int size);
  void shift(Register dst, Immediate shift_amount, int subcode, int size);

  // Shift dst by cl % 64 bits.
  void shift(Register dst, int subcode, int size);
  void shift(Operand dst, int subcode, int size);

  void emit_farith(int b1, int b2, int i);

  // Arithmetics.
  void emit_add(Register dst, Register src, int size) {
    arithmetic_op(0x03, dst, src, size);
  }

  void emit_add(Register dst, Immediate src, int size) {
    immediate_arithmetic_op(0x0, dst, src, size);
  }

  void emit_add(Register dst, const Operand &src, int size) {
    arithmetic_op(0x03, dst, src, size);
  }

  void emit_add(const Operand &dst, Register src, int size) {
    arithmetic_op(0x1, src, dst, size);
  }

  void emit_add(const Operand &dst, Immediate src, int size) {
    immediate_arithmetic_op(0x0, dst, src, size);
  }

  void emit_and(Register dst, Register src, int size) {
    arithmetic_op(0x23, dst, src, size);
  }

  void emit_and(Register dst, const Operand &src, int size) {
    arithmetic_op(0x23, dst, src, size);
  }

  void emit_and(const Operand &dst, Register src, int size) {
    arithmetic_op(0x21, src, dst, size);
  }

  void emit_and(Register dst, Immediate src, int size) {
    immediate_arithmetic_op(0x4, dst, src, size);
  }

  void emit_and(const Operand &dst, Immediate src, int size) {
    immediate_arithmetic_op(0x4, dst, src, size);
  }

  void emit_cmp(Register dst, Register src, int size) {
    arithmetic_op(0x3B, dst, src, size);
  }

  void emit_cmp(Register dst, const Operand &src, int size) {
    arithmetic_op(0x3B, dst, src, size);
  }

  void emit_cmp(const Operand &dst, Register src, int size) {
    arithmetic_op(0x39, src, dst, size);
  }

  void emit_cmp(Register dst, Immediate src, int size) {
    immediate_arithmetic_op(0x7, dst, src, size);
  }

  void emit_cmp(const Operand &dst, Immediate src, int size) {
    immediate_arithmetic_op(0x7, dst, src, size);
  }

  // Compare {al,ax,eax,rax} with src. If equal, set ZF and write dst into
  // src. Otherwise clear ZF and write src into {al,ax,eax,rax}.  This
  // operation is only atomic if prefixed by the lock instruction.
  void emit_cmpxchg(const Operand &dst, Register src, int size);

  // Divide rdx:rax by src. Quotient in rax, remainder in rdx when size is 64.
  // Divide edx:eax by lower 32 bits of src. Quotient in eax, remainder in edx
  // when size is 32.
  void emit_idiv(Register src, int size);
  void emit_idiv(const Operand &src, int size);
  void emit_div(Register src, int size);
  void emit_div(const Operand &src, int size);

  // Signed multiply instructions.
  // rdx:rax = rax * src when size is 64 or edx:eax = eax * src when size is 32.
  void emit_imul(Register src, int size);
  void emit_imul(const Operand &src, int size);
  void emit_imul(Register dst, Register src, int size);
  void emit_imul(Register dst, const Operand &src, int size);
  void emit_imul(Register dst, Register src, Immediate imm, int size);
  void emit_imul(Register dst, const Operand &src, Immediate imm, int size);

  void emit_inc(Register dst, int size);
  void emit_inc(const Operand &dst, int size);
  void emit_dec(Register dst, int size);
  void emit_dec(const Operand &dst, int size);

  void emit_lea(Register dst, const Operand &src, int size);

  void emit_mov(Register dst, const Operand &src, int size);
  void emit_mov(Register dst, Register src, int size);
  void emit_mov(const Operand &dst, Register src, int size);
  void emit_mov(Register dst, Immediate value, int size);
  void emit_mov(const Operand &dst, Immediate value, int size);

  void emit_movzxb(Register dst, const Operand &src, int size);
  void emit_movzxb(Register dst, Register src, int size);
  void emit_movzxw(Register dst, const Operand &src, int size);
  void emit_movzxw(Register dst, Register src, int size);

  void emit_neg(Register dst, int size);
  void emit_neg(const Operand &dst, int size);

  void emit_not(Register dst, int size);
  void emit_not(const Operand &dst, int size);

  void emit_or(Register dst, Register src, int size) {
    arithmetic_op(0x0B, dst, src, size);
  }

  void emit_or(Register dst, const Operand &src, int size) {
    arithmetic_op(0x0B, dst, src, size);
  }

  void emit_or(const Operand &dst, Register src, int size) {
    arithmetic_op(0x9, src, dst, size);
  }

  void emit_or(Register dst, Immediate src, int size) {
    immediate_arithmetic_op(0x1, dst, src, size);
  }

  void emit_or(const Operand &dst, Immediate src, int size) {
    immediate_arithmetic_op(0x1, dst, src, size);
  }

  void emit_repmovs(int size);
  void emit_repstos(int size);

  void emit_sbb(Register dst, Register src, int size) {
    arithmetic_op(0x1b, dst, src, size);
  }

  void emit_sub(Register dst, Register src, int size) {
    arithmetic_op(0x2B, dst, src, size);
  }

  void emit_sub(Register dst, Immediate src, int size) {
    immediate_arithmetic_op(0x5, dst, src, size);
  }

  void emit_sub(Register dst, const Operand &src, int size) {
    arithmetic_op(0x2B, dst, src, size);
  }

  void emit_sub(const Operand &dst, Register src, int size) {
    arithmetic_op(0x29, src, dst, size);
  }

  void emit_sub(const Operand &dst, Immediate src, int size) {
    immediate_arithmetic_op(0x5, dst, src, size);
  }

  void emit_test(Register dst, Register src, int size);
  void emit_test(Register reg, Immediate mask, int size);
  void emit_test(const Operand &op, Register reg, int size);
  void emit_test(const Operand &op, Immediate mask, int size);
  void emit_test(Register reg, const Operand &op, int size) {
    return emit_test(op, reg, size);
  }

  void emit_xchg(Register dst, Register src, int size);
  void emit_xchg(Register dst, const Operand &src, int size);

  void emit_xor(Register dst, Register src, int size) {
    arithmetic_op(0x33, dst, src, size);
  }

  void emit_xor(Register dst, const Operand &src, int size) {
    arithmetic_op(0x33, dst, src, size);
  }

  void emit_xor(Register dst, Immediate src, int size) {
    immediate_arithmetic_op(0x6, dst, src, size);
  }

  void emit_xor(const Operand &dst, Immediate src, int size) {
    immediate_arithmetic_op(0x6, dst, src, size);
  }

  void emit_xor(const Operand &dst, Register src, int size) {
    arithmetic_op(0x31, src, dst, size);
  }

  void emit_prefetch(const Operand &src, int subcode);

  // SSE2 instruction encoding.
  void sse2_instr(XMMRegister dst, XMMRegister src, byte prefix, byte escape,
                  byte opcode);
  void sse2_instr(XMMRegister dst, const Operand &src, byte prefix, byte escape,
                  byte opcode);

  // SSE3 instruction encoding.
  void ssse3_instr(XMMRegister dst, XMMRegister src, byte prefix, byte escape1,
                   byte escape2, byte opcode);
  void ssse3_instr(XMMRegister dst, const Operand &src, byte prefix,
                   byte escape1, byte escape2, byte opcode);

  // SSE4 instruction encoding.
  void sse4_instr(XMMRegister dst, XMMRegister src, byte prefix, byte escape1,
                  byte escape2, byte opcode);
  void sse4_instr(XMMRegister dst, const Operand &src, byte prefix,
                  byte escape1, byte escape2, byte opcode);

  // AVX instruction encoding.
  void vinstr(byte op, XMMRegister dst, XMMRegister src1, XMMRegister src2,
              SIMDPrefix pp, LeadingOpcode m, VexW w);
  void vinstr(byte op, XMMRegister dst, XMMRegister src1, const Operand &src2,
              SIMDPrefix pp, LeadingOpcode m, VexW w, int sl = 0);
  void vinstr(byte op, YMMRegister dst, YMMRegister src1, YMMRegister src2,
              SIMDPrefix pp, LeadingOpcode m, VexW w);
  void vinstr(byte op, YMMRegister dst, YMMRegister src1, const Operand &src2,
              SIMDPrefix pp, LeadingOpcode m, VexW w, int sl = 0);

  void vsd(byte op, XMMRegister dst, XMMRegister src1, XMMRegister src2) {
    vinstr(op, dst, src1, src2, kF2, k0F, kWIG);
  }
  void vsd(byte op, XMMRegister dst, XMMRegister src1, const Operand &src2,
           int sl = 0) {
    vinstr(op, dst, src1, src2, kF2, k0F, kWIG, sl);
  }
  void vsd(byte op, YMMRegister dst, YMMRegister src1, YMMRegister src2) {
    vinstr(op, dst, src1, src2, kF2, k0F, kWIG);
  }
  void vsd(byte op, YMMRegister dst, YMMRegister src1, const Operand &src2,
           int sl = 0) {
    vinstr(op, dst, src1, src2, kF2, k0F, kWIG, sl);
  }

  void vss(byte op, XMMRegister dst, XMMRegister src1, XMMRegister src2);
  void vss(byte op, XMMRegister dst, XMMRegister src1, const Operand &src2,
           int sl = 0);
  void vss(byte op, YMMRegister dst, YMMRegister src1, YMMRegister src2);
  void vss(byte op, YMMRegister dst, YMMRegister src1, const Operand &src2,
           int sl = 0);

  void vps(byte op, XMMRegister dst, XMMRegister src1, XMMRegister src2);
  void vps(byte op, XMMRegister dst, XMMRegister src1, const Operand &src2,
           int sl = 0);
  void vps(byte op, YMMRegister dst, YMMRegister src1, YMMRegister src2);
  void vps(byte op, YMMRegister dst, YMMRegister src1, const Operand &src2,
           int sl = 0);

  void vpd(byte op, XMMRegister dst, XMMRegister src1, XMMRegister src2);
  void vpd(byte op, XMMRegister dst, XMMRegister src1, const Operand &src2,
           int sl = 0);
  void vpd(byte op, YMMRegister dst, YMMRegister src1, YMMRegister src2);
  void vpd(byte op, YMMRegister dst, YMMRegister src1, const Operand &src2,
           int sl = 0);

  // FMA instruction encoding.
  void vfmas(byte op, XMMRegister dst, XMMRegister src1, XMMRegister src2);
  void vfmas(byte op, XMMRegister dst, XMMRegister src1, const Operand &src2);
  void vfmas(byte op, YMMRegister dst, YMMRegister src1, YMMRegister src2);
  void vfmas(byte op, YMMRegister dst, YMMRegister src1, const Operand &src2);

  void vfmad(byte op, XMMRegister dst, XMMRegister src1, XMMRegister src2);
  void vfmad(byte op, XMMRegister dst, XMMRegister src1, const Operand &src2);
  void vfmad(byte op, YMMRegister dst, YMMRegister src1, YMMRegister src2);
  void vfmad(byte op, YMMRegister dst, YMMRegister src1, const Operand &src2);

  // AVX-512 opmask instruction encoding.
  void kinstr(byte op, OpmaskRegister k1, OpmaskRegister k2,
              OpmaskRegister k3, SIMDPrefix pp, LeadingOpcode m, VexW w);
  void kinstr(byte op, OpmaskRegister k1, OpmaskRegister k2,
              SIMDPrefix pp, LeadingOpcode m, VexW w);
  void kinstr(byte op, OpmaskRegister k1, OpmaskRegister k2, int8_t imm8,
              SIMDPrefix pp, LeadingOpcode m, VexW w);
  void kinstr(byte op, OpmaskRegister k1, const Operand &src,
              SIMDPrefix pp, LeadingOpcode m, VexW w);
  void kinstr(byte op, const Operand &dst, OpmaskRegister k1,
              SIMDPrefix pp, LeadingOpcode m, VexW w);
  void kinstr(byte op, OpmaskRegister k1, Register src,
              SIMDPrefix pp, LeadingOpcode m, VexW w);
  void kinstr(byte op, Register dst, OpmaskRegister k1,
              SIMDPrefix pp, LeadingOpcode m, VexW w);

  // AVX-512 instruction encoding.
  static int evex_round(RoundingMode er) {
    return er == noround ? 0 : (er * EVEX_R0) | EVEX_ER;
  }

  void zinstr(byte op, ZMMRegister dst, ZMMRegister src,
              int8_t imm8, Mask mask, int flags);
  void zinstr(byte op, ZMMRegister dst, const Operand &src,
              int8_t imm8, Mask mask, int flags);
  void zinstr(byte op, const Operand &dst, ZMMRegister src,
              int8_t imm8, Mask mask, int flags);
  void zinstr(byte op, ZMMRegister dst, ZMMRegister src1, ZMMRegister src2,
              int8_t imm8, Mask mask, int flags);
  void zinstr(byte op, ZMMRegister dst, ZMMRegister src1, const Operand &src2,
              int8_t imm8, Mask mask, int flags);
  void zinstr(byte op, const Operand &dst, ZMMRegister src1, ZMMRegister src2,
              int8_t imm8, Mask mask, int flags);
  void zinstr(byte op, ZMMRegister dst, ZMMRegister src1, Register src2,
              int8_t imm8, Mask mask, int flags);
  void zinstr(byte op, ZMMRegister dst, Register src,
              int8_t imm8, Mask mask, int flags);
  void zinstr(byte op, Register dst, ZMMRegister src,
              int8_t imm8, Mask mask, int flags);
  void zinstr(byte op, Register dst, const Operand &src,
              int8_t imm8, Mask mask, int flags);
  void zinstr(byte op, OpmaskRegister k, ZMMRegister src1, ZMMRegister src2,
              int8_t imm8, Mask mask, int flags);
  void zinstr(byte op, OpmaskRegister k, ZMMRegister src1, const Operand &src2,
              int8_t imm8, Mask mask, int flags);
  void zinstr(byte op, OpmaskRegister k, ZMMRegister src,
              int8_t imm8, Mask mask, int flags);
  void zinstr(byte op, ZMMRegister dst, OpmaskRegister k,
              int8_t imm8, Mask mask, int flags);

  // BMI instruction encoding.
  void bmi1q(byte op, Register reg, Register vreg, Register rm);
  void bmi1q(byte op, Register reg, Register vreg, const Operand &rm);
  void bmi1l(byte op, Register reg, Register vreg, Register rm);
  void bmi1l(byte op, Register reg, Register vreg, const Operand &rm);
  void bmi2q(SIMDPrefix pp, byte op, Register reg, Register vreg, Register rm);
  void bmi2q(SIMDPrefix pp, byte op, Register reg, Register vreg,
             const Operand &rm);
  void bmi2l(SIMDPrefix pp, byte op, Register reg, Register vreg, Register rm);
  void bmi2l(SIMDPrefix pp, byte op, Register reg, Register vreg,
             const Operand &rm);

  // Enabled CPU features.
  unsigned cpu_features_;
};

}  // namespace jit
}  // namespace sling

#endif  // JIT_ASSEMBLER__
