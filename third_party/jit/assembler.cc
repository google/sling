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

#include "third_party/jit/assembler.h"

#include "sling/base/logging.h"
#include "sling/base/macros.h"
#include "third_party/jit/cpu.h"
#include "third_party/jit/memory.h"
#include "third_party/jit/types.h"

namespace sling {
namespace jit {

// bit_cast<Dest,Source> is a template function that implements the
// equivalent of "*reinterpret_cast<Dest*>(&source)".
template <class Dest, class Source>
inline Dest bit_cast(const Source &source) {
  static_assert(sizeof(Dest) == sizeof(Source),
                "Source and destination types should have equal sizes.");

  Dest dest;
  memcpy(&dest, &source, sizeof(dest));
  return dest;
}

// Returns true iff value is a power of 2.
static bool IsPowerOfTwo32(uint32_t value) {
  return value && !(value & (value - 1));
}

Operand::Operand(Register base, int32_t disp, LoadMode load)
    : rex_(0), load_(load) {
  len_ = 1;
  if (base.is(rsp) || base.is(r12)) {
    // SIB byte is needed to encode (rsp + offset) or (r12 + offset).
    set_sib(times_1, rsp, base);
  }

  if (disp == 0 && !base.is(rbp) && !base.is(r13)) {
    set_modrm(0, base);
  } else if (is_int8(disp)) {
    set_modrm(1, base);
    set_disp8(disp);
  } else {
    set_modrm(2, base);
    set_disp32(disp);
  }
}

Operand::Operand(Register base,
                 Register index,
                 ScaleFactor scale,
                 int32_t disp,
                 LoadMode load) : rex_(0), load_(load) {
  DCHECK(!index.is(rsp));
  len_ = 1;
  set_sib(scale, index, base);
  if (disp == 0 && !base.is(rbp) && !base.is(r13)) {
    // This call to set_modrm doesn't overwrite the REX.B (or REX.X) bits
    // possibly set by set_sib.
    set_modrm(0, rsp);
  } else if (is_int8(disp)) {
    set_modrm(1, rsp);
    set_disp8(disp);
  } else {
    set_modrm(2, rsp);
    set_disp32(disp);
  }
}

Operand::Operand(Register index,
                 ScaleFactor scale,
                 int32_t disp,
                 LoadMode load) : rex_(0), load_(load) {
  DCHECK(!index.is(rsp));
  len_ = 1;
  set_modrm(0, rsp);
  set_sib(scale, index, rbp);
  set_disp32(disp);
}

Operand::Operand(Label *label, LoadMode load) : rex_(0), len_(1), load_(load) {
  DCHECK(label != nullptr);
  set_modrm(0, rbp);
  set_disp64(reinterpret_cast<intptr_t>(label));
}

Assembler::Assembler(void *buffer, int buffer_size)
    : CodeGenerator(buffer, buffer_size) {
  cpu_features_ = CPU::SupportedFeatures();

#ifdef DEBUG
  if (own_buffer_) {
    memset(buffer_, 0xCC, buffer_size_);  // int3
  }
#endif
}

void Assembler::Align(int m) {
  DCHECK(IsPowerOfTwo32(m));
  int delta = (m - (pc_offset() & (m - 1))) & (m - 1);
  Nop(delta);
}

void Assembler::DataAlign(int m) {
  DCHECK(m >= 2 && IsPowerOfTwo32(m));
  while ((pc_offset() & (m - 1)) != 0) {
    db(0);
  }
}

void Assembler::CodeTargetAlign() {
  Align(16);  // preferred alignment of jump targets on x64
}

void Assembler::shift(Register dst,
                      Immediate shift_amount,
                      int subcode,
                      int size) {
  EnsureSpace ensure_space(this);
  DCHECK(size == kInt64Size ? is_uint6(shift_amount.value_)
                            : is_uint5(shift_amount.value_));
  if (shift_amount.value_ == 1) {
    emit_rex(dst, size);
    emit(0xD1);
    emit_modrm(subcode, dst);
  } else {
    emit_rex(dst, size);
    emit(0xC1);
    emit_modrm(subcode, dst);
    emit(shift_amount.value_);
  }
}

void Assembler::shift(Operand dst, Immediate shift_amount, int subcode,
                      int size) {
  EnsureSpace ensure_space(this);
  DCHECK(size == kInt64Size ? is_uint6(shift_amount.value_)
                            : is_uint5(shift_amount.value_));
  if (shift_amount.value_ == 1) {
    emit_rex(dst, size);
    emit(0xD1);
    emit_operand(subcode, dst);
  } else {
    emit_rex(dst, size);
    emit(0xC1);
    emit_operand(subcode, dst, 1);
    emit(shift_amount.value_);
  }
}

void Assembler::shift(Register dst, int subcode, int size) {
  EnsureSpace ensure_space(this);
  emit_rex(dst, size);
  emit(0xD3);
  emit_modrm(subcode, dst);
}

void Assembler::shift(Operand dst, int subcode, int size) {
  EnsureSpace ensure_space(this);
  emit_rex(dst, size);
  emit(0xD3);
  emit_operand(subcode, dst);
}

void Assembler::bt(const Operand &dst, Register src) {
  EnsureSpace ensure_space(this);
  emit_rex_64(src, dst);
  emit(0x0F);
  emit(0xA3);
  emit_operand(src, dst);
}

void Assembler::bts(const Operand &dst, Register src) {
  EnsureSpace ensure_space(this);
  emit_rex_64(src, dst);
  emit(0x0F);
  emit(0xAB);
  emit_operand(src, dst);
}

void Assembler::bsrl(Register dst, Register src) {
  EnsureSpace ensure_space(this);
  emit_optional_rex_32(dst, src);
  emit(0x0F);
  emit(0xBD);
  emit_modrm(dst, src);
}

void Assembler::bsrl(Register dst, const Operand &src) {
  EnsureSpace ensure_space(this);
  emit_optional_rex_32(dst, src);
  emit(0x0F);
  emit(0xBD);
  emit_operand(dst, src);
}

void Assembler::bsrq(Register dst, Register src) {
  EnsureSpace ensure_space(this);
  emit_rex_64(dst, src);
  emit(0x0F);
  emit(0xBD);
  emit_modrm(dst, src);
}

void Assembler::bsrq(Register dst, const Operand &src) {
  EnsureSpace ensure_space(this);
  emit_rex_64(dst, src);
  emit(0x0F);
  emit(0xBD);
  emit_operand(dst, src);
}

void Assembler::bsfl(Register dst, Register src) {
  EnsureSpace ensure_space(this);
  emit_optional_rex_32(dst, src);
  emit(0x0F);
  emit(0xBC);
  emit_modrm(dst, src);
}

void Assembler::bsfl(Register dst, const Operand &src) {
  EnsureSpace ensure_space(this);
  emit_optional_rex_32(dst, src);
  emit(0x0F);
  emit(0xBC);
  emit_operand(dst, src);
}

void Assembler::bsfq(Register dst, Register src) {
  EnsureSpace ensure_space(this);
  emit_rex_64(dst, src);
  emit(0x0F);
  emit(0xBC);
  emit_modrm(dst, src);
}

void Assembler::bsfq(Register dst, const Operand &src) {
  EnsureSpace ensure_space(this);
  emit_rex_64(dst, src);
  emit(0x0F);
  emit(0xBC);
  emit_operand(dst, src);
}

void Assembler::call(Label *l) {
  EnsureSpace ensure_space(this);
  // 1110 1000 #32-bit disp.
  emit(0xE8);
  if (l->is_bound()) {
    int offset = l->pos() - pc_offset() - sizeof(int32_t);
    DCHECK(offset <= 0);
    emitl(offset);
  } else if (l->is_linked()) {
    emitl(l->pos());
    l->link_to(pc_offset() - sizeof(int32_t));
  } else {
    DCHECK(l->is_unused());
    int32_t current = pc_offset();
    emitl(current);
    l->link_to(current);
  }
}

void Assembler::call(Register adr) {
  EnsureSpace ensure_space(this);
  // Opcode: FF /2 r64.
  emit_optional_rex_32(adr);
  emit(0xFF);
  emit_modrm(0x2, adr);
}

void Assembler::call(const Operand &op) {
  EnsureSpace ensure_space(this);
  // Opcode: FF /2 m64.
  emit_optional_rex_32(op);
  emit(0xFF);
  emit_operand(0x2, op);
}

// Calls directly to the given address using a relative offset.
// Should only ever be used in Code objects for calls within the
// same Code object. Should not be used when generating new code (use labels),
// but only when patching existing code.
void Assembler::call(Address target) {
  EnsureSpace ensure_space(this);
  // 1110 1000 #32-bit disp.
  emit(0xE8);
  Address source = pc_ + 4;
  intptr_t displacement = target - source;
  DCHECK(is_int32(displacement));
  emitl(static_cast<int32_t>(displacement));
}

void Assembler::call(const void *target, const string &symbol) {
  EnsureSpace ensure_space(this);
  emit(0xE8);
  AddExtern(symbol, static_cast<Address>(const_cast<void *>(target)), true);
  emitl(-4);
}

void Assembler::clc() {
  EnsureSpace ensure_space(this);
  emit(0xF8);
}

void Assembler::cld() {
  EnsureSpace ensure_space(this);
  emit(0xFC);
}

void Assembler::cdq() {
  EnsureSpace ensure_space(this);
  emit(0x99);
}

void Assembler::cbw() {
  EnsureSpace ensure_space(this);
  emit(0x66);
  emit(0x99);
}

void Assembler::cmovq(Condition cc, Register dst, Register src) {
  if (cc == always) {
    movq(dst, src);
  } else if (cc == never) {
    return;
  }
  // No need to check CpuInfo for CMOV support, it's a required part of the
  // 64-bit architecture.
  DCHECK(cc >= 0);  // Use mov for unconditional moves.
  EnsureSpace ensure_space(this);
  // Opcode: REX.W 0f 40 + cc /r.
  emit_rex_64(dst, src);
  emit(0x0f);
  emit(0x40 + cc);
  emit_modrm(dst, src);
}

void Assembler::cmovq(Condition cc, Register dst, const Operand &src) {
  if (cc == always) {
    movq(dst, src);
  } else if (cc == never) {
    return;
  }
  DCHECK(cc >= 0);
  EnsureSpace ensure_space(this);
  // Opcode: REX.W 0f 40 + cc /r.
  emit_rex_64(dst, src);
  emit(0x0f);
  emit(0x40 + cc);
  emit_operand(dst, src);
}

void Assembler::cmovl(Condition cc, Register dst, Register src) {
  if (cc == always) {
    movl(dst, src);
  } else if (cc == never) {
    return;
  }
  DCHECK(cc >= 0);
  EnsureSpace ensure_space(this);
  // Opcode: 0f 40 + cc /r.
  emit_optional_rex_32(dst, src);
  emit(0x0f);
  emit(0x40 + cc);
  emit_modrm(dst, src);
}

void Assembler::cmovl(Condition cc, Register dst, const Operand &src) {
  if (cc == always) {
    movl(dst, src);
  } else if (cc == never) {
    return;
  }
  DCHECK(cc >= 0);
  EnsureSpace ensure_space(this);
  // Opcode: 0f 40 + cc /r.
  emit_optional_rex_32(dst, src);
  emit(0x0f);
  emit(0x40 + cc);
  emit_operand(dst, src);
}

void Assembler::cmpb_al(Immediate imm8) {
  DCHECK(is_int8(imm8.value_) || is_uint8(imm8.value_));
  EnsureSpace ensure_space(this);
  emit(0x3c);
  emit(imm8.value_);
}

void Assembler::lock() {
  EnsureSpace ensure_space(this);
  emit(0xf0);
}

void Assembler::cmpxchgb(const Operand &dst, Register src) {
  EnsureSpace ensure_space(this);
  if (!src.is_byte_register()) {
    // Register is not one of al, bl, cl, dl.  Its encoding needs REX.
    emit_rex_32(src, dst);
  } else {
    emit_optional_rex_32(src, dst);
  }
  emit(0x0f);
  emit(0xb0);
  emit_operand(src, dst);
}

void Assembler::cmpxchgw(const Operand &dst, Register src) {
  EnsureSpace ensure_space(this);
  emit(0x66);
  emit_optional_rex_32(src, dst);
  emit(0x0f);
  emit(0xb1);
  emit_operand(src, dst);
}

void Assembler::emit_cmpxchg(const Operand &dst, Register src, int size) {
  EnsureSpace ensure_space(this);
  emit_rex(src, dst, size);
  emit(0x0f);
  emit(0xb1);
  emit_operand(src, dst);
}

void Assembler::cpuid() {
  EnsureSpace ensure_space(this);
  emit(0x0F);
  emit(0xA2);
}

void Assembler::cqo() {
  EnsureSpace ensure_space(this);
  emit_rex_64();
  emit(0x99);
}

void Assembler::emit_dec(Register dst, int size) {
  EnsureSpace ensure_space(this);
  emit_rex(dst, size);
  emit(0xFF);
  emit_modrm(0x1, dst);
}

void Assembler::emit_dec(const Operand &dst, int size) {
  EnsureSpace ensure_space(this);
  emit_rex(dst, size);
  emit(0xFF);
  emit_operand(1, dst);
}

void Assembler::decb(Register dst) {
  EnsureSpace ensure_space(this);
  if (!dst.is_byte_register()) {
    // Register is not one of al, bl, cl, dl.  Its encoding needs REX.
    emit_rex_32(dst);
  }
  emit(0xFE);
  emit_modrm(0x1, dst);
}

void Assembler::decb(const Operand &dst) {
  EnsureSpace ensure_space(this);
  emit_optional_rex_32(dst);
  emit(0xFE);
  emit_operand(1, dst);
}

void Assembler::enter(Immediate size) {
  EnsureSpace ensure_space(this);
  emit(0xC8);
  emitw(size.value_);  // 16 bit operand, always.
  emit(0);
}

void Assembler::hlt() {
  EnsureSpace ensure_space(this);
  emit(0xF4);
}

void Assembler::emit_idiv(Register src, int size) {
  EnsureSpace ensure_space(this);
  if (size == 2) emit(0x66);  // operand size override prefix
  emit_rex(src, size);
  emit(size == 1 ? 0xF6 : 0xF7);
  emit_modrm(0x7, src);
}

void Assembler::emit_idiv(const Operand &src, int size) {
  EnsureSpace ensure_space(this);
  if (size == 2) emit(0x66);  // operand size override prefix
  emit_rex(src, size);
  emit(size == 1 ? 0xF6 : 0xF7);
  emit_operand(0x7, src);
}

void Assembler::emit_div(Register src, int size) {
  EnsureSpace ensure_space(this);
  if (size == 2) emit(0x66);  // operand size override prefix
  emit_rex(src, size);
  emit(size == 1 ? 0xF6 : 0xF7);
  emit_modrm(0x6, src);
}

void Assembler::emit_div(const Operand &src, int size) {
  EnsureSpace ensure_space(this);
  if (size == 2) emit(0x66);  // operand size override prefix
  emit_rex(src, size);
  emit(size == 1 ? 0xF6 : 0xF7);
  emit_operand(0x6, src);
}

void Assembler::emit_imul(Register src, int size) {
  EnsureSpace ensure_space(this);
  if (size == 2) emit(0x66);  // operand size override prefix
  emit_rex(src, size);
  emit(size == 1 ? 0xF6 : 0xF7);
  emit_modrm(0x5, src);
}

void Assembler::emit_imul(const Operand &src, int size) {
  EnsureSpace ensure_space(this);
  if (size == 2) emit(0x66);  // operand size override prefix
  emit_rex(src, size);
  emit(size == 1 ? 0xF6 : 0xF7);
  emit_operand(0x5, src);
}

void Assembler::emit_imul(Register dst, Register src, int size) {
  EnsureSpace ensure_space(this);
  DCHECK(size != 1);
  if (size == 2) emit(0x66);  // operand size override prefix
  emit_rex(dst, src, size);
  emit(0x0F);
  emit(0xAF);
  emit_modrm(dst, src);
}

void Assembler::emit_imul(Register dst, const Operand &src, int size) {
  EnsureSpace ensure_space(this);
  DCHECK(size != 1);
  if (size == 2) emit(0x66);  // operand size override prefix
  emit_rex(dst, src, size);
  emit(0x0F);
  emit(0xAF);
  emit_operand(dst, src);
}

void Assembler::emit_imul(Register dst, Register src, Immediate imm, int size) {
  EnsureSpace ensure_space(this);
  DCHECK(size != 1);
  if (size == 2) emit(0x66);  // operand size override prefix
  emit_rex(dst, src, size);
  if (is_int8(imm.value_)) {
    emit(0x6B);
    emit_modrm(dst, src);
    emit(imm.value_);
  } else {
    emit(0x69);
    emit_modrm(dst, src);
    emitl(imm.value_);
  }
}

void Assembler::emit_imul(Register dst, const Operand &src, Immediate imm,
                          int size) {
  EnsureSpace ensure_space(this);
  DCHECK(size != 1);
  if (size == 2) emit(0x66);  // operand size override prefix
  emit_rex(dst, src, size);
  if (is_int8(imm.value_)) {
    emit(0x6B);
    emit_operand(dst, src, 1);
    emit(imm.value_);
  } else {
    emit(0x69);
    emit_operand(dst, src, 4);
    emitl(imm.value_);
  }
}

void Assembler::emit_inc(Register dst, int size) {
  EnsureSpace ensure_space(this);
  emit_rex(dst, size);
  emit(0xFF);
  emit_modrm(0x0, dst);
}

void Assembler::emit_inc(const Operand &dst, int size) {
  EnsureSpace ensure_space(this);
  emit_rex(dst, size);
  emit(0xFF);
  emit_operand(0, dst);
}

void Assembler::int3() {
  EnsureSpace ensure_space(this);
  emit(0xCC);
}

void Assembler::j(Condition cc, Label *l, Label::Distance distance) {
  if (cc == always) {
    jmp(l);
    return;
  } else if (cc == never) {
    return;
  }
  EnsureSpace ensure_space(this);
  DCHECK(is_uint4(cc));
  if (l->is_bound()) {
    const int short_size = 2;
    const int long_size  = 6;
    int offs = l->pos() - pc_offset();
    DCHECK(offs <= 0);
    // Determine whether we can use 1-byte offsets for backwards branches,
    // which have a max range of 128 bytes.
    if (is_int8(offs - short_size)) {
      // 0111 tttn #8-bit disp.
      emit(0x70 | cc);
      emit((offs - short_size) & 0xFF);
    } else {
      // 0000 1111 1000 tttn #32-bit disp.
      emit(0x0F);
      emit(0x80 | cc);
      emitl(offs - long_size);
    }
  } else if (distance == Label::kNear) {
    // 0111 tttn #8-bit disp
    emit(0x70 | cc);
    byte disp = 0x00;
    if (l->is_near_linked()) {
      int offset = l->near_link_pos() - pc_offset();
      DCHECK(is_int8(offset));
      disp = static_cast<byte>(offset & 0xFF);
    }
    l->link_to(pc_offset(), Label::kNear);
    emit(disp);
  } else if (l->is_linked()) {
    // 0000 1111 1000 tttn #32-bit disp.
    emit(0x0F);
    emit(0x80 | cc);
    emitl(l->pos());
    l->link_to(pc_offset() - sizeof(int32_t));
  } else {
    DCHECK(l->is_unused());
    emit(0x0F);
    emit(0x80 | cc);
    int32_t current = pc_offset();
    emitl(current);
    l->link_to(current);
  }
}

void Assembler::jmp(Label *l, Label::Distance distance) {
  EnsureSpace ensure_space(this);
  const int short_size = sizeof(int8_t);
  const int long_size = sizeof(int32_t);
  if (l->is_bound()) {
    int offs = l->pos() - pc_offset() - 1;
    DCHECK(offs <= 0);
    if (is_int8(offs - short_size)) {
      // 1110 1011 #8-bit disp.
      emit(0xEB);
      emit((offs - short_size) & 0xFF);
    } else {
      // 1110 1001 #32-bit disp.
      emit(0xE9);
      emitl(offs - long_size);
    }
  } else if (distance == Label::kNear) {
    emit(0xEB);
    byte disp = 0x00;
    if (l->is_near_linked()) {
      int offset = l->near_link_pos() - pc_offset();
      DCHECK(is_int8(offset));
      disp = static_cast<byte>(offset & 0xFF);
    }
    l->link_to(pc_offset(), Label::kNear);
    emit(disp);
  } else if (l->is_linked()) {
    // 1110 1001 #32-bit disp.
    emit(0xE9);
    emitl(l->pos());
    l->link_to(pc_offset() - long_size);
  } else {
    // 1110 1001 #32-bit disp.
    DCHECK(l->is_unused());
    emit(0xE9);
    int32_t current = pc_offset();
    emitl(current);
    l->link_to(current);
  }
}

void Assembler::jmp(Register target) {
  EnsureSpace ensure_space(this);
  // Opcode FF/4 r64.
  emit_optional_rex_32(target);
  emit(0xFF);
  emit_modrm(0x4, target);
}

void Assembler::jmp(const Operand &src) {
  EnsureSpace ensure_space(this);
  // Opcode FF/4 m64.
  emit_optional_rex_32(src);
  emit(0xFF);
  emit_operand(0x4, src);
}

void Assembler::emit_lea(Register dst, const Operand &src, int size) {
  EnsureSpace ensure_space(this);
  emit_rex(dst, src, size);
  emit(0x8D);
  emit_operand(dst, src);
}

void Assembler::load_extern(Register dst, const void *value,
                            const string &symbol, bool pic) {
  EnsureSpace ensure_space(this);
  DCHECK(kPointerSize == kInt64Size);
  if (pic) {
    // PC-relative load, lea dst, [rip+disp].
    emit(0x48 | dst.high_bit() << 2 | 1);  // REX.64
    emit(0x8D);
    emit(0x05 | dst.low_bits() << 3);  // mod:0, reg: dst, rm: 101
    AddExtern(symbol, static_cast<Address>(const_cast<void *>(value)), true);
    emitl(-4);
  } else {
    // Absolute load, mov dst, imm64.
    emit_rex(dst, kPointerSize);
    emit(0xB8 | dst.low_bits());
    AddExtern(symbol, static_cast<Address>(const_cast<void *>(value)));
    emitp(value);
  }
}

void Assembler::load_rax(const void *value) {
  EnsureSpace ensure_space(this);
  if (kPointerSize == kInt64Size) {
    emit(0x48);  // REX.W
    emit(0xA1);
    emitp(value);
  } else {
    DCHECK(kPointerSize == kInt32Size);
    emit(0xA1);
    emitp(value);
    // In 64-bit mode, need to zero extend the operand to 8 bytes.
    // See 2.2.1.4 in Intel64 and IA32 Architectures Software
    // Developer's Manual Volume 2.
    emitl(0);
  }
}

void Assembler::leave() {
  EnsureSpace ensure_space(this);
  emit(0xC9);
}

void Assembler::movb(Register dst, const Operand &src) {
  EnsureSpace ensure_space(this);
  if (!dst.is_byte_register()) {
    // Register is not one of al, bl, cl, dl.  Its encoding needs REX.
    emit_rex_32(dst, src);
  } else {
    emit_optional_rex_32(dst, src);
  }
  emit(0x8A);
  emit_operand(dst, src);
}

void Assembler::movb(Register dst, Immediate imm) {
  EnsureSpace ensure_space(this);
  if (!dst.is_byte_register()) {
    // Register is not one of al, bl, cl, dl.  Its encoding needs REX.
    emit_rex_32(dst);
  }
  emit(0xB0 + dst.low_bits());
  emit(imm.value_);
}

void Assembler::movb(const Operand &dst, Register src) {
  EnsureSpace ensure_space(this);
  if (!src.is_byte_register()) {
    // Register is not one of al, bl, cl, dl.  Its encoding needs REX.
    emit_rex_32(src, dst);
  } else {
    emit_optional_rex_32(src, dst);
  }
  emit(0x88);
  emit_operand(src, dst);
}

void Assembler::movb(const Operand &dst, Immediate imm) {
  EnsureSpace ensure_space(this);
  emit_optional_rex_32(dst);
  emit(0xC6);
  emit_operand(0x0, dst, 1);
  emit(static_cast<byte>(imm.value_));
}

void Assembler::movw(Register dst, const Operand &src) {
  EnsureSpace ensure_space(this);
  emit(0x66);
  emit_optional_rex_32(dst, src);
  emit(0x8B);
  emit_operand(dst, src);
}

void Assembler::movw(Register dst, Immediate imm) {
  EnsureSpace ensure_space(this);
  emit(0x66);
  emit_optional_rex_32(dst);
  emit(0xB8 + dst.low_bits());
  emit(static_cast<byte>(imm.value_ & 0xff));
  emit(static_cast<byte>(imm.value_ >> 8));
}

void Assembler::movw(const Operand &dst, Register src) {
  EnsureSpace ensure_space(this);
  emit(0x66);
  emit_optional_rex_32(src, dst);
  emit(0x89);
  emit_operand(src, dst);
}

void Assembler::movw(const Operand &dst, Immediate imm) {
  EnsureSpace ensure_space(this);
  emit(0x66);
  emit_optional_rex_32(dst);
  emit(0xC7);
  emit_operand(0x0, dst, 2);
  emit(static_cast<byte>(imm.value_ & 0xff));
  emit(static_cast<byte>(imm.value_ >> 8));
}

void Assembler::emit_mov(Register dst, const Operand &src, int size) {
  EnsureSpace ensure_space(this);
  emit_rex(dst, src, size);
  emit(0x8B);
  emit_operand(dst, src);
}

void Assembler::emit_mov(Register dst, Register src, int size) {
  EnsureSpace ensure_space(this);
  if (src.low_bits() == 4) {
    emit_rex(src, dst, size);
    emit(0x89);
    emit_modrm(src, dst);
  } else {
    emit_rex(dst, src, size);
    emit(0x8B);
    emit_modrm(dst, src);
  }
}

void Assembler::emit_mov(const Operand &dst, Register src, int size) {
  EnsureSpace ensure_space(this);
  emit_rex(src, dst, size);
  emit(0x89);
  emit_operand(src, dst);
}

void Assembler::emit_mov(Register dst, Immediate value, int size) {
  EnsureSpace ensure_space(this);
  emit_rex(dst, size);
  if (size == kInt64Size) {
    emit(0xC7);
    emit_modrm(0x0, dst);
  } else {
    DCHECK(size == kInt32Size);
    emit(0xB8 + dst.low_bits());
  }
  emit(value);
}

void Assembler::emit_mov(const Operand &dst, Immediate value, int size) {
  EnsureSpace ensure_space(this);
  emit_rex(dst, size);
  emit(0xC7);
  emit_operand(0x0, dst, 1);
  emit(value);
}

void Assembler::movp(Register dst, const void *value) {
  EnsureSpace ensure_space(this);
  emit_rex(dst, kPointerSize);
  emit(0xB8 | dst.low_bits());
  emitp(value);
}

void Assembler::movq(Register dst, int64_t value) {
  EnsureSpace ensure_space(this);
  emit_rex_64(dst);
  emit(0xB8 | dst.low_bits());
  emitq(value);
}

void Assembler::movq(Register dst, uint64_t value) {
  movq(dst, static_cast<int64_t>(value));
}

// Loads the ip-relative location of the src label into the target location
// (as a 32-bit offset sign extended to 64-bit).
void Assembler::movl(const Operand &dst, Label *src) {
  EnsureSpace ensure_space(this);
  emit_optional_rex_32(dst);
  emit(0xC7);
  emit_operand(0, dst, 4);
  if (src->is_bound()) {
    int offset = src->pos() - pc_offset() - sizeof(int32_t);
    DCHECK(offset <= 0);
    emitl(offset);
  } else if (src->is_linked()) {
    emitl(src->pos());
    src->link_to(pc_offset() - sizeof(int32_t));
  } else {
    DCHECK(src->is_unused());
    int32_t current = pc_offset();
    emitl(current);
    src->link_to(current);
  }
}

void Assembler::movsxbl(Register dst, Register src) {
  EnsureSpace ensure_space(this);
  if (!src.is_byte_register()) {
    // Register is not one of al, bl, cl, dl.  Its encoding needs REX.
    emit_rex_32(dst, src);
  } else {
    emit_optional_rex_32(dst, src);
  }
  emit(0x0F);
  emit(0xBE);
  emit_modrm(dst, src);
}

void Assembler::movsxbl(Register dst, const Operand &src) {
  EnsureSpace ensure_space(this);
  emit_optional_rex_32(dst, src);
  emit(0x0F);
  emit(0xBE);
  emit_operand(dst, src);
}

void Assembler::movsxbq(Register dst, const Operand &src) {
  EnsureSpace ensure_space(this);
  emit_rex_64(dst, src);
  emit(0x0F);
  emit(0xBE);
  emit_operand(dst, src);
}

void Assembler::movsxbq(Register dst, Register src) {
  EnsureSpace ensure_space(this);
  emit_rex_64(dst, src);
  emit(0x0F);
  emit(0xBE);
  emit_modrm(dst, src);
}

void Assembler::movsxwl(Register dst, Register src) {
  EnsureSpace ensure_space(this);
  emit_optional_rex_32(dst, src);
  emit(0x0F);
  emit(0xBF);
  emit_modrm(dst, src);
}

void Assembler::movsxwl(Register dst, const Operand &src) {
  EnsureSpace ensure_space(this);
  emit_optional_rex_32(dst, src);
  emit(0x0F);
  emit(0xBF);
  emit_operand(dst, src);
}

void Assembler::movsxwq(Register dst, const Operand &src) {
  EnsureSpace ensure_space(this);
  emit_rex_64(dst, src);
  emit(0x0F);
  emit(0xBF);
  emit_operand(dst, src);
}

void Assembler::movsxwq(Register dst, Register src) {
  EnsureSpace ensure_space(this);
  emit_rex_64(dst, src);
  emit(0x0F);
  emit(0xBF);
  emit_modrm(dst, src);
}

void Assembler::movsxlq(Register dst, Register src) {
  EnsureSpace ensure_space(this);
  emit_rex_64(dst, src);
  emit(0x63);
  emit_modrm(dst, src);
}

void Assembler::movsxlq(Register dst, const Operand &src) {
  EnsureSpace ensure_space(this);
  emit_rex_64(dst, src);
  emit(0x63);
  emit_operand(dst, src);
}

void Assembler::emit_movzxb(Register dst, const Operand &src, int size) {
  EnsureSpace ensure_space(this);
  // 32 bit operations zero the top 32 bits of 64 bit registers.  Therefore
  // there is no need to make this a 64 bit operation.
  emit_optional_rex_32(dst, src);
  emit(0x0F);
  emit(0xB6);
  emit_operand(dst, src);
}

void Assembler::emit_movzxb(Register dst, Register src, int size) {
  EnsureSpace ensure_space(this);
  // 32 bit operations zero the top 32 bits of 64 bit registers.  Therefore
  // there is no need to make this a 64 bit operation.
  if (!src.is_byte_register()) {
    // Register is not one of al, bl, cl, dl.  Its encoding needs REX.
    emit_rex_32(dst, src);
  } else {
    emit_optional_rex_32(dst, src);
  }
  emit(0x0F);
  emit(0xB6);
  emit_modrm(dst, src);
}

void Assembler::emit_movzxw(Register dst, const Operand &src, int size) {
  EnsureSpace ensure_space(this);
  // 32 bit operations zero the top 32 bits of 64 bit registers.  Therefore
  // there is no need to make this a 64 bit operation.
  emit_optional_rex_32(dst, src);
  emit(0x0F);
  emit(0xB7);
  emit_operand(dst, src);
}

void Assembler::emit_movzxw(Register dst, Register src, int size) {
  EnsureSpace ensure_space(this);
  // 32 bit operations zero the top 32 bits of 64 bit registers.  Therefore
  // there is no need to make this a 64 bit operation.
  emit_optional_rex_32(dst, src);
  emit(0x0F);
  emit(0xB7);
  emit_modrm(dst, src);
}

void Assembler::repmovsb() {
  EnsureSpace ensure_space(this);
  emit(0xF3);
  emit(0xA4);
}

void Assembler::repmovsw() {
  EnsureSpace ensure_space(this);
  emit(0x66);  // operand size override
  emit(0xF3);
  emit(0xA4);
}

void Assembler::emit_repmovs(int size) {
  EnsureSpace ensure_space(this);
  emit(0xF3);
  emit_rex(size);
  emit(0xA5);
}

void Assembler::repstosb() {
  EnsureSpace ensure_space(this);
  emit(0xF3);
  emit(0xAA);
}

void Assembler::repstosw() {
  EnsureSpace ensure_space(this);
  emit(0x66);  // operand size override
  emit(0xF3);
  emit(0xAA);
}

void Assembler::emit_repstos(int size) {
  EnsureSpace ensure_space(this);
  emit(0xF3);
  emit_rex(size);
  emit(0xAB);
}

void Assembler::mull(Register src) {
  EnsureSpace ensure_space(this);
  emit_optional_rex_32(src);
  emit(0xF7);
  emit_modrm(0x4, src);
}

void Assembler::mull(const Operand &src) {
  EnsureSpace ensure_space(this);
  emit_optional_rex_32(src);
  emit(0xF7);
  emit_operand(0x4, src);
}

void Assembler::mulq(Register src) {
  EnsureSpace ensure_space(this);
  emit_rex_64(src);
  emit(0xF7);
  emit_modrm(0x4, src);
}

void Assembler::emit_neg(Register dst, int size) {
  EnsureSpace ensure_space(this);
  emit_rex(dst, size);
  emit(0xF7);
  emit_modrm(0x3, dst);
}

void Assembler::emit_neg(const Operand &dst, int size) {
  EnsureSpace ensure_space(this);
  emit_rex_64(dst);
  emit(0xF7);
  emit_operand(3, dst);
}

void Assembler::nop() {
  EnsureSpace ensure_space(this);
  emit(0x90);
}

void Assembler::emit_not(Register dst, int size) {
  EnsureSpace ensure_space(this);
  emit_rex(dst, size);
  emit(0xF7);
  emit_modrm(0x2, dst);
}

void Assembler::emit_not(const Operand &dst, int size) {
  EnsureSpace ensure_space(this);
  emit_rex(dst, size);
  emit(0xF7);
  emit_operand(2, dst);
}

void Assembler::Nop(int n) {
  // The recommended muti-byte sequences of NOP instructions from the Intel 64
  // and IA-32 Architectures Software Developer's Manual.
  //
  // Length   Assembly                                Byte Sequence
  // 2 bytes  66 NOP                                  66 90H
  // 3 bytes  NOP DWORD ptr [EAX]                     0F 1F 00H
  // 4 bytes  NOP DWORD ptr [EAX + 00H]               0F 1F 40 00H
  // 5 bytes  NOP DWORD ptr [EAX + EAX*1 + 00H]       0F 1F 44 00 00H
  // 6 bytes  66 NOP DWORD ptr [EAX + EAX*1 + 00H]    66 0F 1F 44 00 00H
  // 7 bytes  NOP DWORD ptr [EAX + 00000000H]         0F 1F 80 00 00 00 00H
  // 8 bytes  NOP DWORD ptr [EAX + EAX*1 + 00000000H] 0F 1F 84 00 00 00 00 00H
  // 9 bytes  66 NOP DWORD ptr [EAX + EAX*1 +         66 0F 1F 84 00 00 00 00
  //          00000000H]                              00H

  EnsureSpace ensure_space(this);
  while (n > 0) {
    switch (n) {
      case 2:
        emit(0x66);
        FALLTHROUGH_INTENDED;
      case 1:
        emit(0x90);
        return;
      case 3:
        emit(0x0f);
        emit(0x1f);
        emit(0x00);
        return;
      case 4:
        emit(0x0f);
        emit(0x1f);
        emit(0x40);
        emit(0x00);
        return;
      case 6:
        emit(0x66);
        FALLTHROUGH_INTENDED;
      case 5:
        emit(0x0f);
        emit(0x1f);
        emit(0x44);
        emit(0x00);
        emit(0x00);
        return;
      case 7:
        emit(0x0f);
        emit(0x1f);
        emit(0x80);
        emit(0x00);
        emit(0x00);
        emit(0x00);
        emit(0x00);
        return;
      default:
        FALLTHROUGH_INTENDED;
      case 11:
        emit(0x66);
        n--;
        FALLTHROUGH_INTENDED;
      case 10:
        emit(0x66);
        n--;
        FALLTHROUGH_INTENDED;
      case 9:
        emit(0x66);
        n--;
        FALLTHROUGH_INTENDED;
      case 8:
        emit(0x0f);
        emit(0x1f);
        emit(0x84);
        emit(0x00);
        emit(0x00);
        emit(0x00);
        emit(0x00);
        emit(0x00);
        n -= 8;
    }
  }
}

void Assembler::popq(Register dst) {
  EnsureSpace ensure_space(this);
  emit_optional_rex_32(dst);
  emit(0x58 | dst.low_bits());
}

void Assembler::popq(const Operand &dst) {
  EnsureSpace ensure_space(this);
  emit_optional_rex_32(dst);
  emit(0x8F);
  emit_operand(0, dst);
}

void Assembler::popfq() {
  EnsureSpace ensure_space(this);
  emit(0x9D);
}

void Assembler::pushq(Register src) {
  EnsureSpace ensure_space(this);
  emit_optional_rex_32(src);
  emit(0x50 | src.low_bits());
}

void Assembler::pushq(const Operand &src) {
  EnsureSpace ensure_space(this);
  emit_optional_rex_32(src);
  emit(0xFF);
  emit_operand(6, src);
}

void Assembler::pushq(Immediate value) {
  EnsureSpace ensure_space(this);
  if (is_int8(value.value_)) {
    emit(0x6A);
    emit(value.value_);  // emit low byte of value
  } else {
    emit(0x68);
    emitl(value.value_);
  }
}

void Assembler::pushq_imm32(int32_t imm32) {
  EnsureSpace ensure_space(this);
  emit(0x68);
  emitl(imm32);
}

void Assembler::pushfq() {
  EnsureSpace ensure_space(this);
  emit(0x9C);
}

void Assembler::ret(int imm16) {
  EnsureSpace ensure_space(this);
  DCHECK(is_uint16(imm16));
  if (imm16 == 0) {
    emit(0xC3);
  } else {
    emit(0xC2);
    emit(imm16 & 0xFF);
    emit((imm16 >> 8) & 0xFF);
  }
}

void Assembler::ud2() {
  EnsureSpace ensure_space(this);
  emit(0x0F);
  emit(0x0B);
}

void Assembler::setcc(Condition cc, Register reg) {
  if (cc > last_condition) {
    movb(reg, Immediate(cc == always ? 1 : 0));
    return;
  }
  EnsureSpace ensure_space(this);
  DCHECK(is_uint4(cc));
  if (!reg.is_byte_register()) {
    // Register is not one of al, bl, cl, dl.  Its encoding needs REX.
    emit_rex_32(reg);
  }
  emit(0x0F);
  emit(0x90 | cc);
  emit_modrm(0x0, reg);
}

void Assembler::rdtsc() {
  EnsureSpace ensure_space(this);
  emit(0x0F);
  emit(0x31);
}

void Assembler::shld(Register dst, Register src) {
  EnsureSpace ensure_space(this);
  emit_rex_64(src, dst);
  emit(0x0F);
  emit(0xA5);
  emit_modrm(src, dst);
}

void Assembler::shrd(Register dst, Register src) {
  EnsureSpace ensure_space(this);
  emit_rex_64(src, dst);
  emit(0x0F);
  emit(0xAD);
  emit_modrm(src, dst);
}

void Assembler::xchgb(Register reg, const Operand &op) {
  EnsureSpace ensure_space(this);
  if (!reg.is_byte_register()) {
    // Register is not one of al, bl, cl, dl.  Its encoding needs REX.
    emit_rex_32(reg, op);
  } else {
    emit_optional_rex_32(reg, op);
  }
  emit(0x86);
  emit_operand(reg, op);
}

void Assembler::xchgw(Register reg, const Operand &op) {
  EnsureSpace ensure_space(this);
  emit(0x66);
  emit_optional_rex_32(reg, op);
  emit(0x87);
  emit_operand(reg, op);
}

void Assembler::emit_xchg(Register dst, Register src, int size) {
  EnsureSpace ensure_space(this);
  if (src.is(rax) || dst.is(rax)) {  // single-byte encoding
    Register other = src.is(rax) ? dst : src;
    emit_rex(other, size);
    emit(0x90 | other.low_bits());
  } else if (dst.low_bits() == 4) {
    emit_rex(dst, src, size);
    emit(0x87);
    emit_modrm(dst, src);
  } else {
    emit_rex(src, dst, size);
    emit(0x87);
    emit_modrm(src, dst);
  }
}

void Assembler::emit_xchg(Register dst, const Operand &src, int size) {
  EnsureSpace ensure_space(this);
  emit_rex(dst, src, size);
  emit(0x87);
  emit_operand(dst, src);
}

void Assembler::store_rax(const void *dst) {
  EnsureSpace ensure_space(this);
  if (kPointerSize == kInt64Size) {
    emit(0x48);  // REX.W
    emit(0xA3);
    emitp(dst);
  } else {
    DCHECK(kPointerSize == kInt32Size);
    emit(0xA3);
    emitp(dst);
    // In 64-bit mode, need to zero extend the operand to 8 bytes.
    // See 2.2.1.4 in Intel64 and IA32 Architectures Software
    // Developer's Manual Volume 2.
    emitl(0);
  }
}

void Assembler::testb(Register dst, Register src) {
  EnsureSpace ensure_space(this);
  if (src.low_bits() == 4) {
    emit_rex_32(src, dst);
    emit(0x84);
    emit_modrm(src, dst);
  } else {
    if (!dst.is_byte_register() || !src.is_byte_register()) {
      // Register is not one of al, bl, cl, dl.  Its encoding needs REX.
      emit_rex_32(dst, src);
    }
    emit(0x84);
    emit_modrm(dst, src);
  }
}

void Assembler::testb(Register reg, Immediate mask) {
  DCHECK(is_int8(mask.value_) || is_uint8(mask.value_));
  EnsureSpace ensure_space(this);
  if (reg.is(rax)) {
    emit(0xA8);
    emit(mask.value_);  // Low byte emitted.
  } else {
    if (!reg.is_byte_register()) {
      // Register is not one of al, bl, cl, dl.  Its encoding needs REX.
      emit_rex_32(reg);
    }
    emit(0xF6);
    emit_modrm(0x0, reg);
    emit(mask.value_);  // low byte emitted
  }
}

void Assembler::testb(const Operand &op, Immediate mask) {
  DCHECK(is_int8(mask.value_) || is_uint8(mask.value_));
  EnsureSpace ensure_space(this);
  emit_optional_rex_32(rax, op);
  emit(0xF6);
  emit_operand(rax, op, 1);  // operation code 0
  emit(mask.value_);  // low byte emitted
}

void Assembler::testb(const Operand &op, Register reg) {
  EnsureSpace ensure_space(this);
  if (!reg.is_byte_register()) {
    // Register is not one of al, bl, cl, dl.  Its encoding needs REX.
    emit_rex_32(reg, op);
  } else {
    emit_optional_rex_32(reg, op);
  }
  emit(0x84);
  emit_operand(reg, op);
}

void Assembler::testw(Register dst, Register src) {
  EnsureSpace ensure_space(this);
  emit(0x66);
  if (src.low_bits() == 4) {
    emit_rex_32(src, dst);
  }
  emit(0x85);
  emit_modrm(src, dst);
}

void Assembler::testw(Register reg, Immediate mask) {
  DCHECK(is_int16(mask.value_) || is_uint16(mask.value_));
  EnsureSpace ensure_space(this);
  emit(0x66);
  if (reg.is(rax)) {
    emit(0xA9);
    emitw(mask.value_);
  } else {
    if (reg.low_bits() == 4) {
      emit_rex_32(reg);
    }
    emit(0xF7);
    emit_modrm(0x0, reg);
    emitw(mask.value_);
  }
}

void Assembler::testw(const Operand &op, Immediate mask) {
  DCHECK(is_int16(mask.value_) || is_uint16(mask.value_));
  EnsureSpace ensure_space(this);
  emit(0x66);
  emit_optional_rex_32(rax, op);
  emit(0xF7);
  emit_operand(rax, op, 2);
  emitw(mask.value_);
}

void Assembler::testw(const Operand &op, Register reg) {
  EnsureSpace ensure_space(this);
  emit(0x66);
  emit_optional_rex_32(reg, op);
  emit(0x85);
  emit_operand(rax, op);
}

void Assembler::emit_test(Register dst, Register src, int size) {
  EnsureSpace ensure_space(this);
  if (src.low_bits() == 4) {
    emit_rex(src, dst, size);
    emit(0x85);
    emit_modrm(src, dst);
  } else {
    emit_rex(dst, src, size);
    emit(0x85);
    emit_modrm(dst, src);
  }
}

void Assembler::emit_test(Register reg, Immediate mask, int size) {
  // testl with a mask that fits in the low byte is exactly testb.
  if (is_uint8(mask.value_)) {
    testb(reg, mask);
    return;
  }
  EnsureSpace ensure_space(this);
  if (reg.is(rax)) {
    emit_rex(rax, size);
    emit(0xA9);
    emit(mask);
  } else {
    emit_rex(reg, size);
    emit(0xF7);
    emit_modrm(0x0, reg);
    emit(mask);
  }
}

void Assembler::emit_test(const Operand &op, Immediate mask, int size) {
  // testl with a mask that fits in the low byte is exactly testb.
  if (is_uint8(mask.value_)) {
    testb(op, mask);
    return;
  }
  EnsureSpace ensure_space(this);
  emit_rex(rax, op, size);
  emit(0xF7);
  emit_operand(rax, op, 1);  // operation code 0
  emit(mask);
}

void Assembler::emit_test(const Operand &op, Register reg, int size) {
  EnsureSpace ensure_space(this);
  emit_rex(reg, op, size);
  emit(0x85);
  emit_operand(reg, op);
}

void Assembler::emit_prefetch(const Operand &src, int subcode) {
  EnsureSpace ensure_space(this);
  emit(0x0F);
  emit(0x18);
  emit_operand(subcode, src);
}

// FPU instructions.

void Assembler::fld(int i) {
  EnsureSpace ensure_space(this);
  emit_farith(0xD9, 0xC0, i);
}

void Assembler::fld1() {
  EnsureSpace ensure_space(this);
  emit(0xD9);
  emit(0xE8);
}

void Assembler::fldz() {
  EnsureSpace ensure_space(this);
  emit(0xD9);
  emit(0xEE);
}

void Assembler::fldpi() {
  EnsureSpace ensure_space(this);
  emit(0xD9);
  emit(0xEB);
}

void Assembler::fldln2() {
  EnsureSpace ensure_space(this);
  emit(0xD9);
  emit(0xED);
}

void Assembler::fld_s(const Operand &adr) {
  EnsureSpace ensure_space(this);
  emit_optional_rex_32(adr);
  emit(0xD9);
  emit_operand(0, adr);
}

void Assembler::fld_d(const Operand &adr) {
  EnsureSpace ensure_space(this);
  emit_optional_rex_32(adr);
  emit(0xDD);
  emit_operand(0, adr);
}

void Assembler::fstp_s(const Operand &adr) {
  EnsureSpace ensure_space(this);
  emit_optional_rex_32(adr);
  emit(0xD9);
  emit_operand(3, adr);
}

void Assembler::fstp_d(const Operand &adr) {
  EnsureSpace ensure_space(this);
  emit_optional_rex_32(adr);
  emit(0xDD);
  emit_operand(3, adr);
}

void Assembler::fstp(int index) {
  DCHECK(is_uint3(index));
  EnsureSpace ensure_space(this);
  emit_farith(0xDD, 0xD8, index);
}

void Assembler::fild_s(const Operand &adr) {
  EnsureSpace ensure_space(this);
  emit_optional_rex_32(adr);
  emit(0xDB);
  emit_operand(0, adr);
}

void Assembler::fild_d(const Operand &adr) {
  EnsureSpace ensure_space(this);
  emit_optional_rex_32(adr);
  emit(0xDF);
  emit_operand(5, adr);
}

void Assembler::fistp_s(const Operand &adr) {
  EnsureSpace ensure_space(this);
  emit_optional_rex_32(adr);
  emit(0xDB);
  emit_operand(3, adr);
}

void Assembler::fisttp_s(const Operand &adr) {
  EnsureSpace ensure_space(this);
  emit_optional_rex_32(adr);
  emit(0xDB);
  emit_operand(1, adr);
}

void Assembler::fisttp_d(const Operand &adr) {
  EnsureSpace ensure_space(this);
  emit_optional_rex_32(adr);
  emit(0xDD);
  emit_operand(1, adr);
}

void Assembler::fist_s(const Operand &adr) {
  EnsureSpace ensure_space(this);
  emit_optional_rex_32(adr);
  emit(0xDB);
  emit_operand(2, adr);
}

void Assembler::fistp_d(const Operand &adr) {
  EnsureSpace ensure_space(this);
  emit_optional_rex_32(adr);
  emit(0xDF);
  emit_operand(7, adr);
}

void Assembler::fabs() {
  EnsureSpace ensure_space(this);
  emit(0xD9);
  emit(0xE1);
}

void Assembler::fchs() {
  EnsureSpace ensure_space(this);
  emit(0xD9);
  emit(0xE0);
}

void Assembler::fcos() {
  EnsureSpace ensure_space(this);
  emit(0xD9);
  emit(0xFF);
}

void Assembler::fsin() {
  EnsureSpace ensure_space(this);
  emit(0xD9);
  emit(0xFE);
}

void Assembler::fptan() {
  EnsureSpace ensure_space(this);
  emit(0xD9);
  emit(0xF2);
}

void Assembler::fyl2x() {
  EnsureSpace ensure_space(this);
  emit(0xD9);
  emit(0xF1);
}

void Assembler::f2xm1() {
  EnsureSpace ensure_space(this);
  emit(0xD9);
  emit(0xF0);
}

void Assembler::fscale() {
  EnsureSpace ensure_space(this);
  emit(0xD9);
  emit(0xFD);
}

void Assembler::fninit() {
  EnsureSpace ensure_space(this);
  emit(0xDB);
  emit(0xE3);
}

void Assembler::fadd(int i) {
  EnsureSpace ensure_space(this);
  emit_farith(0xDC, 0xC0, i);
}

void Assembler::fsub(int i) {
  EnsureSpace ensure_space(this);
  emit_farith(0xDC, 0xE8, i);
}

void Assembler::fisub_s(const Operand &adr) {
  EnsureSpace ensure_space(this);
  emit_optional_rex_32(adr);
  emit(0xDA);
  emit_operand(4, adr);
}

void Assembler::fmul(int i) {
  EnsureSpace ensure_space(this);
  emit_farith(0xDC, 0xC8, i);
}

void Assembler::fdiv(int i) {
  EnsureSpace ensure_space(this);
  emit_farith(0xDC, 0xF8, i);
}

void Assembler::faddp(int i) {
  EnsureSpace ensure_space(this);
  emit_farith(0xDE, 0xC0, i);
}

void Assembler::fsubp(int i) {
  EnsureSpace ensure_space(this);
  emit_farith(0xDE, 0xE8, i);
}

void Assembler::fsubrp(int i) {
  EnsureSpace ensure_space(this);
  emit_farith(0xDE, 0xE0, i);
}

void Assembler::fmulp(int i) {
  EnsureSpace ensure_space(this);
  emit_farith(0xDE, 0xC8, i);
}

void Assembler::fdivp(int i) {
  EnsureSpace ensure_space(this);
  emit_farith(0xDE, 0xF8, i);
}

void Assembler::fprem() {
  EnsureSpace ensure_space(this);
  emit(0xD9);
  emit(0xF8);
}

void Assembler::fprem1() {
  EnsureSpace ensure_space(this);
  emit(0xD9);
  emit(0xF5);
}

void Assembler::fxch(int i) {
  EnsureSpace ensure_space(this);
  emit_farith(0xD9, 0xC8, i);
}

void Assembler::fincstp() {
  EnsureSpace ensure_space(this);
  emit(0xD9);
  emit(0xF7);
}

void Assembler::ffree(int i) {
  EnsureSpace ensure_space(this);
  emit_farith(0xDD, 0xC0, i);
}

void Assembler::ftst() {
  EnsureSpace ensure_space(this);
  emit(0xD9);
  emit(0xE4);
}

void Assembler::fucomp(int i) {
  EnsureSpace ensure_space(this);
  emit_farith(0xDD, 0xE8, i);
}

void Assembler::fucompp() {
  EnsureSpace ensure_space(this);
  emit(0xDA);
  emit(0xE9);
}

void Assembler::fucomi(int i) {
  EnsureSpace ensure_space(this);
  emit(0xDB);
  emit(0xE8 + i);
}

void Assembler::fucomip() {
  EnsureSpace ensure_space(this);
  emit(0xDF);
  emit(0xE9);
}

void Assembler::fcompp() {
  EnsureSpace ensure_space(this);
  emit(0xDE);
  emit(0xD9);
}

void Assembler::fnstsw_ax() {
  EnsureSpace ensure_space(this);
  emit(0xDF);
  emit(0xE0);
}

void Assembler::fwait() {
  EnsureSpace ensure_space(this);
  emit(0x9B);
}

void Assembler::frndint() {
  EnsureSpace ensure_space(this);
  emit(0xD9);
  emit(0xFC);
}

void Assembler::fnclex() {
  EnsureSpace ensure_space(this);
  emit(0xDB);
  emit(0xE2);
}

void Assembler::sahf() {
  DCHECK(Enabled(SAHF));
  EnsureSpace ensure_space(this);
  emit(0x9E);
}

void Assembler::emit_farith(int b1, int b2, int i) {
  DCHECK(is_uint8(b1) && is_uint8(b2));  // wrong opcode
  DCHECK(is_uint3(i));  // illegal stack offset
  emit(b1);
  emit(b2 + i);
}

void Assembler::andps(XMMRegister dst, XMMRegister src) {
  EnsureSpace ensure_space(this);
  emit_optional_rex_32(dst, src);
  emit(0x0F);
  emit(0x54);
  emit_sse_operand(dst, src);
}

void Assembler::andps(XMMRegister dst, const Operand &src) {
  EnsureSpace ensure_space(this);
  emit_optional_rex_32(dst, src);
  emit(0x0F);
  emit(0x54);
  emit_sse_operand(dst, src);
}

void Assembler::orps(XMMRegister dst, XMMRegister src) {
  EnsureSpace ensure_space(this);
  emit_optional_rex_32(dst, src);
  emit(0x0F);
  emit(0x56);
  emit_sse_operand(dst, src);
}

void Assembler::orps(XMMRegister dst, const Operand &src) {
  EnsureSpace ensure_space(this);
  emit_optional_rex_32(dst, src);
  emit(0x0F);
  emit(0x56);
  emit_sse_operand(dst, src);
}

void Assembler::xorps(XMMRegister dst, XMMRegister src) {
  DCHECK(Enabled(SSE));
  EnsureSpace ensure_space(this);
  emit_optional_rex_32(dst, src);
  emit(0x0F);
  emit(0x57);
  emit_sse_operand(dst, src);
}

void Assembler::xorps(XMMRegister dst, const Operand &src) {
  DCHECK(Enabled(SSE));
  EnsureSpace ensure_space(this);
  emit_optional_rex_32(dst, src);
  emit(0x0F);
  emit(0x57);
  emit_sse_operand(dst, src);
}

void Assembler::andnps(XMMRegister dst, XMMRegister src) {
  EnsureSpace ensure_space(this);
  emit_optional_rex_32(dst, src);
  emit(0x0F);
  emit(0x55);
  emit_sse_operand(dst, src);
}

void Assembler::andnps(XMMRegister dst, const Operand &src) {
  EnsureSpace ensure_space(this);
  emit_optional_rex_32(dst, src);
  emit(0x0F);
  emit(0x55);
  emit_sse_operand(dst, src);
}

void Assembler::addps(XMMRegister dst, XMMRegister src) {
  EnsureSpace ensure_space(this);
  emit_optional_rex_32(dst, src);
  emit(0x0F);
  emit(0x58);
  emit_sse_operand(dst, src);
}

void Assembler::addps(XMMRegister dst, const Operand &src) {
  EnsureSpace ensure_space(this);
  emit_optional_rex_32(dst, src);
  emit(0x0F);
  emit(0x58);
  emit_sse_operand(dst, src);
}

void Assembler::subps(XMMRegister dst, XMMRegister src) {
  EnsureSpace ensure_space(this);
  emit_optional_rex_32(dst, src);
  emit(0x0F);
  emit(0x5C);
  emit_sse_operand(dst, src);
}

void Assembler::subps(XMMRegister dst, const Operand &src) {
  EnsureSpace ensure_space(this);
  emit_optional_rex_32(dst, src);
  emit(0x0F);
  emit(0x5C);
  emit_sse_operand(dst, src);
}

void Assembler::mulps(XMMRegister dst, XMMRegister src) {
  EnsureSpace ensure_space(this);
  emit_optional_rex_32(dst, src);
  emit(0x0F);
  emit(0x59);
  emit_sse_operand(dst, src);
}

void Assembler::mulps(XMMRegister dst, const Operand &src) {
  EnsureSpace ensure_space(this);
  emit_optional_rex_32(dst, src);
  emit(0x0F);
  emit(0x59);
  emit_sse_operand(dst, src);
}

void Assembler::divps(XMMRegister dst, XMMRegister src) {
  EnsureSpace ensure_space(this);
  emit_optional_rex_32(dst, src);
  emit(0x0F);
  emit(0x5E);
  emit_sse_operand(dst, src);
}

void Assembler::divps(XMMRegister dst, const Operand &src) {
  EnsureSpace ensure_space(this);
  emit_optional_rex_32(dst, src);
  emit(0x0F);
  emit(0x5E);
  emit_sse_operand(dst, src);
}

void Assembler::minps(XMMRegister dst, XMMRegister src) {
  EnsureSpace ensure_space(this);
  emit_optional_rex_32(dst, src);
  emit(0x0F);
  emit(0x5D);
  emit_sse_operand(dst, src);
}

void Assembler::minps(XMMRegister dst, const Operand &src) {
  EnsureSpace ensure_space(this);
  emit_optional_rex_32(dst, src);
  emit(0x0F);
  emit(0x5D);
  emit_sse_operand(dst, src);
}

void Assembler::maxps(XMMRegister dst, XMMRegister src) {
  EnsureSpace ensure_space(this);
  emit_optional_rex_32(dst, src);
  emit(0x0F);
  emit(0x5F);
  emit_sse_operand(dst, src);
}

void Assembler::maxps(XMMRegister dst, const Operand &src) {
  EnsureSpace ensure_space(this);
  emit_optional_rex_32(dst, src);
  emit(0x0F);
  emit(0x5F);
  emit_sse_operand(dst, src);
}

void Assembler::addpd(XMMRegister dst, XMMRegister src) {
  DCHECK(Enabled(SSE2));
  EnsureSpace ensure_space(this);
  emit(0x66);
  emit_optional_rex_32(dst, src);
  emit(0x0F);
  emit(0x58);
  emit_sse_operand(dst, src);
}

void Assembler::addpd(XMMRegister dst, const Operand &src) {
  DCHECK(Enabled(SSE2));
  EnsureSpace ensure_space(this);
  emit(0x66);
  emit_optional_rex_32(dst, src);
  emit(0x0F);
  emit(0x58);
  emit_sse_operand(dst, src);
}

void Assembler::subpd(XMMRegister dst, XMMRegister src) {
  DCHECK(Enabled(SSE2));
  EnsureSpace ensure_space(this);
  emit(0x66);
  emit_optional_rex_32(dst, src);
  emit(0x0F);
  emit(0x5C);
  emit_sse_operand(dst, src);
}

void Assembler::subpd(XMMRegister dst, const Operand &src) {
  DCHECK(Enabled(SSE2));
  EnsureSpace ensure_space(this);
  emit(0x66);
  emit_optional_rex_32(dst, src);
  emit(0x0F);
  emit(0x5C);
  emit_sse_operand(dst, src);
}

void Assembler::mulpd(XMMRegister dst, XMMRegister src) {
  DCHECK(Enabled(SSE2));
  EnsureSpace ensure_space(this);
  emit(0x66);
  emit_optional_rex_32(dst, src);
  emit(0x0F);
  emit(0x59);
  emit_sse_operand(dst, src);
}

void Assembler::mulpd(XMMRegister dst, const Operand &src) {
  DCHECK(Enabled(SSE2));
  EnsureSpace ensure_space(this);
  emit(0x66);
  emit_optional_rex_32(dst, src);
  emit(0x0F);
  emit(0x59);
  emit_sse_operand(dst, src);
}

void Assembler::divpd(XMMRegister dst, XMMRegister src) {
  DCHECK(Enabled(SSE2));
  EnsureSpace ensure_space(this);
  emit(0x66);
  emit_optional_rex_32(dst, src);
  emit(0x0F);
  emit(0x5E);
  emit_sse_operand(dst, src);
}

void Assembler::divpd(XMMRegister dst, const Operand &src) {
  DCHECK(Enabled(SSE2));
  EnsureSpace ensure_space(this);
  emit(0x66);
  emit_optional_rex_32(dst, src);
  emit(0x0F);
  emit(0x5E);
  emit_sse_operand(dst, src);
}

void Assembler::minpd(XMMRegister dst, XMMRegister src) {
  DCHECK(Enabled(SSE2));
  EnsureSpace ensure_space(this);
  emit(0x66);
  emit_optional_rex_32(dst, src);
  emit(0x0F);
  emit(0x5D);
  emit_sse_operand(dst, src);
}

void Assembler::minpd(XMMRegister dst, const Operand &src) {
  DCHECK(Enabled(SSE2));
  EnsureSpace ensure_space(this);
  emit(0x66);
  emit_optional_rex_32(dst, src);
  emit(0x0F);
  emit(0x5D);
  emit_sse_operand(dst, src);
}

void Assembler::maxpd(XMMRegister dst, XMMRegister src) {
  DCHECK(Enabled(SSE2));
  EnsureSpace ensure_space(this);
  emit(0x66);
  emit_optional_rex_32(dst, src);
  emit(0x0F);
  emit(0x5F);
  emit_sse_operand(dst, src);
}

void Assembler::maxpd(XMMRegister dst, const Operand &src) {
  DCHECK(Enabled(SSE2));
  EnsureSpace ensure_space(this);
  emit(0x66);
  emit_optional_rex_32(dst, src);
  emit(0x0F);
  emit(0x5F);
  emit_sse_operand(dst, src);
}

void Assembler::movd(XMMRegister dst, Register src) {
  DCHECK(Enabled(MMX));
  EnsureSpace ensure_space(this);
  emit(0x66);
  emit_optional_rex_32(dst, src);
  emit(0x0F);
  emit(0x6E);
  emit_sse_operand(dst, src);
}

void Assembler::movd(XMMRegister dst, const Operand &src) {
  DCHECK(Enabled(MMX));
  EnsureSpace ensure_space(this);
  emit(0x66);
  emit_optional_rex_32(dst, src);
  emit(0x0F);
  emit(0x6E);
  emit_sse_operand(dst, src);
}

void Assembler::movd(Register dst, XMMRegister src) {
  DCHECK(Enabled(MMX));
  EnsureSpace ensure_space(this);
  emit(0x66);
  emit_optional_rex_32(src, dst);
  emit(0x0F);
  emit(0x7E);
  emit_sse_operand(src, dst);
}

void Assembler::movd(const Operand &dst, XMMRegister src) {
  DCHECK(Enabled(SSE2));
  EnsureSpace ensure_space(this);
  emit(0x66);
  emit_optional_rex_32(src, dst);
  emit(0x0F);
  emit(0x7E);
  emit_sse_operand(src, dst);
}

void Assembler::movq(XMMRegister dst, Register src) {
  DCHECK(Enabled(MMX));
  EnsureSpace ensure_space(this);
  emit(0x66);
  emit_rex_64(dst, src);
  emit(0x0F);
  emit(0x6E);
  emit_sse_operand(dst, src);
}

void Assembler::movq(Register dst, XMMRegister src) {
  DCHECK(Enabled(MMX));
  EnsureSpace ensure_space(this);
  emit(0x66);
  emit_rex_64(src, dst);
  emit(0x0F);
  emit(0x7E);
  emit_sse_operand(src, dst);
}

void Assembler::movq(const Operand &dst, XMMRegister src) {
  DCHECK(Enabled(SSE2));
  EnsureSpace ensure_space(this);
  emit(0x66);
  emit_rex_64(src, dst);
  emit(0x0F);
  emit(0x7E);
  emit_sse_operand(src, dst);
}

void Assembler::movq(XMMRegister dst, XMMRegister src) {
  DCHECK(Enabled(MMX));
  EnsureSpace ensure_space(this);
  if (dst.low_bits() == 4) {
    // Avoid unnecessary SIB byte.
    emit(0xF3);
    emit_optional_rex_32(dst, src);
    emit(0x0F);
    emit(0x7E);
    emit_sse_operand(dst, src);
  } else {
    emit(0x66);
    emit_optional_rex_32(src, dst);
    emit(0x0F);
    emit(0xD6);
    emit_sse_operand(src, dst);
  }
}

void Assembler::movdqa(XMMRegister dst, XMMRegister src) {
  DCHECK(Enabled(SSE2));
  EnsureSpace ensure_space(this);
  emit(0x66);
  emit_optional_rex_32(dst, src);
  emit(0x0F);
  emit(0x6f);
  emit_sse_operand(dst, src);
}

void Assembler::movdqa(const Operand &dst, XMMRegister src) {
  DCHECK(Enabled(SSE2));
  EnsureSpace ensure_space(this);
  emit(0x66);
  emit_optional_rex_32(src, dst);
  emit(0x0F);
  emit(0x7F);
  emit_sse_operand(src, dst);
}

void Assembler::movdqa(XMMRegister dst, const Operand &src) {
  DCHECK(Enabled(SSE2));
  EnsureSpace ensure_space(this);
  emit(0x66);
  emit_optional_rex_32(dst, src);
  emit(0x0F);
  emit(0x6F);
  emit_sse_operand(dst, src);
}

void Assembler::movdqu(XMMRegister dst, XMMRegister src) {
  DCHECK(Enabled(SSE2));
  EnsureSpace ensure_space(this);
  emit(0xF3);
  emit_optional_rex_32(dst, src);
  emit(0x0F);
  emit(0x6F);
  emit_sse_operand(dst, src);
}

void Assembler::movdqu(const Operand &dst, XMMRegister src) {
  EnsureSpace ensure_space(this);
  emit(0xF3);
  emit_optional_rex_32(src, dst);
  emit(0x0F);
  emit(0x7F);
  emit_sse_operand(src, dst);
}

void Assembler::movdqu(XMMRegister dst, const Operand &src) {
  EnsureSpace ensure_space(this);
  emit(0xF3);
  emit_optional_rex_32(dst, src);
  emit(0x0F);
  emit(0x6F);
  emit_sse_operand(dst, src);
}

void Assembler::extractps(Register dst, XMMRegister src, byte imm8) {
  DCHECK(Enabled(SSE4_1));
  DCHECK(is_uint8(imm8));
  EnsureSpace ensure_space(this);
  emit(0x66);
  emit_optional_rex_32(src, dst);
  emit(0x0F);
  emit(0x3A);
  emit(0x17);
  emit_sse_operand(src, dst);
  emit(imm8);
}

void Assembler::pextrb(Register dst, XMMRegister src, int8_t imm8) {
  DCHECK(Enabled(SSE4_1));
  DCHECK(is_uint8(imm8));
  EnsureSpace ensure_space(this);
  emit(0x66);
  emit_optional_rex_32(src, dst);
  emit(0x0F);
  emit(0x3A);
  emit(0x14);
  emit_sse_operand(src, dst);
  emit(imm8);
}

void Assembler::pextrb(const Operand &dst, XMMRegister src, int8_t imm8) {
  DCHECK(Enabled(SSE4_1));
  DCHECK(is_uint8(imm8));
  EnsureSpace ensure_space(this);
  emit(0x66);
  emit_optional_rex_32(src, dst);
  emit(0x0F);
  emit(0x3A);
  emit(0x14);
  emit_sse_operand(src, dst, 1);
  emit(imm8);
}

void Assembler::pinsrb(XMMRegister dst, Register src, int8_t imm8) {
  DCHECK(Enabled(SSE4_1));
  EnsureSpace ensure_space(this);
  emit(0x66);
  emit_optional_rex_32(dst, src);
  emit(0x0F);
  emit(0x3A);
  emit(0x20);
  emit_sse_operand(dst, src);
  emit(imm8);
}

void Assembler::pinsrb(XMMRegister dst, const Operand &src, int8_t imm8) {
  DCHECK(Enabled(SSE4_1));
  EnsureSpace ensure_space(this);
  emit(0x66);
  emit_optional_rex_32(dst, src);
  emit(0x0F);
  emit(0x3A);
  emit(0x20);
  emit_sse_operand(dst, src, 1);
  emit(imm8);
}

void Assembler::pinsrw(XMMRegister dst, Register src, int8_t imm8) {
  DCHECK(is_uint8(imm8));
  EnsureSpace ensure_space(this);
  emit(0x66);
  emit_optional_rex_32(dst, src);
  emit(0x0F);
  emit(0xC4);
  emit_sse_operand(dst, src);
  emit(imm8);
}

void Assembler::pinsrw(XMMRegister dst, const Operand &src, int8_t imm8) {
  DCHECK(is_uint8(imm8));
  EnsureSpace ensure_space(this);
  emit(0x66);
  emit_optional_rex_32(dst, src);
  emit(0x0F);
  emit(0xC4);
  emit_sse_operand(dst, src, 1);
  emit(imm8);
}

void Assembler::pextrw(Register dst, XMMRegister src, int8_t imm8) {
  DCHECK(is_uint8(imm8));
  EnsureSpace ensure_space(this);
  emit(0x66);
  emit_optional_rex_32(src, dst);
  emit(0x0F);
  emit(0xC5);
  emit_sse_operand(src, dst);
  emit(imm8);
}

void Assembler::pextrw(const Operand &dst, XMMRegister src, int8_t imm8) {
  DCHECK(Enabled(SSE4_1));
  DCHECK(is_uint8(imm8));
  EnsureSpace ensure_space(this);
  emit(0x66);
  emit_optional_rex_32(src, dst);
  emit(0x0F);
  emit(0x3A);
  emit(0x15);
  emit_sse_operand(src, dst, 1);
  emit(imm8);
}

void Assembler::pextrd(Register dst, XMMRegister src, int8_t imm8) {
  DCHECK(Enabled(SSE4_1));
  EnsureSpace ensure_space(this);
  emit(0x66);
  emit_optional_rex_32(src, dst);
  emit(0x0F);
  emit(0x3A);
  emit(0x16);
  emit_sse_operand(src, dst);
  emit(imm8);
}

void Assembler::pextrd(const Operand &dst, XMMRegister src, int8_t imm8) {
  DCHECK(Enabled(SSE4_1));
  EnsureSpace ensure_space(this);
  emit(0x66);
  emit_optional_rex_32(src, dst);
  emit(0x0F);
  emit(0x3A);
  emit(0x16);
  emit_sse_operand(src, dst, 1);
  emit(imm8);
}

void Assembler::pinsrd(XMMRegister dst, Register src, int8_t imm8) {
  DCHECK(Enabled(SSE4_1));
  EnsureSpace ensure_space(this);
  emit(0x66);
  emit_optional_rex_32(dst, src);
  emit(0x0F);
  emit(0x3A);
  emit(0x22);
  emit_sse_operand(dst, src);
  emit(imm8);
}

void Assembler::pinsrd(XMMRegister dst, const Operand &src, int8_t imm8) {
  DCHECK(Enabled(SSE4_1));
  EnsureSpace ensure_space(this);
  emit(0x66);
  emit_optional_rex_32(dst, src);
  emit(0x0F);
  emit(0x3A);
  emit(0x22);
  emit_sse_operand(dst, src, 1);
  emit(imm8);
}

void Assembler::pextrq(Register dst, XMMRegister src, int8_t imm8) {
  DCHECK(Enabled(SSE4_1));
  EnsureSpace ensure_space(this);
  emit(0x66);
  emit_rex_64(dst, src);
  emit(0x0F);
  emit(0x3A);
  emit(0x16);
  emit_sse_operand(src, dst);
  emit(imm8);
}

void Assembler::pextrq(const Operand &dst, XMMRegister src, int8_t imm8) {
  DCHECK(Enabled(SSE4_1));
  EnsureSpace ensure_space(this);
  emit(0x66);
  emit_rex_64(src, dst);
  emit(0x0F);
  emit(0x3A);
  emit(0x16);
  emit_sse_operand(src, dst, 1);
  emit(imm8);
}

void Assembler::pinsrq(XMMRegister dst, Register src, int8_t imm8) {
  DCHECK(Enabled(SSE4_1));
  EnsureSpace ensure_space(this);
  emit(0x66);
  emit_rex_64(dst, src);
  emit(0x0F);
  emit(0x3A);
  emit(0x22);
  emit_sse_operand(dst, src);
  emit(imm8);
}

void Assembler::pinsrq(XMMRegister dst, const Operand &src, int8_t imm8) {
  DCHECK(Enabled(SSE4_1));
  EnsureSpace ensure_space(this);
  emit(0x66);
  emit_rex_64(dst, src);
  emit(0x0F);
  emit(0x3A);
  emit(0x22);
  emit_sse_operand(dst, src, 1);
  emit(imm8);
}

void Assembler::insertps(XMMRegister dst, XMMRegister src, byte imm8) {
  DCHECK(Enabled(SSE4_1));
  DCHECK(is_uint8(imm8));
  EnsureSpace ensure_space(this);
  emit(0x66);
  emit_optional_rex_32(dst, src);
  emit(0x0F);
  emit(0x3A);
  emit(0x21);
  emit_sse_operand(dst, src);
  emit(imm8);
}

void Assembler::movsd(const Operand &dst, XMMRegister src) {
  DCHECK(Enabled(MMX));
  EnsureSpace ensure_space(this);
  emit(0xF2);  // double
  emit_optional_rex_32(src, dst);
  emit(0x0F);
  emit(0x11);  // store
  emit_sse_operand(src, dst);
}

void Assembler::movsd(XMMRegister dst, XMMRegister src) {
  DCHECK(Enabled(MMX));
  EnsureSpace ensure_space(this);
  emit(0xF2);  // double
  emit_optional_rex_32(dst, src);
  emit(0x0F);
  emit(0x10);  // load
  emit_sse_operand(dst, src);
}

void Assembler::movsd(XMMRegister dst, const Operand &src) {
  DCHECK(Enabled(MMX));
  EnsureSpace ensure_space(this);
  emit(0xF2);  // double
  emit_optional_rex_32(dst, src);
  emit(0x0F);
  emit(0x10);  // load
  emit_sse_operand(dst, src);
}

void Assembler::movaps(XMMRegister dst, XMMRegister src) {
  DCHECK(Enabled(SSE));
  EnsureSpace ensure_space(this);
  if (src.low_bits() == 4) {
    // Try to avoid an unnecessary SIB byte.
    emit_optional_rex_32(src, dst);
    emit(0x0F);
    emit(0x29);
    emit_sse_operand(src, dst);
  } else {
    emit_optional_rex_32(dst, src);
    emit(0x0F);
    emit(0x28);
    emit_sse_operand(dst, src);
  }
}

void Assembler::movaps(XMMRegister dst, const Operand &src) {
  DCHECK(Enabled(SSE));
  EnsureSpace ensure_space(this);
  emit_optional_rex_32(dst, src);
  emit(0x0F);
  emit(0x28);
  emit_sse_operand(dst, src);
}

void Assembler::movaps(const Operand &dst, XMMRegister src) {
  DCHECK(Enabled(SSE));
  EnsureSpace ensure_space(this);
  emit_optional_rex_32(src, dst);
  emit(0x0F);
  emit(0x29);
  emit_sse_operand(src, dst);
}

void Assembler::shufps(XMMRegister dst, XMMRegister src, byte imm8) {
  DCHECK(is_uint8(imm8));
  EnsureSpace ensure_space(this);
  emit_optional_rex_32(src, dst);
  emit(0x0F);
  emit(0xC6);
  emit_sse_operand(dst, src);
  emit(imm8);
}

void Assembler::shufps(XMMRegister dst, const Operand &src, byte imm8) {
  DCHECK(is_uint8(imm8));
  EnsureSpace ensure_space(this);
  emit_optional_rex_32(dst, src);
  emit(0x0F);
  emit(0xC6);
  emit_sse_operand(dst, src);
  emit(imm8);
}

void Assembler::shufpd(XMMRegister dst, XMMRegister src, byte imm8) {
  DCHECK(is_uint8(imm8));
  EnsureSpace ensure_space(this);
  emit(0x66);
  emit_optional_rex_32(dst, src);
  emit(0x0F);
  emit(0xC6);
  emit_sse_operand(dst, src);
  emit(imm8);
}

void Assembler::shufpd(XMMRegister dst, const Operand &src, byte imm8) {
  DCHECK(is_uint8(imm8));
  EnsureSpace ensure_space(this);
  emit(0x66);
  emit_optional_rex_32(dst, src);
  emit(0x0F);
  emit(0xC6);
  emit_sse_operand(dst, src);
  emit(imm8);
}

void Assembler::movapd(XMMRegister dst, XMMRegister src) {
  DCHECK(Enabled(SSE2));
  EnsureSpace ensure_space(this);
  if (src.low_bits() == 4) {
    // Try to avoid an unnecessary SIB byte.
    emit(0x66);
    emit_optional_rex_32(src, dst);
    emit(0x0F);
    emit(0x29);
    emit_sse_operand(src, dst);
  } else {
    emit(0x66);
    emit_optional_rex_32(dst, src);
    emit(0x0F);
    emit(0x28);
    emit_sse_operand(dst, src);
  }
}

void Assembler::movapd(XMMRegister dst, const Operand &src) {
  DCHECK(Enabled(SSE2));
  EnsureSpace ensure_space(this);
  emit(0x66);
  emit_optional_rex_32(dst, src);
  emit(0x0F);
  emit(0x28);
  emit_sse_operand(dst, src);
}

void Assembler::movapd(const Operand &dst, XMMRegister src) {
  DCHECK(Enabled(SSE2));
  EnsureSpace ensure_space(this);
  emit(0x66);
  emit_optional_rex_32(src, dst);
  emit(0x0F);
  emit(0x29);
  emit_sse_operand(src, dst);
}

void Assembler::movupd(XMMRegister dst, XMMRegister src) {
  DCHECK(Enabled(SSE2));
  EnsureSpace ensure_space(this);
  if (src.low_bits() == 4) {
    // Try to avoid an unnecessary SIB byte.
    emit(0x66);
    emit_optional_rex_32(src, dst);
    emit(0x0F);
    emit(0x11);
    emit_sse_operand(src, dst);
  } else {
    emit(0x66);
    emit_optional_rex_32(dst, src);
    emit(0x0F);
    emit(0x10);
    emit_sse_operand(dst, src);
  }
}

void Assembler::movupd(XMMRegister dst, const Operand &src) {
  DCHECK(Enabled(SSE2));
  EnsureSpace ensure_space(this);
  emit(0x66);
  emit_optional_rex_32(dst, src);
  emit(0x0F);
  emit(0x10);
  emit_sse_operand(dst, src);
}

void Assembler::movupd(const Operand &dst, XMMRegister src) {
  DCHECK(Enabled(SSE2));
  EnsureSpace ensure_space(this);
  emit(0x66);
  emit_optional_rex_32(src, dst);
  emit(0x0F);
  emit(0x11);
  emit_sse_operand(src, dst);
}

void Assembler::addss(XMMRegister dst, XMMRegister src) {
  EnsureSpace ensure_space(this);
  emit(0xF3);
  emit_optional_rex_32(dst, src);
  emit(0x0F);
  emit(0x58);
  emit_sse_operand(dst, src);
}

void Assembler::addss(XMMRegister dst, const Operand &src) {
  EnsureSpace ensure_space(this);
  emit(0xF3);
  emit_optional_rex_32(dst, src);
  emit(0x0F);
  emit(0x58);
  emit_sse_operand(dst, src);
}

void Assembler::subss(XMMRegister dst, XMMRegister src) {
  EnsureSpace ensure_space(this);
  emit(0xF3);
  emit_optional_rex_32(dst, src);
  emit(0x0F);
  emit(0x5C);
  emit_sse_operand(dst, src);
}

void Assembler::subss(XMMRegister dst, const Operand &src) {
  EnsureSpace ensure_space(this);
  emit(0xF3);
  emit_optional_rex_32(dst, src);
  emit(0x0F);
  emit(0x5C);
  emit_sse_operand(dst, src);
}

void Assembler::mulss(XMMRegister dst, XMMRegister src) {
  EnsureSpace ensure_space(this);
  emit(0xF3);
  emit_optional_rex_32(dst, src);
  emit(0x0F);
  emit(0x59);
  emit_sse_operand(dst, src);
}

void Assembler::mulss(XMMRegister dst, const Operand &src) {
  EnsureSpace ensure_space(this);
  emit(0xF3);
  emit_optional_rex_32(dst, src);
  emit(0x0F);
  emit(0x59);
  emit_sse_operand(dst, src);
}

void Assembler::divss(XMMRegister dst, XMMRegister src) {
  EnsureSpace ensure_space(this);
  emit(0xF3);
  emit_optional_rex_32(dst, src);
  emit(0x0F);
  emit(0x5E);
  emit_sse_operand(dst, src);
}

void Assembler::divss(XMMRegister dst, const Operand &src) {
  EnsureSpace ensure_space(this);
  emit(0xF3);
  emit_optional_rex_32(dst, src);
  emit(0x0F);
  emit(0x5E);
  emit_sse_operand(dst, src);
}

void Assembler::maxss(XMMRegister dst, XMMRegister src) {
  EnsureSpace ensure_space(this);
  emit(0xF3);
  emit_optional_rex_32(dst, src);
  emit(0x0F);
  emit(0x5F);
  emit_sse_operand(dst, src);
}

void Assembler::maxss(XMMRegister dst, const Operand &src) {
  EnsureSpace ensure_space(this);
  emit(0xF3);
  emit_optional_rex_32(dst, src);
  emit(0x0F);
  emit(0x5F);
  emit_sse_operand(dst, src);
}

void Assembler::minss(XMMRegister dst, XMMRegister src) {
  EnsureSpace ensure_space(this);
  emit(0xF3);
  emit_optional_rex_32(dst, src);
  emit(0x0F);
  emit(0x5D);
  emit_sse_operand(dst, src);
}

void Assembler::minss(XMMRegister dst, const Operand &src) {
  EnsureSpace ensure_space(this);
  emit(0xF3);
  emit_optional_rex_32(dst, src);
  emit(0x0F);
  emit(0x5D);
  emit_sse_operand(dst, src);
}

void Assembler::cmpss(XMMRegister dst, XMMRegister src, int8_t cmp) {
  EnsureSpace ensure_space(this);
  emit(0xF3);
  emit_optional_rex_32(dst, src);
  emit(0x0F);
  emit(0xC2);
  emit_sse_operand(dst, src);
  emit(cmp);
}

void Assembler::cmpss(XMMRegister dst, const Operand &src, int8_t cmp) {
  EnsureSpace ensure_space(this);
  emit(0xF3);
  emit_optional_rex_32(dst, src);
  emit(0x0F);
  emit(0xC2);
  emit_sse_operand(dst, src, 1);
  emit(cmp);
}

void Assembler::sqrtss(XMMRegister dst, XMMRegister src) {
  EnsureSpace ensure_space(this);
  emit(0xF3);
  emit_optional_rex_32(dst, src);
  emit(0x0F);
  emit(0x51);
  emit_sse_operand(dst, src);
}

void Assembler::sqrtss(XMMRegister dst, const Operand &src) {
  EnsureSpace ensure_space(this);
  emit(0xF3);
  emit_optional_rex_32(dst, src);
  emit(0x0F);
  emit(0x51);
  emit_sse_operand(dst, src);
}

void Assembler::rsqrtss(XMMRegister dst, XMMRegister src) {
  EnsureSpace ensure_space(this);
  emit(0xF3);
  emit_optional_rex_32(dst, src);
  emit(0x0F);
  emit(0x52);
  emit_sse_operand(dst, src);
}

void Assembler::rsqrtss(XMMRegister dst, const Operand &src) {
  EnsureSpace ensure_space(this);
  emit(0xF3);
  emit_optional_rex_32(dst, src);
  emit(0x0F);
  emit(0x52);
  emit_sse_operand(dst, src);
}

void Assembler::ucomiss(XMMRegister dst, XMMRegister src) {
  DCHECK(Enabled(MMX));
  EnsureSpace ensure_space(this);
  emit_optional_rex_32(dst, src);
  emit(0x0f);
  emit(0x2e);
  emit_sse_operand(dst, src);
}

void Assembler::ucomiss(XMMRegister dst, const Operand &src) {
  DCHECK(Enabled(MMX));
  EnsureSpace ensure_space(this);
  emit_optional_rex_32(dst, src);
  emit(0x0f);
  emit(0x2e);
  emit_sse_operand(dst, src);
}

void Assembler::movss(XMMRegister dst, XMMRegister src) {
  DCHECK(Enabled(MMX));
  EnsureSpace ensure_space(this);
  emit(0xF3);  // single
  emit_optional_rex_32(dst, src);
  emit(0x0F);
  emit(0x10);  // load
  emit_sse_operand(dst, src);
}

void Assembler::movss(XMMRegister dst, const Operand &src) {
  DCHECK(Enabled(SSE));
  EnsureSpace ensure_space(this);
  emit(0xF3);  // single
  emit_optional_rex_32(dst, src);
  emit(0x0F);
  emit(0x10);  // load
  emit_sse_operand(dst, src);
}

void Assembler::movss(const Operand &src, XMMRegister dst) {
  DCHECK(Enabled(SSE));
  EnsureSpace ensure_space(this);
  emit(0xF3);  // single
  emit_optional_rex_32(dst, src);
  emit(0x0F);
  emit(0x11);  // store
  emit_sse_operand(dst, src);
}

void Assembler::psllq(XMMRegister reg, byte imm8) {
  DCHECK(Enabled(MMX));
  EnsureSpace ensure_space(this);
  emit(0x66);
  emit_optional_rex_32(reg);
  emit(0x0F);
  emit(0x73);
  emit_sse_operand(rsi, reg);  // rsi == 6
  emit(imm8);
}

void Assembler::psrlq(XMMRegister reg, byte imm8) {
  DCHECK(Enabled(MMX));
  EnsureSpace ensure_space(this);
  emit(0x66);
  emit_optional_rex_32(reg);
  emit(0x0F);
  emit(0x73);
  emit_sse_operand(rdx, reg);  // rdx == 2
  emit(imm8);
}

void Assembler::psllw(XMMRegister reg, byte imm8) {
  EnsureSpace ensure_space(this);
  emit(0x66);
  emit_optional_rex_32(reg);
  emit(0x0F);
  emit(0x71);
  emit_sse_operand(rsi, reg);  // rsi == 6
  emit(imm8);
}

void Assembler::pslld(XMMRegister reg, byte imm8) {
  EnsureSpace ensure_space(this);
  emit(0x66);
  emit_optional_rex_32(reg);
  emit(0x0F);
  emit(0x72);
  emit_sse_operand(rsi, reg);  // rsi == 6
  emit(imm8);
}

void Assembler::psrlw(XMMRegister reg, byte imm8) {
  EnsureSpace ensure_space(this);
  emit(0x66);
  emit_optional_rex_32(reg);
  emit(0x0F);
  emit(0x71);
  emit_sse_operand(rdx, reg);  // rdx == 2
  emit(imm8);
}

void Assembler::psrld(XMMRegister reg, byte imm8) {
  EnsureSpace ensure_space(this);
  emit(0x66);
  emit_optional_rex_32(reg);
  emit(0x0F);
  emit(0x72);
  emit_sse_operand(rdx, reg);  // rdx == 2
  emit(imm8);
}

void Assembler::psraw(XMMRegister reg, byte imm8) {
  EnsureSpace ensure_space(this);
  emit(0x66);
  emit_optional_rex_32(reg);
  emit(0x0F);
  emit(0x71);
  emit_sse_operand(rsp, reg);  // rsp == 4
  emit(imm8);
}

void Assembler::psrad(XMMRegister reg, byte imm8) {
  EnsureSpace ensure_space(this);
  emit(0x66);
  emit_optional_rex_32(reg);
  emit(0x0F);
  emit(0x72);
  emit_sse_operand(rsp, reg);  // rsp == 4
  emit(imm8);
}

void Assembler::cmpps(XMMRegister dst, XMMRegister src, int8_t cmp) {
  EnsureSpace ensure_space(this);
  emit_optional_rex_32(dst, src);
  emit(0x0F);
  emit(0xC2);
  emit_sse_operand(dst, src);
  emit(cmp);
}

void Assembler::cmpps(XMMRegister dst, const Operand &src, int8_t cmp) {
  EnsureSpace ensure_space(this);
  emit_optional_rex_32(dst, src);
  emit(0x0F);
  emit(0xC2);
  emit_sse_operand(dst, src, 1);
  emit(cmp);
}

void Assembler::cmppd(XMMRegister dst, XMMRegister src, int8_t cmp) {
  EnsureSpace ensure_space(this);
  emit(0x66);
  emit_optional_rex_32(dst, src);
  emit(0x0F);
  emit(0xC2);
  emit_sse_operand(dst, src);
  emit(cmp);
}

void Assembler::cmppd(XMMRegister dst, const Operand &src, int8_t cmp) {
  EnsureSpace ensure_space(this);
  emit(0x66);
  emit_optional_rex_32(dst, src);
  emit(0x0F);
  emit(0xC2);
  emit_sse_operand(dst, src, 1);
  emit(cmp);
}

void Assembler::cvttss2si(Register dst, const Operand &src) {
  DCHECK(Enabled(MMX));
  EnsureSpace ensure_space(this);
  emit(0xF3);
  emit_optional_rex_32(dst, src);
  emit(0x0F);
  emit(0x2C);
  emit_operand(dst, src);
}

void Assembler::cvttss2si(Register dst, XMMRegister src) {
  DCHECK(Enabled(MMX));
  EnsureSpace ensure_space(this);
  emit(0xF3);
  emit_optional_rex_32(dst, src);
  emit(0x0F);
  emit(0x2C);
  emit_sse_operand(dst, src);
}

void Assembler::cvttsd2si(Register dst, const Operand &src) {
  DCHECK(Enabled(MMX));
  EnsureSpace ensure_space(this);
  emit(0xF2);
  emit_optional_rex_32(dst, src);
  emit(0x0F);
  emit(0x2C);
  emit_operand(dst, src);
}

void Assembler::cvttsd2si(Register dst, XMMRegister src) {
  DCHECK(Enabled(MMX));
  EnsureSpace ensure_space(this);
  emit(0xF2);
  emit_optional_rex_32(dst, src);
  emit(0x0F);
  emit(0x2C);
  emit_sse_operand(dst, src);
}

void Assembler::cvttss2siq(Register dst, XMMRegister src) {
  DCHECK(Enabled(MMX));
  EnsureSpace ensure_space(this);
  emit(0xF3);
  emit_rex_64(dst, src);
  emit(0x0F);
  emit(0x2C);
  emit_sse_operand(dst, src);
}

void Assembler::cvttss2siq(Register dst, const Operand &src) {
  DCHECK(Enabled(MMX));
  EnsureSpace ensure_space(this);
  emit(0xF3);
  emit_rex_64(dst, src);
  emit(0x0F);
  emit(0x2C);
  emit_sse_operand(dst, src);
}

void Assembler::cvttsd2siq(Register dst, XMMRegister src) {
  DCHECK(Enabled(MMX));
  EnsureSpace ensure_space(this);
  emit(0xF2);
  emit_rex_64(dst, src);
  emit(0x0F);
  emit(0x2C);
  emit_sse_operand(dst, src);
}

void Assembler::cvttsd2siq(Register dst, const Operand &src) {
  DCHECK(Enabled(MMX));
  EnsureSpace ensure_space(this);
  emit(0xF2);
  emit_rex_64(dst, src);
  emit(0x0F);
  emit(0x2C);
  emit_sse_operand(dst, src);
}

void Assembler::cvtlsi2sd(XMMRegister dst, const Operand &src) {
  DCHECK(Enabled(MMX));
  EnsureSpace ensure_space(this);
  emit(0xF2);
  emit_optional_rex_32(dst, src);
  emit(0x0F);
  emit(0x2A);
  emit_sse_operand(dst, src);
}

void Assembler::cvtlsi2sd(XMMRegister dst, Register src) {
  DCHECK(Enabled(MMX));
  EnsureSpace ensure_space(this);
  emit(0xF2);
  emit_optional_rex_32(dst, src);
  emit(0x0F);
  emit(0x2A);
  emit_sse_operand(dst, src);
}

void Assembler::cvtlsi2ss(XMMRegister dst, const Operand &src) {
  DCHECK(Enabled(MMX));
  EnsureSpace ensure_space(this);
  emit(0xF3);
  emit_optional_rex_32(dst, src);
  emit(0x0F);
  emit(0x2A);
  emit_sse_operand(dst, src);
}

void Assembler::cvtlsi2ss(XMMRegister dst, Register src) {
  EnsureSpace ensure_space(this);
  emit(0xF3);
  emit_optional_rex_32(dst, src);
  emit(0x0F);
  emit(0x2A);
  emit_sse_operand(dst, src);
}

void Assembler::cvtqsi2ss(XMMRegister dst, const Operand &src) {
  DCHECK(Enabled(MMX));
  EnsureSpace ensure_space(this);
  emit(0xF3);
  emit_rex_64(dst, src);
  emit(0x0F);
  emit(0x2A);
  emit_sse_operand(dst, src);
}

void Assembler::cvtqsi2ss(XMMRegister dst, Register src) {
  DCHECK(Enabled(MMX));
  EnsureSpace ensure_space(this);
  emit(0xF3);
  emit_rex_64(dst, src);
  emit(0x0F);
  emit(0x2A);
  emit_sse_operand(dst, src);
}

void Assembler::cvtqsi2sd(XMMRegister dst, const Operand &src) {
  DCHECK(Enabled(MMX));
  EnsureSpace ensure_space(this);
  emit(0xF2);
  emit_rex_64(dst, src);
  emit(0x0F);
  emit(0x2A);
  emit_sse_operand(dst, src);
}

void Assembler::cvtqsi2sd(XMMRegister dst, Register src) {
  DCHECK(Enabled(MMX));
  EnsureSpace ensure_space(this);
  emit(0xF2);
  emit_rex_64(dst, src);
  emit(0x0F);
  emit(0x2A);
  emit_sse_operand(dst, src);
}

void Assembler::cvtss2sd(XMMRegister dst, XMMRegister src) {
  DCHECK(Enabled(MMX));
  EnsureSpace ensure_space(this);
  emit(0xF3);
  emit_optional_rex_32(dst, src);
  emit(0x0F);
  emit(0x5A);
  emit_sse_operand(dst, src);
}

void Assembler::cvtss2sd(XMMRegister dst, const Operand &src) {
  DCHECK(Enabled(MMX));
  EnsureSpace ensure_space(this);
  emit(0xF3);
  emit_optional_rex_32(dst, src);
  emit(0x0F);
  emit(0x5A);
  emit_sse_operand(dst, src);
}

void Assembler::cvtsd2ss(XMMRegister dst, XMMRegister src) {
  DCHECK(Enabled(MMX));
  EnsureSpace ensure_space(this);
  emit(0xF2);
  emit_optional_rex_32(dst, src);
  emit(0x0F);
  emit(0x5A);
  emit_sse_operand(dst, src);
}

void Assembler::cvtsd2ss(XMMRegister dst, const Operand &src) {
  DCHECK(Enabled(MMX));
  EnsureSpace ensure_space(this);
  emit(0xF2);
  emit_optional_rex_32(dst, src);
  emit(0x0F);
  emit(0x5A);
  emit_sse_operand(dst, src);
}

void Assembler::cvtsd2si(Register dst, XMMRegister src) {
  DCHECK(Enabled(MMX));
  EnsureSpace ensure_space(this);
  emit(0xF2);
  emit_optional_rex_32(dst, src);
  emit(0x0F);
  emit(0x2D);
  emit_sse_operand(dst, src);
}

void Assembler::cvtsd2siq(Register dst, XMMRegister src) {
  DCHECK(Enabled(MMX));
  EnsureSpace ensure_space(this);
  emit(0xF2);
  emit_rex_64(dst, src);
  emit(0x0F);
  emit(0x2D);
  emit_sse_operand(dst, src);
}

void Assembler::cvttps2dq(XMMRegister dst, XMMRegister src) {
  DCHECK(Enabled(SSE2));
  EnsureSpace ensure_space(this);
  emit(0xF3);
  emit_optional_rex_32(dst, src);
  emit(0x0F);
  emit(0x5B);
  emit_sse_operand(dst, src);
}

void Assembler::cvttps2dq(XMMRegister dst, const Operand &src) {
  DCHECK(Enabled(SSE2));
  EnsureSpace ensure_space(this);
  emit(0xF3);
  emit_optional_rex_32(dst, src);
  emit(0x0F);
  emit(0x5B);
  emit_sse_operand(dst, src);
}

void Assembler::cvttpd2dq(XMMRegister dst, XMMRegister src) {
  DCHECK(Enabled(SSE2));
  EnsureSpace ensure_space(this);
  emit(0x66);
  emit_optional_rex_32(dst, src);
  emit(0x0F);
  emit(0xE6);
  emit_sse_operand(dst, src);
}

void Assembler::cvttpd2dq(XMMRegister dst, const Operand &src) {
  DCHECK(Enabled(SSE2));
  EnsureSpace ensure_space(this);
  emit(0x66);
  emit_optional_rex_32(dst, src);
  emit(0x0F);
  emit(0xE6);
  emit_sse_operand(dst, src);
}

void Assembler::addsd(XMMRegister dst, XMMRegister src) {
  EnsureSpace ensure_space(this);
  emit(0xF2);
  emit_optional_rex_32(dst, src);
  emit(0x0F);
  emit(0x58);
  emit_sse_operand(dst, src);
}

void Assembler::addsd(XMMRegister dst, const Operand &src) {
  EnsureSpace ensure_space(this);
  emit(0xF2);
  emit_optional_rex_32(dst, src);
  emit(0x0F);
  emit(0x58);
  emit_sse_operand(dst, src);
}

void Assembler::mulsd(XMMRegister dst, XMMRegister src) {
  EnsureSpace ensure_space(this);
  emit(0xF2);
  emit_optional_rex_32(dst, src);
  emit(0x0F);
  emit(0x59);
  emit_sse_operand(dst, src);
}

void Assembler::mulsd(XMMRegister dst, const Operand &src) {
  EnsureSpace ensure_space(this);
  emit(0xF2);
  emit_optional_rex_32(dst, src);
  emit(0x0F);
  emit(0x59);
  emit_sse_operand(dst, src);
}

void Assembler::subsd(XMMRegister dst, XMMRegister src) {
  EnsureSpace ensure_space(this);
  emit(0xF2);
  emit_optional_rex_32(dst, src);
  emit(0x0F);
  emit(0x5C);
  emit_sse_operand(dst, src);
}

void Assembler::subsd(XMMRegister dst, const Operand &src) {
  EnsureSpace ensure_space(this);
  emit(0xF2);
  emit_optional_rex_32(dst, src);
  emit(0x0F);
  emit(0x5C);
  emit_sse_operand(dst, src);
}

void Assembler::divsd(XMMRegister dst, XMMRegister src) {
  EnsureSpace ensure_space(this);
  emit(0xF2);
  emit_optional_rex_32(dst, src);
  emit(0x0F);
  emit(0x5E);
  emit_sse_operand(dst, src);
}

void Assembler::divsd(XMMRegister dst, const Operand &src) {
  EnsureSpace ensure_space(this);
  emit(0xF2);
  emit_optional_rex_32(dst, src);
  emit(0x0F);
  emit(0x5E);
  emit_sse_operand(dst, src);
}

void Assembler::cmpsd(XMMRegister dst, XMMRegister src, int8_t cmp) {
  EnsureSpace ensure_space(this);
  emit(0xF2);
  emit_optional_rex_32(dst, src);
  emit(0x0F);
  emit(0xC2);
  emit_sse_operand(dst, src);
  emit(cmp);
}

void Assembler::cmpsd(XMMRegister dst, const Operand &src, int8_t cmp) {
  EnsureSpace ensure_space(this);
  emit(0xF2);
  emit_optional_rex_32(dst, src);
  emit(0x0F);
  emit(0xC2);
  emit_sse_operand(dst, src, 1);
  emit(cmp);
}

void Assembler::maxsd(XMMRegister dst, XMMRegister src) {
  EnsureSpace ensure_space(this);
  emit(0xF2);
  emit_optional_rex_32(dst, src);
  emit(0x0F);
  emit(0x5F);
  emit_sse_operand(dst, src);
}

void Assembler::maxsd(XMMRegister dst, const Operand &src) {
  EnsureSpace ensure_space(this);
  emit(0xF2);
  emit_optional_rex_32(dst, src);
  emit(0x0F);
  emit(0x5F);
  emit_sse_operand(dst, src);
}

void Assembler::minsd(XMMRegister dst, XMMRegister src) {
  EnsureSpace ensure_space(this);
  emit(0xF2);
  emit_optional_rex_32(dst, src);
  emit(0x0F);
  emit(0x5D);
  emit_sse_operand(dst, src);
}

void Assembler::minsd(XMMRegister dst, const Operand &src) {
  EnsureSpace ensure_space(this);
  emit(0xF2);
  emit_optional_rex_32(dst, src);
  emit(0x0F);
  emit(0x5D);
  emit_sse_operand(dst, src);
}

void Assembler::andpd(XMMRegister dst, XMMRegister src) {
  EnsureSpace ensure_space(this);
  emit(0x66);
  emit_optional_rex_32(dst, src);
  emit(0x0F);
  emit(0x54);
  emit_sse_operand(dst, src);
}

void Assembler::andpd(XMMRegister dst, const Operand &src) {
  EnsureSpace ensure_space(this);
  emit(0x66);
  emit_optional_rex_32(dst, src);
  emit(0x0F);
  emit(0x54);
  emit_sse_operand(dst, src);
}

void Assembler::orpd(XMMRegister dst, XMMRegister src) {
  EnsureSpace ensure_space(this);
  emit(0x66);
  emit_optional_rex_32(dst, src);
  emit(0x0F);
  emit(0x56);
  emit_sse_operand(dst, src);
}

void Assembler::orpd(XMMRegister dst, const Operand &src) {
  EnsureSpace ensure_space(this);
  emit(0x66);
  emit_optional_rex_32(dst, src);
  emit(0x0F);
  emit(0x56);
  emit_sse_operand(dst, src);
}

void Assembler::xorpd(XMMRegister dst, XMMRegister src) {
  DCHECK(Enabled(MMX));
  EnsureSpace ensure_space(this);
  emit(0x66);
  emit_optional_rex_32(dst, src);
  emit(0x0F);
  emit(0x57);
  emit_sse_operand(dst, src);
}

void Assembler::xorpd(XMMRegister dst, const Operand &src) {
  DCHECK(Enabled(MMX));
  EnsureSpace ensure_space(this);
  emit(0x66);
  emit_optional_rex_32(dst, src);
  emit(0x0F);
  emit(0x57);
  emit_sse_operand(dst, src);
}

void Assembler::andnpd(XMMRegister dst, XMMRegister src) {
  EnsureSpace ensure_space(this);
  emit(0x66);
  emit_optional_rex_32(dst, src);
  emit(0x0F);
  emit(0x55);
  emit_sse_operand(dst, src);
}

void Assembler::andnpd(XMMRegister dst, const Operand &src) {
  EnsureSpace ensure_space(this);
  emit(0x66);
  emit_optional_rex_32(dst, src);
  emit(0x0F);
  emit(0x55);
  emit_sse_operand(dst, src);
}

void Assembler::sqrtsd(XMMRegister dst, XMMRegister src) {
  DCHECK(Enabled(MMX));
  EnsureSpace ensure_space(this);
  emit(0xF2);
  emit_optional_rex_32(dst, src);
  emit(0x0F);
  emit(0x51);
  emit_sse_operand(dst, src);
}

void Assembler::sqrtsd(XMMRegister dst, const Operand &src) {
  DCHECK(Enabled(MMX));
  EnsureSpace ensure_space(this);
  emit(0xF2);
  emit_optional_rex_32(dst, src);
  emit(0x0F);
  emit(0x51);
  emit_sse_operand(dst, src);
}

void Assembler::ucomisd(XMMRegister dst, XMMRegister src) {
  DCHECK(Enabled(MMX));
  EnsureSpace ensure_space(this);
  emit(0x66);
  emit_optional_rex_32(dst, src);
  emit(0x0f);
  emit(0x2e);
  emit_sse_operand(dst, src);
}

void Assembler::ucomisd(XMMRegister dst, const Operand &src) {
  DCHECK(Enabled(MMX));
  EnsureSpace ensure_space(this);
  emit(0x66);
  emit_optional_rex_32(dst, src);
  emit(0x0f);
  emit(0x2e);
  emit_sse_operand(dst, src);
}

void Assembler::cmpltsd(XMMRegister dst, XMMRegister src) {
  EnsureSpace ensure_space(this);
  emit(0xF2);
  emit_optional_rex_32(dst, src);
  emit(0x0F);
  emit(0xC2);
  emit_sse_operand(dst, src);
  emit(0x01);  // LT == 1
}

void Assembler::roundss(XMMRegister dst, XMMRegister src, int8_t mode) {
  DCHECK(Enabled(SSE4_1));
  EnsureSpace ensure_space(this);
  emit(0x66);
  emit_optional_rex_32(dst, src);
  emit(0x0f);
  emit(0x3a);
  emit(0x0a);
  emit_sse_operand(dst, src);
  // Mask precision exception.
  emit(mode | 0x8);
}

void Assembler::roundss(XMMRegister dst, const Operand &src, int8_t mode) {
  DCHECK(Enabled(SSE4_1));
  EnsureSpace ensure_space(this);
  emit(0x66);
  emit_optional_rex_32(dst, src);
  emit(0x0f);
  emit(0x3a);
  emit(0x0a);
  emit_sse_operand(dst, src, 1);
  // Mask precision exception.
  emit(mode | 0x8);
}

void Assembler::roundsd(XMMRegister dst, XMMRegister src, int8_t mode) {
  DCHECK(Enabled(SSE4_1));
  EnsureSpace ensure_space(this);
  emit(0x66);
  emit_optional_rex_32(dst, src);
  emit(0x0f);
  emit(0x3a);
  emit(0x0b);
  emit_sse_operand(dst, src);
  // Mask precision exception.
  emit(mode | 0x8);
}

void Assembler::roundsd(XMMRegister dst, const Operand &src, int8_t mode) {
  DCHECK(Enabled(SSE4_1));
  EnsureSpace ensure_space(this);
  emit(0x66);
  emit_optional_rex_32(dst, src);
  emit(0x0f);
  emit(0x3a);
  emit(0x0b);
  emit_sse_operand(dst, src, 1);
  // Mask precision exception.
  emit(mode | 0x8);
}

void Assembler::roundps(XMMRegister dst, XMMRegister src, int8_t mode) {
  DCHECK(Enabled(SSE4_1));
  EnsureSpace ensure_space(this);
  emit(0x66);
  emit_optional_rex_32(dst, src);
  emit(0x0f);
  emit(0x3a);
  emit(0x08);
  emit_sse_operand(dst, src);
  // Mask precision exception.
  emit(mode | 0x8);
}

void Assembler::roundps(XMMRegister dst, const Operand &src, int8_t mode) {
  DCHECK(Enabled(SSE4_1));
  EnsureSpace ensure_space(this);
  emit(0x66);
  emit_optional_rex_32(dst, src);
  emit(0x0f);
  emit(0x3a);
  emit(0x08);
  emit_sse_operand(dst, src, 1);
  // Mask precision exception.
  emit(mode | 0x8);
}

void Assembler::roundpd(XMMRegister dst, XMMRegister src, int8_t mode) {
  DCHECK(Enabled(SSE4_1));
  EnsureSpace ensure_space(this);
  emit(0x66);
  emit_optional_rex_32(dst, src);
  emit(0x0f);
  emit(0x3a);
  emit(0x09);
  emit_sse_operand(dst, src);
  // Mask precision exception.
  emit(mode | 0x8);
}

void Assembler::roundpd(XMMRegister dst, const Operand &src, int8_t mode) {
  DCHECK(Enabled(SSE4_1));
  EnsureSpace ensure_space(this);
  emit(0x66);
  emit_optional_rex_32(dst, src);
  emit(0x0f);
  emit(0x3a);
  emit(0x09);
  emit_sse_operand(dst, src, 1);
  // Mask precision exception.
  emit(mode | 0x8);
}

void Assembler::blendps(XMMRegister dst, XMMRegister src, int8_t mask) {
  DCHECK(Enabled(SSE4_1));
  EnsureSpace ensure_space(this);
  emit(0x66);
  emit_optional_rex_32(dst, src);
  emit(0x0f);
  emit(0x3a);
  emit(0x0c);
  emit_sse_operand(dst, src);
  emit(mask);
}

void Assembler::blendps(XMMRegister dst, const Operand &src, int8_t mask) {
  DCHECK(Enabled(SSE4_1));
  EnsureSpace ensure_space(this);
  emit(0x66);
  emit_optional_rex_32(dst, src);
  emit(0x0f);
  emit(0x3a);
  emit(0x0c);
  emit_sse_operand(dst, src, 1);
  emit(mask);
}

void Assembler::blendpd(XMMRegister dst, XMMRegister src, int8_t mask) {
  DCHECK(Enabled(SSE4_1));
  EnsureSpace ensure_space(this);
  emit(0x66);
  emit_optional_rex_32(dst, src);
  emit(0x0f);
  emit(0x3a);
  emit(0x0d);
  emit_sse_operand(dst, src);
  emit(mask);
}

void Assembler::blendpd(XMMRegister dst, const Operand &src, int8_t mask) {
  DCHECK(Enabled(SSE4_1));
  EnsureSpace ensure_space(this);
  emit(0x66);
  emit_optional_rex_32(dst, src);
  emit(0x0f);
  emit(0x3a);
  emit(0x0d);
  emit_sse_operand(dst, src, 1);
  emit(mask);
}

void Assembler::movmskpd(Register dst, XMMRegister src) {
  EnsureSpace ensure_space(this);
  emit(0x66);
  emit_optional_rex_32(dst, src);
  emit(0x0f);
  emit(0x50);
  emit_sse_operand(dst, src);
}

void Assembler::movmskps(Register dst, XMMRegister src) {
  EnsureSpace ensure_space(this);
  emit_optional_rex_32(dst, src);
  emit(0x0f);
  emit(0x50);
  emit_sse_operand(dst, src);
}

void Assembler::movhlps(XMMRegister dst, XMMRegister src) {
  EnsureSpace ensure_space(this);
  emit_optional_rex_32(dst, src);
  emit(0x0f);
  emit(0x12);
  emit_sse_operand(dst, src);
}

void Assembler::punpckldq(XMMRegister dst, XMMRegister src) {
  EnsureSpace ensure_space(this);
  emit(0x66);
  emit_optional_rex_32(dst, src);
  emit(0x0F);
  emit(0x62);
  emit_sse_operand(dst, src);
}

void Assembler::punpckldq(XMMRegister dst, const Operand &src) {
  EnsureSpace ensure_space(this);
  emit(0x66);
  emit_optional_rex_32(dst, src);
  emit(0x0F);
  emit(0x62);
  emit_sse_operand(dst, src);
}

void Assembler::punpckhdq(XMMRegister dst, XMMRegister src) {
  EnsureSpace ensure_space(this);
  emit(0x66);
  emit_optional_rex_32(dst, src);
  emit(0x0F);
  emit(0x6A);
  emit_sse_operand(dst, src);
}

void Assembler::vmovd(XMMRegister dst, Register src) {
  DCHECK(Enabled(AVX));
  EnsureSpace ensure_space(this);
  XMMRegister isrc = {src.code()};
  emit_vex_prefix(dst, xmm0, isrc, kL128, k66, k0F, kW0);
  emit(0x6e);
  emit_sse_operand(dst, src);
}

void Assembler::vmovd(XMMRegister dst, const Operand &src) {
  DCHECK(Enabled(AVX));
  EnsureSpace ensure_space(this);
  emit_vex_prefix(dst, xmm0, src, kL128, k66, k0F, kW0);
  emit(0x6e);
  emit_sse_operand(dst, src);
}

void Assembler::vmovd(Register dst, XMMRegister src) {
  DCHECK(Enabled(AVX));
  EnsureSpace ensure_space(this);
  XMMRegister idst = {dst.code()};
  emit_vex_prefix(src, xmm0, idst, kL128, k66, k0F, kW0);
  emit(0x7e);
  emit_sse_operand(src, dst);
}

void Assembler::vmovd(const Operand &dst, XMMRegister src) {
  DCHECK(Enabled(AVX));
  EnsureSpace ensure_space(this);
  emit_vex_prefix(src, xmm0, dst, kL128, k66, k0F, kW0);
  emit(0x7e);
  emit_sse_operand(src, dst);
}

void Assembler::vmovq(XMMRegister dst, Register src) {
  DCHECK(Enabled(AVX));
  EnsureSpace ensure_space(this);
  XMMRegister isrc = {src.code()};
  emit_vex_prefix(dst, xmm0, isrc, kL128, k66, k0F, kW1);
  emit(0x6e);
  emit_sse_operand(dst, src);
}

void Assembler::vmovq(XMMRegister dst, const Operand &src) {
  DCHECK(Enabled(AVX));
  EnsureSpace ensure_space(this);
  emit_vex_prefix(dst, xmm0, src, kL128, k66, k0F, kW1);
  emit(0x6e);
  emit_sse_operand(dst, src);
}

void Assembler::vmovq(Register dst, XMMRegister src) {
  DCHECK(Enabled(AVX));
  EnsureSpace ensure_space(this);
  XMMRegister idst = {dst.code()};
  emit_vex_prefix(src, xmm0, idst, kL128, k66, k0F, kW1);
  emit(0x7e);
  emit_sse_operand(src, dst);
}

void Assembler::vmovq(const Operand &dst, XMMRegister src) {
  DCHECK(Enabled(AVX));
  EnsureSpace ensure_space(this);
  emit_vex_prefix(src, xmm0, dst, kL128, k66, k0F, kW1);
  emit(0x7e);
  emit_sse_operand(src, dst);
}

void Assembler::vzeroall() {
  DCHECK(Enabled(AVX));
  EnsureSpace ensure_space(this);
  emit_vex_prefix(xmm0, xmm0, xmm0, kL256, kNone, k0F, kWIG);
  emit(0x77);
}

void Assembler::vzeroupper() {
  DCHECK(Enabled(AVX));
  EnsureSpace ensure_space(this);
  emit_vex_prefix(xmm0, xmm0, xmm0, kL128, kNone, k0F, kWIG);
  emit(0x77);
}

void Assembler::vucomiss(XMMRegister dst, XMMRegister src) {
  DCHECK(Enabled(AVX));
  EnsureSpace ensure_space(this);
  emit_vex_prefix(dst, xmm0, src, kLIG, kNone, k0F, kWIG);
  emit(0x2e);
  emit_sse_operand(dst, src);
}

void Assembler::vucomiss(XMMRegister dst, const Operand &src) {
  DCHECK(Enabled(AVX));
  EnsureSpace ensure_space(this);
  emit_vex_prefix(dst, xmm0, src, kLIG, kNone, k0F, kWIG);
  emit(0x2e);
  emit_sse_operand(dst, src);
}

void Assembler::bmi1q(byte op, Register reg, Register vreg, Register rm) {
  DCHECK(Enabled(BMI1));
  EnsureSpace ensure_space(this);
  emit_vex_prefix(reg, vreg, rm, kLZ, kNone, k0F38, kW1);
  emit(op);
  emit_modrm(reg, rm);
}

void Assembler::bmi1q(byte op, Register reg, Register vreg, const Operand &rm) {
  DCHECK(Enabled(BMI1));
  EnsureSpace ensure_space(this);
  emit_vex_prefix(reg, vreg, rm, kLZ, kNone, k0F38, kW1);
  emit(op);
  emit_operand(reg, rm);
}

void Assembler::bmi1l(byte op, Register reg, Register vreg, Register rm) {
  DCHECK(Enabled(BMI1));
  EnsureSpace ensure_space(this);
  emit_vex_prefix(reg, vreg, rm, kLZ, kNone, k0F38, kW0);
  emit(op);
  emit_modrm(reg, rm);
}

void Assembler::bmi1l(byte op, Register reg, Register vreg, const Operand &rm) {
  DCHECK(Enabled(BMI1));
  EnsureSpace ensure_space(this);
  emit_vex_prefix(reg, vreg, rm, kLZ, kNone, k0F38, kW0);
  emit(op);
  emit_operand(reg, rm);
}

void Assembler::tzcntq(Register dst, Register src) {
  DCHECK(Enabled(BMI1));
  EnsureSpace ensure_space(this);
  emit(0xF3);
  emit_rex_64(dst, src);
  emit(0x0F);
  emit(0xBC);
  emit_modrm(dst, src);
}

void Assembler::tzcntq(Register dst, const Operand &src) {
  DCHECK(Enabled(BMI1));
  EnsureSpace ensure_space(this);
  emit(0xF3);
  emit_rex_64(dst, src);
  emit(0x0F);
  emit(0xBC);
  emit_operand(dst, src);
}

void Assembler::tzcntl(Register dst, Register src) {
  DCHECK(Enabled(BMI1));
  EnsureSpace ensure_space(this);
  emit(0xF3);
  emit_optional_rex_32(dst, src);
  emit(0x0F);
  emit(0xBC);
  emit_modrm(dst, src);
}

void Assembler::tzcntl(Register dst, const Operand &src) {
  DCHECK(Enabled(BMI1));
  EnsureSpace ensure_space(this);
  emit(0xF3);
  emit_optional_rex_32(dst, src);
  emit(0x0F);
  emit(0xBC);
  emit_operand(dst, src);
}

void Assembler::lzcntq(Register dst, Register src) {
  DCHECK(Enabled(LZCNT));
  EnsureSpace ensure_space(this);
  emit(0xF3);
  emit_rex_64(dst, src);
  emit(0x0F);
  emit(0xBD);
  emit_modrm(dst, src);
}

void Assembler::lzcntq(Register dst, const Operand &src) {
  DCHECK(Enabled(LZCNT));
  EnsureSpace ensure_space(this);
  emit(0xF3);
  emit_rex_64(dst, src);
  emit(0x0F);
  emit(0xBD);
  emit_operand(dst, src);
}

void Assembler::lzcntl(Register dst, Register src) {
  DCHECK(Enabled(LZCNT));
  EnsureSpace ensure_space(this);
  emit(0xF3);
  emit_optional_rex_32(dst, src);
  emit(0x0F);
  emit(0xBD);
  emit_modrm(dst, src);
}

void Assembler::lzcntl(Register dst, const Operand &src) {
  DCHECK(Enabled(LZCNT));
  EnsureSpace ensure_space(this);
  emit(0xF3);
  emit_optional_rex_32(dst, src);
  emit(0x0F);
  emit(0xBD);
  emit_operand(dst, src);
}

void Assembler::popcntq(Register dst, Register src) {
  DCHECK(Enabled(POPCNT));
  EnsureSpace ensure_space(this);
  emit(0xF3);
  emit_rex_64(dst, src);
  emit(0x0F);
  emit(0xB8);
  emit_modrm(dst, src);
}

void Assembler::popcntq(Register dst, const Operand &src) {
  DCHECK(Enabled(POPCNT));
  EnsureSpace ensure_space(this);
  emit(0xF3);
  emit_rex_64(dst, src);
  emit(0x0F);
  emit(0xB8);
  emit_operand(dst, src);
}

void Assembler::popcntl(Register dst, Register src) {
  DCHECK(Enabled(POPCNT));
  EnsureSpace ensure_space(this);
  emit(0xF3);
  emit_optional_rex_32(dst, src);
  emit(0x0F);
  emit(0xB8);
  emit_modrm(dst, src);
}

void Assembler::popcntl(Register dst, const Operand &src) {
  DCHECK(Enabled(POPCNT));
  EnsureSpace ensure_space(this);
  emit(0xF3);
  emit_optional_rex_32(dst, src);
  emit(0x0F);
  emit(0xB8);
  emit_operand(dst, src);
}

void Assembler::bmi2q(SIMDPrefix pp, byte op, Register reg, Register vreg,
                      Register rm) {
  DCHECK(Enabled(BMI2));
  EnsureSpace ensure_space(this);
  emit_vex_prefix(reg, vreg, rm, kLZ, pp, k0F38, kW1);
  emit(op);
  emit_modrm(reg, rm);
}

void Assembler::bmi2q(SIMDPrefix pp, byte op, Register reg, Register vreg,
                      const Operand &rm) {
  DCHECK(Enabled(BMI2));
  EnsureSpace ensure_space(this);
  emit_vex_prefix(reg, vreg, rm, kLZ, pp, k0F38, kW1);
  emit(op);
  emit_operand(reg, rm);
}

void Assembler::bmi2l(SIMDPrefix pp, byte op, Register reg, Register vreg,
                      Register rm) {
  DCHECK(Enabled(BMI2));
  EnsureSpace ensure_space(this);
  emit_vex_prefix(reg, vreg, rm, kLZ, pp, k0F38, kW0);
  emit(op);
  emit_modrm(reg, rm);
}

void Assembler::bmi2l(SIMDPrefix pp, byte op, Register reg, Register vreg,
                      const Operand &rm) {
  DCHECK(Enabled(BMI2));
  EnsureSpace ensure_space(this);
  emit_vex_prefix(reg, vreg, rm, kLZ, pp, k0F38, kW0);
  emit(op);
  emit_operand(reg, rm);
}

void Assembler::rorxq(Register dst, Register src, byte imm8) {
  DCHECK(Enabled(BMI2));
  DCHECK(is_uint8(imm8));
  Register vreg = {0};  // VEX.vvvv unused
  EnsureSpace ensure_space(this);
  emit_vex_prefix(dst, vreg, src, kLZ, kF2, k0F3A, kW1);
  emit(0xF0);
  emit_modrm(dst, src);
  emit(imm8);
}

void Assembler::rorxq(Register dst, const Operand &src, byte imm8) {
  DCHECK(Enabled(BMI2));
  DCHECK(is_uint8(imm8));
  Register vreg = {0};  // VEX.vvvv unused
  EnsureSpace ensure_space(this);
  emit_vex_prefix(dst, vreg, src, kLZ, kF2, k0F3A, kW1);
  emit(0xF0);
  emit_operand(dst, src, 1);
  emit(imm8);
}

void Assembler::rorxl(Register dst, Register src, byte imm8) {
  DCHECK(Enabled(BMI2));
  DCHECK(is_uint8(imm8));
  Register vreg = {0};  // VEX.vvvv unused
  EnsureSpace ensure_space(this);
  emit_vex_prefix(dst, vreg, src, kLZ, kF2, k0F3A, kW0);
  emit(0xF0);
  emit_modrm(dst, src);
  emit(imm8);
}

void Assembler::rorxl(Register dst, const Operand &src, byte imm8) {
  DCHECK(Enabled(BMI2));
  DCHECK(is_uint8(imm8));
  Register vreg = {0};  // VEX.vvvv unused
  EnsureSpace ensure_space(this);
  emit_vex_prefix(dst, vreg, src, kLZ, kF2, k0F3A, kW0);
  emit(0xF0);
  emit_operand(dst, src, 1);
  emit(imm8);
}

void Assembler::rcpss(XMMRegister dst, XMMRegister src) {
  EnsureSpace ensure_space(this);
  emit(0xF3);
  emit_optional_rex_32(dst, src);
  emit(0x0F);
  emit(0x53);
  emit_sse_operand(dst, src);
}

void Assembler::rcpss(XMMRegister dst, const Operand &src) {
  EnsureSpace ensure_space(this);
  emit(0xF3);
  emit_optional_rex_32(dst, src);
  emit(0x0F);
  emit(0x53);
  emit_sse_operand(dst, src);
}

void Assembler::rcpps(XMMRegister dst, XMMRegister src) {
  EnsureSpace ensure_space(this);
  emit_optional_rex_32(dst, src);
  emit(0x0F);
  emit(0x53);
  emit_sse_operand(dst, src);
}

void Assembler::rcpps(XMMRegister dst, const Operand &src) {
  EnsureSpace ensure_space(this);
  emit_optional_rex_32(dst, src);
  emit(0x0F);
  emit(0x53);
  emit_sse_operand(dst, src);
}

void Assembler::sqrtps(XMMRegister dst, XMMRegister src) {
  EnsureSpace ensure_space(this);
  emit_optional_rex_32(dst, src);
  emit(0x0F);
  emit(0x51);
  emit_sse_operand(dst, src);
}

void Assembler::sqrtps(XMMRegister dst, const Operand &src) {
  EnsureSpace ensure_space(this);
  emit_optional_rex_32(dst, src);
  emit(0x0F);
  emit(0x51);
  emit_sse_operand(dst, src);
}

void Assembler::sqrtpd(XMMRegister dst, XMMRegister src) {
  EnsureSpace ensure_space(this);
  emit(0x66);
  emit_optional_rex_32(dst, src);
  emit(0x0F);
  emit(0x51);
  emit_sse_operand(dst, src);
}

void Assembler::sqrtpd(XMMRegister dst, const Operand &src) {
  EnsureSpace ensure_space(this);
  emit(0x66);
  emit_optional_rex_32(dst, src);
  emit(0x0F);
  emit(0x51);
  emit_sse_operand(dst, src);
}

void Assembler::rsqrtps(XMMRegister dst, XMMRegister src) {
  EnsureSpace ensure_space(this);
  emit_optional_rex_32(dst, src);
  emit(0x0F);
  emit(0x52);
  emit_sse_operand(dst, src);
}

void Assembler::rsqrtps(XMMRegister dst, const Operand &src) {
  EnsureSpace ensure_space(this);
  emit_optional_rex_32(dst, src);
  emit(0x0F);
  emit(0x52);
  emit_sse_operand(dst, src);
}

void Assembler::cvtdq2ps(XMMRegister dst, XMMRegister src) {
  EnsureSpace ensure_space(this);
  emit_optional_rex_32(dst, src);
  emit(0x0F);
  emit(0x5B);
  emit_sse_operand(dst, src);
}

void Assembler::cvtdq2ps(XMMRegister dst, const Operand &src) {
  EnsureSpace ensure_space(this);
  emit_optional_rex_32(dst, src);
  emit(0x0F);
  emit(0x5B);
  emit_sse_operand(dst, src);
}

void Assembler::cvtdq2pd(XMMRegister dst, XMMRegister src) {
  EnsureSpace ensure_space(this);
  emit(0xF3);
  emit_optional_rex_32(dst, src);
  emit(0x0F);
  emit(0xE6);
  emit_sse_operand(dst, src);
}

void Assembler::cvtdq2pd(XMMRegister dst, const Operand &src) {
  EnsureSpace ensure_space(this);
  emit(0xF3);
  emit_optional_rex_32(dst, src);
  emit(0x0F);
  emit(0xE6);
  emit_sse_operand(dst, src);
}

void Assembler::movups(XMMRegister dst, XMMRegister src) {
  EnsureSpace ensure_space(this);
  if (src.low_bits() == 4) {
    // Try to avoid an unnecessary SIB byte.
    emit_optional_rex_32(src, dst);
    emit(0x0F);
    emit(0x11);
    emit_sse_operand(src, dst);
  } else {
    emit_optional_rex_32(dst, src);
    emit(0x0F);
    emit(0x10);
    emit_sse_operand(dst, src);
  }
}

void Assembler::movups(XMMRegister dst, const Operand &src) {
  EnsureSpace ensure_space(this);
  emit_optional_rex_32(dst, src);
  emit(0x0F);
  emit(0x10);
  emit_sse_operand(dst, src);
}

void Assembler::movups(const Operand &dst, XMMRegister src) {
  EnsureSpace ensure_space(this);
  emit_optional_rex_32(src, dst);
  emit(0x0F);
  emit(0x11);
  emit_sse_operand(src, dst);
}

void Assembler::haddps(XMMRegister dst, XMMRegister src) {
  DCHECK(Enabled(SSE3));
  EnsureSpace ensure_space(this);
  emit(0xF2);
  emit_optional_rex_32(dst, src);
  emit(0x0F);
  emit(0x7C);
  emit_sse_operand(dst, src);
}

void Assembler::haddps(XMMRegister dst, const Operand &src) {
  DCHECK(Enabled(SSE3));
  EnsureSpace ensure_space(this);
  emit(0xF2);
  emit_optional_rex_32(dst, src);
  emit(0x0F);
  emit(0x7C);
  emit_sse_operand(dst, src);
}

void Assembler::movshdup(XMMRegister dst, XMMRegister src) {
  DCHECK(Enabled(SSE3));
  EnsureSpace ensure_space(this);
  emit(0xF3);
  emit_optional_rex_32(dst, src);
  emit(0x0F);
  emit(0x16);
  emit_sse_operand(dst, src);
}

void Assembler::movshdup(XMMRegister dst, const Operand &src) {
  DCHECK(Enabled(SSE3));
  EnsureSpace ensure_space(this);
  emit(0xF3);
  emit_optional_rex_32(dst, src);
  emit(0x0F);
  emit(0x16);
  emit_sse_operand(dst, src);
}

void Assembler::lddqu(XMMRegister dst, const Operand &src) {
  DCHECK(Enabled(SSE3));
  EnsureSpace ensure_space(this);
  emit(0xF2);
  emit_optional_rex_32(dst, src);
  emit(0x0F);
  emit(0xF0);
  emit_sse_operand(dst, src);
}

void Assembler::psrldq(XMMRegister dst, uint8_t shift) {
  EnsureSpace ensure_space(this);
  emit(0x66);
  emit_optional_rex_32(dst);
  emit(0x0F);
  emit(0x73);
  emit_sse_operand(dst);
  emit(shift);
}

void Assembler::pshufd(XMMRegister dst, XMMRegister src, uint8_t shuffle) {
  EnsureSpace ensure_space(this);
  emit(0x66);
  emit_optional_rex_32(dst, src);
  emit(0x0F);
  emit(0x70);
  emit_sse_operand(dst, src);
  emit(shuffle);
}

void Assembler::pshufd(XMMRegister dst, const Operand &src, uint8_t shuffle) {
  EnsureSpace ensure_space(this);
  emit(0x66);
  emit_optional_rex_32(dst, src);
  emit(0x0F);
  emit(0x70);
  emit_sse_operand(dst, src);
  emit(shuffle);
}

void Assembler::pshuflw(XMMRegister dst, XMMRegister src, uint8_t shuffle) {
  EnsureSpace ensure_space(this);
  emit_optional_rex_32(dst, src);
  emit(0xF2);
  emit(0x0F);
  emit(0x70);
  emit_sse_operand(dst, src);
  emit(shuffle);
}

void Assembler::pshuflw(XMMRegister dst, const Operand &src, uint8_t shuffle) {
  EnsureSpace ensure_space(this);
  emit_optional_rex_32(dst, src);
  emit(0xF2);
  emit(0x0F);
  emit(0x70);
  emit_sse_operand(dst, src);
  emit(shuffle);
}

// EVEX prefix format:
//      7  6  5  4  3  2  0  1
// P0:  R  X  B  R' 0  0  m  m
// P1:  W  v  v  v  v  1  p  p
// P2:  z  L' L  b  V' a  a  a
void Assembler::emit_evex_prefix(ZMMRegister reg, ZMMRegister vreg,
                                 ZMMRegister rm, Mask mask, int flags) {
  CHECK(Enabled(AVX512F));
  byte p0 = 0;
  byte p1 = 0x04;
  byte p2 = 0;

  // Leading opcode (mm).
  if (flags & EVEX_M0F) p0 |= 0x01;
  if (flags & EVEX_M0F38) p0 |= 0x02;
  if (flags & EVEX_M0F3A) p0 |= 0x03;

  // Prefix (pp).
  if (flags & EVEX_P66) p1 |= 0x01;
  if (flags & EVEX_PF3) p1 |= 0x02;
  if (flags & EVEX_PF2) p1 |= 0x03;

  // Operation size (L' and L) and width (W).
  if (!(flags & EVEX_LIG) && !(flags & EVEX_ER)) p2 |= reg.size_bits() << 5;
  if (flags & EVEX_W1) p1 |= 0x80;

  // Register specifiers.
  int rxbr = ((reg.mid_bit() << 3) | (rm.high_bits() << 1) | reg.high_bit());
  p0 |= (~rxbr & 0x0F) << 4;               // RXBR'
  p1 |= (~vreg.code() & 0x0F) << 3;        // vvvv
  p2 |= (~vreg.high_bit() & 0x01) << 3;    // V'

  // Masking (z and aaa).
  p2 |= (mask.op() << 7) | mask.reg().code();

  // Rounding and exception supression (b).
  if (flags & EVEX_ER) {
    // Static rounding; set rounding mode in LL.
    p2 |= 0x10;
    if (flags & EVEX_R0) p2 |= 0x20;
    if (flags & EVEX_R1) p2 |= 0x40;
  } else if (flags & EVEX_SAE) {
    // Suppress all exceptions.
    p2 |= 0x10;
  }

  // Emit four-byte EVEX prefix.
  emit(0x62);
  emit(p0);
  emit(p1);
  emit(p2);
}

void Assembler::emit_evex_prefix(ZMMRegister reg, ZMMRegister vreg,
                                 const Operand &rm, Mask mask, int flags) {
  CHECK(Enabled(AVX512F));
  byte p0 = 0;
  byte p1 = 0x04;
  byte p2 = 0;

  // Leading opcode (mm).
  if (flags & EVEX_M0F) p0 |= 0x01;
  if (flags & EVEX_M0F38) p0 |= 0x02;
  if (flags & EVEX_M0F3A) p0 |= 0x03;

  // Prefix (pp).
  if (flags & EVEX_P66) p1 |= 0x01;
  if (flags & EVEX_PF3) p1 |= 0x02;
  if (flags & EVEX_PF2) p1 |= 0x03;

  // Operation size (L' and L) and width (W).
  if (!(flags & EVEX_LIG) && !(flags & EVEX_ER)) p2 |= reg.size_bits() << 5;
  if (flags & EVEX_W1) p1 |= 0x80;

  // Register specifiers.
  int rxbr = ((reg.mid_bit() << 3) | (rm.rex_ << 1) | reg.high_bit());
  p0 |= (~rxbr & 0x0F) << 4;               // RXBR'
  p1 |= (~vreg.code() & 0x0F) << 3;        // vvvv
  p2 |= (~vreg.high_bit() & 0x01) << 3;    // V'

  // Masking (z and aaa).
  p2 |= (mask.op() << 7) | mask.reg().code();

  // Broadcasting, rounding and exception supression (b).
  if (flags & EVEX_BCST) {
    // Broadcast memory source operand.
    if (rm.load() == broadcast) p2 |= 0x10;
  } else if (flags & EVEX_ER) {
    // Static rounding; set rounding mode in LL.
    p2 |= 0x10;
    if (flags & EVEX_R0) p2 |= 0x20;
    if (flags & EVEX_R1) p2 |= 0x40;
  } else if (flags & EVEX_SAE) {
    // Suppress all exceptions.
    p2 |= 0x10;
  }

  // Emit four-byte EVEX prefix.
  emit(0x62);
  emit(p0);
  emit(p1);
  emit(p2);
}

void Assembler::emit_operand(int code, const Operand &adr, int sl, int ts) {
  DCHECK(is_uint3(code));
  unsigned length = adr.len_;
  DCHECK(length > 0);
  uint8_t modrm = adr.buf_[0];

  if (modrm == 5) {
    // Emit RIP relative addressing.
    DCHECK_EQ(9u, length);
    CHECK(sl == 0 || sl == 1);
    emit(modrm | code << 3);
    Label *label = *bit_cast<Label *const *>(&adr.buf_[1]);
    if (label->is_bound()) {
      int offset = label->pos() - (pc_offset() + sl) - sizeof(int32_t);
      DCHECK_GE(0, offset);
      emitl(offset);
    } else if (label->is_linked()) {
      emitl(sl == 1 ? -label->pos() : label->pos());
      label->link_to(pc_offset() - sizeof(int32_t));
    } else {
      DCHECK(label->is_unused());
      int32_t current = pc_offset();
      emitl(sl == 1 ? -current : current);
      label->link_to(current);
    }
  } else {
    int32_t disp = 0;
    bool disp8 = false;
    bool disp32 = false;

    // Update ModR/M with the given register.
    DCHECK((modrm & 0x38) == 0);
    modrm |= code << 3;

    uint8_t mod = modrm >> 6;
    if (ts > 0 && (mod == 1 || mod == 2)) {
      // Encode displacement with disp8*N encoding.
      if (mod == 1) {
        length -= 1;
        disp = *bit_cast<const int8_t *>(&adr.buf_[length]);
      } else {
        length -= 4;
        disp = *bit_cast<const int32_t *>(&adr.buf_[length]);
      }
      if (disp % ts == 0 && is_int8(disp / ts)) {
        // Use disp8*N compressed encoding.
        disp = disp / ts;
        disp8 = true;
        modrm = (modrm & 0x3F) | 0x40;
      } else {
        // Use disp32 encoding.
        disp32 = true;
        modrm = (modrm & 0x3F) | 0x80;
      }
    }

    // Emit ModR/M byte.
    emit(modrm);

    // Emit the rest of the encoded operand.
    for (unsigned i = 1; i < length; i++) emit(adr.buf_[i]);

    // Emit displacement.
    if (disp8) emit(disp);
    if (disp32) emitl(disp);
  }
}

void Assembler::arithmetic_op(byte opcode,
                              Register reg,
                              const Operand &op,
                              int size) {
  if (size == 1) {
    arithmetic_op_8(opcode - 1, reg, op);
  } else if (size == 2) {
    arithmetic_op_16(opcode, reg, op);
  } else {
    EnsureSpace ensure_space(this);
    emit_rex(reg, op, size);
    emit(opcode);
    emit_operand(reg, op);
  }
}

void Assembler::arithmetic_op(byte opcode,
                              Register reg,
                              Register rm_reg,
                              int size) {
  if (size == 1) {
    arithmetic_op_8(opcode - 1, reg, rm_reg);
  } else if (size == 2) {
    arithmetic_op_16(opcode, reg, rm_reg);
  } else {
    EnsureSpace ensure_space(this);
    DCHECK((opcode & 0xC6) == 2);
    if (rm_reg.low_bits() == 4)  {  // Forces SIB byte.
      // Swap reg and rm_reg and change opcode operand order.
      emit_rex(rm_reg, reg, size);
      emit(opcode ^ 0x02);
      emit_modrm(rm_reg, reg);
    } else {
      emit_rex(reg, rm_reg, size);
      emit(opcode);
      emit_modrm(reg, rm_reg);
    }
  }
}

void Assembler::arithmetic_op_16(byte opcode, Register reg, Register rm_reg) {
  EnsureSpace ensure_space(this);
  DCHECK((opcode & 0xC6) == 2);
  if (rm_reg.low_bits() == 4) {  // Forces SIB byte.
    // Swap reg and rm_reg and change opcode operand order.
    emit(0x66);
    emit_optional_rex_32(rm_reg, reg);
    emit(opcode ^ 0x02);
    emit_modrm(rm_reg, reg);
  } else {
    emit(0x66);
    emit_optional_rex_32(reg, rm_reg);
    emit(opcode);
    emit_modrm(reg, rm_reg);
  }
}

void Assembler::arithmetic_op_16(byte opcode,
                                 Register reg,
                                 const Operand &rm_reg) {
  EnsureSpace ensure_space(this);
  emit(0x66);
  emit_optional_rex_32(reg, rm_reg);
  emit(opcode);
  emit_operand(reg, rm_reg);
}

void Assembler::arithmetic_op_8(byte opcode, Register reg, const Operand &op) {
  EnsureSpace ensure_space(this);
  if (!reg.is_byte_register()) {
    emit_rex_32(reg, op);
  } else {
    emit_optional_rex_32(reg, op);
  }
  emit(opcode);
  emit_operand(reg, op);
}

void Assembler::arithmetic_op_8(byte opcode, Register reg, Register rm_reg) {
  EnsureSpace ensure_space(this);
  DCHECK((opcode & 0xC6) == 2);
  if (rm_reg.low_bits() == 4)  {  // forces SIB byte
    // Swap reg and rm_reg and change opcode operand order.
    if (!rm_reg.is_byte_register() || !reg.is_byte_register()) {
      // Register is not one of al, bl, cl, dl.  Its encoding needs REX.
      emit_rex_32(rm_reg, reg);
    }
    emit(opcode ^ 0x02);
    emit_modrm(rm_reg, reg);
  } else {
    if (!reg.is_byte_register() || !rm_reg.is_byte_register()) {
      // Register is not one of al, bl, cl, dl.  Its encoding needs REX.
      emit_rex_32(reg, rm_reg);
    }
    emit(opcode);
    emit_modrm(reg, rm_reg);
  }
}

void Assembler::immediate_arithmetic_op(byte subcode,
                                        Register dst,
                                        Immediate src,
                                        int size) {
  if (size == 1) {
    immediate_arithmetic_op_8(subcode, dst, src);
  } else if (size == 2) {
    immediate_arithmetic_op_16(subcode, dst, src);
  } else {
    EnsureSpace ensure_space(this);
    emit_rex(dst, size);
    if (is_int8(src.value_)) {
      emit(0x83);
      emit_modrm(subcode, dst);
      emit(src.value_);
    } else if (dst.is(rax)) {
      emit(0x05 | (subcode << 3));
      emit(src);
    } else {
      emit(0x81);
      emit_modrm(subcode, dst);
      emit(src);
    }
  }
}

void Assembler::immediate_arithmetic_op(byte subcode,
                                        const Operand &dst,
                                        Immediate src,
                                        int size) {
  if (size == 1) {
    immediate_arithmetic_op_8(subcode - 1, dst, src);
  } else if (size == 2) {
    immediate_arithmetic_op_16(subcode, dst, src);
  } else {
    EnsureSpace ensure_space(this);
    emit_rex(dst, size);
    if (is_int8(src.value_)) {
      emit(0x83);
      emit_operand(subcode, dst, 1);
      emit(src.value_);
    } else {
      emit(0x81);
      emit_operand(subcode, dst, 4);
      emit(src);
    }
  }
}

void Assembler::immediate_arithmetic_op_16(byte subcode,
                                           Register dst,
                                           Immediate src) {
  EnsureSpace ensure_space(this);
  emit(0x66);  // operand size override prefix
  emit_optional_rex_32(dst);
  if (is_int8(src.value_)) {
    emit(0x83);
    emit_modrm(subcode, dst);
    emit(src.value_);
  } else if (dst.is(rax)) {
    emit(0x05 | (subcode << 3));
    emitw(src.value_);
  } else {
    emit(0x81);
    emit_modrm(subcode, dst);
    emitw(src.value_);
  }
}

void Assembler::immediate_arithmetic_op_16(byte subcode,
                                           const Operand &dst,
                                           Immediate src) {
  EnsureSpace ensure_space(this);
  emit(0x66);  // operand size override prefix
  emit_optional_rex_32(dst);
  if (is_int8(src.value_)) {
    emit(0x83);
    emit_operand(subcode, dst, 1);
    emit(src.value_);
  } else {
    emit(0x81);
    emit_operand(subcode, dst, 4);
    emitw(src.value_);
  }
}

void Assembler::immediate_arithmetic_op_8(byte subcode,
                                          const Operand &dst,
                                          Immediate src) {
  EnsureSpace ensure_space(this);
  emit_optional_rex_32(dst);
  DCHECK(is_int8(src.value_) || is_uint8(src.value_));
  emit(0x80);
  emit_operand(subcode, dst, 1);
  emit(src.value_);
}

void Assembler::immediate_arithmetic_op_8(byte subcode,
                                          Register dst,
                                          Immediate src) {
  EnsureSpace ensure_space(this);
  if (!dst.is_byte_register()) {
    // Register is not one of al, bl, cl, dl.  Its encoding needs REX.
    emit_rex_32(dst);
  }
  DCHECK(is_int8(src.value_) || is_uint8(src.value_));
  emit(0x80);
  emit_modrm(subcode, dst);
  emit(src.value_);
}

void Assembler::sse2_instr(XMMRegister dst, XMMRegister src, byte prefix,
                           byte escape, byte opcode) {
  EnsureSpace ensure_space(this);
  emit(prefix);
  emit_optional_rex_32(dst, src);
  emit(escape);
  emit(opcode);
  emit_sse_operand(dst, src);
}

void Assembler::sse2_instr(XMMRegister dst, const Operand &src, byte prefix,
                           byte escape, byte opcode) {
  EnsureSpace ensure_space(this);
  emit(prefix);
  emit_optional_rex_32(dst, src);
  emit(escape);
  emit(opcode);
  emit_sse_operand(dst, src);
}

void Assembler::ssse3_instr(XMMRegister dst, XMMRegister src, byte prefix,
                            byte escape1, byte escape2, byte opcode) {
  DCHECK(Enabled(SSSE3));
  EnsureSpace ensure_space(this);
  emit(prefix);
  emit_optional_rex_32(dst, src);
  emit(escape1);
  emit(escape2);
  emit(opcode);
  emit_sse_operand(dst, src);
}

void Assembler::ssse3_instr(XMMRegister dst, const Operand &src, byte prefix,
                            byte escape1, byte escape2, byte opcode) {
  DCHECK(Enabled(SSSE3));
  EnsureSpace ensure_space(this);
  emit(prefix);
  emit_optional_rex_32(dst, src);
  emit(escape1);
  emit(escape2);
  emit(opcode);
  emit_sse_operand(dst, src);
}

void Assembler::sse4_instr(XMMRegister dst, XMMRegister src, byte prefix,
                           byte escape1, byte escape2, byte opcode) {
  DCHECK(Enabled(SSE4_1));
  EnsureSpace ensure_space(this);
  emit(prefix);
  emit_optional_rex_32(dst, src);
  emit(escape1);
  emit(escape2);
  emit(opcode);
  emit_sse_operand(dst, src);
}

void Assembler::sse4_instr(XMMRegister dst, const Operand &src, byte prefix,
                           byte escape1, byte escape2, byte opcode) {
  DCHECK(Enabled(SSE4_1));
  EnsureSpace ensure_space(this);
  emit(prefix);
  emit_optional_rex_32(dst, src);
  emit(escape1);
  emit(escape2);
  emit(opcode);
  emit_sse_operand(dst, src);
}

void Assembler::emit_sse_operand(XMMRegister reg, const Operand &adr, int sl) {
  Register ireg = { reg.code() };
  emit_operand(ireg, adr, sl);
}

void Assembler::emit_sse_operand(Register reg, const Operand &adr, int sl) {
  Register ireg = {reg.code()};
  emit_operand(ireg, adr, sl);
}

void Assembler::emit_sse_operand(XMMRegister dst, XMMRegister src) {
  emit(0xC0 | (dst.low_bits() << 3) | src.low_bits());
}

void Assembler::emit_sse_operand(XMMRegister dst, Register src) {
  emit(0xC0 | (dst.low_bits() << 3) | src.low_bits());
}

void Assembler::emit_sse_operand(Register dst, XMMRegister src) {
  emit(0xC0 | (dst.low_bits() << 3) | src.low_bits());
}

void Assembler::emit_sse_operand(XMMRegister dst) {
  emit(0xD8 | dst.low_bits());
}

void Assembler::emit_sse_operand(ZMMRegister dst, ZMMRegister src) {
  emit(0xC0 | (dst.low_bits() << 3) | src.low_bits());
}

void Assembler::emit_sse_operand(ZMMRegister reg, const Operand &adr,
                                 int flags) {
  // Get register size.
  int regsize;
  switch (reg.size()) {
    case ZMMRegister::VectorLength128: regsize = 16; break;
    case ZMMRegister::VectorLength256: regsize = 32; break;
    case ZMMRegister::VectorLength512: regsize = 64; break;
    default: regsize = 0;
  }

  // Use broadcast type for broadcasts.
  int ts = regsize;
  if (adr.load() == broadcast) {
    if (flags & EVEX_BT4) {
      ts = 4;
    } else if (flags & EVEX_BT8) {
      ts = 8;
    }
  } else {
    if (flags & EVEX_DT1) {
      ts = 1;
    } else if (flags & EVEX_DT2) {
      ts = 2;
    } else if (flags & EVEX_DT2) {
      ts = 2;
    } else if (flags & EVEX_DT4) {
      ts = 4;
    } else if (flags & EVEX_DT8) {
      ts = 8;
    } else if (flags & EVEX_DT16) {
      ts = 16;
    } else if (flags & EVEX_DT32) {
      ts = 32;
    } else if (flags & EVEX_DT64) {
      ts = 64;
    }
  }

  // Suffix length is one if there is an immediate byte.
  int sl = (flags & EVEX_IMM) ? 1 : 0;

  // Emit operand for EVEX instruction.
  emit_operand(reg.low_bits(), adr, sl, ts);
}

void Assembler::emit_sse_operand(ZMMRegister dst, Register src) {
  emit(0xC0 | (dst.low_bits() << 3) | src.low_bits());
}

void Assembler::emit_sse_operand(Register dst, ZMMRegister src) {
  emit(0xC0 | (dst.low_bits() << 3) | src.low_bits());
}

void Assembler::vinstr(byte op, XMMRegister dst, XMMRegister src1,
                       XMMRegister src2, SIMDPrefix pp, LeadingOpcode m,
                       VexW w) {
  DCHECK(Enabled(AVX));
  EnsureSpace ensure_space(this);
  emit_vex_prefix(dst, src1, src2, kL128, pp, m, w);
  emit(op);
  emit_sse_operand(dst, src2);
}

void Assembler::vinstr(byte op, XMMRegister dst, XMMRegister src1,
                       const Operand &src2, SIMDPrefix pp, LeadingOpcode m,
                       VexW w, int sl) {
  DCHECK(Enabled(AVX));
  EnsureSpace ensure_space(this);
  emit_vex_prefix(dst, src1, src2, kL128, pp, m, w);
  emit(op);
  emit_sse_operand(dst, src2, sl);
}

void Assembler::vinstr(byte op, YMMRegister dst, YMMRegister src1,
                       YMMRegister src2, SIMDPrefix pp, LeadingOpcode m,
                       VexW w) {
  DCHECK(Enabled(AVX));
  EnsureSpace ensure_space(this);
  emit_vex_prefix(dst.xmm(), src1.xmm(), src2.xmm(), kL256, pp, m, w);
  emit(op);
  emit_sse_operand(dst.xmm(), src2.xmm());
}

void Assembler::vinstr(byte op, YMMRegister dst, YMMRegister src1,
                       const Operand &src2, SIMDPrefix pp, LeadingOpcode m,
                       VexW w, int sl) {
  DCHECK(Enabled(AVX));
  EnsureSpace ensure_space(this);
  emit_vex_prefix(dst.xmm(), src1.xmm(), src2, kL256, pp, m, w);
  emit(op);
  emit_sse_operand(dst.xmm(), src2, sl);
}

void Assembler::vps(byte op, XMMRegister dst, XMMRegister src1,
                    XMMRegister src2) {
  DCHECK(Enabled(AVX));
  EnsureSpace ensure_space(this);
  emit_vex_prefix(dst, src1, src2, kL128, kNone, k0F, kWIG);
  emit(op);
  emit_sse_operand(dst, src2);
}

void Assembler::vps(byte op, XMMRegister dst, XMMRegister src1,
                    const Operand &src2, int sl) {
  DCHECK(Enabled(AVX));
  EnsureSpace ensure_space(this);
  emit_vex_prefix(dst, src1, src2, kL128, kNone, k0F, kWIG);
  emit(op);
  emit_sse_operand(dst, src2, sl);
}

void Assembler::vps(byte op, YMMRegister dst, YMMRegister src1,
                    YMMRegister src2) {
  DCHECK(Enabled(AVX));
  EnsureSpace ensure_space(this);
  emit_vex_prefix(dst.xmm(), src1.xmm(), src2.xmm(), kL256, kNone, k0F, kWIG);
  emit(op);
  emit_sse_operand(dst.xmm(), src2.xmm());
}

void Assembler::vps(byte op, YMMRegister dst, YMMRegister src1,
                    const Operand &src2, int sl) {
  DCHECK(Enabled(AVX));
  EnsureSpace ensure_space(this);
  emit_vex_prefix(dst.xmm(), src1.xmm(), src2, kL256, kNone, k0F, kWIG);
  emit(op);
  emit_sse_operand(dst.xmm(), src2, sl);
}

void Assembler::vpd(byte op, XMMRegister dst, XMMRegister src1,
                    XMMRegister src2) {
  DCHECK(Enabled(AVX));
  EnsureSpace ensure_space(this);
  emit_vex_prefix(dst, src1, src2, kL128, k66, k0F, kWIG);
  emit(op);
  emit_sse_operand(dst, src2);
}

void Assembler::vpd(byte op, XMMRegister dst, XMMRegister src1,
                    const Operand &src2, int sl) {
  DCHECK(Enabled(AVX));
  EnsureSpace ensure_space(this);
  emit_vex_prefix(dst, src1, src2, kL128, k66, k0F, kWIG);
  emit(op);
  emit_sse_operand(dst, src2, sl);
}

void Assembler::vpd(byte op, YMMRegister dst, YMMRegister src1,
                    YMMRegister src2) {
  DCHECK(Enabled(AVX2));
  EnsureSpace ensure_space(this);
  emit_vex_prefix(dst.xmm(), src1.xmm(), src2.xmm(), kL256, k66, k0F, kWIG);
  emit(op);
  emit_sse_operand(dst.xmm(), src2.xmm());
}

void Assembler::vpd(byte op, YMMRegister dst, YMMRegister src1,
                    const Operand &src2, int sl) {
  DCHECK(Enabled(AVX2));
  EnsureSpace ensure_space(this);
  emit_vex_prefix(dst.xmm(), src1.xmm(), src2, kL256, k66, k0F, kWIG);
  emit(op);
  emit_sse_operand(dst.xmm(), src2, sl);
}

void Assembler::vss(byte op, XMMRegister dst, XMMRegister src1,
                    XMMRegister src2) {
  DCHECK(Enabled(AVX));
  EnsureSpace ensure_space(this);
  emit_vex_prefix(dst, src1, src2, kL128, kF3, k0F, kWIG);
  emit(op);
  emit_sse_operand(dst, src2);
}

void Assembler::vss(byte op, XMMRegister dst, XMMRegister src1,
                    const Operand &src2, int sl) {
  DCHECK(Enabled(AVX));
  EnsureSpace ensure_space(this);
  emit_vex_prefix(dst, src1, src2, kL128, kF3, k0F, kWIG);
  emit(op);
  emit_sse_operand(dst, src2, sl);
}

void Assembler::vss(byte op, YMMRegister dst, YMMRegister src1,
                    YMMRegister src2) {
  DCHECK(Enabled(AVX));
  EnsureSpace ensure_space(this);
  emit_vex_prefix(dst.xmm(), src1.xmm(), src2.xmm(), kL256, kF3, k0F, kWIG);
  emit(op);
  emit_sse_operand(dst.xmm(), src2.xmm());
}

void Assembler::vss(byte op, YMMRegister dst, YMMRegister src1,
                    const Operand &src2, int sl) {
  DCHECK(Enabled(AVX));
  EnsureSpace ensure_space(this);
  emit_vex_prefix(dst.xmm(), src1.xmm(), src2, kL256, kF3, k0F, kWIG);
  emit(op);
  emit_sse_operand(dst.xmm(), src2, sl);
}

void Assembler::vfmas(byte op, XMMRegister dst, XMMRegister src1,
                      XMMRegister src2) {
  DCHECK(Enabled(FMA3));
  EnsureSpace ensure_space(this);
  emit_vex_prefix(dst, src1, src2, kL128, k66, k0F38, kW0);
  emit(op);
  emit_sse_operand(dst, src2);
}

void Assembler::vfmas(byte op, XMMRegister dst, XMMRegister src1,
                      const Operand &src2) {
  DCHECK(Enabled(FMA3));
  EnsureSpace ensure_space(this);
  emit_vex_prefix(dst, src1, src2, kL128, k66, k0F38, kW0);
  emit(op);
  emit_sse_operand(dst, src2);
}

void Assembler::vfmas(byte op, YMMRegister dst, YMMRegister src1,
                      YMMRegister src2) {
  DCHECK(Enabled(FMA3));
  EnsureSpace ensure_space(this);
  emit_vex_prefix(dst.xmm(), src1.xmm(), src2.xmm(), kL256, k66, k0F38, kW0);
  emit(op);
  emit_sse_operand(dst.xmm(), src2.xmm());
}

void Assembler::vfmas(byte op, YMMRegister dst, YMMRegister src1,
                      const Operand &src2) {
  DCHECK(Enabled(FMA3));
  EnsureSpace ensure_space(this);
  emit_vex_prefix(dst.xmm(), src1.xmm(), src2, kL256, k66, k0F38, kW0);
  emit(op);
  emit_sse_operand(dst.xmm(), src2);
}

void Assembler::vfmad(byte op, XMMRegister dst, XMMRegister src1,
                      XMMRegister src2) {
  DCHECK(Enabled(FMA3));
  EnsureSpace ensure_space(this);
  emit_vex_prefix(dst, src1, src2, kL128, k66, k0F38, kW1);
  emit(op);
  emit_sse_operand(dst, src2);
}

void Assembler::vfmad(byte op, XMMRegister dst, XMMRegister src1,
                      const Operand &src2) {
  DCHECK(Enabled(FMA3));
  EnsureSpace ensure_space(this);
  emit_vex_prefix(dst, src1, src2, kL128, k66, k0F38, kW1);
  emit(op);
  emit_sse_operand(dst, src2);
}

void Assembler::vfmad(byte op, YMMRegister dst, YMMRegister src1,
                      YMMRegister src2) {
  DCHECK(Enabled(FMA3));
  EnsureSpace ensure_space(this);
  emit_vex_prefix(dst.xmm(), src1.xmm(), src2.xmm(), kL256, k66, k0F38, kW1);
  emit(op);
  emit_sse_operand(dst.xmm(), src2.xmm());
}

void Assembler::vfmad(byte op, YMMRegister dst, YMMRegister src1,
                      const Operand &src2) {
  DCHECK(Enabled(FMA3));
  EnsureSpace ensure_space(this);
  emit_vex_prefix(dst.xmm(), src1.xmm(), src2, kL256, k66, k0F38, kW1);
  emit(op);
  emit_sse_operand(dst.xmm(), src2);
}

void Assembler::kinstr(byte op, OpmaskRegister k1, OpmaskRegister k2,
                       OpmaskRegister k3, SIMDPrefix pp, LeadingOpcode m,
                       VexW w) {
  DCHECK(Enabled(AVX512F));
  EnsureSpace ensure_space(this);
  XMMRegister ik1 = {k1.code()};
  XMMRegister ik2 = {k2.code()};
  XMMRegister ik3 = {k3.code()};
  emit_vex_prefix(ik1, ik2, ik3, kL256, pp, m, w);
  emit(op);
  emit_sse_operand(ik1, ik3);
}

void Assembler::kinstr(byte op, OpmaskRegister k1, OpmaskRegister k2,
                       SIMDPrefix pp, LeadingOpcode m, VexW w) {
  DCHECK(Enabled(AVX512F));
  EnsureSpace ensure_space(this);
  XMMRegister ik1 = {k1.code()};
  XMMRegister ik2 = {k2.code()};
  emit_vex_prefix(ik1, xmm0, ik2, kL128, pp, m, w);
  emit(op);
  emit_sse_operand(ik1, ik2);
}

void Assembler::kinstr(byte op, OpmaskRegister k1, OpmaskRegister k2,
                       int8_t imm8, SIMDPrefix pp, LeadingOpcode m, VexW w) {
  DCHECK(Enabled(AVX512F));
  EnsureSpace ensure_space(this);
  XMMRegister ik1 = {k1.code()};
  XMMRegister ik2 = {k2.code()};
  emit_vex_prefix(ik1, xmm0, ik2, kL128, pp, m, w);
  emit(op);
  emit_sse_operand(ik1, ik2);
  emit(imm8);
}

void Assembler::kinstr(byte op, OpmaskRegister k1, const Operand &src,
                       SIMDPrefix pp, LeadingOpcode m, VexW w) {
  DCHECK(Enabled(AVX512F));
  EnsureSpace ensure_space(this);
  XMMRegister ik1 = {k1.code()};
  emit_vex_prefix(ik1, xmm0, src, kL128, pp, m, w);
  emit(op);
  emit_sse_operand(ik1, src);
}

void Assembler::kinstr(byte op, const Operand &dst, OpmaskRegister k1,
                       SIMDPrefix pp, LeadingOpcode m, VexW w) {
  DCHECK(Enabled(AVX512F));
  EnsureSpace ensure_space(this);
  XMMRegister ik1 = {k1.code()};
  emit_vex_prefix(ik1, xmm0, dst, kL128, pp, m, w);
  emit(op);
  emit_sse_operand(ik1, dst);
}

void Assembler::kinstr(byte op, OpmaskRegister k1, Register src,
                       SIMDPrefix pp, LeadingOpcode m, VexW w) {
  DCHECK(Enabled(AVX512F));
  EnsureSpace ensure_space(this);
  XMMRegister ik1 = {k1.code()};
  XMMRegister isrc = {src.code()};
  emit_vex_prefix(ik1, xmm0, isrc, kL128, pp, m, w);
  emit(op);
  emit_sse_operand(ik1, isrc);
}

void Assembler::kinstr(byte op, Register dst, OpmaskRegister k1,
                       SIMDPrefix pp, LeadingOpcode m, VexW w) {
  DCHECK(Enabled(AVX512F));
  EnsureSpace ensure_space(this);
  XMMRegister ik1 = {k1.code()};
  XMMRegister idst = {dst.code()};
  emit_vex_prefix(idst, xmm0, ik1, kL128, pp, m, w);
  emit(op);
  emit_sse_operand(idst, ik1);
}

void Assembler::zinstr(byte op, ZMMRegister dst, ZMMRegister src,
                       int8_t imm8, Mask mask, int flags) {
  EnsureSpace ensure_space(this);
  emit_evex_prefix(dst, zmm0, src, mask, flags);
  emit(op);
  emit_sse_operand(dst, src);
  if (flags & EVEX_IMM) emit(imm8);
}

void Assembler::zinstr(byte op, ZMMRegister dst, const Operand &src,
                       int8_t imm8, Mask mask, int flags) {
  EnsureSpace ensure_space(this);
  emit_evex_prefix(dst, zmm0, src, mask, flags);
  emit(op);
  emit_sse_operand(dst, src, flags);
  if (flags & EVEX_IMM) emit(imm8);
}

void Assembler::zinstr(byte op, const Operand &dst, ZMMRegister src,
                       int8_t imm8, Mask mask, int flags) {
  EnsureSpace ensure_space(this);
  emit_evex_prefix(src, zmm0, dst, mask, flags);
  emit(op);
  emit_sse_operand(src, dst, flags);
  if (flags & EVEX_IMM) emit(imm8);
}

void Assembler::zinstr(byte op, ZMMRegister dst, ZMMRegister src1,
                       ZMMRegister src2,
                       int8_t imm8, Mask mask, int flags) {
  EnsureSpace ensure_space(this);
  emit_evex_prefix(dst, src1, src2, mask, flags);
  emit(op);
  emit_sse_operand(dst, src2);
  if (flags & EVEX_IMM) emit(imm8);
}

void Assembler::zinstr(byte op, ZMMRegister dst, ZMMRegister src1,
                       const Operand &src2,
                       int8_t imm8, Mask mask, int flags) {
  EnsureSpace ensure_space(this);
  emit_evex_prefix(dst, src1, src2, mask, flags);
  emit(op);
  emit_sse_operand(dst, src2, flags);
  if (flags & EVEX_IMM) emit(imm8);
}

void Assembler::zinstr(byte op, const Operand &dst, ZMMRegister src1,
                       ZMMRegister src2,
                       int8_t imm8, Mask mask, int flags) {
  EnsureSpace ensure_space(this);
  emit_evex_prefix(src2, src1, dst, mask, flags);
  emit(op);
  emit_sse_operand(src2, dst, flags);
  if (flags & EVEX_IMM) emit(imm8);
}

void Assembler::zinstr(byte op, ZMMRegister dst, ZMMRegister src1,
                       Register src2,
                       int8_t imm8, Mask mask, int flags) {
  EnsureSpace ensure_space(this);
  ZMMRegister isrc2 = ZMMRegister::from_code(src2.code());
  emit_evex_prefix(dst, src1, isrc2, mask, flags);
  emit(op);
  emit_sse_operand(dst, isrc2);
  if (flags & EVEX_IMM) emit(imm8);
}

void Assembler::zinstr(byte op, ZMMRegister dst, Register src,
                       int8_t imm8, Mask mask, int flags) {
  EnsureSpace ensure_space(this);
  ZMMRegister isrc = ZMMRegister::from_code(src.code());
  emit_evex_prefix(dst, zmm0, isrc, mask, flags);
  emit(op);
  emit_sse_operand(dst, isrc);
  if (flags & EVEX_IMM) emit(imm8);
}

void Assembler::zinstr(byte op, Register dst, ZMMRegister src,
                        int8_t imm8, Mask mask, int flags) {
  EnsureSpace ensure_space(this);
  ZMMRegister idst = ZMMRegister::from_code(dst.code());
  emit_evex_prefix(idst, zmm0, src, mask, flags);
  emit(op);
  emit_sse_operand(idst, src);
  if (flags & EVEX_IMM) emit(imm8);
}

void Assembler::zinstr(byte op, Register dst, const Operand &src,
                       int8_t imm8, Mask mask, int flags) {
  EnsureSpace ensure_space(this);
  ZMMRegister idst = ZMMRegister::from_code(dst.code());
  emit_evex_prefix(idst, zmm0, src, mask, flags);
  emit(op);
  emit_sse_operand(idst, src, flags);
  if (flags & EVEX_IMM) emit(imm8);
}

void Assembler::zinstr(byte op, OpmaskRegister k, ZMMRegister src1,
                       ZMMRegister src2,
                       int8_t imm8, Mask mask, int flags) {
  EnsureSpace ensure_space(this);
  ZMMRegister ik = ZMMRegister::from_code(k.code());
  emit_evex_prefix(ik, src1, src2, mask, flags);
  emit(op);
  emit_sse_operand(ik, src2);
  if (flags & EVEX_IMM) emit(imm8);
}

void Assembler::zinstr(byte op, OpmaskRegister k, ZMMRegister src1,
                       const Operand &src2,
                       int8_t imm8, Mask mask, int flags) {
  EnsureSpace ensure_space(this);
  ZMMRegister ik = ZMMRegister::from_code(k.code());
  emit_evex_prefix(ik, src1, src2, mask, flags);
  emit(op);
  emit_sse_operand(ik, src2, flags);
  if (flags & EVEX_IMM) emit(imm8);
}

void Assembler::zinstr(byte op, OpmaskRegister k, ZMMRegister src,
                       int8_t imm8, Mask mask, int flags) {
  EnsureSpace ensure_space(this);
  ZMMRegister ik = ZMMRegister::from_code(k.code());
  emit_evex_prefix(ik, zmm0, src, mask, flags);
  emit(op);
  emit_sse_operand(ik, src);
  if (flags & EVEX_IMM) emit(imm8);
}

void Assembler::zinstr(byte op, ZMMRegister dst, OpmaskRegister k,
                       int8_t imm8, Mask mask, int flags) {
  EnsureSpace ensure_space(this);
  ZMMRegister ik = ZMMRegister::from_code(k.code());
  emit_evex_prefix(dst, zmm0, ik, mask, flags);
  emit(op);
  emit_sse_operand(dst, ik);
  if (flags & EVEX_IMM) emit(imm8);
}

void Assembler::db(uint8_t data) {
  EnsureSpace ensure_space(this);
  emit(data);
}

void Assembler::dd(uint32_t data) {
  EnsureSpace ensure_space(this);
  emitl(data);
}

void Assembler::dq(uint64_t data) {
  EnsureSpace ensure_space(this);
  emitq(data);
}

void Assembler::dq(Label *label) {
  EnsureSpace ensure_space(this);
  if (label->is_bound()) {
    refs_.push_back(pc_offset());
    emitp(buffer_ + label->pos());
  } else {
    emitl(0);  // zero for the first 32bit marks it as 64bit absolute address
    if (label->is_linked()) {
      emitl(label->pos());
      label->link_to(pc_offset() - sizeof(int32_t));
    } else {
      DCHECK(label->is_unused());
      int32_t current = pc_offset();
      emitl(current);
      label->link_to(current);
    }
  }
}

}  // namespace jit
}  // namespace sling

