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

#include "myelin/macro-assembler.h"

#include <stddef.h>

#include "base/logging.h"
#include "base/macros.h"
#include "myelin/compute.h"

namespace sling {
namespace myelin {

using namespace jit;

// Register used for profile timestamp.
static Register tsreg = r15;

// Use base register for data instance.
static Register datareg = rbp;

Register Registers::try_alloc() {
  for (int r = 0; r < kNumRegisters; ++r) {
    if (!used(r)) {
      use(r);
      return Register::from_code(r);
    }
  }
  return no_reg;
}

Register Registers::alloc() {
  Register r = try_alloc();
  CHECK(r.is_valid()) << "Register overflow";
  return r;
}

Register Registers::try_alloc_preserved() {
  for (int r = 0; r < kNumRegisters; ++r) {
    if (!used(r) && preserved(r)) {
      use(r);
      return Register::from_code(r);
    }
  }
  return no_reg;
}

Register Registers::alloc_preserved() {
  Register r = try_alloc_preserved();
  CHECK(r.is_valid()) << "Register overflow";
  return r;
}

Register Registers::alloc_preferred(Register r) {
  if (!used(r)) {
    use(r);
    return r;
  } else {
    return alloc();
  }
}

Register Registers::alloc_fixed(Register r) {
  CHECK(!used(r)) << "Register already used";
  use(r);
  return r;
}

Register Registers::alloc_temp() {
  if (!used(r10)) return alloc_fixed(r10);
  if (!used(r11)) return alloc_fixed(r11);
  LOG(FATAL) << "Temp register overflow";
  return no_reg;
}

Register Registers::arg(int n) {
  Register r;
  switch (n) {
    case 0: r = rax; break;
    case 1: r = arg_reg_1; break;
    case 2: r = arg_reg_2; break;
    case 3: r = arg_reg_3; break;
    case 4: r = arg_reg_4; break;
    case 5: r = arg_reg_5; break;
    case 6: r = arg_reg_6; break;
    default: LOG(FATAL) << "Only six argument registers";
  }
  return alloc_fixed(r);
}

void Registers::reserve(int r) {
  CHECK(!saved(r)) << r;
  CHECK(used(r)) << r;
  saved_regs_ |= (1 << r);
  used_regs_ &= ~(1 << r);
}

void Registers::free(int r) {
  CHECK(saved(r)) << r;
  CHECK(!used(r)) << r;
  saved_regs_ &= ~(1 << r);
  used_regs_ |= (1 << r);
}

bool Registers::usage(int n) {
  switch (n) {
    case 13: reserve(r15); FALLTHROUGH_INTENDED;
    case 12: reserve(r14); FALLTHROUGH_INTENDED;
    case 11: reserve(r13); FALLTHROUGH_INTENDED;
    case 10: reserve(r12); FALLTHROUGH_INTENDED;
    case 9: reserve(rbx); FALLTHROUGH_INTENDED;
    case 8: case 7: case 6: case 5: case 4: case 3: case 2: case 1: case 0:
      return true;
  }
  return false;
}

int Registers::num_free() const {
  int n = 0;
  for (int r = 0; r < kNumRegisters; ++r) {
    if (!used(r)) n++;
  }
  return n;
}

int SIMDRegisters::try_alloc() {
  for (int r = 0; r < kNumRegisters; ++r) {
    if ((used_regs_ & (1 << r)) == 0) {
      use(r);
      return r;
    }
  }
  return -1;
}

int SIMDRegisters::alloc() {
  int r = try_alloc();
  CHECK(r != -1) << "SIMD register overflow";
  return r;
}

void StaticData::AddData(const void *buffer, int size, int repeat) {
  const uint8 *ptr = static_cast<const uint8 *>(buffer);
  for (int n = 0; n < repeat; ++n) {
    data_.insert(data_.end(), ptr, ptr + size);
  }
}

bool StaticData::Equals(const void *data, int size, int repeat) const {
  if (size * repeat != data_.size()) return false;
  const uint8 *p1 = data_.data();
  for (int i = 0; i < repeat; ++i) {
    const uint8 *p2 = static_cast<const uint8 *>(data);
    for (int j = 0; j < size; ++j) {
      if (*p1++ != *p2++) return false;
    }
  }
  return true;
}

void StaticData::Generate(MacroAssembler *masm) {
  // Align output.
  masm->DataAlign(alignment_);

  // Bind label to the address of the generated data block.
  masm->bind(&location_);

  // Emit data block.
  for (uint8 byte : data_) masm->db(byte);
}

MacroAssembler::MacroAssembler(void *buffer, int buffer_size)
    : Assembler(buffer, buffer_size) {}

MacroAssembler::~MacroAssembler() {
  for (auto *d : data_blocks_) delete d;
}

Register MacroAssembler::instance() const {
  return datareg;
}

void MacroAssembler::Prologue() {
  // Zero upper part of YMM register if CPU needs it to avoid AVX-SSE transition
  // penalties.
  if (CPU::VZeroNeeded()) {
    vzeroupper();
  }

  // Reserve timestamp register.
  if (timing_) {
    rr_.reserve(tsreg);
    rr_.use(tsreg);
  }

  // Get argument.
  pushq(datareg);
  movq(datareg, arg_reg_1);

  // Save preserved registers on stack.
  if (rr_.saved(rbx)) pushq(rbx);
  if (rr_.saved(r12)) pushq(r12);
  if (rr_.saved(r13)) pushq(r13);
  if (rr_.saved(r14)) pushq(r14);
  if (rr_.saved(r15)) pushq(r15);

  // Get initial timestamp counter if timing instrumentation is active.
  if (timing_) {
    rdtsc();
    shlq(rdx, Immediate(32));
    orq(rax, rdx);
    movq(tsreg, rax);
  }
}

void MacroAssembler::Epilogue() {
  // Restore preserved registers from stack.
  if (rr_.saved(r15)) popq(r15);
  if (rr_.saved(r14)) popq(r14);
  if (rr_.saved(r13)) popq(r13);
  if (rr_.saved(r12)) popq(r12);
  if (rr_.saved(rbx)) popq(rbx);

  // Restore instance data register.
  popq(datareg);

  // Zero upper part of YMM register if CPU needs it to avoid AVX-SSE transition
  // penalties.
  if (CPU::VZeroNeeded()) {
    vzeroupper();
  }

  // Generate return instruction.
  ret(0);

  // Release timing register.
  if (timing_) {
    rr_.release(tsreg);
    rr_.free(tsreg);
  }
}

StaticData *MacroAssembler::CreateDataBlock(int alignment) {
  StaticData *data = new StaticData(alignment);
  data_blocks_.push_back(data);
  return data;
}

StaticData *MacroAssembler::FindDataBlock(
    const void *data, int size, int repeat) {
  for (StaticData *sd : data_blocks_) {
    if (sd->Equals(data, size, repeat)) return sd;
  }
  return nullptr;
}

void MacroAssembler::GenerateDataBlocks() {
  for (StaticData *sd : data_blocks_) {
    sd->Generate(this);
  }
}

void MacroAssembler::LoopStart(jit::Label *label) {
  //CodeTargetAlign();
  bind(label);
}

void MacroAssembler::LoadTensorAddress(Register dst, Tensor *tensor) {
  if (tensor->IsConstant()) {
    movp(dst, tensor->data());
    if (tensor->ref()) {
      movq(dst, Operand(dst));
    }
  } else if (tensor->offset() == 0) {
    if (tensor->ref()) {
      movq(dst, Operand(datareg));
    } else {
      movq(dst, datareg);
    }
  } else {
    DCHECK(tensor->offset() != -1) << tensor->name();
    if (tensor->ref()) {
      movq(dst, Operand(datareg, tensor->offset()));
    } else {
      leaq(dst, Operand(datareg, tensor->offset()));
    }
  }
}

void MacroAssembler::Copy(Register dst, int ddisp,
                          Register src, int sdisp,
                          int size) {
  // Save registers if needed.
  if (rr_.used(rsi)) pushq(rsi);
  if (rr_.used(rdi)) pushq(rdi);
  if (rr_.used(rcx)) pushq(rcx);

  // Set up source and destination.
  if (src.is(rdi) && dst.is(rdi)) {
    xchgq(dst, src);
    if (ddisp != 0) addq(rdi, Immediate(ddisp));
    if (sdisp != 0) addq(rsi, Immediate(sdisp));
  } else {
    if (dst.is(rdi)) {
      if (ddisp != 0) addq(rdi, Immediate(ddisp));
    } else {
      if (ddisp != 0) {
        leaq(rdi, Operand(dst, ddisp));
      } else {
        movq(rdi, dst);
      }
    }

    if (src.is(rsi)) {
      if (sdisp != 0) addq(rsi, Immediate(sdisp));
    } else {
      if (sdisp != 0) {
        leaq(rsi, Operand(src, sdisp));
      } else {
        movq(rsi, src);
      }
    }
  }

  // Set up size.
  movq(rcx, Immediate(size));

  // Copy data.
  repmovsb();

  // Restore registers if needed.
  if (rr_.used(rcx)) popq(rcx);
  if (rr_.used(rdi)) popq(rdi);
  if (rr_.used(rsi)) popq(rsi);
}

void MacroAssembler::LoadInteger(jit::Register dst, jit::Register base,
                                 jit::Register index, Type type) {
  switch (type) {
    case DT_INT8:
      movsxbq(dst, Operand(base, index, times_1));
      break;

    case DT_UINT8:
      movb(dst, Operand(base, index, times_1));
      break;

    case DT_INT16:
      movsxwq(dst, Operand(base, index, times_2));
      break;

    case DT_UINT16:
      movw(dst, Operand(base, index, times_2));
      break;

    case DT_INT32:
      movsxlq(dst, Operand(base, index, times_4));
      break;

    case DT_INT64:
      movq(dst, Operand(base, index, times_8));
      break;

    default:
      LOG(FATAL) << "Invalid integer type: " << type;
  }
}

void MacroAssembler::StoreInteger(jit::Register base, jit::Register index,
                                  jit::Register src, Type type) {
  switch (type) {
    case DT_INT8:
    case DT_UINT8:
      movb(Operand(base, index, times_1), src);
      break;

    case DT_INT16:
    case DT_UINT16:
      movw(Operand(base, index, times_2), src);
      break;

    case DT_INT32:
      movl(Operand(base, index, times_4), src);
      break;

    case DT_INT64:
      movq(Operand(base, index, times_8), src);
      break;

    default:
      LOG(FATAL) << "Invalid integer type: " << type;
  }
}

void MacroAssembler::Multiply(jit::Register reg, int64 scalar) {
  if (scalar == 0) {
    xorq(reg, reg);
  } else if (scalar < 0) {
    imulq(reg, reg, Immediate(scalar));
  } else if (scalar != 1) {
    // Check if scalar is power of two.
    int shift = 0;
    int64 value = 1;
    while (value < scalar) {
      value <<= 1;
      shift++;
    }
    if (value == scalar) {
      shlq(reg, Immediate(shift));
    } else {
      imulq(reg, reg, Immediate(scalar));
    }
  }
}

void MacroAssembler::StartTask(int offset, int32 id, int32 index,
                               jit::Label *entry) {
  // Check that runtime supports parallel execution.
  CHECK(runtime_->SupportsAsync())
      << "Runtime does not support asynchronous execution";

  // Fill out task structure.
  Register acc = rr_.alloc();
  leaq(arg_reg_1, Operand(datareg, offset));
  leaq(acc, Operand(entry));
  movq(Operand(arg_reg_1, offsetof(Task, func)), acc);
  movq(Operand(arg_reg_1, offsetof(Task, arg)), datareg);
  movl(Operand(arg_reg_1, offsetof(Task, id)), Immediate(id));
  movl(Operand(arg_reg_1, offsetof(Task, index)), Immediate(index));

  // Call runtime to start task.
  movp(acc, reinterpret_cast<void *>(runtime_->StartTaskFunc()));
  call(acc);

  rr_.release(acc);
}

void MacroAssembler::WaitForTask(int offset) {
  // Check that runtime supports parallel execution.
  CHECK(runtime_->SupportsAsync())
      << "Runtime does not support asynchronous execution";

  // Call runtime to wait for task to complete.
  Register acc = rr_.alloc();
  leaq(arg_reg_1, Operand(datareg, offset));
  movp(acc, reinterpret_cast<void *>(runtime_->WaitTaskFunc()));
  call(acc);
  rr_.release(acc);
}

void MacroAssembler::CallInstanceFunction(void (*func)(void *)) {
  if (func != nullptr) {
    Register acc = rr_.alloc();
    movq(arg_reg_1, datareg);
    movp(acc, reinterpret_cast<void *>(func));
    call(acc);
    rr_.release(acc);
  }
}

void MacroAssembler::IncrementInvocations(int offset) {
  incq(Operand(datareg, offset));
}

void MacroAssembler::TimeStep(int offset) {
  // Timing instrumentation must be active.
  CHECK(timing_);
  CHECK(!rr_.used(rax));
  CHECK(!rr_.used(rdx));

  // Get current time stamp (rax).
  rdtsc();
  shlq(rdx, Immediate(32));
  orq(rax, rdx);

  // Compute time elapsed (rdx).
  movq(rdx, rax);
  subq(rdx, tsreg);

  // Add elapsed time to timing block.
  addq(Operand(datareg, offset), rdx);

  // Store new timestamp.
  movq(tsreg, rax);
}

void MacroAssembler::ResetRegisterUsage() {
  rr_.reset();
  mm_.reset();
  if (timing_) rr_.use(tsreg);
}

}  // namespace myelin
}  // namespace sling

