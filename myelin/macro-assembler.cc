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

Register Registers::alloc() {
  for (int r = 0; r < kNumRegisters; ++r) {
    if (!used(r)) {
      use(r);
      return Register::from_code(r);
    }
  }
  LOG(FATAL) << "Register overflow";
  return no_reg;
}

Register Registers::alloc_preserved() {
  for (int r = 0; r < kNumRegisters; ++r) {
    if (!used(r) && preserved(r)) {
      use(r);
      return Register::from_code(r);
    }
  }
  LOG(FATAL) << "Preserved register overflow";
  return no_reg;
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

int SIMDRegisters::alloc() {
  for (int r = 0; r < kNumRegisters; ++r) {
    if ((used_regs_ & (1 << r)) == 0) {
      use(r);
      return r;
    }
  }
  LOG(FATAL) << "SIMD register overflow";
  return -1;
}

MacroAssembler::MacroAssembler(void *buffer, int buffer_size)
    : Assembler(buffer, buffer_size) {}

Register MacroAssembler::instance() const {
  return datareg;
}

void MacroAssembler::Prolog() {
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

  // Zero upper part of YMM register if CPU needs it.
  if (CPU::VZeroNeeded()) {
    vzeroupper();
  }

  // Get initial timestamp counter if timing instrumentation is active.
  if (timing_) {
    rdtsc();
    shlq(rdx, Immediate(32));
    orq(rax, rdx);
    movq(tsreg, rax);
  }
}

void MacroAssembler::Epilog() {
  // Restore preserved registers from stack.
  if (rr_.saved(r15)) popq(r15);
  if (rr_.saved(r14)) popq(r14);
  if (rr_.saved(r13)) popq(r13);
  if (rr_.saved(r12)) popq(r12);
  if (rr_.saved(rbx)) popq(rbx);

  // Restore instance data register.
  popq(datareg);

  // Generate return instruction.
  ret(0);

  // Release timing register.
  if (timing_) {
    rr_.release(tsreg);
    rr_.free(tsreg);
  }
}

void MacroAssembler::LoopStart(jit::Label *label) {
  //CodeTargetAlign();
  bind(label);
}

void MacroAssembler::LoadTensorAddress(Register dst, Tensor *tensor) {
  if (tensor->IsConstant()) {
    movp(dst, static_cast<void *>(tensor->data()));
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

}  // namespace myelin
}  // namespace sling

