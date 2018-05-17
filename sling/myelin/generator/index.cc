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

#include "sling/myelin/generator/index.h"

#define __ masm->

namespace sling {
namespace myelin {

using namespace jit;

bool IndexGenerator::AllocateRegisters() {
  CHECK(masm_ != nullptr);

  // Allocate fixed registers.
  bool ok = true;
  for (auto r : fixed_) {
    ok |= !masm_->rr().used(r);
    masm_->rr().alloc_fixed(r);
  }

  // Allocate temporary registers.
  for (auto &r : regs_) {
    r = masm_->rr().try_alloc();
    if (!r.is_valid()) ok = false;
  }
  for (auto &m : mmregs_) {
    m = masm_->mm().try_alloc(extended_regs_);
    if (m == -1) ok = false;
  }

  // Allocate auxiliary registers.
  for (auto &r : aux_) {
    r = masm_->rr().try_alloc();
    if (!r.is_valid()) ok = false;
  }
  for (auto &m : mmaux_) {
    m = masm_->mm().try_alloc(extended_regs_);
    if (m == -1) ok = false;
  }

  return ok;
}

void IndexGenerator::ReserveFixedRegister(jit::Register reg) {
  fixed_.push_back(reg);
}

void IndexGenerator::ReserveRegisters(int count) {
  for (int n = 0; n < count; ++n) {
    regs_.push_back(no_reg);
  }
}

void IndexGenerator::ReserveAuxRegisters(int count) {
  for (int n = 0; n < count; ++n) {
    aux_.push_back(no_reg);
  }
}

void IndexGenerator::ReserveXMMRegisters(int count) {
  for (int n = 0; n < count; ++n) {
    mmregs_.push_back(-1);
  }
}

void IndexGenerator::ReserveYMMRegisters(int count) {
  ReserveXMMRegisters(count);
}

void IndexGenerator::ReserveZMMRegisters(int count) {
  ReserveXMMRegisters(count);
}

void IndexGenerator::ReserveAuxXMMRegisters(int count) {
  for (int n = 0; n < count; ++n) {
    mmaux_.push_back(-1);
  }
}

void IndexGenerator::ReserveAuxYMMRegisters(int count) {
  ReserveAuxXMMRegisters(count);
}

void IndexGenerator::ReserveAuxZMMRegisters(int count) {
  ReserveAuxXMMRegisters(count);
}

}  // namespace myelin
}  // namespace sling

