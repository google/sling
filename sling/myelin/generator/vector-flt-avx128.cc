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

#include "sling/myelin/generator/expression.h"

#define __ masm->

namespace sling {
namespace myelin {

using namespace jit;

// Generate vector float expression using AVX and XMM registers.
class VectorFltAVX128Generator : public ExpressionGenerator {
 public:
  VectorFltAVX128Generator() {
    model_.mov_reg_reg = true;
    model_.mov_reg_imm = true;
    model_.mov_reg_mem = true;
    model_.mov_mem_reg = true;
    model_.op_reg_reg_reg = true;
    model_.op_reg_reg_imm = true;
    model_.op_reg_reg_mem = true;
    model_.func_reg_reg = true;
    model_.func_reg_imm = true;
    model_.func_reg_mem = true;
    if (CPU::Enabled(FMA3)) {
      model_.fm_reg_reg_reg = true;
      model_.fm_reg_reg_imm = true;
      model_.fm_reg_reg_mem = true;
    }
  }

  string Name() override { return "VFltAVX128"; }

  int VectorSize() override { return XMMRegSize; }

  void Reserve() override {
    // Reserve XMM registers.
    index_->ReserveXMMRegisters(instructions_.NumRegs());
  }

  void Generate(Express::Op *instr, MacroAssembler *masm) override {
    switch (instr->type) {
      case Express::MOV:
        if (IsLoadZero(instr) && masm->Enabled(ZEROIDIOM)) {
          // Use XOR to zero register instead of loading constant from memory.
          // This uses the floating point version of xor to avoid bypass delays
          // between integer and floating point units.
          switch (type_) {
            case DT_FLOAT:
              __ vxorps(xmm(instr->dst), xmm(instr->dst), xmm(instr->dst));
              break;
            case DT_DOUBLE:
              __ vxorpd(xmm(instr->dst), xmm(instr->dst), xmm(instr->dst));
              break;
            default: UNSUPPORTED;
          }
        } else {
          GenerateXMMVectorMove(instr, masm);
        }
        break;
      case Express::ADD:
        GenerateXMMFltOp(instr,
            &Assembler::vaddps, &Assembler::vaddpd,
            &Assembler::vaddps, &Assembler::vaddpd,
            masm);
        break;
      case Express::SUB:
        GenerateXMMFltOp(instr,
            &Assembler::vsubps, &Assembler::vsubpd,
            &Assembler::vsubps, &Assembler::vsubpd,
            masm);
        break;
      case Express::MUL:
        GenerateXMMFltOp(instr,
            &Assembler::vmulps, &Assembler::vmulpd,
            &Assembler::vmulps, &Assembler::vmulpd,
            masm);
        break;
      case Express::DIV:
        GenerateXMMFltOp(instr,
            &Assembler::vdivps, &Assembler::vdivpd,
            &Assembler::vdivps, &Assembler::vdivpd,
            masm);
        break;
      case Express::MIN:
        GenerateXMMFltOp(instr,
            &Assembler::vminps, &Assembler::vminpd,
            &Assembler::vminps, &Assembler::vminpd,
            masm);
        break;
      case Express::MAX:
        GenerateXMMFltOp(instr,
            &Assembler::vmaxps, &Assembler::vmaxpd,
            &Assembler::vmaxps, &Assembler::vmaxpd,
            masm);
        break;
      case Express::SQRT:
        GenerateXMMFltOp(instr,
            &Assembler::vsqrtps, &Assembler::vsqrtpd,
            &Assembler::vsqrtps, &Assembler::vsqrtpd,
            masm, 0);
        break;
      case Express::MULADD132:
        GenerateXMMFltOp(instr,
            &Assembler::vfmadd132ps, &Assembler::vfmadd132pd,
            &Assembler::vfmadd132ps, &Assembler::vfmadd132pd,
            masm, 2);
        break;
      case Express::MULADD213:
        GenerateXMMFltOp(instr,
            &Assembler::vfmadd213ps, &Assembler::vfmadd213pd,
            &Assembler::vfmadd213ps, &Assembler::vfmadd213pd,
            masm, 2);
        break;
      case Express::MULADD231:
        GenerateXMMFltOp(instr,
            &Assembler::vfmadd231ps, &Assembler::vfmadd231pd,
            &Assembler::vfmadd231ps, &Assembler::vfmadd231pd,
            masm, 2);
        break;
      case Express::MULSUB132:
        GenerateXMMFltOp(instr,
            &Assembler::vfmsub132ps, &Assembler::vfmsub132pd,
            &Assembler::vfmsub132ps, &Assembler::vfmsub132pd,
            masm, 2);
        break;
      case Express::MULSUB213:
        GenerateXMMFltOp(instr,
            &Assembler::vfmsub213ps, &Assembler::vfmsub213pd,
            &Assembler::vfmsub213ps, &Assembler::vfmsub213pd,
            masm, 2);
        break;
      case Express::MULSUB231:
        GenerateXMMFltOp(instr,
            &Assembler::vfmsub231ps, &Assembler::vfmsub231pd,
            &Assembler::vfmsub231ps, &Assembler::vfmsub231pd,
            masm, 2);
        break;
      case Express::CMPEQOQ:
        GenerateCompare(instr, masm, CMP_EQ_OQ);
        break;
      case Express::CMPLTOQ:
        GenerateCompare(instr, masm, CMP_LT_OQ);
        break;
      case Express::CMPGTOQ:
        GenerateCompare(instr, masm, CMP_GT_OQ);
        break;
      case Express::CMPNGEUQ:
        GenerateCompare(instr, masm, CMP_NGE_UQ);
        break;
      case Express::AND:
        GenerateXMMFltOp(instr,
            &Assembler::vandps, &Assembler::vandpd,
            &Assembler::vandps, &Assembler::vandpd,
            masm);
        break;
      case Express::OR:
        GenerateXMMFltOp(instr,
            &Assembler::vorps, &Assembler::vorpd,
            &Assembler::vorps, &Assembler::vorpd,
            masm);
        break;
      case Express::ANDNOT:
        GenerateXMMFltOp(instr,
            &Assembler::vandnps, &Assembler::vandnpd,
            &Assembler::vandnps, &Assembler::vandnpd,
            masm);
        break;
      case Express::SHR23:
        GenerateShift(instr, masm, false, 23);
        break;
      case Express::SHL23:
        GenerateShift(instr, masm, true, 23);
        break;
      case Express::FLOOR:
        GenerateXMMFltOp(instr,
            &Assembler::vroundps, &Assembler::vroundpd,
            &Assembler::vroundps, &Assembler::vroundpd,
            round_down, masm);
        break;
      case Express::CVTFLTINT:
        GenerateXMMFltOp(instr,
            &Assembler::vcvttps2dq, &Assembler::vcvttpd2dq,
            &Assembler::vcvttps2dq, &Assembler::vcvttpd2dq,
            masm);
        break;
      case Express::CVTINTFLT:
        GenerateXMMFltOp(instr,
            &Assembler::vcvtdq2ps, &Assembler::vcvtdq2pd,
            &Assembler::vcvtdq2ps, &Assembler::vcvtdq2pd,
            masm);
        break;
      case Express::SUBINT:
        GenerateXMMFltOp(instr,
            &Assembler::vpsubd, &Assembler::vpsubq,
            &Assembler::vpsubd, &Assembler::vpsubq,
            masm);
        break;
      default: UNSUPPORTED;
    }
  }

  // Generate left/right shift.
  void GenerateShift(Express::Op *instr, MacroAssembler *masm,
                     bool left, int bits) {
    // Make sure source is in a register.
    CHECK(instr->dst != -1);
    int src = instr->src;
    if (instr->src == -1) {
      switch (type_) {
        case DT_FLOAT:
          __ vmovaps(xmm(instr->dst), addr(instr->args[0]));
          break;
        case DT_DOUBLE:
          __ vmovapd(xmm(instr->dst), addr(instr->args[0]));
          break;
        default: UNSUPPORTED;
      }
      src = instr->dst;
    }

    // Shift xmm register.
    switch (type_) {
      case DT_FLOAT:
        if (left) {
          __ vpslld(xmm(instr->dst), xmm(src), bits);
        } else {
          __ vpsrld(xmm(instr->dst), xmm(src), bits);
        }
        break;
      case DT_DOUBLE:
        if (left) {
          __ vpsllq(xmm(instr->dst), xmm(src), bits);
        } else {
          __ vpsrlq(xmm(instr->dst), xmm(src), bits);
        }
        break;
      default: UNSUPPORTED;
    }
  }

  // Generate compare.
  void GenerateCompare(Express::Op *instr, MacroAssembler *masm, int8 code) {
    GenerateXMMFltOp(instr,
        &Assembler::vcmpps, &Assembler::vcmppd,
        &Assembler::vcmpps, &Assembler::vcmppd,
        code, masm);
  }
};

ExpressionGenerator *CreateVectorFltAVX128Generator() {
  return new VectorFltAVX128Generator();
}

}  // namespace myelin
}  // namespace sling

