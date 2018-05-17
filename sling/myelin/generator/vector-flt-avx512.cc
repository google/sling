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

// Generate vector float expression using AVX and YMM registers.
class VectorFltAVX512Generator : public ExpressionGenerator {
 public:
  VectorFltAVX512Generator() {
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

  string Name() override { return "VFltAVX512"; }

  int VectorSize() override { return ZMMRegSize; }

  bool ExtendedRegs() override { return true; }

  void Reserve() override {
    // Reserve ZMM registers.
    index_->ReserveZMMRegisters(instructions_.NumRegs());
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
              __ vxorps(zmm(instr->dst), zmm(instr->dst), zmm(instr->dst));
              break;
            case DT_DOUBLE:
              __ vxorpd(zmm(instr->dst), zmm(instr->dst), zmm(instr->dst));
              break;
            default: UNSUPPORTED;
          }
        } else {
          GenerateZMMVectorMove(instr, masm);
        }
        break;
      case Express::ADD:
        GenerateZMMFltOp(instr,
            nullptr, nullptr,
            &Assembler::vaddps, &Assembler::vaddpd,
            &Assembler::vaddps, &Assembler::vaddpd,
            masm);
        break;
      case Express::SUB:
        GenerateZMMFltOp(instr,
            nullptr, nullptr,
            &Assembler::vsubps, &Assembler::vsubpd,
            &Assembler::vsubps, &Assembler::vsubpd,
            masm);
        break;
      case Express::MUL:
        GenerateZMMFltOp(instr,
            nullptr, nullptr,
            &Assembler::vmulps, &Assembler::vmulpd,
            &Assembler::vmulps, &Assembler::vmulpd,
            masm);
        break;
      case Express::DIV:
        GenerateZMMFltOp(instr,
            nullptr, nullptr,
            &Assembler::vdivps, &Assembler::vdivpd,
            &Assembler::vdivps, &Assembler::vdivpd,
            masm);
        break;
      case Express::MIN:
        GenerateZMMFltOp(instr,
            &Assembler::vminps, &Assembler::vminpd,
            nullptr, nullptr,
            &Assembler::vminps, &Assembler::vminpd,
            masm);
        break;
      case Express::MAX:
        GenerateZMMFltOp(instr,
            &Assembler::vmaxps, &Assembler::vmaxpd,
            nullptr, nullptr,
            &Assembler::vmaxps, &Assembler::vmaxpd,
            masm);
        break;
      case Express::SQRT:
        GenerateZMMFltOp(instr,
            nullptr, nullptr,
            &Assembler::vsqrtps, &Assembler::vsqrtpd,
            &Assembler::vsqrtps, &Assembler::vsqrtpd,
            masm);
        break;
      case Express::MULADD132:
        GenerateZMMFltOp(instr,
            nullptr, nullptr,
            &Assembler::vfmadd132ps, &Assembler::vfmadd132pd,
            &Assembler::vfmadd132ps, &Assembler::vfmadd132pd,
            masm, 2);
        break;
      case Express::MULADD213:
        GenerateZMMFltOp(instr,
            nullptr, nullptr,
            &Assembler::vfmadd213ps, &Assembler::vfmadd213pd,
            &Assembler::vfmadd213ps, &Assembler::vfmadd213pd,
            masm, 2);
        break;
      case Express::MULADD231:
        GenerateZMMFltOp(instr,
            nullptr, nullptr,
            &Assembler::vfmadd231ps, &Assembler::vfmadd231pd,
            &Assembler::vfmadd231ps, &Assembler::vfmadd231pd,
            masm, 2);
        break;
      case Express::MULSUB132:
        GenerateZMMFltOp(instr,
            nullptr, nullptr,
            &Assembler::vfmsub132ps, &Assembler::vfmsub132pd,
            &Assembler::vfmsub132ps, &Assembler::vfmsub132pd,
            masm, 2);
        break;
      case Express::MULSUB213:
        GenerateZMMFltOp(instr,
            nullptr, nullptr,
            &Assembler::vfmsub213ps, &Assembler::vfmsub213pd,
            &Assembler::vfmsub213ps, &Assembler::vfmsub213pd,
            masm, 2);
        break;
      case Express::MULSUB231:
        GenerateZMMFltOp(instr,
            nullptr, nullptr,
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
        GenerateZMMFltOp(instr,
            &Assembler::vandps, &Assembler::vandpd,
            nullptr, nullptr,
            &Assembler::vandps, &Assembler::vandpd,
            masm);
        break;
      case Express::OR:
        GenerateZMMFltOp(instr,
            &Assembler::vorps, &Assembler::vorpd,
            nullptr, nullptr,
            &Assembler::vorps, &Assembler::vorpd,
            masm);
        break;
      case Express::ANDNOT:
        GenerateZMMFltOp(instr,
            &Assembler::vandnps, &Assembler::vandnpd,
            nullptr, nullptr,
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
        GenerateZMMFltOp(instr,
            &Assembler::vrndscaleps, &Assembler::vrndscalepd,
            &Assembler::vrndscaleps, &Assembler::vrndscalepd,
            round_down, masm);
        break;
      case Express::CVTFLTINT:
        GenerateZMMFltOp(instr,
            &Assembler::vcvttps2dq, &Assembler::vcvttpd2dq,
            nullptr, nullptr,
            &Assembler::vcvttps2dq, &Assembler::vcvttpd2dq,
            masm);
        break;
      case Express::CVTINTFLT:
        GenerateZMMFltOp(instr,
            nullptr, &Assembler::vcvtdq2pd,
            &Assembler::vcvtdq2ps, nullptr,
            &Assembler::vcvtdq2ps, &Assembler::vcvtdq2pd,
            masm);
        break;
      case Express::SUBINT:
        GenerateZMMFltOp(instr,
            &Assembler::vpsubd, &Assembler::vpsubq,
            nullptr, nullptr,
            &Assembler::vpsubd, &Assembler::vpsubq,
            masm);
        break;
      default:
        UNSUPPORTED;
    }
  }

  // Generate left/right shift.
  void GenerateShift(Express::Op *instr, MacroAssembler *masm,
                     bool left, int bits) {
    if (instr->dst != -1 && instr->src != -1) {
      switch (type_) {
        case DT_FLOAT:
          if (left) {
            __ vpslld(zmm(instr->dst), zmm(instr->src), bits);
          } else {
            __ vpsrld(zmm(instr->dst), zmm(instr->src), bits);
          }
          break;
        case DT_DOUBLE:
          if (left) {
            __ vpsllq(zmm(instr->dst), zmm(instr->src), bits);
          } else {
            __ vpsrlq(zmm(instr->dst), zmm(instr->src), bits);
          }
          break;
        default: UNSUPPORTED;
      }
    } else if (instr->dst != -1 && instr->src == -1) {
      switch (type_) {
        case DT_FLOAT:
          if (left) {
            __ vpslld(zmm(instr->dst), addr(instr->args[0]), bits);
          } else {
            __ vpsrld(zmm(instr->dst), addr(instr->args[0]), bits);
          }
          break;
        case DT_DOUBLE:
          if (left) {
            __ vpsllq(zmm(instr->dst), addr(instr->args[0]), bits);
          } else {
            __ vpsrlq(zmm(instr->dst), addr(instr->args[0]), bits);
          }
          break;
        default: UNSUPPORTED;
      }
    } else {
      UNSUPPORTED;
    }
  }

  // Generate compare.
  void GenerateCompare(Express::Op *instr, MacroAssembler *masm, int8 code) {
    // Allocate mask register.
    OpmaskRegister mask = masm->kk().alloc();

    // Allocate mask.
    auto *ones = masm->GetConstant<int32>(-1, 16);

    // Compare operands.
    if (instr->src != -1 && instr->src2 != -1) {
      switch (type_) {
        case DT_FLOAT:
          __ vcmpps(mask, zmm(instr->src), zmm(instr->src2), code);
          break;
        case DT_DOUBLE:
          __ vcmppd(mask, zmm(instr->src), zmm(instr->src2), code);
          break;
        default: UNSUPPORTED;
      }
    } else if (instr->src != -1 && instr->src2 == -1) {
      switch (type_) {
        case DT_FLOAT:
          __ vcmpps(mask, zmm(instr->src), addr(instr->args[1]), code);
          break;
        case DT_DOUBLE:
          __ vcmppd(mask, zmm(instr->src), addr(instr->args[1]), code);
          break;
        default: UNSUPPORTED;
      }
    } else {
      UNSUPPORTED;
    }

    if (instr->dst != -1) {
      // Put mask into destination register.
      __ vmovaps(zmm(instr->dst), ones->address(), Mask(mask, zeroing));
    } else {
      UNSUPPORTED;
    }
    masm->kk().release(mask);
  }
};

ExpressionGenerator *CreateVectorFltAVX512Generator() {
  return new VectorFltAVX512Generator();
}

}  // namespace myelin
}  // namespace sling

