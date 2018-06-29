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
class VectorFltAVX256Generator : public ExpressionGenerator {
 public:
  VectorFltAVX256Generator() {
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

  string Name() override { return "VFltAVX256"; }

  int VectorSize() override { return YMMRegSize; }

  void Reserve() override {
    // Reserve YMM registers.
    index_->ReserveYMMRegisters(instructions_.NumRegs());

    // Allocate auxiliary registers.
    int num_mm_aux = 0;
    if (!CPU::Enabled(AVX2)) {
      if (instructions_.Has(Express::SHR23) ||
          instructions_.Has(Express::SHL23)) {
        num_mm_aux = std::max(num_mm_aux, 1);
      }
      if (instructions_.Has(Express::SUBINT)) {
        num_mm_aux = std::max(num_mm_aux, 3);
      }
    }
    if (instructions_.Has(Express::SUM) ||
        instructions_.Has(Express::PRODUCT) ||
        instructions_.Has(Express::MIN) ||
        instructions_.Has(Express::MAX)) {
      num_mm_aux = std::max(num_mm_aux, 1);
    }
    index_->ReserveAuxYMMRegisters(num_mm_aux);
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
              __ vxorps(ymm(instr->dst), ymm(instr->dst), ymm(instr->dst));
              break;
            case DT_DOUBLE:
              __ vxorpd(ymm(instr->dst), ymm(instr->dst), ymm(instr->dst));
              break;
            default: UNSUPPORTED;
          }
        } else {
          GenerateYMMVectorMove(instr, masm);
        }
        break;
      case Express::ADD:
        GenerateYMMFltOp(instr,
            &Assembler::vaddps, &Assembler::vaddpd,
            &Assembler::vaddps, &Assembler::vaddpd,
            masm);
        break;
      case Express::SUB:
        GenerateYMMFltOp(instr,
            &Assembler::vsubps, &Assembler::vsubpd,
            &Assembler::vsubps, &Assembler::vsubpd,
            masm);
        break;
      case Express::MUL:
        GenerateYMMFltOp(instr,
            &Assembler::vmulps, &Assembler::vmulpd,
            &Assembler::vmulps, &Assembler::vmulpd,
            masm);
        break;
      case Express::DIV:
        GenerateYMMFltOp(instr,
            &Assembler::vdivps, &Assembler::vdivpd,
            &Assembler::vdivps, &Assembler::vdivpd,
            masm);
        break;
      case Express::MINIMUM:
        GenerateYMMFltOp(instr,
            &Assembler::vminps, &Assembler::vminpd,
            &Assembler::vminps, &Assembler::vminpd,
            masm);
        break;
      case Express::MAXIMUM:
        GenerateYMMFltOp(instr,
            &Assembler::vmaxps, &Assembler::vmaxpd,
            &Assembler::vmaxps, &Assembler::vmaxpd,
            masm);
        break;
      case Express::SQRT:
        GenerateYMMFltOp(instr,
            &Assembler::vsqrtps, &Assembler::vsqrtpd,
            &Assembler::vsqrtps, &Assembler::vsqrtpd,
            masm);
        break;
      case Express::MULADD132:
        GenerateYMMFltOp(instr,
            &Assembler::vfmadd132ps, &Assembler::vfmadd132pd,
            &Assembler::vfmadd132ps, &Assembler::vfmadd132pd,
            masm, 2);
        break;
      case Express::MULADD213:
        GenerateYMMFltOp(instr,
            &Assembler::vfmadd213ps, &Assembler::vfmadd213pd,
            &Assembler::vfmadd213ps, &Assembler::vfmadd213pd,
            masm, 2);
        break;
      case Express::MULADD231:
        GenerateYMMFltOp(instr,
            &Assembler::vfmadd231ps, &Assembler::vfmadd231pd,
            &Assembler::vfmadd231ps, &Assembler::vfmadd231pd,
            masm, 2);
        break;
      case Express::MULSUB132:
        GenerateYMMFltOp(instr,
            &Assembler::vfmsub132ps, &Assembler::vfmsub132pd,
            &Assembler::vfmsub132ps, &Assembler::vfmsub132pd,
            masm, 2);
        break;
      case Express::MULSUB213:
        GenerateYMMFltOp(instr,
            &Assembler::vfmsub213ps, &Assembler::vfmsub213pd,
            &Assembler::vfmsub213ps, &Assembler::vfmsub213pd,
            masm, 2);
        break;
      case Express::MULSUB231:
        GenerateYMMFltOp(instr,
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
        GenerateYMMFltOp(instr,
            &Assembler::vandps, &Assembler::vandpd,
            &Assembler::vandps, &Assembler::vandpd,
            masm);
        break;
      case Express::OR:
        GenerateYMMFltOp(instr,
            &Assembler::vorps, &Assembler::vorpd,
            &Assembler::vorps, &Assembler::vorpd,
            masm);
        break;
      case Express::ANDNOT:
        GenerateYMMFltOp(instr,
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
        GenerateYMMFltOp(instr,
            &Assembler::vroundps, &Assembler::vroundpd,
            &Assembler::vroundps, &Assembler::vroundpd,
            round_down, masm);
        break;
      case Express::CVTFLTINT:
        GenerateYMMFltOp(instr,
            &Assembler::vcvttps2dq, &Assembler::vcvttpd2dq,
            &Assembler::vcvttps2dq, &Assembler::vcvttpd2dq,
            masm);
        break;
      case Express::CVTINTFLT:
        GenerateYMMFltOp(instr,
            &Assembler::vcvtdq2ps, &Assembler::vcvtdq2pd,
            &Assembler::vcvtdq2ps, &Assembler::vcvtdq2pd,
            masm);
        break;
      case Express::SUBINT:
        GenerateIntegerSubtract(instr, masm);
        break;
      case Express::SUM:
        GenerateYMMFltAccOp(instr,
            &Assembler::vaddps, &Assembler::vaddpd,
            &Assembler::vaddps, &Assembler::vaddpd,
            masm);
        break;
      case Express::PRODUCT:
        GenerateYMMFltAccOp(instr,
            &Assembler::vmulps, &Assembler::vmulpd,
            &Assembler::vmulps, &Assembler::vmulpd,
            masm);
        break;
      case Express::MIN:
        GenerateYMMFltAccOp(instr,
            &Assembler::vminps, &Assembler::vminpd,
            &Assembler::vminps, &Assembler::vminpd,
            masm);
        break;
      case Express::MAX:
        GenerateYMMFltAccOp(instr,
            &Assembler::vmaxps, &Assembler::vmaxpd,
            &Assembler::vmaxps, &Assembler::vmaxpd,
            masm);
        break;
      default:
        UNSUPPORTED;
    }
  }

  // Generate left/right shift.
  void GenerateShift(Express::Op *instr, MacroAssembler *masm,
                     bool left, int bits) {
    // Make sure source is in a register.
    CHECK(instr->dst != -1);
    int src = instr->src;
    if (instr->src == -1) {
      GenerateYMMMoveMemToReg(ymm(instr->dst), addr(instr->args[0]), masm);
      src = instr->dst;
    }

    if (CPU::Enabled(AVX2)) {
      // Shift ymm register.
      switch (type_) {
        case DT_FLOAT:
          if (left) {
            __ vpslld(ymm(instr->dst), ymm(src), bits);
          } else {
            __ vpsrld(ymm(instr->dst), ymm(src), bits);
          }
          break;
        case DT_DOUBLE:
          if (left) {
            __ vpsllq(ymm(instr->dst), ymm(src), bits);
          } else {
            __ vpsrlq(ymm(instr->dst), ymm(src), bits);
          }
          break;
        default: UNSUPPORTED;
      }
    } else {
      // Shift ymm register by shifting lo and hi xmm registers.
      __ vextractf128(xmmaux(0), ymm(src), 1);
      if (left) {
        __ vpslld(xmmaux(0), xmmaux(0), bits);
        __ vpslld(xmm(instr->dst), xmm(src), bits);
      } else {
        __ vpsrld(xmmaux(0), xmmaux(0), bits);
        __ vpsrld(xmm(instr->dst), xmm(src), bits);
      }
      __ vinsertf128(ymm(instr->dst), ymm(instr->dst), xmmaux(0), 1);
    }
  }

  // Generate integer subtract.
  void GenerateIntegerSubtract(Express::Op *instr, MacroAssembler *masm) {
    if (CPU::Enabled(AVX2)) {
      GenerateYMMFltOp(instr,
          &Assembler::vpsubd, &Assembler::vpsubq,
          &Assembler::vpsubd, &Assembler::vpsubq,
          masm);
    } else {
      // Move second operand to register.
      CHECK(instr->dst != -1);
      YMMRegister src2;
      if (instr->src2 != -1) {
        src2 = ymm(instr->src2);
      } else {
        GenerateYMMMoveMemToReg(ymmaux(0), addr(instr->args[1]), masm);
        src2 = ymmaux(0);
      }

      // Subtract upper and lower parts separately.
      __ vextractf128(xmmaux(1), ymm(instr->src), 1);
      __ vextractf128(xmmaux(2), src2, 1);
      switch (type_) {
        case DT_FLOAT:
          __ vpsubd(xmmaux(1), xmmaux(1), xmmaux(2));
          __ vpsubd(xmm(instr->dst), xmm(instr->src), src2.xmm());
          break;
        case DT_DOUBLE:
          __ vpsubq(xmmaux(1), xmmaux(1), xmmaux(2));
          __ vpsubq(xmm(instr->dst), xmm(instr->src), src2.xmm());
          break;
        default: UNSUPPORTED;
      }
      __ vinsertf128(ymm(instr->dst), ymm(instr->dst), xmmaux(1), 1);
    }
  }

  // Generate compare.
  void GenerateCompare(Express::Op *instr, MacroAssembler *masm, int8 code) {
    GenerateYMMFltOp(instr,
        &Assembler::vcmpps, &Assembler::vcmppd,
        &Assembler::vcmpps, &Assembler::vcmppd,
        code, masm);
  }

  // Generate code for reduction operation.
  void GenerateReduce(Express::Op *instr, MacroAssembler *masm) override {
    auto acc = ymm(instr->acc);
    auto aux = ymmaux(0);
    switch (type_) {
      case DT_FLOAT:
        switch (instr->type) {
          case Express::SUM:
            __ vperm2f128(aux, acc, acc, 1);
            __ vhaddps(acc, acc, aux);
            __ vhaddps(acc, acc, acc);
            __ vhaddps(acc, acc, acc);
            break;
          case Express::PRODUCT:
            __ vperm2f128(aux, acc, acc, 1);
            __ vmulps(acc, acc, aux);
            __ vpermilps(aux, acc, 0x0E);
            __ vmulps(acc, acc, aux);
            __ vpermilps(aux, acc, 0x01);
            __ vmulps(acc, acc, aux);
            break;
          case Express::MIN:
            __ vperm2f128(aux, acc, acc, 1);
            __ vminps(acc, acc, aux);
            __ vpermilps(aux, acc, 0x0E);
            __ vminps(acc, acc, aux);
            __ vpermilps(aux, acc, 0x01);
            __ vminps(acc, acc, aux);
            break;
          case Express::MAX:
            __ vperm2f128(aux, acc, acc, 1);
            __ vmaxps(acc, acc, aux);
            __ vpermilps(aux, acc, 0x0E);
            __ vmaxps(acc, acc, aux);
            __ vpermilps(aux, acc, 0x01);
            __ vmaxps(acc, acc, aux);
            break;
          default: UNSUPPORTED;
        }
        if (instr->dst != -1) {
          __ vmovss(xmm(instr->dst), xmm(instr->dst), xmm(instr->acc));
        } else {
          __ vmovss(addr(instr->result), xmm(instr->acc));
        }
        break;
      case DT_DOUBLE:
        switch (instr->type) {
          case Express::SUM:
            __ vperm2f128(aux, acc, acc, 1);
            __ vhaddpd(acc, acc, aux);
            __ vhaddpd(acc, acc, acc);
            break;
          case Express::PRODUCT:
            __ vperm2f128(aux, acc, acc, 1);
            __ vmulpd(acc, acc, aux);
            __ vpermilpd(aux, acc, 1);
            __ vmulpd(acc, acc, aux);
            break;
          case Express::MIN:
            __ vperm2f128(aux, acc, acc, 1);
            __ vminpd(acc, acc, aux);
            __ vpermilpd(aux, acc, 1);
            __ vminpd(acc, acc, aux);
          case Express::MAX:
            __ vperm2f128(aux, acc, acc, 1);
            __ vmaxpd(acc, acc, aux);
            __ vpermilpd(aux, acc, 1);
            __ vmaxpd(acc, acc, aux);
            break;
            break;
          default: UNSUPPORTED;
        }
        if (instr->dst != -1) {
          __ vmovsd(xmm(instr->dst), xmm(instr->dst), xmm(instr->acc));
        } else {
          __ vmovsd(addr(instr->result), xmm(instr->acc));
        }
        break;
      default: UNSUPPORTED;
    }
  }
};

ExpressionGenerator *CreateVectorFltAVX256Generator() {
  return new VectorFltAVX256Generator();
}

}  // namespace myelin
}  // namespace sling

