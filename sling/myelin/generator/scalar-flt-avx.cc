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

// Generate scalar float expression using AXV and XMM registers.
class ScalarFltAVXGenerator : public ExpressionGenerator {
 public:
  ScalarFltAVXGenerator() {
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

  string Name() override { return "FltAVX"; }

  void Reserve() override {
    // Reserve XMM registers.
    index_->ReserveXMMRegisters(instructions_.NumRegs());

    // Allocate auxiliary registers.
    int num_mm_aux = 0;
    if (instructions_.Has(Express::AND) ||
        instructions_.Has(Express::OR) ||
        instructions_.Has(Express::ANDNOT) ||
        instructions_.Has(Express::CVTFLTINT) ||
        instructions_.Has(Express::CVTINTFLT) ||
        instructions_.Has(Express::SUBINT)) {
      num_mm_aux = std::max(num_mm_aux, 1);
    }

    index_->ReserveAuxXMMRegisters(num_mm_aux);
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
          GenerateXMMScalarFltMove(instr, masm);
        }
        break;
      case Express::ADD:
        GenerateXMMFltOp(instr,
            &Assembler::vaddss, &Assembler::vaddsd,
            &Assembler::vaddss, &Assembler::vaddsd,
            masm);
        break;
      case Express::SUB:
        GenerateXMMFltOp(instr,
            &Assembler::vsubss, &Assembler::vsubsd,
            &Assembler::vsubss, &Assembler::vsubsd,
            masm);
        break;
      case Express::MUL:
        GenerateXMMFltOp(instr,
            &Assembler::vmulss, &Assembler::vmulsd,
            &Assembler::vmulss, &Assembler::vmulsd,
            masm);
        break;
      case Express::DIV:
        GenerateXMMFltOp(instr,
            &Assembler::vdivss, &Assembler::vdivsd,
            &Assembler::vdivss, &Assembler::vdivsd,
            masm);
        break;
      case Express::MIN:
        GenerateXMMFltOp(instr,
            &Assembler::vminss, &Assembler::vminsd,
            &Assembler::vminss, &Assembler::vminsd,
            masm);
        break;
      case Express::MAX:
        GenerateXMMFltOp(instr,
            &Assembler::vmaxss, &Assembler::vmaxsd,
            &Assembler::vmaxss, &Assembler::vmaxsd,
            masm);
        break;
      case Express::SQRT:
        GenerateSqrt(instr, masm);
        break;
      case Express::MULADD132:
        GenerateXMMFltOp(instr,
            &Assembler::vfmadd132ss, &Assembler::vfmadd132sd,
            &Assembler::vfmadd132ss, &Assembler::vfmadd132sd,
            masm, 2);
        break;
      case Express::MULADD213:
        GenerateXMMFltOp(instr,
            &Assembler::vfmadd213ss, &Assembler::vfmadd213sd,
            &Assembler::vfmadd213ss, &Assembler::vfmadd213sd,
            masm, 2);
        break;
      case Express::MULADD231:
        GenerateXMMFltOp(instr,
            &Assembler::vfmadd231ss, &Assembler::vfmadd231sd,
            &Assembler::vfmadd231ss, &Assembler::vfmadd231sd,
            masm, 2);
        break;
      case Express::MULSUB132:
        GenerateXMMFltOp(instr,
            &Assembler::vfmsub132ss, &Assembler::vfmsub132sd,
            &Assembler::vfmsub132ss, &Assembler::vfmsub132sd,
            masm, 2);
        break;
      case Express::MULSUB213:
        GenerateXMMFltOp(instr,
            &Assembler::vfmsub213ss, &Assembler::vfmsub213sd,
            &Assembler::vfmsub213ss, &Assembler::vfmsub213sd,
            masm, 2);
        break;
      case Express::MULSUB231:
        GenerateXMMFltOp(instr,
            &Assembler::vfmsub231ss, &Assembler::vfmsub231sd,
            &Assembler::vfmsub231ss, &Assembler::vfmsub231sd,
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
      case Express::OR:
      case Express::ANDNOT:
        GenerateRegisterOp(instr, masm);
        break;
      case Express::SHR23:
        GenerateShift(instr, masm, false, 23);
        break;
      case Express::SHL23:
        GenerateShift(instr, masm, true, 23);
        break;
      case Express::FLOOR:
        GenerateRound(instr, masm, round_down);
        break;
      case Express::CVTFLTINT:
      case Express::CVTINTFLT:
        GenerateRegisterOp(instr, masm, true);
        break;
      case Express::SUBINT:
        GenerateRegisterOp(instr, masm);
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
          __ movss(xmm(instr->dst), addr(instr->args[0]));
          break;
        case DT_DOUBLE:
          __ movsd(xmm(instr->dst), addr(instr->args[0]));
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

  // Generate rounding op. Please notice that GenerateXMMFltOp cannot be used
  // here because it is a three register op but the arguments are in dst and
  // src and not src1 and src2.
  void GenerateRound(Express::Op *instr, MacroAssembler *masm, int8 code) {
    if (instr->dst != -1 && instr->src != -1) {
      switch (type_) {
        case DT_FLOAT:
          __ vroundss(xmm(instr->dst), xmm(instr->dst), xmm(instr->src), code);
          break;
        case DT_DOUBLE:
          __ vroundsd(xmm(instr->dst), xmm(instr->dst), xmm(instr->src), code);
          break;
        default: UNSUPPORTED;
      }
    } else if (instr->dst != -1 && instr->src == -1) {
      switch (type_) {
        case DT_FLOAT:
          __ vroundss(xmm(instr->dst), xmm(instr->dst), addr(instr->args[0]),
                      code);
          break;
        case DT_DOUBLE:
          __ vroundsd(xmm(instr->dst), xmm(instr->dst), addr(instr->args[0]),
                      code);
          break;
        default: UNSUPPORTED;
      }
    } else {
      UNSUPPORTED;
    }
  }

  // Generate square root.
  void GenerateSqrt(Express::Op *instr, MacroAssembler *masm) {
    if (instr->dst != -1 && instr->src != -1) {
      switch (type_) {
        case DT_FLOAT:
          __ vsqrtss(xmm(instr->dst), xmm(instr->dst), xmm(instr->src));
          break;
        case DT_DOUBLE:
          __ vsqrtsd(xmm(instr->dst), xmm(instr->dst), xmm(instr->src));
          break;
        default: UNSUPPORTED;
      }
    } else if (instr->dst != -1 && instr->src == -1) {
      switch (type_) {
        case DT_FLOAT:
          __ vsqrtss(xmm(instr->dst), xmm(instr->dst), addr(instr->args[0]));
          break;
        case DT_DOUBLE:
          __ vsqrtsd(xmm(instr->dst), xmm(instr->dst), addr(instr->args[0]));
          break;
        default: UNSUPPORTED;
      }
    } else {
      UNSUPPORTED;
    }
  }

  // Generate compare.
  void GenerateCompare(Express::Op *instr, MacroAssembler *masm, int8 code) {
    GenerateXMMFltOp(instr,
        &Assembler::vcmpss, &Assembler::vcmpsd,
        &Assembler::vcmpss, &Assembler::vcmpsd,
        code, masm);
  }

  // Generate scalar op that loads memory operands into a register first.
  void GenerateRegisterOp(Express::Op *instr, MacroAssembler *masm,
                          bool unary = false) {
    CHECK(instr->dst != -1);
    XMMRegister dst = xmm(instr->dst);
    XMMRegister src;
    XMMRegister src2;
    if (unary) {
      if (instr->src != -1) {
        src = xmm(instr->src);
      } else {
        src = xmmaux(0);
      }
    } else {
      CHECK(instr->src != -1);
      src = xmm(instr->src);
      if (instr->src2 != -1) {
        src2 = xmm(instr->src2);
      } else {
        src2 = xmmaux(0);
      }
    }

    switch (type_) {
      case DT_FLOAT:
        if (unary && instr->src == -1) {
          __ vmovss(src, addr(instr->args[0]));
        } else if (!unary && instr->src2 == -1) {
          __ vmovss(src2, addr(instr->args[1]));
        }
        switch (instr->type) {
          case Express::AND: __ vandps(dst, src, src2); break;
          case Express::OR: __ vorps(dst, src, src2); break;
          case Express::ANDNOT: __ vandnps(dst, src, src2); break;
          case Express::CVTFLTINT: __ vcvttps2dq(dst, src); break;
          case Express::CVTINTFLT: __ vcvtdq2ps(dst, src); break;
          case Express::SUBINT: __ vpsubd(dst, src, src2); break;
          default: UNSUPPORTED;
        }
        break;
      case DT_DOUBLE:
        if (unary && instr->src == -1) {
          __ vmovsd(src, addr(instr->args[0]));
        } else if (!unary && instr->src2 == -1) {
          __ vmovsd(src2, addr(instr->args[1]));
        }
        switch (instr->type) {
          case Express::AND: __ vandpd(dst, src, src2); break;
          case Express::OR: __ vorpd(dst, src, src2); break;
          case Express::ANDNOT: __ vandnpd(dst, src, src2); break;
          case Express::CVTFLTINT: __ vcvttpd2dq(dst, src); break;
          case Express::CVTINTFLT: __ vcvtdq2pd(dst, src); break;
          case Express::SUBINT: __ vpsubq(dst, src, src2); break;
          default: UNSUPPORTED;
        }
        break;
      default: UNSUPPORTED;
    }
  }
};

ExpressionGenerator *CreateScalarFltAVXGenerator() {
  return new ScalarFltAVXGenerator();
}

}  // namespace myelin
}  // namespace sling

