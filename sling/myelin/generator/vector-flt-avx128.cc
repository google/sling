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
  VectorFltAVX128Generator(Type type) {
    model_.name = "VFltAVX128";
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
      model_.instruction_set({
        Express::MULADD132, Express::MULADD213, Express::MULADD231,
        Express::MULSUB132, Express::MULSUB213, Express::MULSUB231,
      });
    }
    model_.cond_reg_reg_reg = true;
    model_.cond_reg_mem_reg = true;
    model_.instruction_set({
      Express::MOV,
      Express::ADD, Express::SUB, Express::MUL, Express::DIV,
      Express::MINIMUM, Express::MAXIMUM,
      Express::CMPEQOQ, Express::CMPNEUQ, Express::CMPLTOQ,
      Express::CMPLEOQ, Express::CMPGTOQ, Express::CMPGEOQ,
      Express::COND, Express::SELECT,
      Express::BITAND, Express::BITOR, Express::BITXOR, Express::BITANDNOT,
      Express::AND, Express::OR, Express::XOR, Express::ANDNOT,
      Express::BITEQ, Express::QUADSIGN, Express::SQRT,
      Express::CVTFLTINT, Express::CVTINTFLT,
      Express::CVTEXPINT, Express::CVTINTEXP,
      Express::FLOOR, Express::CEIL, Express::ROUND, Express::TRUNC,
      Express::ADDINT, Express::SUBINT,
      Express::SUM, Express::PRODUCT, Express::MIN, Express::MAX,
      Express::ALL, Express::ANY,
    });
    if (type == DT_FLOAT) {
      model_.instruction_set({Express::RECIPROCAL, Express::RSQRT});
    }
  }

  int VectorSize() override { return XMMRegSize; }

  void Reserve() override {
    // Reserve XMM registers.
    index_->ReserveXMMRegisters(instructions_.NumRegs());

    // Allocate auxiliary registers.
    int num_mm_aux = 0;
    if (instructions_.Has({
        Express::SUM, Express::PRODUCT, Express::MIN, Express::MAX,
        Express::ALL, Express::ANY})) {
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
      case Express::MINIMUM:
        GenerateXMMFltOp(instr,
            &Assembler::vminps, &Assembler::vminpd,
            &Assembler::vminps, &Assembler::vminpd,
            masm);
        break;
      case Express::MAXIMUM:
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
      case Express::RSQRT:
        GenerateXMMFltOp(instr,
            &Assembler::vrsqrtps, &Assembler::vrsqrtps,
            &Assembler::vrsqrtps, &Assembler::vrsqrtps,
            masm, 0);
        break;
      case Express::RECIPROCAL:
        GenerateXMMFltOp(instr,
            &Assembler::vrcpps, &Assembler::vrcpps,
            &Assembler::vrcpps, &Assembler::vrcpps,
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
      case Express::CMPNEUQ:
        GenerateCompare(instr, masm, CMP_NEQ_UQ);
        break;
      case Express::CMPLTOQ:
        GenerateCompare(instr, masm, CMP_LT_OQ);
        break;
      case Express::CMPLEOQ:
        GenerateCompare(instr, masm, CMP_LE_OQ);
        break;
      case Express::CMPGTOQ:
        GenerateCompare(instr, masm, CMP_GT_OQ);
        break;
      case Express::CMPGEOQ:
        GenerateCompare(instr, masm, CMP_GE_OQ);
        break;
      case Express::COND:
        GenerateConditional(instr, masm);
        break;
      case Express::SELECT:
        GenerateSelect(instr, masm);
        break;
      case Express::BITAND:
      case Express::AND:
        GenerateXMMFltOp(instr,
            &Assembler::vandps, &Assembler::vandpd,
            &Assembler::vandps, &Assembler::vandpd,
            masm);
        break;
      case Express::OR:
      case Express::BITOR:
        GenerateXMMFltOp(instr,
            &Assembler::vorps, &Assembler::vorpd,
            &Assembler::vorps, &Assembler::vorpd,
            masm);
        break;
      case Express::XOR:
      case Express::BITXOR:
        GenerateXMMFltOp(instr,
            &Assembler::vxorps, &Assembler::vxorpd,
            &Assembler::vxorps, &Assembler::vxorpd,
            masm);
        break;
      case Express::ANDNOT:
      case Express::BITANDNOT:
        GenerateXMMFltOp(instr,
            &Assembler::vandnps, &Assembler::vandnpd,
            &Assembler::vandnps, &Assembler::vandnpd,
            masm);
        break;
      case Express::BITEQ:
        GenerateXMMFltOp(instr,
            &Assembler::vpcmpeqd, &Assembler::vpcmpeqq,
            &Assembler::vpcmpeqd, &Assembler::vpcmpeqq,
            masm);
        break;
      case Express::FLOOR:
        GenerateXMMUnaryFltOp(instr,
            &Assembler::vroundps, &Assembler::vroundpd,
            &Assembler::vroundps, &Assembler::vroundpd,
            round_down, masm);
        break;
      case Express::CEIL:
        GenerateXMMUnaryFltOp(instr,
            &Assembler::vroundps, &Assembler::vroundpd,
            &Assembler::vroundps, &Assembler::vroundpd,
            round_up, masm);
        break;
      case Express::ROUND:
        GenerateXMMUnaryFltOp(instr,
            &Assembler::vroundps, &Assembler::vroundpd,
            &Assembler::vroundps, &Assembler::vroundpd,
            round_nearest, masm);
        break;
      case Express::TRUNC:
        GenerateXMMUnaryFltOp(instr,
            &Assembler::vroundps, &Assembler::vroundpd,
            &Assembler::vroundps, &Assembler::vroundpd,
            round_to_zero, masm);
        break;
      case Express::CVTFLTINT:
        GenerateFltToInt(instr, masm);
        break;
      case Express::CVTINTFLT:
        GenerateIntToFlt(instr, masm);
        break;
      case Express::CVTEXPINT:
        GenerateShift(instr, masm, false, type_ == DT_FLOAT ? 23 : 52);
        break;
      case Express::CVTINTEXP:
        GenerateShift(instr, masm, true, type_ == DT_FLOAT ? 23 : 52);
        break;
      case Express::QUADSIGN:
        GenerateShift(instr, masm, true, type_ == DT_FLOAT ? 29 : 61);
        break;
      case Express::ADDINT:
        GenerateXMMFltOp(instr,
            &Assembler::vpaddd, &Assembler::vpaddq,
            &Assembler::vpaddd, &Assembler::vpaddq,
            masm);
        break;
      case Express::SUBINT:
        GenerateXMMFltOp(instr,
            &Assembler::vpsubd, &Assembler::vpsubq,
            &Assembler::vpsubd, &Assembler::vpsubq,
            masm);
        break;
      case Express::SUM:
        GenerateXMMFltAccOp(instr,
            &Assembler::vaddps, &Assembler::vaddpd,
            &Assembler::vaddps, &Assembler::vaddpd,
            masm);
        break;
      case Express::PRODUCT:
        GenerateXMMFltAccOp(instr,
            &Assembler::vmulps, &Assembler::vmulpd,
            &Assembler::vmulps, &Assembler::vmulpd,
            masm);
        break;
      case Express::MIN:
        GenerateXMMFltAccOp(instr,
            &Assembler::vminps, &Assembler::vminpd,
            &Assembler::vminps, &Assembler::vminpd,
            masm);
        break;
      case Express::MAX:
        GenerateXMMFltAccOp(instr,
            &Assembler::vmaxps, &Assembler::vmaxpd,
            &Assembler::vmaxps, &Assembler::vmaxpd,
            masm);
        break;
      case Express::ALL:
        GenerateXMMFltAccOp(instr,
            &Assembler::vandps, &Assembler::vandpd,
            &Assembler::vandps, &Assembler::vandpd,
            masm);
        break;
      case Express::ANY:
        GenerateXMMFltAccOp(instr,
            &Assembler::vorps, &Assembler::vorpd,
            &Assembler::vorps, &Assembler::vorpd,
            masm);
        break;
      default:
        LOG(FATAL) << "Unsupported instruction: " << instr->AsInstruction();
    }
  }

  // Generate float to integer conversion.
  void GenerateFltToInt(Express::Op *instr, MacroAssembler *masm) {
    // Convert four floats or two doubles to int32.
    GenerateXMMFltOp(instr,
        &Assembler::vcvttps2dq, &Assembler::vcvttpd2dq,
        &Assembler::vcvttps2dq, &Assembler::vcvttpd2dq,
        masm);

    // Convert int32 to int64 for doubles.
    if (type_ == DT_DOUBLE) {
      __ vpmovsxdq(xmm(instr->dst), xmm(instr->dst));
    }
  }

  // Generate integer to float conversion.
  void GenerateIntToFlt(Express::Op *instr, MacroAssembler *masm) {
    if (type_ == DT_FLOAT) {
      // Convert four int32s to floats.
      if (instr->src != -1) {
        __ vcvtdq2ps(xmm(instr->dst), xmm(instr->src));
      } else {
        __ vcvtdq2ps(xmm(instr->dst), addr(instr->args[0]));
      }
    } else if (type_ == DT_DOUBLE) {
      // Make sure source is in a register.
      int src = instr->src;
      if (instr->src == -1) {
        __ vmovdqa(xmm(instr->dst), addr(instr->args[0]));
        src = instr->dst;
      }

      // Convert two int64s to two int32s.
      __ vshufps(xmm(instr->dst), xmm(src), xmm(src), 0xD8);

      // Convert two int32s to doubles.
      __ vcvtdq2pd(xmm(instr->dst), xmm(instr->dst));
    } else {
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

  // Generate conditional.
  void GenerateConditional(Express::Op *instr, MacroAssembler *masm) {
    CHECK(instr->dst != -1);
    CHECK(instr->src2 != -1);
    CHECK(instr->mask != -1);
    if (instr->src != -1) {
      // COND dst[mask],src,src2
      switch (type_) {
        case DT_FLOAT:
          __ vblendvps(xmm(instr->dst), xmm(instr->src2), xmm(instr->src),
                       xmm(instr->mask));
          break;
        case DT_DOUBLE:
          __ vblendvpd(xmm(instr->dst), xmm(instr->src2), xmm(instr->src),
                       xmm(instr->mask));
          break;
        default: UNSUPPORTED;
      }
    } else {
      // COND dst[mask],[mem],src2
      switch (type_) {
        case DT_FLOAT:
          __ vblendvps(xmm(instr->dst), xmm(instr->src2), addr(instr->args[1]),
                       xmm(instr->mask));
          break;
        case DT_DOUBLE:
          __ vblendvpd(xmm(instr->dst), xmm(instr->src2), addr(instr->args[1]),
                       xmm(instr->mask));
          break;
        default: UNSUPPORTED;
      }
    }
  }

  // Generate masked select.
  void GenerateSelect(Express::Op *instr, MacroAssembler *masm) {
    CHECK(instr->dst != -1);
    CHECK(instr->mask != -1);
    if (instr->src != -1) {
      // SELECT dst[mask],src
      switch (type_) {
        case DT_FLOAT:
          __ vandps(xmm(instr->dst), xmm(instr->mask), xmm(instr->src));
          break;
        case DT_DOUBLE:
          __ vandpd(xmm(instr->dst), xmm(instr->mask), xmm(instr->src));
          break;
        default: UNSUPPORTED;
      }
    } else {
      // SELECT dst[mask],[mem]
      switch (type_) {
        case DT_FLOAT:
          __ vandps(xmm(instr->dst), xmm(instr->mask), addr(instr->args[1]));
          break;
        case DT_DOUBLE:
          __ vandpd(xmm(instr->dst), xmm(instr->mask), addr(instr->args[1]));
          break;
        default: UNSUPPORTED;
      }
    }
  }

  // Generate code for reduction operation.
  void GenerateReduce(Express::Op *instr, MacroAssembler *masm) override {
    auto acc = xmm(instr->acc);
    auto aux = xmmaux(0);
    __ Reduce(ReduceOp(instr), type_, acc, aux);

    switch (type_) {
      case DT_FLOAT:
        if (instr->dst != -1) {
          __ vmovss(xmm(instr->dst), xmm(instr->dst), xmm(instr->acc));
        } else {
          __ vmovss(addr(instr->result), xmm(instr->acc));
        }
        break;
      case DT_DOUBLE:
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

ExpressionGenerator *CreateVectorFltAVX128Generator(Type type) {
  return new VectorFltAVX128Generator(type);
}

}  // namespace myelin
}  // namespace sling

