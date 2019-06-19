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

// Generate scalar float expression using SSE and XMM registers.
class ScalarFltSSEGenerator : public ExpressionGenerator {
 public:
  ScalarFltSSEGenerator(Type type) {
    model_.name = "FltSSE";
    model_.mov_reg_reg = true;
    model_.mov_reg_imm = true;
    model_.mov_reg_mem = true;
    model_.mov_mem_reg = true;
    model_.op_reg_reg = true;
    model_.op_reg_imm = true;
    model_.op_reg_mem = true;
    model_.func_reg_reg = true;
    model_.func_reg_imm = true;
    model_.func_reg_mem = true;
    model_.cond_reg_reg_reg = true;
    model_.cond_reg_mem_reg = true;
    model_.cond_reg_reg_mem = true;
    model_.cond_reg_mem_mem = true;
    model_.instruction_set({
      Express::MOV, Express::ADD, Express::SUB, Express::MUL, Express::DIV,
      Express::MINIMUM, Express::MAXIMUM, Express::SQRT,
      Express::CMPEQOQ, Express::CMPNEUQ, Express::CMPLTOQ,
      Express::CMPLEOQ, Express::CMPGTOQ, Express::CMPGEOQ,
      Express::COND, Express::SELECT,
      Express::BITAND, Express::BITOR, Express::BITXOR, Express::BITANDNOT,
      Express::BITEQ, Express::QUADSIGN,
      Express::AND, Express::OR, Express::XOR, Express::ANDNOT,
      Express::FLOOR, Express::CEIL, Express::ROUND, Express::TRUNC,
      Express::CVTFLTINT, Express::CVTINTFLT,
      Express::CVTEXPINT, Express::CVTINTEXP,
      Express::ADDINT, Express::SUBINT,
      Express::SUM, Express::PRODUCT, Express::MIN, Express::MAX,
      Express::ALL, Express::ANY,
    });
    if (type == DT_FLOAT) {
      model_.instruction_set({Express::RECIPROCAL, Express::RSQRT});
    }
  }

  void Reserve() override {
    // Reserve XMM registers.
    index_->ReserveXMMRegisters(instructions_.NumRegs());

    // Allocate auxiliary registers.
    int num_mm_aux = 0;
    int num_rr_aux = 0;
    if (instructions_.Has({
        Express::BITAND, Express::BITOR, Express::BITXOR, Express::BITANDNOT,
        Express::BITEQ, Express::AND, Express::OR, Express::XOR,
        Express::ANDNOT, Express::CVTFLTINT, Express::CVTINTFLT,
        Express::ADDINT, Express::SUBINT, Express::ALL, Express::ANY})) {
      num_mm_aux = std::max(num_mm_aux, 1);
    }
    if (instructions_.Has({Express::SELECT, Express::COND})) {
      num_rr_aux = std::max(num_rr_aux, 1);
    }

    index_->ReserveAuxRegisters(num_rr_aux);
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
              __ xorps(xmm(instr->dst), xmm(instr->dst));
              break;
            case DT_DOUBLE:
              __ xorpd(xmm(instr->dst), xmm(instr->dst));
              break;
            default: UNSUPPORTED;
          }
        } else {
          GenerateXMMScalarFltMove(instr, masm);
        }
        break;
      case Express::ADD:
        GenerateXMMFltOp(instr,
            &Assembler::addss, &Assembler::addsd,
            &Assembler::addss, &Assembler::addsd,
            masm);
        break;
      case Express::SUB:
        GenerateXMMFltOp(instr,
            &Assembler::subss, &Assembler::subsd,
            &Assembler::subss, &Assembler::subsd,
            masm);
        break;
      case Express::MUL:
        GenerateXMMFltOp(instr,
            &Assembler::mulss, &Assembler::mulsd,
            &Assembler::mulss, &Assembler::mulsd,
            masm);
        break;
      case Express::DIV:
        GenerateXMMFltOp(instr,
            &Assembler::divss, &Assembler::divsd,
            &Assembler::divss, &Assembler::divsd,
            masm);
        break;
      case Express::MINIMUM:
        GenerateXMMFltOp(instr,
            &Assembler::minss, &Assembler::minsd,
            &Assembler::minss, &Assembler::minsd,
            masm);
        break;
      case Express::MAXIMUM:
        GenerateXMMFltOp(instr,
            &Assembler::maxss, &Assembler::maxsd,
            &Assembler::maxss, &Assembler::maxsd,
            masm);
        break;
      case Express::SQRT:
        GenerateXMMFltOp(instr,
            &Assembler::sqrtss, &Assembler::sqrtsd,
            &Assembler::sqrtss, &Assembler::sqrtsd,
            masm, 0);
        break;
      case Express::RSQRT:
        GenerateXMMFltOp(instr,
            &Assembler::rsqrtss, &Assembler::rsqrtss,
            &Assembler::rsqrtss, &Assembler::rsqrtss,
            masm, 0);
        break;
      case Express::RECIPROCAL:
        GenerateXMMFltOp(instr,
            &Assembler::rcpss, &Assembler::rcpss,
            &Assembler::rcpss, &Assembler::rcpss,
            masm, 0);
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
      case Express::BITOR:
      case Express::BITXOR:
      case Express::BITANDNOT:
      case Express::BITEQ:
      case Express::AND:
      case Express::OR:
      case Express::XOR:
      case Express::ANDNOT:
        GenerateRegisterOp(instr, masm);
        break;
      case Express::FLOOR:
        GenerateRound(instr, masm, round_down);
        break;
      case Express::CEIL:
        GenerateRound(instr, masm, round_up);
        break;
      case Express::ROUND:
        GenerateRound(instr, masm, round_nearest);
        break;
      case Express::TRUNC:
        GenerateRound(instr, masm, round_to_zero);
        break;
      case Express::CVTFLTINT:
      case Express::CVTINTFLT:
        if (CPU::Enabled(SSE2) && CPU::Enabled(SSE4_1)) {
          GenerateRegisterOp(instr, masm);
        } else {
          UNSUPPORTED;
        }
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
      case Express::SUBINT:
        GenerateRegisterOp(instr, masm);
        break;
      case Express::SUM:
        GenerateXMMFltAccOp(instr,
            &Assembler::addss, &Assembler::addsd,
            &Assembler::addss, &Assembler::addsd,
            masm);
        break;
      case Express::PRODUCT:
        GenerateXMMFltAccOp(instr,
            &Assembler::mulss, &Assembler::mulsd,
            &Assembler::mulss, &Assembler::mulsd,
            masm);
        break;
      case Express::MIN:
        GenerateXMMFltAccOp(instr,
            &Assembler::minss, &Assembler::minsd,
            &Assembler::minss, &Assembler::minsd,
            masm);
        break;
      case Express::MAX:
        GenerateXMMFltAccOp(instr,
            &Assembler::maxss, &Assembler::maxsd,
            &Assembler::maxss, &Assembler::maxsd,
            masm);
        break;
      case Express::ALL:
      case Express::ANY:
        GenerateLogicAccumulate(instr, masm);
        break;
      default:
        LOG(FATAL) << "Unsupported instruction: " << instr->AsInstruction();
    }
  }

  // Generate left/right shift.
  void GenerateShift(Express::Op *instr, MacroAssembler *masm,
                     bool left, int bits) {
    // Move argument to destination register
    CHECK(instr->dst != -1);
    if (instr->src != -1) {
      __ movapd(xmm(instr->dst), xmm(instr->src));
    } else {
      switch (type_) {
        case DT_FLOAT:
          __ movss(xmm(instr->dst), addr(instr->args[0]));
          break;
        case DT_DOUBLE:
          __ movsd(xmm(instr->dst), addr(instr->args[0]));
          break;
        default: UNSUPPORTED;
      }
    }

    // Shift xmm register.
    switch (type_) {
      case DT_FLOAT:
        if (CPU::Enabled(SSE2)) {
          if (left) {
            __ pslld(xmm(instr->dst), bits);
          } else {
            __ psrld(xmm(instr->dst), bits);
          }
        } else {
          UNSUPPORTED;
        }
        break;
      case DT_DOUBLE:
        if (CPU::Enabled(SSE2)) {
          if (left) {
            __ psllq(xmm(instr->dst), bits);
          } else {
            __ psrlq(xmm(instr->dst), bits);
          }
        } else {
          UNSUPPORTED;
        }
        break;
      default: UNSUPPORTED;
    }
  }

  // Generate compare.
  void GenerateCompare(Express::Op *instr, MacroAssembler *masm, int8 code) {
    GenerateXMMFltOp(instr,
        &Assembler::cmpss, &Assembler::cmpsd,
        &Assembler::cmpss, &Assembler::cmpsd,
        code, masm);
  }

  // Generate rounding.
  void GenerateRound(Express::Op *instr, MacroAssembler *masm, int8 code) {
    if (CPU::Enabled(SSE4_1)) {
      GenerateXMMUnaryFltOp(instr,
          &Assembler::roundss, &Assembler::roundsd,
          &Assembler::roundss, &Assembler::roundsd,
          code, masm);
    } else {
      UNSUPPORTED;
    }
  }

  // Generate scalar op that loads memory operands into a register first.
  void GenerateRegisterOp(Express::Op *instr, MacroAssembler *masm) {
    CHECK(instr->dst != -1);
    XMMRegister dst = xmm(instr->dst);
    XMMRegister src;
    if (instr->src != -1) {
      src = xmm(instr->src);
    } else {
      src = xmmaux(0);
    }

    switch (type_) {
      case DT_FLOAT:
        if (instr->src == -1) {
          __ movss(src, addr(instr->args[1]));
        }
        switch (instr->type) {
          case Express::CVTFLTINT:
            __ cvttps2dq(dst, src);
            break;
          case Express::CVTINTFLT:
            __ cvtdq2ps(dst, src);
            break;
          case Express::ADDINT:
            __ paddd(dst, src);
            break;
          case Express::SUBINT:
            __ psubd(dst, src);
            break;
          case Express::BITAND:
          case Express::AND:
            __ andps(dst, src);
            break;
          case Express::BITOR:
          case Express::OR:
            __ orps(dst, src);
            break;
          case Express::XOR:
          case Express::BITXOR:
            __ xorps(dst, src);
            break;
          case Express::ANDNOT:
          case Express::BITANDNOT:
            __ andnps(dst, src);
            break;
          case Express::BITEQ:
            if (CPU::Enabled(SSE2)) {
              __ pcmpeqd(dst, src);
            } else {
              UNSUPPORTED;
            }
            break;
          default: UNSUPPORTED;
        }
        break;
      case DT_DOUBLE:
        if (instr->src == -1) {
          __ movsd(src, addr(instr->args[1]));
        }
        switch (instr->type) {
          case Express::CVTFLTINT:
            __ cvttpd2dq(dst, src);
            __ pmovsxdq(dst, dst);
            break;
          case Express::CVTINTFLT:
            __ cvtdq2pd(dst, src);
            break;
          case Express::ADDINT:
            __ paddq(dst, src);
            break;
          case Express::SUBINT:
            __ psubq(dst, src);
            break;
          case Express::BITAND:
          case Express::AND:
            __ andpd(dst, src);
            break;
          case Express::BITOR:
          case Express::OR:
            __ orpd(dst, src);
            break;
          case Express::XOR:
          case Express::BITXOR:
            __ xorpd(dst, src);
            break;
          case Express::ANDNOT:
          case Express::BITANDNOT:
            __ andnpd(dst, src);
            break;
          case Express::BITEQ:
            if (CPU::Enabled(SSE4_1)) {
              __ pcmpeqq(dst, src);
            } else {
              UNSUPPORTED;
            }
            break;
          default: UNSUPPORTED;
        }
        break;
      default: UNSUPPORTED;
    }
  }

  // Generate conditional.
  void GenerateConditional(Express::Op *instr, MacroAssembler *masm) {
    CHECK(instr->dst != -1);
    CHECK(instr->mask != -1);
    Label l1, l2;
    switch (type_) {
      case DT_FLOAT: {
        __ movd(aux(0), xmm(instr->mask));
        __ testl(aux(0), aux(0));
        __ j(zero, &l1);
        if (instr->src != -1) {
          __ movaps(xmm(instr->dst), xmm(instr->src));
        } else {
          __ movss(xmm(instr->dst), addr(instr->args[1]));
        }
        __ jmp(&l2);
        __ bind(&l1);
        if (instr->src2 != -1) {
          __ movaps(xmm(instr->dst), xmm(instr->src2));
        } else {
          __ movss(xmm(instr->dst), addr(instr->args[2]));
        }
        __ bind(&l2);
        break;
      }
      case DT_DOUBLE: {
        __ movq(aux(0), xmm(instr->mask));
        __ testq(aux(0), aux(0));
        __ j(zero, &l1);
        if (instr->src != -1) {
          __ movapd(xmm(instr->dst), xmm(instr->src));
        } else {
          __ movsd(xmm(instr->dst), addr(instr->args[1]));
        }
        __ jmp(&l2);
        __ bind(&l1);
        if (instr->src2 != -1) {
          __ movapd(xmm(instr->dst), xmm(instr->src2));
        } else {
          __ movsd(xmm(instr->dst), addr(instr->args[2]));
        }
        __ bind(&l2);
        break;
      }
      default: UNSUPPORTED;
    }
  }

  // Generate masked select.
  void GenerateSelect(Express::Op *instr, MacroAssembler *masm) {
    CHECK(instr->dst != -1);
    CHECK(instr->mask != -1);
    Label l1, l2;
    switch (type_) {
      case DT_FLOAT: {
        __ movd(aux(0), xmm(instr->mask));
        __ testl(aux(0), aux(0));
        __ j(not_zero, &l1);
        __ xorps(xmm(instr->dst), xmm(instr->dst));
        if (instr->src == instr->dst) {
          __ bind(&l1);
        } else {
          __ jmp(&l2);
          __ bind(&l1);
          if (instr->src != -1) {
            __ movaps(xmm(instr->dst), xmm(instr->src));
          } else {
            __ movss(xmm(instr->dst), addr(instr->args[1]));
          }
        }
        __ bind(&l2);
        break;
      }
      case DT_DOUBLE: {
        __ movq(aux(0), xmm(instr->mask));
        __ testq(aux(0), aux(0));
        __ j(not_zero, &l1);
        __ xorpd(xmm(instr->dst), xmm(instr->dst));
        if (instr->src == instr->dst) {
          __ bind(&l1);
        } else {
          __ jmp(&l2);
          __ bind(&l1);
          if (instr->src != -1) {
            __ movaps(xmm(instr->dst), xmm(instr->src));
          } else {
            __ movsd(xmm(instr->dst), addr(instr->args[1]));
          }
        }
        __ bind(&l2);
        break;
      }
      default: UNSUPPORTED;
    }
  }

  // Generate logic reduction accumulation.
  void GenerateLogicAccumulate(Express::Op *instr, MacroAssembler *masm) {
    XMMRegister acc = xmm(instr->acc);
    XMMRegister src;
    if (instr->src != -1) {
      src = xmm(instr->src);
    } else {
      src = xmmaux(0);
    }
    switch (type_) {
      case DT_FLOAT:
        if (instr->src == -1) {
          __ movss(src, addr(instr->args[0]));
        }
        switch (instr->type) {
          case Express::ALL:
            __ andps(acc, src);
            break;
          case Express::ANY:
            __ orps(acc, src);
            break;
          default: UNSUPPORTED;
        }
        break;
      case DT_DOUBLE:
        if (instr->src == -1) {
          __ movsd(src, addr(instr->args[0]));
        }
        switch (instr->type) {
          case Express::ALL:
            __ andpd(acc, src);
            break;
          case Express::ANY:
            __ orpd(acc, src);
            break;
          default: UNSUPPORTED;
        }
        break;
      default: UNSUPPORTED;
    }
  }

  // Generate code for reduction operation.
  void GenerateReduce(Express::Op *instr, MacroAssembler *masm) override {
    switch (type_) {
      case DT_FLOAT:
        if (instr->dst != -1) {
          __ movss(xmm(instr->dst), xmm(instr->acc));
        } else {
          __ movss(addr(instr->result), xmm(instr->acc));
        }
        break;
      case DT_DOUBLE:
        if (instr->dst != -1) {
          __ movsd(xmm(instr->dst), xmm(instr->acc));
        } else {
          __ movsd(addr(instr->result), xmm(instr->acc));
        }
        break;
      default: UNSUPPORTED;
    }
  }
};

ExpressionGenerator *CreateScalarFltSSEGenerator(Type type) {
  return new ScalarFltSSEGenerator(type);
}

}  // namespace myelin
}  // namespace sling

