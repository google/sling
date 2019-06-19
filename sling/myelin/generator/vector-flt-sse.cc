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

// Generate vector float expression using SSE and XMM registers.
class VectorFltSSEGenerator : public ExpressionGenerator {
 public:
  VectorFltSSEGenerator(Type type) {
    model_.name = "VFltSSE";
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
      Express::MOV,
      Express::ADD, Express::SUB, Express::MUL, Express::DIV,
      Express::MINIMUM, Express::MAXIMUM, Express::SQRT,
      Express::CMPEQOQ, Express::CMPNEUQ, Express::CMPLTOQ,
      Express::CMPLEOQ, Express::CMPGTOQ, Express::CMPGEOQ,
      Express::COND, Express::SELECT,
      Express::AND, Express::OR, Express::XOR, Express::ANDNOT,
      Express::BITAND, Express::BITOR, Express::BITXOR, Express::BITANDNOT,
      Express::BITEQ, Express::QUADSIGN,
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

  int VectorSize() override { return XMMRegSize; }

  void Reserve() override {
    // Reserve XMM registers.
    index_->ReserveXMMRegisters(instructions_.NumRegs());

    // Allocate auxiliary registers.
    int num_mm_aux = 0;
    if (instructions_.Has({Express::SUM, Express::PRODUCT,
                           Express::MIN, Express::MAX,
                           Express::ALL, Express::ANY})) {
      num_mm_aux = std::max(num_mm_aux, 1);
    }
    if (instructions_.Has(Express::COND)) {
      num_mm_aux = std::max(num_mm_aux, 2);
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
              __ xorps(xmm(instr->dst), xmm(instr->dst));
              break;
            case DT_DOUBLE:
              __ xorpd(xmm(instr->dst), xmm(instr->dst));
              break;
            default: UNSUPPORTED;
          }
        } else {
          GenerateXMMVectorMove(instr, masm);
        }
        break;
      case Express::ADD:
        GenerateXMMFltOp(instr,
            &Assembler::addps, &Assembler::addpd,
            &Assembler::addps, &Assembler::addpd,
            masm);
        break;
      case Express::SUB:
        GenerateXMMFltOp(instr,
            &Assembler::subps, &Assembler::subpd,
            &Assembler::subps, &Assembler::subpd,
            masm);
        break;
      case Express::MUL:
        GenerateXMMFltOp(instr,
            &Assembler::mulps, &Assembler::mulpd,
            &Assembler::mulps, &Assembler::mulpd,
            masm);
        break;
      case Express::DIV:
        GenerateXMMFltOp(instr,
            &Assembler::divps, &Assembler::divpd,
            &Assembler::divps, &Assembler::divpd,
            masm);
        break;
      case Express::MINIMUM:
        GenerateXMMFltOp(instr,
            &Assembler::minps, &Assembler::minpd,
            &Assembler::minps, &Assembler::minpd,
            masm);
        break;
      case Express::MAXIMUM:
        GenerateXMMFltOp(instr,
            &Assembler::maxps, &Assembler::maxpd,
            &Assembler::maxps, &Assembler::maxpd,
            masm);
        break;
      case Express::SQRT:
        GenerateXMMFltOp(instr,
            &Assembler::sqrtps, &Assembler::sqrtpd,
            &Assembler::sqrtps, &Assembler::sqrtpd,
            masm, 0);
        break;
      case Express::RSQRT:
        GenerateXMMFltOp(instr,
            &Assembler::rsqrtps, &Assembler::rsqrtps,
            &Assembler::rsqrtps, &Assembler::rsqrtps,
            masm, 0);
        break;
      case Express::RECIPROCAL:
        GenerateXMMFltOp(instr,
            &Assembler::rcpps, &Assembler::rcpps,
            &Assembler::rcpps, &Assembler::rcpps,
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
      case Express::AND:
      case Express::BITAND:
        GenerateXMMFltOp(instr,
            &Assembler::andps, &Assembler::andpd,
            &Assembler::andps, &Assembler::andpd,
            masm);
        break;
      case Express::OR:
      case Express::BITOR:
        GenerateXMMFltOp(instr,
            &Assembler::orps, &Assembler::orpd,
            &Assembler::orps, &Assembler::orpd,
            masm);
        break;
      case Express::XOR:
      case Express::BITXOR:
        GenerateXMMFltOp(instr,
            &Assembler::xorps, &Assembler::xorpd,
            &Assembler::xorps, &Assembler::xorpd,
            masm);
        break;
      case Express::ANDNOT:
      case Express::BITANDNOT:
        GenerateXMMFltOp(instr,
            &Assembler::andnps, &Assembler::andnpd,
            &Assembler::andnps, &Assembler::andnpd,
            masm);
        break;
      case Express::BITEQ:
        if (CPU::Enabled(SSE4_1)) {
          GenerateXMMFltOp(instr,
              &Assembler::pcmpeqd, &Assembler::pcmpeqq,
              &Assembler::pcmpeqd, &Assembler::pcmpeqq,
              masm);
        } else {
          UNSUPPORTED;
        }
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
            &Assembler::paddd, &Assembler::paddq,
            &Assembler::paddd, &Assembler::paddq,
            masm);
        break;
      case Express::SUBINT:
        GenerateXMMFltOp(instr,
            &Assembler::psubd, &Assembler::psubq,
            &Assembler::psubd, &Assembler::psubq,
            masm);
        break;
      case Express::SUM:
        GenerateXMMFltAccOp(instr,
            &Assembler::addps, &Assembler::addpd,
            &Assembler::addps, &Assembler::addpd,
            masm);
        break;
      case Express::PRODUCT:
        GenerateXMMFltAccOp(instr,
            &Assembler::mulps, &Assembler::mulpd,
            &Assembler::mulps, &Assembler::mulpd,
            masm);
        break;
      case Express::MIN:
        GenerateXMMFltAccOp(instr,
            &Assembler::minps, &Assembler::minpd,
            &Assembler::minps, &Assembler::minpd,
            masm);
        break;
      case Express::MAX:
        GenerateXMMFltAccOp(instr,
            &Assembler::maxps, &Assembler::maxpd,
            &Assembler::maxps, &Assembler::maxpd,
            masm);
        break;
      case Express::ALL:
        GenerateXMMFltAccOp(instr,
            &Assembler::andps, &Assembler::andpd,
            &Assembler::andps, &Assembler::andpd,
            masm);
        break;
      case Express::ANY:
        GenerateXMMFltAccOp(instr,
            &Assembler::orps, &Assembler::orpd,
            &Assembler::orps, &Assembler::orpd,
            masm);
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
          __ movaps(xmm(instr->dst), addr(instr->args[0]));
          break;
        case DT_DOUBLE:
          __ movapd(xmm(instr->dst), addr(instr->args[0]));
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

  // Generate rounding.
  void GenerateRound(Express::Op *instr, MacroAssembler *masm, int8 code) {
    if (CPU::Enabled(SSE4_1)) {
      GenerateXMMUnaryFltOp(instr,
          &Assembler::roundps, &Assembler::roundpd,
          &Assembler::roundps, &Assembler::roundpd,
          code, masm);
    } else {
      UNSUPPORTED;
    }
  }

  // Generate float to integer conversion.
  void GenerateFltToInt(Express::Op *instr, MacroAssembler *masm) {
    if (CPU::Enabled(SSE2) && CPU::Enabled(SSE4_1)) {
      GenerateXMMFltOp(instr,
          &Assembler::cvttps2dq, &Assembler::cvttpd2dq,
          &Assembler::cvttps2dq, &Assembler::cvttpd2dq,
          masm);
      if (type_ == DT_DOUBLE) {
        __ pmovsxdq(xmm(instr->dst), xmm(instr->dst));
      }
    } else {
      UNSUPPORTED;
    }
  }

  // Generate integer to float conversion.
  void GenerateIntToFlt(Express::Op *instr, MacroAssembler *masm) {
    if (type_ == DT_FLOAT && CPU::Enabled(SSE2)) {
      // Convert four int32s to floats.
      if (instr->src != -1) {
        __ cvtdq2ps(xmm(instr->dst), xmm(instr->src));
      } else {
        __ cvtdq2ps(xmm(instr->dst), addr(instr->args[0]));
      }
    } else if (type_ == DT_DOUBLE && CPU::Enabled(SSE2)) {
      // Get argument.
      if (instr->src != -1) {
        __ movdqa(xmm(instr->dst), xmm(instr->src));
      } else {
        __ movdqa(xmm(instr->dst), addr(instr->args[0]));
      }

      // Convert two int64s to two int32s.
      __ shufps(xmm(instr->dst), xmm(instr->dst), 0xD8);

      // Convert two int32s to doubles.
      __ cvtdq2pd(xmm(instr->dst), xmm(instr->dst));

    } else {
      UNSUPPORTED;
    }
  }

  // Generate compare.
  void GenerateCompare(Express::Op *instr, MacroAssembler *masm, int8 code) {
    GenerateXMMFltOp(instr,
        &Assembler::cmpps, &Assembler::cmppd,
        &Assembler::cmpps, &Assembler::cmppd,
        code, masm);
  }

  // Generate conditional.
  void GenerateConditional(Express::Op *instr, MacroAssembler *masm) {
    CHECK(instr->dst != -1);
    CHECK(instr->mask != -1);

    // Mask first argument.
    __ movaps(xmmaux(0), xmm(instr->mask));
    switch (type_) {
      case DT_FLOAT:
        if (instr->src != -1) {
          __ andps(xmmaux(0), xmm(instr->src));
        } else {
          __ andps(xmmaux(0), addr(instr->args[1]));
        }
        break;
      case DT_DOUBLE:
        if (instr->src != -1) {
          __ andpd(xmmaux(0), xmm(instr->src));
        } else {
          __ andpd(xmmaux(0), addr(instr->args[1]));
        }
        break;
      default: UNSUPPORTED;
    }

    // Mask second argument.
    __ movaps(xmmaux(1), xmm(instr->mask));
    switch (type_) {
      case DT_FLOAT:
        if (instr->src2 != -1) {
          __ andnps(xmmaux(1), xmm(instr->src2));
        } else {
          __ andnps(xmmaux(1), addr(instr->args[2]));
        }
        break;
      case DT_DOUBLE:
        if (instr->src2 != -1) {
          __ andnpd(xmmaux(1), xmm(instr->src2));
        } else {
          __ andnpd(xmmaux(1), addr(instr->args[2]));
        }
        break;
      default: UNSUPPORTED;
    }

    // Merge masked values.
    __ movaps(xmm(instr->dst), xmmaux(0));
    switch (type_) {
      case DT_FLOAT:
        __ orps(xmm(instr->dst), xmmaux(1));
        break;
      case DT_DOUBLE:
        __ orpd(xmm(instr->dst), xmmaux(1));
        break;
      default: UNSUPPORTED;
    }
  }

  // Generate masked select.
  void GenerateSelect(Express::Op *instr, MacroAssembler *masm) {
    CHECK(instr->dst != -1);
    CHECK(instr->mask != -1);
    switch (type_) {
      case DT_FLOAT:
        if (instr->dst != instr->mask) {
          __ movaps(xmm(instr->dst), xmm(instr->mask));
        }
        if (instr->src != -1) {
          __ andps(xmm(instr->dst), xmm(instr->src));
        } else {
          __ andps(xmm(instr->dst), addr(instr->args[1]));
        }
        break;
      case DT_DOUBLE:
        if (instr->dst != instr->mask) {
          __ movapd(xmm(instr->dst), xmm(instr->mask));
        }
        if (instr->src != -1) {
          __ andpd(xmm(instr->dst), xmm(instr->src));
        } else {
          __ andpd(xmm(instr->dst), addr(instr->args[1]));
        }
        break;
      default: UNSUPPORTED;
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

ExpressionGenerator *CreateVectorFltSSEGenerator(Type type) {
  return new VectorFltSSEGenerator(type);
}

}  // namespace myelin
}  // namespace sling

