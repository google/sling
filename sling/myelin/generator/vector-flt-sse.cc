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
  VectorFltSSEGenerator() {
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
  }

  string Name() override { return "VFltSSE"; }

  int VectorSize() override { return XMMRegSize; }

  void Reserve() override {
    // Reserve XMM registers.
    index_->ReserveXMMRegisters(instructions_.NumRegs());

    // Allocate auxiliary registers.
    int num_mm_aux = 0;
    if (instructions_.Has(Express::SUM) ||
        instructions_.Has(Express::PRODUCT) ||
        instructions_.Has(Express::MIN) ||
        instructions_.Has(Express::MAX)) {
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
        GenerateXMMFltOp(instr,
            &Assembler::xorps, &Assembler::xorpd,
            &Assembler::xorps, &Assembler::xorpd,
            masm);
        break;
      case Express::ANDNOT:
        GenerateXMMFltOp(instr,
            &Assembler::andnps, &Assembler::andnpd,
            &Assembler::andnps, &Assembler::andnpd,
            masm);
        break;
      case Express::NOT:
        GenerateNot(instr, masm);
        break;
      case Express::FLOOR:
        GenerateFloor(instr, masm);
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
      default:
        LOG(INFO) << "Unsupported: " << instr->AsInstruction();
        UNSUPPORTED;
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

  // Generate floor rounding.
  void GenerateFloor(Express::Op *instr, MacroAssembler *masm) {
    if (CPU::Enabled(SSE4_1)) {
      GenerateXMMFltOp(instr,
          &Assembler::roundps, &Assembler::roundpd,
          &Assembler::roundps, &Assembler::roundpd,
          round_down, masm);
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
      // Make sure source is in a register.
      int src = instr->src;
      if (instr->src == -1) {
        __ movdqa(xmm(instr->dst), addr(instr->args[0]));
        src = instr->dst;
      }

      // Convert two int64s to two int32s.
      __ shufps(xmm(instr->src), xmm(instr->src), 0xD8);

      // Convert two int32s to doubles.
      __ cvtdq2pd(xmm(instr->dst), xmm(src));
    } else {
      UNSUPPORTED;
    }
  }

  // Generate logical not.
  void GenerateNot(Express::Op *instr, MacroAssembler *masm) {
    // Compute not(x) = xor(1,x).
    __ pcmpeqd(xmm(instr->dst), xmm(instr->dst));
    GenerateXMMFltOp(instr,
        &Assembler::xorps, &Assembler::xorpd,
        &Assembler::xorps, &Assembler::xorpd,
        masm);
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
    if (instr->src != -1) {
      __ movaps(xmmaux(0), xmm(instr->src));
    } else {
      __ movaps(xmmaux(0), addr(instr->args[1]));
    }
    switch (type_) {
      case DT_FLOAT:
        __ andps(xmmaux(0), xmm(instr->mask));
        break;
      case DT_DOUBLE:
        __ andpd(xmmaux(0), xmm(instr->mask));
        break;
      default: UNSUPPORTED;
    }

    // Mask second argument.
    if (instr->src2 != -1) {
      __ movaps(xmmaux(1), xmm(instr->src2));
    } else {
      __ movaps(xmmaux(1), addr(instr->args[2]));
    }
    switch (type_) {
      case DT_FLOAT:
        __ andnps(xmmaux(1), xmm(instr->mask));
        break;
      case DT_DOUBLE:
        __ andnpd(xmmaux(1), xmm(instr->mask));
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
        __ movaps(xmm(instr->dst), xmm(instr->mask));
        if (instr->src != -1) {
          __ andps(xmm(instr->dst), xmm(instr->src));
        } else {
          __ andps(xmm(instr->dst), addr(instr->args[1]));
        }
        break;
      case DT_DOUBLE:
        __ movapd(xmm(instr->dst), xmm(instr->mask));
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
    switch (type_) {
      case DT_FLOAT:
        switch (instr->type) {
          case Express::SUM:
            if (CPU::Enabled(SSE3)) {
              __ movshdup(aux, acc);
              __ addps(acc, aux);
              __ movhlps(aux, acc);
              __ addss(acc, aux);
            } else {
              __ movaps(aux, acc);
              __ shufps(aux, acc, 0xB1);
              __ addps(acc, aux);
                if (CPU::Enabled(SSE2)) {
                  __ movhlps(aux, acc);
                } else {
                  __ movaps(aux, acc);
                  __ shufps(aux, acc, 0x03);
                }
              __ addss(acc, aux);
            }
            break;
          case Express::PRODUCT:
            if (CPU::Enabled(SSE3)) {
              __ movshdup(aux, acc);
              __ mulps(acc, aux);
              __ movhlps(aux, acc);
              __ mulss(acc, aux);
            } else {
              __ movaps(aux, acc);
              __ shufps(aux, acc, 0xB1);
              __ mulps(acc, aux);
                if (CPU::Enabled(SSE2)) {
                  __ movhlps(aux, acc);
                } else {
                  __ movaps(aux, acc);
                  __ shufps(aux, acc, 0x03);
                }
              __ mulss(acc, aux);
            }
            break;
          case Express::MIN:
            if (CPU::Enabled(SSE3)) {
              __ movshdup(aux, acc);
              __ minps(acc, aux);
              __ movhlps(aux, acc);
              __ minss(acc, aux);
            } else {
              __ movaps(aux, acc);
              __ shufps(aux, acc, 0xB1);
              __ minps(acc, aux);
                if (CPU::Enabled(SSE2)) {
                  __ movhlps(aux, acc);
                } else {
                  __ movaps(aux, acc);
                  __ shufps(aux, acc, 0x03);
                }
              __ minss(acc, aux);
            }
            break;
          case Express::MAX:
            if (CPU::Enabled(SSE3)) {
              __ movshdup(aux, acc);
              __ maxps(acc, aux);
              __ movhlps(aux, acc);
              __ maxss(acc, aux);
            } else {
              __ movaps(aux, acc);
              __ shufps(aux, acc, 0xB1);
              __ maxps(acc, aux);
                if (CPU::Enabled(SSE2)) {
                  __ movhlps(aux, acc);
                } else {
                  __ movaps(aux, acc);
                  __ shufps(aux, acc, 0x03);
                }
              __ maxss(acc, aux);
            }
            break;
          default: UNSUPPORTED;
        }
        if (instr->dst != -1) {
          __ movss(xmm(instr->dst), xmm(instr->acc));
        } else {
          __ movss(addr(instr->result), xmm(instr->acc));
        }
        break;
      case DT_DOUBLE:
        if (CPU::Enabled(SSE2)) {
          __ movapd(aux, acc);
          __ shufpd(aux, acc, 1);
          switch (instr->type) {
            case Express::SUM:
              __ addsd(acc, aux);
              break;
            case Express::PRODUCT:
              __ mulsd(acc, aux);
              break;
            case Express::MIN:
              __ minsd(acc, aux);
              break;
            case Express::MAX:
              __ maxsd(acc, aux);
              break;
            default: UNSUPPORTED;
          }
          if (instr->dst != -1) {
            __ movsd(xmm(instr->dst), xmm(instr->acc));
          } else {
            __ movsd(addr(instr->result), xmm(instr->acc));
          }
        } else {
          UNSUPPORTED;
        }
        break;
      default: UNSUPPORTED;
    }
  }
};

ExpressionGenerator *CreateVectorFltSSEGenerator() {
  return new VectorFltSSEGenerator();
}

}  // namespace myelin
}  // namespace sling

