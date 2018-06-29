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
            &Assembler::andps, &Assembler::andpd,
            &Assembler::andps, &Assembler::andpd,
            masm);
        break;
      case Express::OR:
        GenerateXMMFltOp(instr,
            &Assembler::orps, &Assembler::orpd,
            &Assembler::orps, &Assembler::orpd,
            masm);
        break;
      case Express::ANDNOT:
        if (CPU::Enabled(SSE2)) {
          GenerateXMMFltOp(instr,
              &Assembler::andnps, &Assembler::andnpd,
              &Assembler::andnps, &Assembler::andnpd,
              masm);
        } else {
          UNSUPPORTED;
        }
        break;
      case Express::SHR23:
        GenerateShift(instr, masm, false, 23);
        break;
      case Express::SHL23:
        GenerateShift(instr, masm, true, 23);
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
    if (CPU::Enabled(SSE2)) {
      GenerateXMMFltOp(instr,
          &Assembler::cvttps2dq, &Assembler::cvttpd2dq,
          &Assembler::cvttps2dq, &Assembler::cvttpd2dq,
          masm);
    } else {
      UNSUPPORTED;
    }
  }

  // Generate integer to float conversion.
  void GenerateIntToFlt(Express::Op *instr, MacroAssembler *masm) {
    if (CPU::Enabled(SSE2)) {
      GenerateXMMFltOp(instr,
          &Assembler::cvtdq2ps, &Assembler::cvtdq2pd,
          &Assembler::cvtdq2ps, &Assembler::cvtdq2pd,
          masm);
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

  // Generate code for reduction operation.
  void GenerateReduce(Express::Op *instr, MacroAssembler *masm) override {
    auto acc = xmm(instr->acc);
    auto aux = xmmaux(0);
    switch (type_) {
      case DT_FLOAT:
        switch (instr->type) {
          case Express::SUM:
            __ haddps(acc, acc);
            __ haddps(acc, acc);
            break;
          case Express::PRODUCT:
            __ shufps(aux, acc, 0x0E);
            __ mulps(acc, aux);
            __ shufps(aux, acc, 0x01);
            __ mulps(acc, aux);
            break;
          case Express::MIN:
            __ shufps(aux, acc, 0x0E);
            __ minps(acc, aux);
            __ shufps(aux, acc, 0x01);
            __ minps(acc, aux);
            break;
          case Express::MAX:
            __ shufps(aux, acc, 0x0E);
            __ maxps(acc, aux);
            __ shufps(aux, acc, 0x01);
            __ maxps(acc, aux);
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
        switch (instr->type) {
          case Express::SUM:
            __ shufpd(aux, acc, 1);
            __ addpd(acc, aux);
            break;
          case Express::PRODUCT:
            __ shufpd(aux, acc, 1);
            __ mulpd(acc, aux);
            break;
          case Express::MIN:
            __ shufpd(aux, acc, 1);
            __ minpd(acc, aux);
            break;
          case Express::MAX:
            __ shufpd(aux, acc, 1);
            __ maxpd(acc, aux);
            break;
          default: UNSUPPORTED;
        }
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

ExpressionGenerator *CreateVectorFltSSEGenerator() {
  return new VectorFltSSEGenerator();
}

}  // namespace myelin
}  // namespace sling

