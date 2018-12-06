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
    model_.cond_reg_reg_reg = true;
    model_.cond_reg_mem_reg = true;
    model_.predicate_regs = true;
    model_.logic_in_regs = true;
  }

  string Name() override { return "VFltAVX512"; }

  int VectorSize() override { return ZMMRegSize; }

  bool ExtendedRegs() override { return true; }

  void Reserve() override {
    // Reserve ZMM and opmask registers.
    index_->ReserveExpressionRegisters(instructions_);

    // Allocate auxiliary registers.
    int num_mm_aux = 0;
    if (instructions_.Has(Express::MOV) ||
        instructions_.Has(Express::SUM) ||
        instructions_.Has(Express::PRODUCT) ||
        instructions_.Has(Express::MIN) ||
        instructions_.Has(Express::MAX)) {
      num_mm_aux = std::max(num_mm_aux, 1);
    }
    index_->ReserveAuxZMMRegisters(num_mm_aux);
  }

  void Generate(Express::Op *instr, MacroAssembler *masm) override {
    switch (instr->type) {
      case Express::MOV:
        if (instr->result->predicate || instr->args[0]->predicate) {
          GeneratePredicateMove(instr, masm);
        } else if (IsLoadZero(instr) && masm->Enabled(ZEROIDIOM)) {
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
      case Express::MINIMUM:
        GenerateZMMFltOp(instr,
            &Assembler::vminps, &Assembler::vminpd,
            nullptr, nullptr,
            &Assembler::vminps, &Assembler::vminpd,
            masm);
        break;
      case Express::MAXIMUM:
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
      case Express::OR:
      case Express::XOR:
      case Express::ANDNOT:
      case Express::NOT:
        GenerateMaskOp(instr, masm);
        break;
      case Express::BITAND:
        GenerateZMMFltOp(instr,
            &Assembler::vandps, &Assembler::vandpd,
            nullptr, nullptr,
            &Assembler::vandps, &Assembler::vandpd,
            masm);
        break;
      case Express::BITOR:
        GenerateZMMFltOp(instr,
            &Assembler::vorps, &Assembler::vorpd,
            nullptr, nullptr,
            &Assembler::vorps, &Assembler::vorpd,
            masm);
        break;
      case Express::FLOOR:
        GenerateZMMFltOp(instr,
            &Assembler::vrndscaleps, &Assembler::vrndscalepd,
            &Assembler::vrndscaleps, &Assembler::vrndscalepd,
            round_down, masm);
        break;
      case Express::CVTFLTINT:
        GenerateZMMFltOp(instr,
            &Assembler::vcvttps2dq, &Assembler::vcvttpd2qq,
            nullptr, nullptr,
            &Assembler::vcvttps2dq, &Assembler::vcvttpd2qq,
            masm);
        break;
      case Express::CVTINTFLT:
        GenerateZMMFltOp(instr,
            nullptr, nullptr,
            &Assembler::vcvtdq2ps, &Assembler::vcvtqq2pd,
            &Assembler::vcvtdq2ps, &Assembler::vcvtqq2pd,
            masm);
        break;
      case Express::CVTEXPINT:
        GenerateShift(instr, masm, false, type_ == DT_FLOAT ? 23 : 52);
        break;
      case Express::CVTINTEXP:
        GenerateShift(instr, masm, true, type_ == DT_FLOAT ? 23 : 52);
        break;
      case Express::SUBINT:
        GenerateZMMFltOp(instr,
            &Assembler::vpsubd, &Assembler::vpsubq,
            nullptr, nullptr,
            &Assembler::vpsubd, &Assembler::vpsubq,
            masm);
        break;
      case Express::SUM:
        GenerateZMMFltAccOp(instr,
            nullptr, nullptr,
            &Assembler::vaddps, &Assembler::vaddpd,
            &Assembler::vaddps, &Assembler::vaddpd,
            masm);
        break;
      case Express::PRODUCT:
        GenerateZMMFltAccOp(instr,
            nullptr, nullptr,
            &Assembler::vmulps, &Assembler::vmulpd,
            &Assembler::vmulps, &Assembler::vmulpd,
            masm);
        break;
      case Express::MIN:
        GenerateZMMFltAccOp(instr,
            &Assembler::vminps, &Assembler::vminpd,
            nullptr, nullptr,
            &Assembler::vminps, &Assembler::vminpd,
            masm);
        break;
      case Express::MAX:
        GenerateZMMFltAccOp(instr,
            &Assembler::vmaxps, &Assembler::vmaxpd,
            nullptr, nullptr,
            &Assembler::vmaxps, &Assembler::vmaxpd,
            masm);
        break;
      default:
        UNSUPPORTED;
    }
  }

  // Generate loading and storing of opmask registers.
  void GeneratePredicateMove(Express::Op *instr, MacroAssembler *masm) {
    if (instr->dst != -1) {
      if (instr->src != -1) {
        __ kmovw(kk(instr->dst), kk(instr->src));
      } else {
        switch (type_) {
          case DT_FLOAT:
            __ vmovaps(zmmaux(0), addr(instr->args[0]));
            __ vpmovd2m(kk(instr->dst), zmmaux(0));
            break;
          case DT_DOUBLE:
            __ vmovapd(zmmaux(0), addr(instr->args[0]));
            __ vpmovq2m(kk(instr->dst), zmmaux(0));
            break;
          default: UNSUPPORTED;
        }
      }
    } else if (instr->src != -1) {
      switch (type_) {
        case DT_FLOAT:
          __ vpmovm2d(zmmaux(0), kk(instr->src));
          __ vmovaps(addr(instr->result), zmmaux(0));
          break;
        case DT_DOUBLE:
          __ vpmovm2q(zmmaux(0), kk(instr->src));
          __ vmovapd(addr(instr->result), zmmaux(0));
          break;
        default: UNSUPPORTED;
      }
    } else {
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

  // Generate logical op with mask registers.
  void GenerateMaskOp(Express::Op *instr, MacroAssembler *masm) {
    bool unary = instr->arity() == 1;
    CHECK(instr->dst != -1);
    CHECK(instr->src != -1);
    CHECK(unary || instr->src2 != -1);
    switch (instr->type) {
      case Express::AND:
        __ kandw(kk(instr->dst), kk(instr->src), kk(instr->src2));
        break;
      case Express::OR:
        __ korw(kk(instr->dst), kk(instr->src), kk(instr->src2));
        break;
      case Express::XOR:
        __ kxorw(kk(instr->dst), kk(instr->src), kk(instr->src2));
        break;
      case Express::ANDNOT:
        __ kandnw(kk(instr->dst), kk(instr->src), kk(instr->src2));
        break;
      case Express::NOT:
        __ knotw(kk(instr->dst), kk(instr->src));
        break;
      default: UNSUPPORTED;
    }
  }

  // Generate compare.
  void GenerateCompare(Express::Op *instr, MacroAssembler *masm, int8 code) {
    if (instr->src != -1 && instr->src2 != -1) {
      switch (type_) {
        case DT_FLOAT:
          __ vcmpps(kk(instr->dst), zmm(instr->src), zmm(instr->src2), code);
          break;
        case DT_DOUBLE:
          __ vcmppd(kk(instr->dst), zmm(instr->src), zmm(instr->src2), code);
          break;
        default: UNSUPPORTED;
      }
    } else if (instr->src != -1 && instr->src2 == -1) {
      switch (type_) {
        case DT_FLOAT:
          __ vcmpps(kk(instr->dst), zmm(instr->src), addr(instr->args[1]),
                    code);
          break;
        case DT_DOUBLE:
          __ vcmppd(kk(instr->dst), zmm(instr->src), addr(instr->args[1]),
                    code);
          break;
        default: UNSUPPORTED;
      }
    } else {
      UNSUPPORTED;
    }
  }

  // Generate conditional.
  void GenerateConditional(Express::Op *instr, MacroAssembler *masm) {
    CHECK(instr->dst != -1);
    CHECK(instr->src2 != -1);
    CHECK(instr->mask != -1);
    if (instr->src != -1) {
      switch (type_) {
        case DT_FLOAT:
          __ vblendmps(zmm(instr->dst), zmm(instr->src2), zmm(instr->src),
                       Mask(kk(instr->mask), merging));
          break;
        case DT_DOUBLE:
          __ vblendmpd(zmm(instr->dst), zmm(instr->src2), zmm(instr->src),
                       Mask(kk(instr->mask), merging));
          break;
        default: UNSUPPORTED;
      }
    } else {
      switch (type_) {
        case DT_FLOAT:
          __ vblendmps(zmm(instr->dst), zmm(instr->src2), addr(instr->args[1]),
                       Mask(kk(instr->mask), merging));
          break;
        case DT_DOUBLE:
          __ vblendmpd(zmm(instr->dst), zmm(instr->src2), addr(instr->args[1]),
                       Mask(kk(instr->mask), merging));
          break;
        default: UNSUPPORTED;
      }
    }
  }

  // Generate masked select.
  void GenerateSelect(Express::Op *instr, MacroAssembler *masm) {
    CHECK(instr->dst != -1);
    Mask mask(kk(instr->mask), zeroing);
    if (instr->src != -1) {
      switch (type_) {
        case DT_FLOAT:
          __ vmovaps(zmm(instr->dst), zmm(instr->src), mask);
          break;
        case DT_DOUBLE:
          __ vmovapd(zmm(instr->dst), zmm(instr->src), mask);
          break;
        default: UNSUPPORTED;
      }
    } else {
      switch (type_) {
        case DT_FLOAT:
          __ vmovaps(zmm(instr->dst), addr(instr->args[1]), mask);
          break;
        case DT_DOUBLE:
          __ vmovapd(zmm(instr->dst), addr(instr->args[1]), mask);
          break;
        default: UNSUPPORTED;
      }
    }
  }

  // Generate code for reduction operation.
  void GenerateReduce(Express::Op *instr, MacroAssembler *masm) override {
    auto acc = zmm(instr->acc);
    auto aux = zmmaux(0);
    __ Reduce(ReduceOp(instr), type_, acc, aux);

    switch (type_) {
      case DT_FLOAT:
        if (instr->dst != -1) {
          __ vmovss(zmm(instr->dst).x(), zmm(instr->dst).x(),
                    zmm(instr->acc).x());
        } else {
          __ vmovss(addr(instr->result), zmm(instr->acc).x());
        }
        break;
      case DT_DOUBLE:
        if (instr->dst != -1) {
          __ vmovsd(zmm(instr->dst).x(), zmm(instr->dst).x(),
                    zmm(instr->acc).x());
        } else {
          __ vmovsd(addr(instr->result), zmm(instr->acc).x());
        }
        break;
      default: UNSUPPORTED;
    }
  }
};

ExpressionGenerator *CreateVectorFltAVX512Generator() {
  return new VectorFltAVX512Generator();
}

}  // namespace myelin
}  // namespace sling

