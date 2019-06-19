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
  VectorFltAVX512Generator(Type type) {
    model_.name = "VFltAVX512";
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
    model_.predicate_regs = true;
    model_.logic_in_regs = true;
    model_.instruction_set({
      Express::MOV,
      Express::ADD, Express::SUB, Express::MUL, Express::DIV,
      Express::RECIPROCAL, Express::SQRT, Express::RSQRT,
      Express::MINIMUM, Express::MAXIMUM,
      Express::CMPEQOQ, Express::CMPNEUQ, Express::CMPLTOQ,
      Express::CMPLEOQ, Express::CMPGTOQ, Express::CMPGEOQ,
      Express::COND, Express::SELECT, Express::QUADSIGN, Express::BITEQ,
      Express::AND, Express::OR, Express::XOR, Express::ANDNOT, Express::NOT,
      Express::BITAND, Express::BITOR, Express::BITXOR, Express::BITANDNOT,
      Express::FLOOR, Express::CEIL, Express::ROUND, Express::TRUNC,
      Express::CVTFLTINT, Express::CVTINTFLT,
      Express::CVTEXPINT, Express::CVTINTEXP,
      Express::ADDINT, Express::SUBINT,
      Express::SUM, Express::PRODUCT, Express::MIN, Express::MAX,
      Express::ALL, Express::ANY,
    });
  }

  int VectorSize() override { return ZMMRegSize; }

  bool ExtendedRegs() override { return true; }

  void Reserve() override {
    // Reserve ZMM and opmask registers.
    index_->ReserveExpressionRegisters(instructions_);

    // Allocate auxiliary registers.
    int num_mm_aux = 0;
    int num_rr_aux = 0;
    if (instructions_.Has({Express::SUM, Express::PRODUCT,
                           Express::MIN, Express::MAX})) {
      num_mm_aux = std::max(num_mm_aux, 1);
    }
    if (instructions_.Has({Express::ALL, Express::ANY})) {
      num_rr_aux = std::max(num_rr_aux, 1);
    }
    if (instructions_.Has(Express::MOV)) {
      bool pred_mov = false;
      for (auto *op : instructions_.ops()) {
        if (op->result->predicate || op->args[0]->predicate) {
          pred_mov = true;
          break;
        }
      }
      if (pred_mov) num_mm_aux = std::max(num_mm_aux, 1);
    }
    index_->ReserveAuxRegisters(num_rr_aux);
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
      case Express::RSQRT:
        GenerateZMMFltOp(instr,
            &Assembler::vrsqrt14ps, &Assembler::vrsqrt14pd,
            nullptr, nullptr,
            &Assembler::vrsqrt14ps, &Assembler::vrsqrt14pd,
            masm);
        break;
      case Express::RECIPROCAL:
        GenerateZMMFltOp(instr,
            &Assembler::vrcp14ps, &Assembler::vrcp14ps,
            nullptr, nullptr,
            &Assembler::vrcp14ps, &Assembler::vrcp14ps,
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
      case Express::BITXOR:
        GenerateZMMFltOp(instr,
            &Assembler::vxorps, &Assembler::vxorpd,
            nullptr, nullptr,
            &Assembler::vxorps, &Assembler::vxorpd,
            masm);
        break;
      case Express::BITANDNOT:
        GenerateZMMFltOp(instr,
            &Assembler::vandnps, &Assembler::vandnpd,
            nullptr, nullptr,
            &Assembler::vandnps, &Assembler::vandnpd,
            masm);
        break;
      case Express::BITEQ:
        GenerateBitEqual(instr, masm);
        break;
      case Express::FLOOR:
        GenerateZMMFltOp(instr,
            &Assembler::vrndscaleps, &Assembler::vrndscalepd,
            &Assembler::vrndscaleps, &Assembler::vrndscalepd,
            round_down, masm);
        break;
      case Express::CEIL:
        GenerateZMMFltOp(instr,
            &Assembler::vrndscaleps, &Assembler::vrndscalepd,
            &Assembler::vrndscaleps, &Assembler::vrndscalepd,
            round_up, masm);
        break;
      case Express::ROUND:
        GenerateZMMFltOp(instr,
            &Assembler::vrndscaleps, &Assembler::vrndscalepd,
            &Assembler::vrndscaleps, &Assembler::vrndscalepd,
            round_nearest, masm);
        break;
      case Express::TRUNC:
        GenerateZMMFltOp(instr,
            &Assembler::vrndscaleps, &Assembler::vrndscalepd,
            &Assembler::vrndscaleps, &Assembler::vrndscalepd,
            round_to_zero, masm);
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
      case Express::QUADSIGN:
        GenerateShift(instr, masm, true, type_ == DT_FLOAT ? 29 : 61);
        break;
      case Express::ADDINT:
        GenerateZMMFltOp(instr,
            &Assembler::vpaddd, &Assembler::vpaddq,
            nullptr, nullptr,
            &Assembler::vpaddd, &Assembler::vpaddq,
            masm);
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
      case Express::ALL:
        __ kandw(kk(instr->acc), kk(instr->acc), kk(instr->src));
        break;
      case Express::ANY:
        __ korw(kk(instr->acc), kk(instr->acc), kk(instr->src));
        break;
      default:
        LOG(FATAL) << "Unsupported instruction: " << instr->AsInstruction();
    }
  }

  // Generate loading and storing of opmask registers.
  void GeneratePredicateMove(Express::Op *instr, MacroAssembler *masm) {
    if (instr->dst != -1) {
      if (instr->src != -1) {
        __ kmovw(kk(instr->dst), kk(instr->src));
      } else {
        // Handle all zeroes or ones as special case.
        auto *value = instr->args[0];
        bool zeroes = false;
        bool ones = false;
        if (value->type == Express::NUMBER) {
          if (value->id == Express::ZERO) zeroes = true;
          if (value->id == Express::QNAN) ones = true;
        }
        if (zeroes) {
          __ kxorw(kk(instr->dst), kk(instr->dst), kk(instr->dst));
        } else if (ones) {
          __ kxnorw(kk(instr->dst), kk(instr->dst), kk(instr->dst));
        } else {
          switch (type_) {
            case DT_FLOAT:
              __ vmovaps(zmmaux(0), addr(value));
              __ vpmovd2m(kk(instr->dst), zmmaux(0));
              break;
            case DT_DOUBLE:
              __ vmovapd(zmmaux(0), addr(value));
              __ vpmovq2m(kk(instr->dst), zmmaux(0));
              break;
            default: UNSUPPORTED;
          }
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

  // Generate bitwise compare equal.
  void GenerateBitEqual(Express::Op *instr, MacroAssembler *masm) {
    if (instr->src != -1 && instr->src2 != -1) {
      switch (type_) {
        case DT_FLOAT:
          __ vpcmpeqd(kk(instr->dst), zmm(instr->src), zmm(instr->src2));
          break;
        case DT_DOUBLE:
          __ vpcmpeqq(kk(instr->dst), zmm(instr->src), zmm(instr->src2));
          break;
        default: UNSUPPORTED;
      }
    } else if (instr->src != -1 && instr->src2 == -1) {
      switch (type_) {
        case DT_FLOAT:
          __ vpcmpeqd(kk(instr->dst), zmm(instr->src), addr(instr->args[1]));
          break;
        case DT_DOUBLE:
          __ vpcmpeqq(kk(instr->dst), zmm(instr->src), addr(instr->args[1]));
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
    if (instr->preduction()) {
      // Check if all mask bits are set.
      CHECK(instr->dst == -1);
      Label l1;
      switch (type_) {
        case DT_FLOAT:
          __ xorq(aux(0), aux(0));
          __ kortestw(kk(instr->acc), kk(instr->acc));
          __ j(instr->type == Express::ALL ? not_carry : zero, &l1);
          __ decq(aux(0));
          __ bind(&l1);
          __ movl(addr(instr->result), aux(0));
          break;
        case DT_DOUBLE:
          __ xorq(aux(0), aux(0));
          __ kortestb(kk(instr->acc), kk(instr->acc));
          __ j(instr->type == Express::ALL ? not_carry : zero, &l1);
          __ decq(aux(0));
          __ bind(&l1);
          __ movq(addr(instr->result), aux(0));
          break;
        default: UNSUPPORTED;
      }
    } else {
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
  }
};

ExpressionGenerator *CreateVectorFltAVX512Generator(Type type) {
  return new VectorFltAVX512Generator(type);
}

}  // namespace myelin
}  // namespace sling

