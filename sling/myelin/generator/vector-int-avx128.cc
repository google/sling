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

// Generate vector int expression using AVX and XMM registers.
class VectorIntAVX128Generator : public ExpressionGenerator {
 public:
  VectorIntAVX128Generator(Type type) {
    model_.name = "VIntAVX128";
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
    model_.instruction_set({
      Express::MOV,
      Express::ADD, Express::SUB, Express::MUL, Express::DIV,
      Express::MINIMUM, Express::MAXIMUM,
    });
  }

  int VectorSize() override { return XMMRegSize; }

  void Reserve() override {
    // Reserve XMM registers for temps.
    index_->ReserveXMMRegisters(instructions_.NumRegs());

    // Reserve auxiliary registers.
    int num_rr_aux = 0;
    int num_mm_aux = 0;
    if (instructions_.Has(Express::MUL)) {
      if (type_ == DT_INT8) {
        num_mm_aux = std::max(num_mm_aux, 3);
      }
      if (type_ == DT_INT64) {
        num_rr_aux = std::max(num_rr_aux, 2);
        num_mm_aux = std::max(num_mm_aux, 1);
      }
    }
    if (instructions_.Has({Express::MINIMUM, Express::MAXIMUM})) {
      if (type_ == DT_INT64) {
        num_rr_aux = std::max(num_rr_aux, 2);
        num_mm_aux = std::max(num_mm_aux, 1);
      }
    }
    index_->ReserveAuxRegisters(num_rr_aux);
    index_->ReserveAuxXMMRegisters(num_mm_aux);
  }

  void Generate(Express::Op *instr, MacroAssembler *masm) override {
    switch (instr->type) {
      case Express::MOV:
        if (IsLoadZero(instr) && masm->Enabled(ZEROIDIOM)) {
          // Use XOR to zero register instead of loading constant from memory.
          __ vpxor(xmm(instr->dst), xmm(instr->dst), xmm(instr->dst));
        } else {
          GenerateXMMVectorIntMove(instr, masm);
        }
        break;
      case Express::ADD:
        GenerateXMMIntOp(instr,
            &Assembler::vpaddb, &Assembler::vpaddb,
            &Assembler::vpaddw, &Assembler::vpaddw,
            &Assembler::vpaddd, &Assembler::vpaddd,
            &Assembler::vpaddq, &Assembler::vpaddq,
            masm);
        break;
      case Express::SUB:
        GenerateXMMIntOp(instr,
            &Assembler::vpsubb, &Assembler::vpsubb,
            &Assembler::vpsubw, &Assembler::vpsubw,
            &Assembler::vpsubd, &Assembler::vpsubd,
            &Assembler::vpsubq, &Assembler::vpsubq,
            masm);
        break;
      case Express::MUL:
        switch (type_) {
          case DT_INT8:
            GenerateMulInt8(instr, masm);
            break;
          case DT_INT16:
          case DT_INT32:
            GenerateXMMIntOp(instr,
                &Assembler::vpmullw, &Assembler::vpmullw,  // dummy
                &Assembler::vpmullw, &Assembler::vpmullw,
                &Assembler::vpmulld, &Assembler::vpmulld,
                &Assembler::vpmulld, &Assembler::vpmulld,  // dummy
                masm);
            break;
          case DT_INT64:
            GenerateMulInt64(instr, masm);
            break;
          default:
            UNSUPPORTED;
        }
        break;
      case Express::DIV:
        UNSUPPORTED;
        break;
      case Express::MINIMUM:
        if (type_ == DT_INT64) {
          GenerateMinInt64(instr, masm);
        } else {
          GenerateXMMIntOp(instr,
              &Assembler::vpminsb, &Assembler::vpminsb,
              &Assembler::vpminsw, &Assembler::vpminsw,
              &Assembler::vpminsd, &Assembler::vpminsd,
              &Assembler::vpminsd, &Assembler::vpminsd,
              masm);
        }
        break;
      case Express::MAXIMUM:
        if (type_ == DT_INT64) {
          GenerateMaxInt64(instr, masm);
        } else {
          GenerateXMMIntOp(instr,
              &Assembler::vpmaxsb, &Assembler::vpmaxsb,
              &Assembler::vpmaxsw, &Assembler::vpmaxsw,
              &Assembler::vpmaxsd, &Assembler::vpmaxsd,
              &Assembler::vpmaxsd, &Assembler::vpmaxsd,  // dummy
              masm);
        }
        break;
      default:
        LOG(FATAL) << "Unsupported instruction: " << instr->AsInstruction();
    }
  }

  // Generate 8-bit multiply.
  void GenerateMulInt8(Express::Op *instr, MacroAssembler *masm) {
    // Multiply even and odd bytes and merge results.
    // See https://stackoverflow.com/a/29155682 for the details.
    // First load operands.
    CHECK(instr->dst != -1);
    CHECK(instr->src != -1);
    XMMRegister src = xmm(instr->src);
    if (instr->dst == instr->src) {
      src = xmmaux(2);
      __ vmovdqa(src, xmm(instr->src));
    }
    if (instr->src2 != -1) {
      __ vmovdqa(xmmaux(1), xmm(instr->src2));
    } else {
      __ vmovdqa(xmmaux(1), addr(instr->args[1]));
    }

    // Multiply even bytes.
    __ vpmullw(xmm(instr->dst), src, xmmaux(1));

    // Multiply odd bytes.
    __ vpsraw(xmmaux(0), src, 8);
    __ vpsraw(xmmaux(1), xmmaux(1), 8);
    __ vpmullw(xmmaux(0), xmmaux(0), xmmaux(1));
    __ vpsllw(xmmaux(0), xmmaux(0), 8);

    // Combine even and odd results.
    __ vpcmpeqw(xmmaux(1), xmmaux(1), xmmaux(1));
    __ vpsrlw(xmmaux(1), xmmaux(1), 8);  // constant 8 times 0x00FF
    __ vpand(xmm(instr->dst), xmm(instr->dst), xmmaux(1));
    __ vpor(xmm(instr->dst), xmm(instr->dst), xmmaux(0));
  }

  // Generate 64-bit mul.
  void GenerateMulInt64(Express::Op *instr, MacroAssembler *masm) {
    // Multiply each XMM element using x86 multiply.
    CHECK(instr->dst != -1);
    CHECK(instr->src != -1);
    XMMRegister src2;
    if (instr->src2 != -1) {
      src2 = xmm(instr->src2);
    } else {
      src2 = xmmaux(0);
      __ vmovdqa(src2, addr(instr->args[1]));
    }
    for (int n = 0; n < 2; ++n) {
      __ vpextrq(aux(0), xmm(instr->src), n);
      __ vpextrq(aux(1), src2, n);
      __ imulq(aux(0), aux(1));
      __ vpinsrq(xmm(instr->dst), xmm(instr->dst), aux(0), n);
    }
  }

  // Generate 64-bit min.
  void GenerateMinInt64(Express::Op *instr, MacroAssembler *masm) {
    CHECK(instr->dst != -1);
    CHECK(instr->src != -1);
    XMMRegister src2;
    if (instr->src2 != -1) {
      src2 = xmm(instr->src2);
    } else {
      src2 = xmmaux(0);
      __ vmovdqa(src2, addr(instr->args[1]));
    }
    for (int n = 0; n < 2; ++n) {
      __ vpextrq(aux(0), xmm(instr->src), n);
      __ vpextrq(aux(1), src2, n);
      __ cmpq(aux(0), aux(1));
      __ cmovq(greater, aux(0), aux(1));
      __ vpinsrq(xmm(instr->dst), xmm(instr->dst), aux(0), n);
    }
  }

  // Generate 64-bit max.
  void GenerateMaxInt64(Express::Op *instr, MacroAssembler *masm) {
    CHECK(instr->dst != -1);
    CHECK(instr->src != -1);
    XMMRegister src2;
    if (instr->src2 != -1) {
      src2 = xmm(instr->src2);
    } else {
      src2 = xmmaux(0);
      __ vmovdqa(src2, addr(instr->args[1]));
    }
    for (int n = 0; n < 2; ++n) {
      __ vpextrq(aux(0), xmm(instr->src), n);
      __ vpextrq(aux(1), src2, n);
      __ cmpq(aux(0), aux(1));
      __ cmovq(less, aux(0), aux(1));
      __ vpinsrq(xmm(instr->dst), xmm(instr->dst), aux(0), n);
    }
  }
};

ExpressionGenerator *CreateVectorIntAVX128Generator(Type type) {
  return new VectorIntAVX128Generator(type);
}

}  // namespace myelin
}  // namespace sling

