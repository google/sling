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

// Generate vector int expression using AVX and YMM registers.
class VectorIntAVX256Generator : public ExpressionGenerator {
 public:
  VectorIntAVX256Generator(Type type) {
    model_.name = "VIntAVX256";
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

  int VectorSize() override { return YMMRegSize; }

  void Reserve() override {
    // Reserve YMM registers for temps.
    index_->ReserveYMMRegisters(instructions_.NumRegs());

    // Allocate auxiliary registers.
    int num_mm_aux = 0;
    if (instructions_.Has(Express::MUL) && type_ == DT_INT8) {
      num_mm_aux = std::max(num_mm_aux, 3);
    }
    index_->ReserveAuxYMMRegisters(num_mm_aux);
  }

  void Generate(Express::Op *instr, MacroAssembler *masm) override {
    switch (instr->type) {
      case Express::MOV:
        if (IsLoadZero(instr) && masm->Enabled(ZEROIDIOM)) {
          // Use XOR to zero register instead of loading constant from memory.
          __ vpxor(ymm(instr->dst), ymm(instr->dst), ymm(instr->dst));
        } else {
          GenerateYMMVectorIntMove(instr, masm);
        }
        break;
      case Express::ADD:
        GenerateYMMIntOp(instr,
            &Assembler::vpaddb, &Assembler::vpaddb,
            &Assembler::vpaddw, &Assembler::vpaddw,
            &Assembler::vpaddd, &Assembler::vpaddd,
            &Assembler::vpaddq, &Assembler::vpaddq,
            masm);
        break;
      case Express::SUB:
        GenerateYMMIntOp(instr,
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
            GenerateYMMIntOp(instr,
                &Assembler::vpmullw, &Assembler::vpmullw,  // dummy
                &Assembler::vpmullw, &Assembler::vpmullw,
                &Assembler::vpmulld, &Assembler::vpmulld,
                &Assembler::vpmulld, &Assembler::vpmulld,  // dummy
                masm);
            break;
          case DT_INT64:
            UNSUPPORTED;
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
          UNSUPPORTED;
        } else {
          GenerateYMMIntOp(instr,
              &Assembler::vpminsb, &Assembler::vpminsb,
              &Assembler::vpminsw, &Assembler::vpminsw,
              &Assembler::vpminsd, &Assembler::vpminsd,
              &Assembler::vpminsd, &Assembler::vpminsd,
              masm);
        }
        break;
      case Express::MAXIMUM:
        if (type_ == DT_INT64) {
          UNSUPPORTED;
        } else {
          GenerateYMMIntOp(instr,
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
    YMMRegister src = ymm(instr->src);
    if (instr->dst == instr->src) {
      src = ymmaux(2);
      __ vmovdqa(src, ymm(instr->src));
    }
    if (instr->src2 != -1) {
      __ vmovdqa(ymmaux(1), ymm(instr->src2));
    } else {
      __ vmovdqa(ymmaux(1), addr(instr->args[1]));
    }

    // Multiply even bytes.
    __ vpmullw(ymm(instr->dst), src, ymmaux(1));

    // Multiply odd bytes.
    __ vpsraw(ymmaux(0), src, 8);
    __ vpsraw(ymmaux(1), ymmaux(1), 8);
    __ vpmullw(ymmaux(0), ymmaux(0), ymmaux(1));
    __ vpsllw(ymmaux(0), ymmaux(0), 8);

    // Combine even and odd results.
    __ vpcmpeqw(ymmaux(1), ymmaux(1), ymmaux(1));
    __ vpsrlw(ymmaux(1), ymmaux(1), 8);  // constant 8 times 0x00FF
    __ vpand(ymm(instr->dst), ymm(instr->dst), ymmaux(1));
    __ vpor(ymm(instr->dst), ymm(instr->dst), ymmaux(0));
  }
};

ExpressionGenerator *CreateVectorIntAVX256Generator(Type type) {
  return new VectorIntAVX256Generator(type);
}

}  // namespace myelin
}  // namespace sling

