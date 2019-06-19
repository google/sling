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

// Generate vector int expression using SSE and XMM registers.
class VectorIntSSEGenerator : public ExpressionGenerator {
 public:
  VectorIntSSEGenerator(Type type) {
    model_.name = "VIntSSE";
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
        num_mm_aux = std::max(num_mm_aux, 2);
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
          __ pxor(xmm(instr->dst), xmm(instr->dst));
        } else {
          GenerateXMMVectorIntMove(instr, masm);
        }
        break;
      case Express::ADD:
        GenerateXMMIntOp(instr,
            &Assembler::paddb, &Assembler::paddb,
            &Assembler::paddw, &Assembler::paddw,
            &Assembler::paddd, &Assembler::paddd,
            &Assembler::paddq, &Assembler::paddq,
            masm);
        break;
      case Express::SUB:
        GenerateXMMIntOp(instr,
            &Assembler::psubb, &Assembler::psubb,
            &Assembler::psubw, &Assembler::psubw,
            &Assembler::psubd, &Assembler::psubd,
            &Assembler::psubq, &Assembler::psubq,  // dummy
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
                &Assembler::pmullw, &Assembler::pmullw,  // dummy
                &Assembler::pmullw, &Assembler::pmullw,
                &Assembler::pmulld, &Assembler::pmulld,  // only sse 4.1
                &Assembler::pmulld, &Assembler::pmulld,  // dummy
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
              &Assembler::pminsb, &Assembler::pminsb,
              &Assembler::pminsw, &Assembler::pminsw,
              &Assembler::pminsd, &Assembler::pminsd,
              &Assembler::pminsd, &Assembler::pminsd,
              masm);
        }
        break;
      case Express::MAXIMUM:
        if (type_ == DT_INT64) {
          GenerateMaxInt64(instr, masm);
        } else {
          GenerateXMMIntOp(instr,
              &Assembler::pmaxsb, &Assembler::pmaxsb,
              &Assembler::pmaxsw, &Assembler::pmaxsw,
              &Assembler::pmaxsd, &Assembler::pmaxsd,
              &Assembler::pmaxsd, &Assembler::pmaxsd,  // dummy
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
    __ movdqa(xmmaux(0), xmm(instr->dst));
    if (instr->src != -1) {
      __ movdqa(xmmaux(1), xmm(instr->src));
    } else {
      __ movdqa(xmmaux(1), addr(instr->args[1]));
    }

    // Multiply even bytes.
    __ pmullw(xmm(instr->dst), xmmaux(1));

    // Multiply odd bytes.
    __ psraw(xmmaux(0), 8);
    __ psraw(xmmaux(1), 8);
    __ pmullw(xmmaux(0), xmmaux(1));
    __ psllw(xmmaux(0), 8);

    // Combine even and odd results.
    __ pcmpeqw(xmmaux(1), xmmaux(1));
    __ psrlw(xmmaux(1), 8);  // constant 8 times 0x00FF
    __ pand(xmm(instr->dst), xmmaux(1));
    __ por(xmm(instr->dst), xmmaux(0));
  }

  // Generate 64-bit mul.
  void GenerateMulInt64(Express::Op *instr, MacroAssembler *masm) {
    // Multiply each XMM element using x86 multiply.
    CHECK(instr->dst != -1);
    XMMRegister src;
    if (instr->src != -1) {
      src = xmm(instr->src);
    } else {
      src = xmmaux(0);
      __ movdqa(src, addr(instr->args[1]));
    }
    for (int n = 0; n < 2; ++n) {
      __ pextrq(aux(0), xmm(instr->dst), n);
      __ pextrq(aux(1), src, n);
      __ imulq(aux(0), aux(1));
      __ pinsrq(xmm(instr->dst), aux(0), n);
    }
  }

  // Generate 64-bit min.
  void GenerateMinInt64(Express::Op *instr, MacroAssembler *masm) {
    CHECK(instr->dst != -1);
    XMMRegister src;
    if (instr->src != -1) {
      src = xmm(instr->src);
    } else {
      src = xmmaux(0);
      __ movdqa(src, addr(instr->args[1]));
    }
    for (int n = 0; n < 2; ++n) {
      __ pextrq(aux(0), xmm(instr->dst), n);
      __ pextrq(aux(1), src, n);
      __ cmpq(aux(0), aux(1));
      __ cmovq(greater, aux(0), aux(1));
      __ pinsrq(xmm(instr->dst), aux(0), n);
    }
  }

  // Generate 64-bit max.
  void GenerateMaxInt64(Express::Op *instr, MacroAssembler *masm) {
    CHECK(instr->dst != -1);
    XMMRegister src;
    if (instr->src != -1) {
      src = xmm(instr->src);
    } else {
      src = xmmaux(0);
      __ movdqa(src, addr(instr->args[1]));
    }
    for (int n = 0; n < 2; ++n) {
      __ pextrq(aux(0), xmm(instr->dst), n);
      __ pextrq(aux(1), src, n);
      __ cmpq(aux(0), aux(1));
      __ cmovq(less, aux(0), aux(1));
      __ pinsrq(xmm(instr->dst), aux(0), n);
    }
  }
};

ExpressionGenerator *CreateVectorIntSSEGenerator(Type type) {
  return new VectorIntSSEGenerator(type);
}

}  // namespace myelin
}  // namespace sling

