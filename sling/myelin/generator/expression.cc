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

#include "sling/myelin/flow.h"
#include "sling/myelin/generator/expression.h"

#define __ masm->

namespace sling {
namespace myelin {

using namespace jit;

// Expression generator factory methods.
ExpressionGenerator *CreateScalarFltSSEGenerator();
ExpressionGenerator *CreateVectorFltSSEGenerator();
ExpressionGenerator *CreateScalarFltAVXGenerator();
ExpressionGenerator *CreateVectorFltAVX128Generator();
ExpressionGenerator *CreateVectorFltAVX256Generator();
ExpressionGenerator *CreateVectorFltAVX512Generator();
ExpressionGenerator *CreateScalarIntGenerator();
ExpressionGenerator *CreateVectorIntSSEGenerator();
ExpressionGenerator *CreateVectorIntAVX128Generator();
ExpressionGenerator *CreateVectorIntAVX256Generator();

void ExpressionGenerator::Initialize(const Express &expression,
                                     Type type,
                                     int spare_regs,
                                     IndexGenerator *index) {
  // Copy expression.
  expression_.Copy(expression);
  type_ = type;
  index_ = index;
  index_->set_extended_regs(ExtendedRegs());

  // Optimize expression.
  bool fma = model_.fm_reg_reg_reg;
  int spare = ImmediateOperands() ? 0 : spare_regs;
  expression_.Optimize(fma, spare);

  // Convert expression to instructions using instruction model.
  CHECK(expression_.Generate(model_, &instructions_));

  // Initialize index generator.
  index->Initialize(VectorSize());

  // Reserve registers.
  Reserve();
}

void ExpressionGenerator::GenerateInit(MacroAssembler *masm) {
  auto &ops = instructions_.ops();
  int body = instructions_.body();
  for (int i = 0; i < body; ++i) {
    if (!ops[i]->nop()) Generate(ops[i], masm);
  }
}

void ExpressionGenerator::GenerateBody(MacroAssembler *masm) {
  auto &ops = instructions_.ops();
  int body = instructions_.body();
  for (int i = body; i < ops.size(); ++i) {
    if (!ops[i]->nop()) Generate(ops[i], masm);
  }
}

int ExpressionGenerator::RegisterNumber(Express::VarType type, int id) const {
  Express::Var *v = instructions_.Lookup(type, id);
  return v == nullptr ? -1 : v->reg;
}

ExpressionGenerator *ExpressionGenerator::Select(const Express &expr,
                                                 Type type, int size) {
  ExpressionGenerator *generator = nullptr;
  switch (type) {
    case DT_FLOAT:
      if (CPU::Enabled(AVX512F)) {
        if (IsVector(size, 16)) {
          generator = CreateVectorFltAVX512Generator();
        } else if (IsVector(size, 8)) {
          generator = CreateVectorFltAVX256Generator();
        } else if (IsVector(size, 4)) {
          generator = CreateVectorFltAVX128Generator();
        } else {
          generator = CreateScalarFltAVXGenerator();
        }
      } else if (CPU::Enabled(AVX)) {
        if (IsVector(size, 8)) {
          generator = CreateVectorFltAVX256Generator();
        } else if (IsVector(size, 4)) {
          generator = CreateVectorFltAVX128Generator();
        } else {
          generator = CreateScalarFltAVXGenerator();
        }
      } else if (CPU::Enabled(SSE)) {
        if (IsVector(size, 4)) {
          generator = CreateVectorFltSSEGenerator();
        } else {
          generator = CreateScalarFltSSEGenerator();
        }
      }
      break;

    case DT_DOUBLE:
      if (CPU::Enabled(AVX512F)) {
        if (IsVector(size, 8)) {
          generator = CreateVectorFltAVX512Generator();
        } else if (IsVector(size, 4)) {
          generator = CreateVectorFltAVX256Generator();
        } else if (IsVector(size, 2)) {
          generator = CreateVectorFltAVX128Generator();
        } else {
          generator = CreateScalarFltAVXGenerator();
        }
      } else if (CPU::Enabled(AVX)) {
        if (IsVector(size, 4)) {
          generator = CreateVectorFltAVX256Generator();
        } else if (IsVector(size, 2)) {
          generator = CreateVectorFltAVX128Generator();
        } else {
          generator = CreateScalarFltAVXGenerator();
        }
      } else if (CPU::Enabled(SSE)) {
        if (CPU::Enabled(SSE2) && IsVector(size, 2)) {
          generator = CreateVectorFltSSEGenerator();
        } else {
          generator = CreateScalarFltSSEGenerator();
        }
      }
      break;

    case DT_INT8:
      if (expr.Has(Express::DIV)) {
        generator = CreateScalarIntGenerator();
      } else if (CPU::Enabled(AVX2) && IsVector(size, 32)) {
        generator = CreateVectorIntAVX256Generator();
      } else if (CPU::Enabled(AVX) && IsVector(size, 16)) {
        generator = CreateVectorIntAVX128Generator();
      } else if (CPU::Enabled(SSE4_1) && IsVector(size, 16)) {
        generator = CreateVectorIntSSEGenerator();
      } else {
        generator = CreateScalarIntGenerator();
      }
      break;

    case DT_INT16:
      if (expr.Has(Express::DIV)) {
        generator = CreateScalarIntGenerator();
      } else if (CPU::Enabled(AVX2) && IsVector(size, 16)) {
        generator = CreateVectorIntAVX256Generator();
      } else if (CPU::Enabled(AVX) && IsVector(size, 8)) {
        generator = CreateVectorIntAVX128Generator();
      } else if (CPU::Enabled(SSE4_1) && IsVector(size, 8)) {
        generator = CreateVectorIntSSEGenerator();
      } else {
        generator = CreateScalarIntGenerator();
      }
      break;

    case DT_INT32:
      if (expr.Has(Express::DIV)) {
        generator = CreateScalarIntGenerator();
      } else if (CPU::Enabled(AVX2) && IsVector(size, 8)) {
        generator = CreateVectorIntAVX256Generator();
      } else if (CPU::Enabled(AVX) && IsVector(size, 4)) {
        generator = CreateVectorIntAVX128Generator();
      } else if (CPU::Enabled(SSE4_1) && IsVector(size, 4)) {
        generator = CreateVectorIntSSEGenerator();
      } else {
        generator = CreateScalarIntGenerator();
      }
      break;

    case DT_INT64:
      if (expr.Has(Express::DIV)) {
        generator = CreateScalarIntGenerator();
      } else if (CPU::Enabled(AVX) && IsVector(size, 2)) {
        generator = CreateVectorIntAVX128Generator();
      } else if (CPU::Enabled(SSE4_1) && IsVector(size, 2)) {
        generator = CreateVectorIntSSEGenerator();
      } else {
        generator = CreateScalarIntGenerator();
      }
      break;

    default:
      generator = nullptr;
  }

  return generator;
}

void ExpressionGenerator::GenerateXMMScalarFltMove(
    Express::Op *instr,
    MacroAssembler *masm) {
  if (instr->dst != -1 && instr->src != -1) {
    // MOV reg,reg
    switch (type_) {
      case DT_FLOAT:
        if (CPU::Enabled(AVX)) {
          __ vmovss(xmm(instr->dst), xmm(instr->dst), xmm(instr->src));
        } else {
          __ movss(xmm(instr->dst), xmm(instr->src));
        }
        break;
      case DT_DOUBLE:
        if (CPU::Enabled(AVX)) {
          __ vmovsd(xmm(instr->dst), xmm(instr->dst), xmm(instr->src));
        } else {
          __ movsd(xmm(instr->dst), xmm(instr->src));
        }
        break;
      default: UNSUPPORTED;
    }
  } else if (instr->dst != -1 && instr->src == -1) {
    // MOV reg,[mem]
    switch (type_) {
      case DT_FLOAT:
        if (CPU::Enabled(AVX)) {
          __ vmovss(xmm(instr->dst), addr(instr->args[0]));
        } else {
          __ movss(xmm(instr->dst), addr(instr->args[0]));
        }
        break;
      case DT_DOUBLE:
        if (CPU::Enabled(AVX)) {
          __ vmovsd(xmm(instr->dst), addr(instr->args[0]));
        } else {
          __ movsd(xmm(instr->dst), addr(instr->args[0]));
        }
        break;
      default: UNSUPPORTED;
    }
  } else if (instr->dst == -1 && instr->src != -1) {
    // MOV [mem],reg
    switch (type_) {
      case DT_FLOAT:
        if (CPU::Enabled(AVX)) {
          __ vmovss(addr(instr->result), xmm(instr->src));
        } else {
          __ movss(addr(instr->result), xmm(instr->src));
        }
        break;
      case DT_DOUBLE:
        if (CPU::Enabled(AVX)) {
          __ vmovsd(addr(instr->result), xmm(instr->src));
        } else {
          __ movsd(addr(instr->result), xmm(instr->src));
        }
        break;
      default: UNSUPPORTED;
    }
  } else {
    UNSUPPORTED;
  }
}

void ExpressionGenerator::GenerateXMMVectorMove(
    Express::Op *instr,
    MacroAssembler *masm) {
  if (instr->dst != -1 && instr->src != -1) {
    // MOV reg,reg
    switch (type_) {
      case DT_FLOAT:
        if (CPU::Enabled(AVX)) {
          __ vmovaps(xmm(instr->dst), xmm(instr->src));
        } else {
          __ movaps(xmm(instr->dst), xmm(instr->src));
        }
        break;
      case DT_DOUBLE:
        if (CPU::Enabled(AVX)) {
          __ vmovapd(xmm(instr->dst), xmm(instr->src));
        } else if (CPU::Enabled(SSE2)) {
          __ movapd(xmm(instr->dst), xmm(instr->src));
        } else {
          UNSUPPORTED;
        }
        break;
      default: UNSUPPORTED;
    }
  } else if (instr->dst != -1 && instr->src == -1) {
    // MOV reg,[mem]
    if (index_->NeedsBroadcast(instr->args[0])) {
      switch (type_) {
        case DT_FLOAT:
          if (CPU::Enabled(AVX)) {
            __ vbroadcastss(xmm(instr->dst), addr(instr->args[0]));
          } else {
            __ movss(xmm(instr->dst), addr(instr->args[0]));
            __ shufps(xmm(instr->dst), xmm(instr->dst), 0);
          }
          break;
        case DT_DOUBLE:
          if (CPU::Enabled(AVX)) {
            __ vbroadcastsd(ymm(instr->dst), addr(instr->args[0]));
          } else if (CPU::Enabled(SSE2)) {
            __ movss(xmm(instr->dst), addr(instr->args[0]));
            __ shufpd(xmm(instr->dst), xmm(instr->dst), 0);
          } else {
            UNSUPPORTED;
          }
          break;
        default: UNSUPPORTED;
      }
    } else {
      switch (type_) {
        case DT_FLOAT:
          if (CPU::Enabled(AVX)) {
            __ vmovaps(xmm(instr->dst), addr(instr->args[0]));
          } else {
            __ movaps(xmm(instr->dst), addr(instr->args[0]));
          }
          break;
        case DT_DOUBLE:
          if (CPU::Enabled(AVX)) {
            __ vmovapd(xmm(instr->dst), addr(instr->args[0]));
          } else if (CPU::Enabled(SSE2)) {
            __ movapd(xmm(instr->dst), addr(instr->args[0]));
          } else {
            UNSUPPORTED;
          }
          break;
        default: UNSUPPORTED;
      }
    }
  } else if (instr->dst == -1 && instr->src != -1) {
    // MOV [mem],reg
    switch (type_) {
      case DT_FLOAT:
        if (CPU::Enabled(AVX)) {
          __ vmovaps(addr(instr->result), xmm(instr->src));
        } else {
          __ movaps(addr(instr->result), xmm(instr->src));
        }
        break;
      case DT_DOUBLE:
        if (CPU::Enabled(AVX)) {
          __ vmovapd(addr(instr->result), xmm(instr->src));
        } else if (CPU::Enabled(SSE2)) {
          __ movapd(addr(instr->result), xmm(instr->src));
        } else {
          UNSUPPORTED;
        }
        break;
      default: UNSUPPORTED;
    }
  } else {
    UNSUPPORTED;
  }
}

void ExpressionGenerator::GenerateYMMMoveMemToReg(
    YMMRegister dst,
    const Operand &src,
    MacroAssembler *masm) {
  switch (type_) {
    case DT_FLOAT:
      __ vmovaps(dst, src);
      break;
    case DT_DOUBLE:
      __ vmovapd(dst, src);
      break;
    default: UNSUPPORTED;
  }
}

void ExpressionGenerator::GenerateYMMVectorMove(
    Express::Op *instr,
    MacroAssembler *masm) {
  if (instr->dst != -1 && instr->src != -1) {
    // MOV reg,reg
    switch (type_) {
      case DT_FLOAT:
        __ vmovaps(ymm(instr->dst), ymm(instr->src));
        break;
      case DT_DOUBLE:
        __ vmovapd(ymm(instr->dst), ymm(instr->src));
        break;
      default: UNSUPPORTED;
    }
  } else if (instr->dst != -1 && instr->src == -1) {
    // MOV reg,[mem]
    if (index_->NeedsBroadcast(instr->args[0])) {
      switch (type_) {
        case DT_FLOAT:
          __ vbroadcastss(ymm(instr->dst), addr(instr->args[0]));
          break;
        case DT_DOUBLE:
          __ vbroadcastsd(ymm(instr->dst), addr(instr->args[0]));
          break;
        default: UNSUPPORTED;
      }
    } else {
      GenerateYMMMoveMemToReg(ymm(instr->dst), addr(instr->args[0]), masm);
    }
  } else if (instr->dst == -1 && instr->src != -1) {
    // MOV [mem],reg
    switch (type_) {
      case DT_FLOAT:
        __ vmovaps(addr(instr->result), ymm(instr->src));
        break;
      case DT_DOUBLE:
        __ vmovapd(addr(instr->result), ymm(instr->src));
        break;
      default: UNSUPPORTED;
    }
  } else {
    UNSUPPORTED;
  }
}

void ExpressionGenerator::GenerateZMMMoveMemToReg(
    ZMMRegister dst,
    const Operand &src,
    MacroAssembler *masm) {
  switch (type_) {
    case DT_FLOAT:
      __ vmovaps(dst, src);
      break;
    case DT_DOUBLE:
      __ vmovapd(dst, src);
      break;
    default: UNSUPPORTED;
  }
}

void ExpressionGenerator::GenerateZMMVectorMove(
    Express::Op *instr,
    MacroAssembler *masm) {
  if (instr->dst != -1 && instr->src != -1) {
    // MOV reg,reg
    switch (type_) {
      case DT_FLOAT:
        __ vmovaps(zmm(instr->dst), zmm(instr->src));
        break;
      case DT_DOUBLE:
        __ vmovapd(zmm(instr->dst), zmm(instr->src));
        break;
      default: UNSUPPORTED;
    }
  } else if (instr->dst != -1 && instr->src == -1) {
    // MOV reg,[mem]
    if (index_->NeedsBroadcast(instr->args[0])) {
      switch (type_) {
        case DT_FLOAT:
          __ vbroadcastss(zmm(instr->dst), addr(instr->args[0]));
          break;
        case DT_DOUBLE:
          __ vbroadcastsd(zmm(instr->dst), addr(instr->args[0]));
          break;
        default: UNSUPPORTED;
      }
    } else {
      GenerateZMMMoveMemToReg(zmm(instr->dst), addr(instr->args[0]), masm);
    }
  } else if (instr->dst == -1 && instr->src != -1) {
    // MOV [mem],reg
    switch (type_) {
      case DT_FLOAT:
        __ vmovaps(addr(instr->result), zmm(instr->src));
        break;
      case DT_DOUBLE:
        __ vmovapd(addr(instr->result), zmm(instr->src));
        break;
      default: UNSUPPORTED;
    }
  } else {
    UNSUPPORTED;
  }
}

void ExpressionGenerator::GenerateIntMoveMemToReg(
    Register dst, const Operand &src,
    MacroAssembler *masm) {
  switch (type_) {
    case DT_INT8:
      __ movb(dst, src);
      break;
    case DT_INT16:
      __ movw(dst, src);
      break;
    case DT_INT32:
      __ movl(dst, src);
      break;
    case DT_INT64:
      __ movq(dst, src);
      break;
    default: UNSUPPORTED;
  }
}

void ExpressionGenerator::GenerateIntMoveRegToMem(
    const Operand &dst, Register src,
    MacroAssembler *masm) {
  switch (type_) {
    case DT_INT8:
      __ movb(dst, src);
      break;
    case DT_INT16:
      __ movw(dst, src);
      break;
    case DT_INT32:
      __ movl(dst, src);
      break;
    case DT_INT64:
      __ movq(dst, src);
      break;
    default: UNSUPPORTED;
  }
}

void ExpressionGenerator::GenerateScalarIntMove(
    Express::Op *instr,
    MacroAssembler *masm) {
  if (instr->dst != -1 && instr->src != -1) {
    // MOV reg,reg
    __ movq(reg(instr->dst), reg(instr->src));
  } else if (instr->dst != -1 && instr->src == -1) {
    // MOV reg,[mem]
    GenerateIntMoveMemToReg(reg(instr->dst), addr(instr->args[0]), masm);
  } else if (instr->dst == -1 && instr->src != -1) {
    // MOV [mem],reg
    GenerateIntMoveRegToMem(addr(instr->result), reg(instr->src), masm);
  } else {
    UNSUPPORTED;
  }
}

void ExpressionGenerator::GenerateXMMVectorIntMove(
    Express::Op *instr,
    MacroAssembler *masm) {
  if (instr->dst != -1 && instr->src != -1) {
    // MOV reg,reg
    if (CPU::Enabled(AVX)) {
      __ vmovdqa(xmm(instr->dst), xmm(instr->src));
    } else if (CPU::Enabled(SSE2)) {
      __ movdqa(xmm(instr->dst), xmm(instr->src));
    } else {
      __ movaps(xmm(instr->dst), xmm(instr->src));
    }
  } else if (instr->dst != -1 && instr->src == -1) {
    // MOV reg,[mem]
    if (index_->NeedsBroadcast(instr->args[0])) {
      // TODO: implement vector int broadcast load.
      UNSUPPORTED;
    } else {
      if (CPU::Enabled(AVX)) {
        __ vmovdqa(xmm(instr->dst), addr(instr->args[0]));
      } else if (CPU::Enabled(SSE2)) {
        __ movdqa(xmm(instr->dst), addr(instr->args[0]));
      } else {
        __ movaps(xmm(instr->dst), addr(instr->args[0]));
      }
    }
  } else if (instr->dst == -1 && instr->src != -1) {
    // MOV [mem],reg
    if (CPU::Enabled(AVX)) {
      __ vmovdqa(addr(instr->result), xmm(instr->src));
    } else if (CPU::Enabled(SSE2)) {
      __ movdqa(addr(instr->result), xmm(instr->src));
    } else {
      __ movaps(addr(instr->result), xmm(instr->src));
    }
  } else {
    UNSUPPORTED;
  }
}

void ExpressionGenerator::GenerateYMMVectorIntMove(
    Express::Op *instr,
    MacroAssembler *masm) {
  if (instr->dst != -1 && instr->src != -1) {
    // MOV reg,reg
    __ vmovdqa(ymm(instr->dst), ymm(instr->src));
  } else if (instr->dst != -1 && instr->src == -1) {
    // MOV reg,[mem]
    if (index_->NeedsBroadcast(instr->args[0])) {
      // TODO: implement vector int broadcast load.
      UNSUPPORTED;
    } else {
      __ vmovdqa(ymm(instr->dst), addr(instr->args[0]));
    }
  } else if (instr->dst == -1 && instr->src != -1) {
    // MOV [mem],reg
    __ vmovdqa(addr(instr->result), ymm(instr->src));
  } else {
    UNSUPPORTED;
  }
}

void ExpressionGenerator::GenerateXMMFltOp(
    Express::Op *instr,
    OpXMMRegReg fltopreg, OpXMMRegReg dblopreg,
    OpXMMRegMem fltopmem, OpXMMRegMem dblopmem,
    MacroAssembler *masm, int argnum) {
  if (instr->dst != -1 && instr->src != -1) {
    // OP reg,reg
    switch (type_) {
      case DT_FLOAT:
        (masm->*fltopreg)(xmm(instr->dst), xmm(instr->src));
        break;
      case DT_DOUBLE:
        (masm->*dblopreg)(xmm(instr->dst), xmm(instr->src));
        break;
      default: UNSUPPORTED;
    }
  } else if (instr->dst != -1 && instr->src == -1) {
    // OP reg,[mem]
    switch (type_) {
      case DT_FLOAT:
        (masm->*fltopmem)(xmm(instr->dst), addr(instr->args[argnum]));
        break;
      case DT_DOUBLE:
        (masm->*dblopmem)(xmm(instr->dst), addr(instr->args[argnum]));
        break;
      default: UNSUPPORTED;
    }
  } else {
    UNSUPPORTED;
  }
}

void ExpressionGenerator::GenerateXMMFltOp(
    Express::Op *instr,
    OpXMMRegRegImm fltopreg, OpXMMRegRegImm dblopreg,
    OpXMMRegMemImm fltopmem, OpXMMRegMemImm dblopmem,
    int8 imm,
    MacroAssembler *masm) {
  if (instr->dst != -1 && instr->src != -1) {
    // OP reg,reg
    switch (type_) {
      case DT_FLOAT:
        (masm->*fltopreg)(xmm(instr->dst), xmm(instr->src), imm);
        break;
      case DT_DOUBLE:
        (masm->*dblopreg)(xmm(instr->dst), xmm(instr->src), imm);
        break;
      default: UNSUPPORTED;
    }
  } else if (instr->dst != -1 && instr->src == -1) {
    // OP reg,[mem]
    switch (type_) {
      case DT_FLOAT:
        (masm->*fltopmem)(xmm(instr->dst), addr(instr->args[1]), imm);
        break;
      case DT_DOUBLE:
        (masm->*dblopmem)(xmm(instr->dst), addr(instr->args[1]), imm);
        break;
      default: UNSUPPORTED;
    }
  } else {
    UNSUPPORTED;
  }
}

void ExpressionGenerator::GenerateXMMUnaryFltOp(
    Express::Op *instr,
    OpXMMRegRegReg fltopreg, OpXMMRegRegReg dblopreg,
    OpXMMRegRegMem fltopmem, OpXMMRegRegMem dblopmem,
    MacroAssembler *masm) {
  if (instr->dst != -1 && instr->src != -1) {
    // OP reg,reg,reg
    switch (type_) {
      case DT_FLOAT:
        (masm->*fltopreg)(xmm(instr->dst), xmm(instr->dst), xmm(instr->src));
        break;
      case DT_DOUBLE:
        (masm->*dblopreg)(xmm(instr->dst), xmm(instr->dst), xmm(instr->src));
        break;
      default: UNSUPPORTED;
    }
  } else if (instr->dst != -1 && instr->src == -1) {
    // OP reg,reg,[mem]
    switch (type_) {
      case DT_FLOAT:
        (masm->*fltopmem)(xmm(instr->dst), xmm(instr->dst),
                          addr(instr->args[0]));
        break;
      case DT_DOUBLE:
        (masm->*dblopmem)(xmm(instr->dst), xmm(instr->dst),
                          addr(instr->args[0]));
        break;
      default: UNSUPPORTED;
    }
  } else {
    UNSUPPORTED;
  }
}

void ExpressionGenerator::GenerateXMMFltOp(
    Express::Op *instr,
    OpXMMRegRegReg fltopreg, OpXMMRegRegReg dblopreg,
    OpXMMRegRegMem fltopmem, OpXMMRegRegMem dblopmem,
    MacroAssembler *masm, int argnum) {
  if (instr->dst != -1 && instr->src != -1 && instr->src2 != -1) {
    // OP reg,reg,reg
    switch (type_) {
      case DT_FLOAT:
        (masm->*fltopreg)(xmm(instr->dst), xmm(instr->src), xmm(instr->src2));
        break;
      case DT_DOUBLE:
        (masm->*dblopreg)(xmm(instr->dst), xmm(instr->src), xmm(instr->src2));
        break;
      default: UNSUPPORTED;
    }
  } else if (instr->dst != -1 && instr->src != -1 && instr->src2 == -1) {
    // OP reg,reg,[mem]
    switch (type_) {
      case DT_FLOAT:
        (masm->*fltopmem)(xmm(instr->dst), xmm(instr->src),
                          addr(instr->args[argnum]));
        break;
      case DT_DOUBLE:
        (masm->*dblopmem)(xmm(instr->dst), xmm(instr->src),
                          addr(instr->args[argnum]));
        break;
      default: UNSUPPORTED;
    }
  } else {
    UNSUPPORTED;
  }
}

void ExpressionGenerator::GenerateXMMFltOp(
    Express::Op *instr,
    OpXMMRegRegRegImm fltopreg, OpXMMRegRegRegImm dblopreg,
    OpXMMRegRegMemImm fltopmem, OpXMMRegRegMemImm dblopmem,
    int8 imm,
    MacroAssembler *masm, int argnum) {
  if (instr->dst != -1 && instr->src != -1 && instr->src2 != -1) {
    // OP reg,reg,reg
    switch (type_) {
      case DT_FLOAT:
        (masm->*fltopreg)(xmm(instr->dst), xmm(instr->src), xmm(instr->src2),
                          imm);
        break;
      case DT_DOUBLE:
        (masm->*dblopreg)(xmm(instr->dst), xmm(instr->src), xmm(instr->src2),
                          imm);
        break;
      default: UNSUPPORTED;
    }
  } else if (instr->dst != -1 && instr->src != -1 && instr->src2 == -1) {
    // OP reg,reg,[mem]
    switch (type_) {
      case DT_FLOAT:
        (masm->*fltopmem)(xmm(instr->dst), xmm(instr->src),
                          addr(instr->args[argnum]), imm);
        break;
      case DT_DOUBLE:
        (masm->*dblopmem)(xmm(instr->dst), xmm(instr->src),
                          addr(instr->args[argnum]), imm);
        break;
      default: UNSUPPORTED;
    }
  } else {
    UNSUPPORTED;
  }
}

void ExpressionGenerator::GenerateYMMUnaryFltOp(
    Express::Op *instr,
    OpYMMRegRegReg fltopreg, OpYMMRegRegReg dblopreg,
    OpYMMRegRegMem fltopmem, OpYMMRegRegMem dblopmem,
    MacroAssembler *masm) {
  if (instr->dst != -1 && instr->src != -1) {
    // OP reg,reg
    switch (type_) {
      case DT_FLOAT:
        (masm->*fltopreg)(ymm(instr->dst), ymm(instr->dst), ymm(instr->src));
        break;
      case DT_DOUBLE:
        (masm->*dblopreg)(ymm(instr->dst), ymm(instr->dst), ymm(instr->src));
        break;
      default: UNSUPPORTED;
    }
  } else if (instr->dst != -1 && instr->src == -1) {
    // OP reg,[mem]
    switch (type_) {
      case DT_FLOAT:
        (masm->*fltopmem)(ymm(instr->dst), ymm(instr->dst),
                          addr(instr->args[0]));
        break;
      case DT_DOUBLE:
        (masm->*dblopmem)(ymm(instr->dst), ymm(instr->dst),
                          addr(instr->args[0]));
        break;
      default: UNSUPPORTED;
    }
  } else {
    UNSUPPORTED;
  }
}

void ExpressionGenerator::GenerateYMMFltOp(
    Express::Op *instr,
    OpYMMRegReg fltopreg, OpYMMRegReg dblopreg,
    OpYMMRegMem fltopmem, OpYMMRegMem dblopmem,
    MacroAssembler *masm, int argnum) {
  if (instr->dst != -1 && instr->src != -1) {
    // OP reg,reg
    switch (type_) {
      case DT_FLOAT:
        (masm->*fltopreg)(ymm(instr->dst), ymm(instr->src));
        break;
      case DT_DOUBLE:
        (masm->*dblopreg)(ymm(instr->dst), ymm(instr->src));
        break;
      default: UNSUPPORTED;
    }
  } else if (instr->dst != -1 && instr->src == -1) {
    // OP reg,[mem]
    switch (type_) {
      case DT_FLOAT:
        (masm->*fltopmem)(ymm(instr->dst), addr(instr->args[argnum]));
        break;
      case DT_DOUBLE:
        (masm->*dblopmem)(ymm(instr->dst), addr(instr->args[argnum]));
        break;
      default: UNSUPPORTED;
    }
  } else {
    UNSUPPORTED;
  }
}

void ExpressionGenerator::GenerateYMMFltOp(
    Express::Op *instr,
    OpYMMRegRegImm fltopreg, OpYMMRegRegImm dblopreg,
    OpYMMRegMemImm fltopmem, OpYMMRegMemImm dblopmem,
    int8 imm,
    MacroAssembler *masm, int argnum) {
  if (instr->dst != -1 && instr->src != -1) {
    // OP reg,reg,imm
    switch (type_) {
      case DT_FLOAT:
        (masm->*fltopreg)(ymm(instr->dst), ymm(instr->src), imm);
        break;
      case DT_DOUBLE:
        (masm->*dblopreg)(ymm(instr->dst), ymm(instr->src), imm);
        break;
      default: UNSUPPORTED;
    }
  } else if (instr->dst != -1 && instr->src == -1) {
    // OP reg,reg,[mem]
    switch (type_) {
      case DT_FLOAT:
        (masm->*fltopmem)(ymm(instr->dst), addr(instr->args[argnum]), imm);
        break;
      case DT_DOUBLE:
        (masm->*dblopmem)(ymm(instr->dst), addr(instr->args[argnum]), imm);
        break;
      default: UNSUPPORTED;
    }
  } else {
    UNSUPPORTED;
  }
}

void ExpressionGenerator::GenerateYMMFltOp(
    Express::Op *instr,
    OpYMMRegRegReg fltopreg, OpYMMRegRegReg dblopreg,
    OpYMMRegRegMem fltopmem, OpYMMRegRegMem dblopmem,
    MacroAssembler *masm, int argnum) {
  if (instr->dst != -1 && instr->src != -1 && instr->src2 != -1) {
    // OP reg,reg,reg
    switch (type_) {
      case DT_FLOAT:
        (masm->*fltopreg)(ymm(instr->dst), ymm(instr->src), ymm(instr->src2));
        break;
      case DT_DOUBLE:
        (masm->*dblopreg)(ymm(instr->dst), ymm(instr->src), ymm(instr->src2));
        break;
      default: UNSUPPORTED;
    }
  } else if (instr->dst != -1 && instr->src != -1 && instr->src2 == -1) {
    // OP reg,reg,[mem]
    switch (type_) {
      case DT_FLOAT:
        (masm->*fltopmem)(ymm(instr->dst), ymm(instr->src),
                          addr(instr->args[argnum]));
        break;
      case DT_DOUBLE:
        (masm->*dblopmem)(ymm(instr->dst), ymm(instr->src),
                          addr(instr->args[argnum]));
        break;
      default: UNSUPPORTED;
    }
  } else {
    UNSUPPORTED;
  }
}

void ExpressionGenerator::GenerateYMMFltOp(
    Express::Op *instr,
    OpYMMRegRegRegImm fltopreg, OpYMMRegRegRegImm dblopreg,
    OpYMMRegRegMemImm fltopmem, OpYMMRegRegMemImm dblopmem,
    int8 imm,
    MacroAssembler *masm, int argnum) {
  if (instr->dst != -1 && instr->src != -1 && instr->src2 != -1) {
    // OP reg,reg,reg
    switch (type_) {
      case DT_FLOAT:
        (masm->*fltopreg)(ymm(instr->dst), ymm(instr->src), ymm(instr->src2),
                          imm);
        break;
      case DT_DOUBLE:
        (masm->*dblopreg)(ymm(instr->dst), ymm(instr->src), ymm(instr->src2),
                          imm);
        break;
      default: UNSUPPORTED;
    }
  } else if (instr->dst != -1 && instr->src != -1 && instr->src2 == -1) {
    // OP reg,reg,[mem]
    switch (type_) {
      case DT_FLOAT:
        (masm->*fltopmem)(ymm(instr->dst), ymm(instr->src),
                          addr(instr->args[argnum]), imm);
        break;
      case DT_DOUBLE:
        (masm->*dblopmem)(ymm(instr->dst), ymm(instr->src),
                          addr(instr->args[argnum]), imm);
        break;
      default: UNSUPPORTED;
    }
  } else {
    UNSUPPORTED;
  }
}

void ExpressionGenerator::GenerateZMMFltOp(
    Express::Op *instr,
    OpZMMRegReg fltopreg, OpZMMRegReg dblopreg,
    OpZMMRegRegR fltopregr, OpZMMRegRegR dblopregr,
    OpZMMRegMem fltopmem, OpZMMRegMem dblopmem,
    MacroAssembler *masm, int argnum) {
  if (instr->dst != -1 && instr->src != -1) {
    // OP reg,reg
    switch (type_) {
      case DT_FLOAT:
        if (fltopreg != nullptr) {
          (masm->*fltopreg)(zmm(instr->dst), zmm(instr->src), nomask);
        } else {
          (masm->*fltopregr)(zmm(instr->dst), zmm(instr->src), nomask, noround);
        }
        break;
      case DT_DOUBLE:
        if (dblopreg != nullptr) {
          (masm->*dblopreg)(zmm(instr->dst), zmm(instr->src), nomask);
        } else {
          (masm->*dblopregr)(zmm(instr->dst), zmm(instr->src), nomask, noround);
        }
        break;
      default: UNSUPPORTED;
    }
  } else if (instr->dst != -1 && instr->src == -1) {
    // OP reg,[mem]
    switch (type_) {
      case DT_FLOAT:
        (masm->*fltopmem)(zmm(instr->dst), addr(instr->args[argnum]), nomask);
        break;
      case DT_DOUBLE:
        (masm->*dblopmem)(zmm(instr->dst), addr(instr->args[argnum]), nomask);
        break;
      default: UNSUPPORTED;
    }
  } else {
    UNSUPPORTED;
  }
}

void ExpressionGenerator::GenerateZMMFltOp(
    Express::Op *instr,
    OpZMMRegRegImm fltopreg, OpZMMRegRegImm dblopreg,
    OpZMMRegMemImm fltopmem, OpZMMRegMemImm dblopmem,
    int8 imm,
    MacroAssembler *masm, int argnum) {
  if (instr->dst != -1 && instr->src != -1) {
    // OP reg,reg,imm
    switch (type_) {
      case DT_FLOAT:
        (masm->*fltopreg)(zmm(instr->dst), zmm(instr->src), imm, nomask);
        break;
      case DT_DOUBLE:
        (masm->*dblopreg)(zmm(instr->dst), zmm(instr->src), imm, nomask);
        break;
      default: UNSUPPORTED;
    }
  } else if (instr->dst != -1 && instr->src == -1) {
    // OP reg,reg,[mem]
    switch (type_) {
      case DT_FLOAT:
        (masm->*fltopmem)(zmm(instr->dst), addr(instr->args[argnum]), imm,
                          nomask);
        break;
      case DT_DOUBLE:
        (masm->*dblopmem)(zmm(instr->dst), addr(instr->args[argnum]), imm,
                          nomask);
        break;
      default: UNSUPPORTED;
    }
  } else {
    UNSUPPORTED;
  }
}

void ExpressionGenerator::GenerateZMMFltOp(
    Express::Op *instr,
    OpZMMRegRegReg fltopreg, OpZMMRegRegReg dblopreg,
    OpZMMRegRegRegR fltopregr, OpZMMRegRegRegR dblopregr,
    OpZMMRegRegMem fltopmem, OpZMMRegRegMem dblopmem,
    MacroAssembler *masm, int argnum) {
  if (instr->dst != -1 && instr->src != -1 && instr->src2 != -1) {
    // OP reg,reg,reg
    switch (type_) {
      case DT_FLOAT:
        if (fltopreg != nullptr) {
          (masm->*fltopreg)(zmm(instr->dst), zmm(instr->src), zmm(instr->src2),
                            nomask);
        } else {
          (masm->*fltopregr)(zmm(instr->dst), zmm(instr->src), zmm(instr->src2),
                             nomask, noround);
        }
        break;
      case DT_DOUBLE:
        if (dblopreg != nullptr) {
          (masm->*dblopreg)(zmm(instr->dst), zmm(instr->src), zmm(instr->src2),
                            nomask);
        } else {
          (masm->*dblopregr)(zmm(instr->dst), zmm(instr->src), zmm(instr->src2),
                            nomask, noround);
        }
        break;
      default: UNSUPPORTED;
    }
  } else if (instr->dst != -1 && instr->src != -1 && instr->src2 == -1) {
    // OP reg,reg,[mem]
    switch (type_) {
      case DT_FLOAT:
        (masm->*fltopmem)(zmm(instr->dst), zmm(instr->src),
                          addr(instr->args[argnum]), nomask);
        break;
      case DT_DOUBLE:
        (masm->*dblopmem)(zmm(instr->dst), zmm(instr->src),
                          addr(instr->args[argnum]), nomask);
        break;
      default: UNSUPPORTED;
    }
  } else {
    UNSUPPORTED;
  }
}

void ExpressionGenerator::GenerateIntUnaryOp(
    Express::Op *instr,
    OpReg opregb, OpMem opmemb,
    OpReg opregw, OpMem opmemw,
    OpReg opregd, OpMem opmemd,
    OpReg opregq, OpMem opmemq,
    MacroAssembler *masm, int argnum) {
  if (instr->dst != -1 && instr->src != -1) {
    // OP reg,reg
    switch (type_) {
      case DT_INT8:
        (masm->*opregb)(reg(instr->src));
        break;
      case DT_INT16:
        (masm->*opregw)(reg(instr->src));
        break;
      case DT_INT32:
        (masm->*opregd)(reg(instr->src));
        break;
      case DT_INT64:
        (masm->*opregq)(reg(instr->src));
        break;
      default: UNSUPPORTED;
    }
  } else if (instr->dst != -1 && instr->src == -1) {
    // OP reg,[mem]
    switch (type_) {
      case DT_INT8:
        (masm->*opmemb)(addr(instr->args[argnum]));
        break;
      case DT_INT16:
        (masm->*opmemw)(addr(instr->args[argnum]));
        break;
      case DT_INT32:
        (masm->*opmemd)(addr(instr->args[argnum]));
        break;
      case DT_INT64:
        (masm->*opmemq)(addr(instr->args[argnum]));
        break;
      default: UNSUPPORTED;
    }
  } else {
    UNSUPPORTED;
  }
}

void ExpressionGenerator::GenerateIntBinaryOp(
    Express::Op *instr,
    OpRegReg opregb, OpRegMem opmemb,
    OpRegReg opregw, OpRegMem opmemw,
    OpRegReg opregd, OpRegMem opmemd,
    OpRegReg opregq, OpRegMem opmemq,
    MacroAssembler *masm, int argnum) {
  if (instr->dst != -1 && instr->src != -1) {
    // OP reg,reg
    switch (type_) {
      case DT_INT8:
        (masm->*opregb)(reg(instr->dst), reg(instr->src));
        break;
      case DT_INT16:
        (masm->*opregw)(reg(instr->dst), reg(instr->src));
        break;
      case DT_INT32:
        (masm->*opregd)(reg(instr->dst), reg(instr->src));
        break;
      case DT_INT64:
        (masm->*opregq)(reg(instr->dst), reg(instr->src));
        break;
      default: UNSUPPORTED;
    }
  } else if (instr->dst != -1 && instr->src == -1) {
    // OP reg,[mem]
    switch (type_) {
      case DT_INT8:
        (masm->*opmemb)(reg(instr->dst), addr(instr->args[argnum]));
        break;
      case DT_INT16:
        (masm->*opmemw)(reg(instr->dst), addr(instr->args[argnum]));
        break;
      case DT_INT32:
        (masm->*opmemd)(reg(instr->dst), addr(instr->args[argnum]));
        break;
      case DT_INT64:
        (masm->*opmemq)(reg(instr->dst), addr(instr->args[argnum]));
        break;
      default: UNSUPPORTED;
    }
  } else {
    UNSUPPORTED;
  }
}

void ExpressionGenerator::GenerateXMMIntOp(
    Express::Op *instr,
    OpXMMRegReg opregb, OpXMMRegMem opmemb,
    OpXMMRegReg opregw, OpXMMRegMem opmemw,
    OpXMMRegReg opregd, OpXMMRegMem opmemd,
    OpXMMRegReg opregq, OpXMMRegMem opmemq,
    MacroAssembler *masm, int argnum) {
  if (instr->dst != -1 && instr->src != -1) {
    // OP reg,reg
    switch (type_) {
      case DT_INT8:
        (masm->*opregb)(xmm(instr->dst), xmm(instr->src));
        break;
      case DT_INT16:
        (masm->*opregw)(xmm(instr->dst), xmm(instr->src));
        break;
      case DT_INT32:
        (masm->*opregd)(xmm(instr->dst), xmm(instr->src));
        break;
      case DT_INT64:
        (masm->*opregq)(xmm(instr->dst), xmm(instr->src));
        break;
      default: UNSUPPORTED;
    }
  } else if (instr->dst != -1 && instr->src == -1) {
    // OP reg,[mem]
    switch (type_) {
      case DT_INT8:
        (masm->*opmemb)(xmm(instr->dst), addr(instr->args[argnum]));
        break;
      case DT_INT16:
        (masm->*opmemw)(xmm(instr->dst), addr(instr->args[argnum]));
        break;
      case DT_INT32:
        (masm->*opmemd)(xmm(instr->dst), addr(instr->args[argnum]));
        break;
      case DT_INT64:
        (masm->*opmemq)(xmm(instr->dst), addr(instr->args[argnum]));
        break;
      default: UNSUPPORTED;
    }
  } else {
    UNSUPPORTED;
  }
}

void ExpressionGenerator::GenerateXMMIntOp(
    Express::Op *instr,
    OpXMMRegRegReg opregb, OpXMMRegRegMem opmemb,
    OpXMMRegRegReg opregw, OpXMMRegRegMem opmemw,
    OpXMMRegRegReg opregd, OpXMMRegRegMem opmemd,
    OpXMMRegRegReg opregq, OpXMMRegRegMem opmemq,
    MacroAssembler *masm, int argnum) {
  if (instr->dst != -1 && instr->src != -1 && instr->src2 != -1) {
    // OP reg,reg,reg
    switch (type_) {
      case DT_INT8:
        (masm->*opregb)(xmm(instr->dst), xmm(instr->src), xmm(instr->src2));
        break;
      case DT_INT16:
        (masm->*opregw)(xmm(instr->dst), xmm(instr->src), xmm(instr->src2));
        break;
      case DT_INT32:
        (masm->*opregd)(xmm(instr->dst), xmm(instr->src), xmm(instr->src2));
        break;
      case DT_INT64:
        (masm->*opregq)(xmm(instr->dst), xmm(instr->src), xmm(instr->src2));
        break;
      default: UNSUPPORTED;
    }
  } else if (instr->dst != -1 && instr->src != -1 && instr->src2 == -1) {
    // OP reg,reg,[mem]
    switch (type_) {
      case DT_INT8:
        (masm->*opmemb)(xmm(instr->dst), xmm(instr->src),
                        addr(instr->args[argnum]));
        break;
      case DT_INT16:
        (masm->*opmemw)(xmm(instr->dst), xmm(instr->src),
                        addr(instr->args[argnum]));
        break;
      case DT_INT32:
        (masm->*opmemd)(xmm(instr->dst), xmm(instr->src),
                        addr(instr->args[argnum]));
        break;
      case DT_INT64:
        (masm->*opmemq)(xmm(instr->dst), xmm(instr->src),
                        addr(instr->args[argnum]));
        break;
      default: UNSUPPORTED;
    }
  } else {
    UNSUPPORTED;
  }
}

void ExpressionGenerator::GenerateYMMIntOp(
    Express::Op *instr,
    OpYMMRegRegReg opregb, OpYMMRegRegMem opmemb,
    OpYMMRegRegReg opregw, OpYMMRegRegMem opmemw,
    OpYMMRegRegReg opregd, OpYMMRegRegMem opmemd,
    OpYMMRegRegReg opregq, OpYMMRegRegMem opmemq,
    MacroAssembler *masm, int argnum) {
  if (instr->dst != -1 && instr->src != -1 && instr->src2 != -1) {
    // OP reg,reg,reg
    switch (type_) {
      case DT_INT8:
        (masm->*opregb)(ymm(instr->dst), ymm(instr->src), ymm(instr->src2));
        break;
      case DT_INT16:
        (masm->*opregw)(ymm(instr->dst), ymm(instr->src), ymm(instr->src2));
        break;
      case DT_INT32:
        (masm->*opregd)(ymm(instr->dst), ymm(instr->src), ymm(instr->src2));
        break;
      case DT_INT64:
        (masm->*opregq)(ymm(instr->dst), ymm(instr->src), ymm(instr->src2));
        break;
      default: UNSUPPORTED;
    }
  } else if (instr->dst != -1 && instr->src != -1 && instr->src2 == -1) {
    // OP reg,reg,[mem]
    switch (type_) {
      case DT_INT8:
        (masm->*opmemb)(ymm(instr->dst), ymm(instr->src),
                        addr(instr->args[argnum]));
        break;
      case DT_INT16:
        (masm->*opmemw)(ymm(instr->dst), ymm(instr->src),
                        addr(instr->args[argnum]));
        break;
      case DT_INT32:
        (masm->*opmemd)(ymm(instr->dst), ymm(instr->src),
                        addr(instr->args[argnum]));
        break;
      case DT_INT64:
        (masm->*opmemq)(ymm(instr->dst), ymm(instr->src),
                        addr(instr->args[argnum]));
        break;
      default: UNSUPPORTED;
    }
  } else {
    UNSUPPORTED;
  }
}

void UnsupportedOperation(const char *file, int line) {
  LOG(FATAL) << "Unsupported operation (" << file << " line " << line << ")";
}

}  // namespace myelin
}  // namespace sling

