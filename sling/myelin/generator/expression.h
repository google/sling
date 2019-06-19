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

#ifndef SLING_MYELIN_GENERATOR_EXPRESSION_H_
#define SLING_MYELIN_GENERATOR_EXPRESSION_H_

#include <vector>

#include "sling/myelin/compute.h"
#include "sling/myelin/express.h"
#include "sling/myelin/generator/index.h"

namespace sling {
namespace myelin {

class ExpressionGenerator {
 public:
  // Assembler type definitions.
  typedef jit::Assembler Assembler;
  typedef jit::Operand Operand;
  typedef jit::Register Register;
  typedef jit::Immediate Immediate;
  typedef jit::Mask Mask;
  typedef jit::RoundingMode RoundingMode;
  typedef jit::XMMRegister XMMRegister;
  typedef jit::YMMRegister YMMRegister;
  typedef jit::ZMMRegister ZMMRegister;
  typedef jit::OpmaskRegister OpmaskRegister;

  // Register sizes in bytes.
  const static int XMMRegSize = 16;
  const static int YMMRegSize = 32;
  const static int ZMMRegSize = 64;

  virtual ~ExpressionGenerator() = default;

  // Return vector size in bytes.
  virtual int VectorSize() { return TypeTraits::of(type_).size(); }

  // Check if generator can use extended SIMD registers.
  virtual bool ExtendedRegs() { return false; }

  // Whether instruction model supports immediate operands.
  virtual bool ImmediateOperands() { return false; }

  // Reserve all the registers needed by the generator.
  virtual void Reserve() = 0;

  // Generate code for instruction.
  virtual void Generate(Express::Op *instr, MacroAssembler *masm) = 0;

  // Generate code for reduction operation.
  virtual void GenerateReduce(Express::Op *instr, MacroAssembler *masm);

  // Initialize expression generator.
  void Initialize(const Express &expression,
                  Type type,
                  int spare_regs,
                  IndexGenerator *index);

  // Generate code for loop-invariant prologue for expression.
  void GenerateInit(MacroAssembler *masm);

  // Generate code for loop body.
  void GenerateBody(MacroAssembler *masm);

  // Generate code for loop-invariant epilogue for expression.
  void GenerateEnd(MacroAssembler *masm);

  // Return register number for variable.
  int RegisterNumber(Express::VarType type, int id) const;

  // Return generator name.
  string Name() { return model_.name; }

  // Select expression generator for expression that is supported by the CPU.
  static ExpressionGenerator *Select(const Express &expr, Type type, int size);

  // Set approximate math flag.
  bool approx() const { return approx_; }
  void set_approx(bool approx) { approx_ = approx; }

 protected:
  // Comparison types. These are Intel comparison predicates used by CMPSS.
  enum Comparison {
    CMP_EQ_OQ    = 0,
    CMP_LT_OS    = 1,
    CMP_LE_OS    = 2,
    CMP_UNORD_Q  = 3,
    CMP_NEQ_UQ   = 4,
    CMP_NLT_US   = 5,
    CMP_NLE_US   = 6,
    CMP_ORD_Q    = 7,
    CMP_EQ_UQ    = 8,
    CMP_NGE_US   = 9,
    CMP_NGT_US   = 10,
    CMP_FALSE_OQ = 11,
    CMP_NEQ_OQ   = 12,
    CMP_GE_OS    = 13,
    CMP_GT_OS    = 14,
    CMP_TRUE_UQ  = 15,
    CMP_EQ_OS    = 16,
    CMP_LT_OQ    = 17,
    CMP_LE_OQ    = 18,
    CMP_UNORD_S  = 19,
    CMP_NEQ_US   = 20,
    CMP_NLT_UQ   = 21,
    CMP_NLE_UQ   = 22,
    CMP_ORD_S    = 23,
    CMP_EQ_US    = 24,
    CMP_NGE_UQ    = 25,
    CMP_NGT_UQ    = 26,
    CMP_FALSE_OS  = 27,
    CMP_NEQ_OS    = 28,
    CMP_GE_OQ     = 29,
    CMP_GT_OQ     = 30,
    CMP_TRUE_US   = 31,
  };

  // Assembler instruction methods for different instruction formats.
  typedef void (Assembler::*OpReg)(Register);
  typedef void (Assembler::*OpMem)(const Operand &);
  typedef void (Assembler::*OpRegReg)(Register, Register);
  typedef void (Assembler::*OpRegMem)(Register, const Operand &);
  typedef void (Assembler::*OpRegImm)(Register, Immediate);

  typedef void (Assembler::*OpXMMRegReg)(XMMRegister,
                                         XMMRegister);
  typedef void (Assembler::*OpXMMRegRegImm)(XMMRegister,
                                            XMMRegister,
                                            int8);
  typedef void (Assembler::*OpXMMRegMem)(XMMRegister,
                                         const Operand &);
  typedef void (Assembler::*OpXMMRegMemImm)(XMMRegister,
                                            const Operand &,
                                            int8);

  typedef void (Assembler::*OpXMMRegRegReg)(XMMRegister,
                                            XMMRegister,
                                            XMMRegister);
  typedef void (Assembler::*OpXMMRegRegRegImm)(XMMRegister,
                                               XMMRegister,
                                               XMMRegister,
                                               int8);
  typedef void (Assembler::*OpXMMRegRegMem)(XMMRegister,
                                            XMMRegister,
                                            const Operand &);
  typedef void (Assembler::*OpXMMRegRegMemImm)(XMMRegister,
                                               XMMRegister,
                                               const Operand &,
                                               int8);

  typedef void (Assembler::*OpYMMRegReg)(YMMRegister,
                                         YMMRegister);
  typedef void (Assembler::*OpYMMRegMem)(YMMRegister,
                                         const Operand &);
  typedef void (Assembler::*OpYMMRegRegImm)(YMMRegister,
                                            YMMRegister,
                                            int8);
  typedef void (Assembler::*OpYMMRegMemImm)(YMMRegister,
                                            const Operand &,
                                            int8);
  typedef void (Assembler::*OpYMMRegRegReg)(YMMRegister,
                                            YMMRegister,
                                            YMMRegister);
  typedef void (Assembler::*OpYMMRegRegRegImm)(YMMRegister,
                                               YMMRegister,
                                               YMMRegister,
                                               int8);
  typedef void (Assembler::*OpYMMRegRegMem)(YMMRegister,
                                            YMMRegister,
                                            const Operand &);
  typedef void (Assembler::*OpYMMRegRegMemImm)(YMMRegister,
                                               YMMRegister,
                                               const Operand &,
                                               int8);

  typedef void (Assembler::*OpZMMRegReg)(ZMMRegister,
                                         ZMMRegister,
                                         Mask);
  typedef void (Assembler::*OpZMMRegRegR)(ZMMRegister,
                                          ZMMRegister,
                                          Mask,
                                          RoundingMode);
  typedef void (Assembler::*OpZMMRegMem)(ZMMRegister,
                                         const Operand &,
                                         Mask);
  typedef void (Assembler::*OpZMMRegRegImm)(ZMMRegister,
                                            ZMMRegister,
                                            int8,
                                            Mask);
  typedef void (Assembler::*OpZMMRegMemImm)(ZMMRegister,
                                            const Operand &,
                                            int8,
                                            Mask);
  typedef void (Assembler::*OpZMMRegRegReg)(ZMMRegister,
                                            ZMMRegister,
                                            ZMMRegister,
                                            Mask);
  typedef void (Assembler::*OpZMMRegRegRegR)(ZMMRegister,
                                             ZMMRegister,
                                             ZMMRegister,
                                             Mask,
                                             RoundingMode);
  typedef void (Assembler::*OpZMMRegRegRegImm)(ZMMRegister,
                                               ZMMRegister,
                                               ZMMRegister,
                                               int8,
                                               Mask);
  typedef void (Assembler::*OpZMMRegRegMem)(ZMMRegister,
                                            ZMMRegister,
                                            const Operand &,
                                            Mask);
  typedef void (Assembler::*OpZMMRegRegMemImm)(ZMMRegister,
                                               ZMMRegister,
                                               const Operand &,
                                               int8,
                                               Mask);

  // Check if size is a multiple of the vector size.
  static bool IsVector(int size, int vecsize) {
    return size > 1 && size % vecsize == 0;
  }

  // Return operand for accessing memory variable.
  Operand addr(Express::Var *var) { return index_->addr(var); }

  // Return pointer to constant data.
  const void *data(Express::Var *var) { return index_->data(var); }

  // Return constant variable value.
  template<typename T> const T value(Express::Var *var) {
    return *reinterpret_cast<const T *>(data(var));
  }

  // Return register for temporary variable.
  Register reg(int idx) { return index_->reg(idx); }
  XMMRegister xmm(int idx) { return index_->xmm(idx); }
  YMMRegister ymm(int idx) { return index_->ymm(idx); }
  ZMMRegister zmm(int idx) { return index_->zmm(idx); }
  OpmaskRegister kk(int idx) { return index_->kk(idx); }

  // Return register for auxiliary variable.
  Register aux(int idx) { return index_->aux(idx); }
  XMMRegister xmmaux(int idx) { return index_->xmmaux(idx); }
  YMMRegister ymmaux(int idx) { return index_->ymmaux(idx); }
  ZMMRegister zmmaux(int idx) { return index_->zmmaux(idx); }

  // Generate XMM scalar float move.
  void GenerateXMMScalarFltMove(Express::Op *instr, MacroAssembler *masm);

  // Generate XMM vector move.
  void GenerateXMMVectorMove(Express::Op *instr, MacroAssembler *masm);

  // Generate move of YMM vector operand to register.
  void GenerateYMMMoveMemToReg(YMMRegister dst, const Operand &src,
                               MacroAssembler *masm);

  // Generate YMM vector move.
  void GenerateYMMVectorMove(Express::Op *instr, MacroAssembler *masm);

  // Generate move of ZMM vector operand to register.
  void GenerateZMMMoveMemToReg(ZMMRegister dst, const Operand &src,
                               MacroAssembler *masm);

  // Generate ZMM vector move.
  void GenerateZMMVectorMove(Express::Op *instr, MacroAssembler *masm);

  // Generate move of x64 operand to register.
  void GenerateIntMoveMemToReg(Register dst, const Operand &src,
                               MacroAssembler *masm);

  // Generate move of x64 register to operand.
  void GenerateIntMoveRegToMem(const Operand &dst, Register src,
                               MacroAssembler *masm);

  // Generate x64 scalar int move.
  void GenerateScalarIntMove(Express::Op *instr, MacroAssembler *masm);

  // Generate XMM vector int move.
  void GenerateXMMVectorIntMove(Express::Op *instr, MacroAssembler *masm);

  // Generate YMM vector int move.
  void GenerateYMMVectorIntMove(Express::Op *instr, MacroAssembler *masm);

  // Generate two-operand XMM float op.
  void GenerateXMMFltOp(
    Express::Op *instr,
    OpXMMRegReg fltopreg, OpXMMRegReg dblopreg,
    OpXMMRegMem fltopmem, OpXMMRegMem dblopmem,
    MacroAssembler *masm, int argnum = 1);

  // Generate two-operand XMM float op with immediate.
  void GenerateXMMFltOp(
    Express::Op *instr,
    OpXMMRegRegImm fltopreg, OpXMMRegRegImm dblopreg,
    OpXMMRegMemImm fltopmem, OpXMMRegMemImm dblopmem,
    int8 imm,
    MacroAssembler *masm);

  // Generate two-operand XMM float accumulate op.
  void GenerateXMMFltAccOp(
    Express::Op *instr,
    OpXMMRegReg fltopreg, OpXMMRegReg dblopreg,
    OpXMMRegMem fltopmem, OpXMMRegMem dblopmem,
    MacroAssembler *masm);

  // Generate unary XMM float op with immediate.
  void GenerateXMMUnaryFltOp(
      Express::Op *instr,
      OpXMMRegRegImm fltopreg, OpXMMRegRegImm dblopreg,
      OpXMMRegMemImm fltopmem, OpXMMRegMemImm dblopmem,
      int8 imm, MacroAssembler *masm);

  // Generate three-operand XMM float op.
  void GenerateXMMFltOp(
      Express::Op *instr,
      OpXMMRegRegReg fltopreg, OpXMMRegRegReg dblopreg,
      OpXMMRegRegMem fltopmem, OpXMMRegRegMem dblopmem,
      MacroAssembler *masm, int argnum = 1);

  // Generate three-operand XMM float accumulate op.
  void GenerateXMMFltAccOp(
      Express::Op *instr,
      OpXMMRegRegReg fltopreg, OpXMMRegRegReg dblopreg,
      OpXMMRegRegMem fltopmem, OpXMMRegRegMem dblopmem,
      MacroAssembler *masm);

  // Generate three-operand XMM float op with immediate.
  void GenerateXMMFltOp(
      Express::Op *instr,
      OpXMMRegRegRegImm fltopreg, OpXMMRegRegRegImm dblopreg,
      OpXMMRegRegMemImm fltopmem, OpXMMRegRegMemImm dblopmem,
      int8 imm,
      MacroAssembler *masm, int argnum = 1);

  // Generate unary YMM float op.
  void GenerateYMMUnaryFltOp(
      Express::Op *instr,
      OpYMMRegRegReg fltopreg, OpYMMRegRegReg dblopreg,
      OpYMMRegRegMem fltopmem, OpYMMRegRegMem dblopmem,
      MacroAssembler *masm);

  // Generate two-operand YMM float op.
  void GenerateYMMFltOp(
      Express::Op *instr,
      OpYMMRegReg fltopreg, OpYMMRegReg dblopreg,
      OpYMMRegMem fltopmem, OpYMMRegMem dblopmem,
      MacroAssembler *masm, int argnum = 0);

  // Generate two-operand YMM float op with immediate.
  void GenerateYMMFltOp(
      Express::Op *instr,
      OpYMMRegRegImm fltopreg, OpYMMRegRegImm dblopreg,
      OpYMMRegMemImm fltopmem, OpYMMRegMemImm dblopmem,
      int8 imm,
      MacroAssembler *masm, int argnum = 0);

  // Generate three-operand YMM float op.
  void GenerateYMMFltOp(
      Express::Op *instr,
      OpYMMRegRegReg fltopreg, OpYMMRegRegReg dblopreg,
      OpYMMRegRegMem fltopmem, OpYMMRegRegMem dblopmem,
      MacroAssembler *masm, int argnum = 1);

  // Generate three-operand YMM float op with immediate.
  void GenerateYMMFltOp(
      Express::Op *instr,
      OpYMMRegRegRegImm fltopreg, OpYMMRegRegRegImm dblopreg,
      OpYMMRegRegMemImm fltopmem, OpYMMRegRegMemImm dblopmem,
      int8 imm,
      MacroAssembler *masm, int argnum = 1);

  // Generate three-operand YMM float accumulate op.
  void GenerateYMMFltAccOp(
      Express::Op *instr,
      OpYMMRegRegReg fltopreg, OpYMMRegRegReg dblopreg,
      OpYMMRegRegMem fltopmem, OpYMMRegRegMem dblopmem,
      MacroAssembler *masm);

  // Generate two-operand ZMM float op.
  void GenerateZMMFltOp(
      Express::Op *instr,
      OpZMMRegReg fltopreg, OpZMMRegReg dblopreg,
      OpZMMRegRegR fltopregr, OpZMMRegRegR dblopregr,
      OpZMMRegMem fltopmem, OpZMMRegMem dblopmem,
      MacroAssembler *masm, int argnum = 0);

  // Generate two-operand ZMM float op with immediate.
  void GenerateZMMFltOp(
      Express::Op *instr,
      OpZMMRegRegImm fltopreg, OpZMMRegRegImm dblopreg,
      OpZMMRegMemImm fltopmem, OpZMMRegMemImm dblopmem,
      int8 imm,
      MacroAssembler *masm, int argnum = 0);

  // Generate three-operand ZMM float op.
  void GenerateZMMFltOp(
      Express::Op *instr,
      OpZMMRegRegReg fltopreg, OpZMMRegRegReg dblopreg,
      OpZMMRegRegRegR fltopregr, OpZMMRegRegRegR dblopregr,
      OpZMMRegRegMem fltopmem, OpZMMRegRegMem dblopmem,
      MacroAssembler *masm, int argnum = 1);

  // Generate three-operand ZMM float accumulate op.
  void GenerateZMMFltAccOp(
      Express::Op *instr,
      OpZMMRegRegReg fltopreg, OpZMMRegRegReg dblopreg,
      OpZMMRegRegRegR fltopregr, OpZMMRegRegRegR dblopregr,
      OpZMMRegRegMem fltopmem, OpZMMRegRegMem dblopmem,
      MacroAssembler *masm);

  // Generate one-operand x64 int op.
  void GenerateIntUnaryOp(
      Express::Op *instr,
      OpReg opregb, OpMem opmemb,
      OpReg opregw, OpMem opmemw,
      OpReg opregd, OpMem opmemd,
      OpReg opregq, OpMem opmemq,
      MacroAssembler *masm, int argnum = 0);

  // Generate two-operand x64 int op.
  void GenerateIntBinaryOp(
      Express::Op *instr,
      OpRegReg opregb, OpRegMem opmemb,
      OpRegReg opregw, OpRegMem opmemw,
      OpRegReg opregd, OpRegMem opmemd,
      OpRegReg opregq, OpRegMem opmemq,
      MacroAssembler *masm, int argnum = 1);

  // Generate two-operand XMM int op.
  void GenerateXMMIntOp(
      Express::Op *instr,
      OpXMMRegReg opregb, OpXMMRegMem opmemb,
      OpXMMRegReg opregw, OpXMMRegMem opmemw,
      OpXMMRegReg opregd, OpXMMRegMem opmemd,
      OpXMMRegReg opregq, OpXMMRegMem opmemq,
      MacroAssembler *masm, int argnum = 1);

  // Generate three-operand XMM int op.
  void GenerateXMMIntOp(
      Express::Op *instr,
      OpXMMRegRegReg opregb, OpXMMRegRegMem opmemb,
      OpXMMRegRegReg opregw, OpXMMRegRegMem opmemw,
      OpXMMRegRegReg opregd, OpXMMRegRegMem opmemd,
      OpXMMRegRegReg opregq, OpXMMRegRegMem opmemq,
      MacroAssembler *masm, int argnum = 1);

  // Generate three-operand YMM int op.
  void GenerateYMMIntOp(
      Express::Op *instr,
      OpYMMRegRegReg opregb, OpYMMRegRegMem opmemb,
      OpYMMRegRegReg opregw, OpYMMRegRegMem opmemw,
      OpYMMRegRegReg opregd, OpYMMRegRegMem opmemd,
      OpYMMRegRegReg opregq, OpYMMRegRegMem opmemq,
      MacroAssembler *masm, int argnum = 1);

  // Check if instruction is MOV reg,0.
  static bool IsLoadZero(Express::Op *instr) {
    return instr->type == Express::MOV &&
           instr->dst != -1 &&
           instr->args[0]->type == Express::NUMBER &&
           instr->args[0]->id == Express::ZERO;
  }

  // Check if instruction is MOV reg,1.
  static bool IsLoadOne(Express::Op *instr) {
    return instr->type == Express::MOV &&
           instr->dst != -1 &&
           instr->args[0]->type == Express::NUMBER &&
           instr->args[0]->id == Express::ONE;
  }

  // Index generator for expression.
  IndexGenerator *index_ = nullptr;

  // Type for expression.
  Type type_;

  // Instruction model for instruction set used by generator.
  Express::Model model_;

  // Expression that should be generated.
  Express expression_{&model_};

  // Instructions for generating expression.
  Express instructions_{&model_};

  // Allow approximate math functions.
  bool approx_ = false;
};

// Return reduction operator for reduction instruction.
Reduction ReduceOp(Express::Op *instr);

// Error handler for unsupported operations.
void UnsupportedOperation(const char *file, int line);

#define UNSUPPORTED UnsupportedOperation(__FILE__, __LINE__);

}  // namespace myelin
}  // namespace sling

#endif  // SLING_MYELIN_GENERATOR_EXPRESSION_H_

