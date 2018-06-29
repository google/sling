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

#include "sling/myelin/kernel/cuda.h"

#include <string>

#include "sling/myelin/express.h"
#include "sling/myelin/cuda/cuda-kernel.h"
#include "sling/myelin/kernel/arithmetic.h"

namespace sling {
namespace myelin {

// CUDA PTX instruction model.
Express::Model ptx_model;

// Kernel for computing arithmetic expressions on GPU using CUDA.
class CUDACalculate : public CUDAKernel {
 public:
  // Compilation state.
  struct Compilation {
    Step *step;               // step being compiled
    PTXMacroAssembler *ptx;   // assembler for code generation
    Type dtype;               // element data type
    const char *type;         // PTX type of elements
    int size;                 // element size
    std::vector<PTXReg> reg;  // temporary registers
    PTXReg offset;            // element offset register
    PTXReg addr;              // address register
  };

  CUDACalculate(const string &name, const string &operation, int arity = -1)
      : name_(name), operation_(operation), arity_(arity) {}

  string Name() override { return name_; }
  string Operation() override { return operation_; }

  bool Supports(Step *step) override {
    // Requires CUDA support.
    if (!CUDAKernel::Supports(step)) return false;

    // Check that operation is compatible.
    if (step->type() != operation_) return false;
    if (arity_ != -1 && step->indegree() != arity_) return false;

    // Check that inputs and outputs have the compatible types and shapes.
    if (step->indegree() < 1 || step->outdegree() < 1) return false;
    Type type = step->output(0)->type();
    const Shape &shape = step->output(0)->shape();
    for (auto *input : step->inputs()) {
      if (input->type() != type) return false;
      if (!input->Compatible(step->output(0))) return false;

      // NB: general broadcasting not yet supported.
      if (input->elements() != shape.elements() && input->elements() != 1) {
        return false;
      }
    }
    for (auto *output : step->outputs()) {
      if (output->type() != type) return false;
      if (output->shape() != shape) return false;
      if (output->elements() != shape.elements()) return false;
    }

    // Check that element type is supported by CUDA.
    if (TypeTraits::of(type).ptx() == nullptr) return false;

    // Dense encoding required.
    for (auto *input : step->inputs()) input->RequireDense();
    for (auto *output : step->outputs()) output->RequireDense();

    return true;
  }

  void Adjust(Step *step) override {
    // Inputs and ouputs must be in standard format.
   for (auto *input : step->inputs()) {
      input->RequireDense();
      input->RequireStandardOrder();
    }
    for (auto *output : step->outputs()) {
      output->RequireDense();
      output->RequireStandardOrder();
    }

    // Enable sharing of inputs and outputs.
    for (int i = 0; i < step->indegree(); ++i) {
      for (int j = 0; j < step->outdegree(); ++j) {
        if (step->input(i)->shape() == step->output(j)->shape()) {
          if (step->AllowInPlace(i, j)) break;
        }
      }
    }
  }

  void GeneratePTX(Step *step, PTXMacroAssembler *ptx)  override {
    // Parse expression for evaluation.
    Express expr(Express::NVIDIA);
    InitExpression(step, &expr, true);

    // Set grid size. Use one thread for each element.
    Tensor *output = step->output(0);
    int size = output->elements();
    ptx->set_grid_dims(size);

    // Set up compilation state.
    Compilation comp;
    comp.step = step;
    comp.ptx = ptx;

    // Get element type.
    comp.dtype = output->type();
    const TypeTraits &traits = TypeTraits::of(comp.dtype);
    comp.type = traits.ptx();
    comp.size = traits.size();

    // Optimize expression.
    expr.EliminateCommonSubexpressions();
    expr.CacheResults();

    // Rewrite expression.
    Express instrs;
    CHECK(expr.Rewrite(ptx_model, &instrs));
    instrs.ComputeLiveRanges();
    int regs = instrs.AllocateRegisters();

    // Get grid location.
    ptx_decl(b32, idx);
    ptx->GetThreadIndex(idx, 0);

    // Check bounds.
    ptx_decl(pred, outside);
    ptx_emit(setp.ge.u32, outside, idx, PTXImm(size));
    ptx_if(outside);
    ptx_jump(done);
    ptx_endif();

    // Compute element offset.
    ptx_decl(b64, offset);
    comp.offset = offset;
    ptx_emit(mul.wide.u32, offset, idx, PTXImm(comp.size));
    ptx_decl(b64, addr);
    comp.addr = addr;

    // Allocate registers.
    comp.reg.resize(regs);
    for (int i = 0; i < regs; ++i) {
      comp.reg[i] = ptx->reg(comp.type, "r", i);
    }

    // Generate code for each instruction in expression.
    for (auto *instr : instrs.ops()) {
      if (instr->nop()) continue;
      switch (instr->type) {
        case Express::MOV:
          if (instr->dst != -1 && instr->src != -1) {
            ptx->emit(PTXInstr("mov", comp.type),
                      comp.reg[instr->dst],
                      comp.reg[instr->src]);
          } else if (instr->dst != -1) {
            GenerateLoad(instr, &comp);
          } else {
            GenerateStore(instr, &comp);
          }
          break;
        case Express::ADD:
          GenerateBinaryOp("add", instr, &comp);
          break;
        case Express::SUB:
          GenerateBinaryOp("sub", instr, &comp);
          break;
        case Express::MUL:
          if (IsFloat(comp.dtype)) {
            GenerateBinaryOp("mul", instr, &comp);
          } else {
            GenerateBinaryOp("mul.lo", instr, &comp);
          }
          break;
        case Express::DIV:
          if (IsFloat(comp.dtype)) {
            GenerateBinaryOp("div.approx", instr, &comp);
          } else {
            GenerateBinaryOp("div", instr, &comp);
          }
          break;
        case Express::MINIMUM:
          GenerateBinaryOp("min", instr, &comp);
          break;
        case Express::MAXIMUM:
          GenerateBinaryOp("max", instr, &comp);
          break;
        case Express::NEG:
          GenerateUnaryOp("neg", instr, &comp);
          break;
        case Express::ABS:
          GenerateUnaryOp("abs", instr, &comp);
          break;
        case Express::SQRT:
          GenerateUnaryOp("sqrt.approx", instr, &comp);
          break;
        case Express::RECIPROCAL:
          GenerateUnaryOp("rcp.approx", instr, &comp);
          break;
        case Express::LOG2:
          GenerateUnaryOp("lg2.approx", instr, &comp);
          break;
        case Express::EXP2:
          GenerateUnaryOp("ex2.approx", instr, &comp);
          break;
        default:
          LOG(FATAL) << "Instruction not supported in CUDA: "
                     <<  instr->AsInstruction();
      }
    }

    // Done.
    ptx_label(done);
    ptx_ret();
  }

  int64 Complexity(const Step *step) override {
    Express expr(Express::NVIDIA);
    InitExpression(step, &expr, true);
    Tensor *output = step->output(0);
    return output->shape().elements() * expr.Complexity();
  }

  void GenerateLoad(Express::Op *instr, Compilation *comp) {
    PTXMacroAssembler *ptx = comp->ptx;
    CHECK_EQ(instr->arity(), 1);
    CHECK_EQ(instr->result->type, Express::TEMP);
    PTXReg &dst = comp->reg[instr->dst];
    switch (instr->args[0]->type) {
      case Express::INPUT: {
        // mov reg, [ptr].
        Tensor *input = comp->step->input(instr->args[0]->id);
        if (input->constant() && input->elements() == 1) {
          // Load scalar constant.
          switch (comp->dtype) {
            case DT_FLOAT:
              ptx->emit(PTXInstr("mov", comp->type), dst,
                        PTXFloat(input->value<float>()));
              break;
            case DT_DOUBLE:
              ptx->emit(PTXInstr("mov", comp->type), dst,
                        PTXFloat(input->value<double>()));
              break;
            case DT_INT8:
              ptx->emit(PTXInstr("mov", comp->type), dst,
                        PTXImm(input->value<int8>()));
              break;
            case DT_INT16:
              ptx->emit(PTXInstr("mov", comp->type), dst,
                        PTXImm(input->value<int16>()));
              break;
            case DT_INT32:
              ptx->emit(PTXInstr("mov", comp->type), dst,
                        PTXImm(input->value<int32>()));
              break;
            case DT_INT64:
              ptx->emit(PTXInstr("mov", comp->type), dst,
                        PTXImm(input->value<int64>()));
              break;
            default:
              LOG(FATAL) << "Unsupported: " << instr->AsInstruction();
          }
        } else {
          // Load from tensor.
          ptx->LoadTensorAddress(comp->addr, input);
          if (input->elements() != 1) {
            ptx->emit("add.u64", comp->addr, comp->addr, comp->offset);
          }
          ptx->emit(PTXInstr("ld.global", comp->type), dst,
                    PTXAddr(comp->addr));
        }
        break;
      }

      case Express::NUMBER:
        // mov reg, imm.
        if (IsFloat(comp->dtype)) {
          ptx->emit(PTXInstr("mov", comp->type), dst,
                    PTXFloat(Express::NumericFlt32(instr->args[0]->id)));
        } else {
          ptx->emit(PTXInstr("mov", comp->type), dst,
                    PTXImm(Express::NumericFlt32(instr->args[0]->id)));
        }
        break;

      default:
        LOG(FATAL) << "Unsupported: " << instr->AsInstruction();
    }
  }

  void GenerateStore(Express::Op *instr, Compilation *comp) {
    PTXMacroAssembler *ptx = comp->ptx;
    CHECK_EQ(instr->arity(), 1);
    CHECK_EQ(instr->args[0]->type, Express::TEMP);
    CHECK_EQ(instr->result->type, Express::OUTPUT);
    PTXReg &src = comp->reg[instr->src];
    Tensor *output = comp->step->output(instr->result->id);
    CHECK(!output->constant());
    ptx->LoadTensorAddress(comp->addr, output);
    if (output->elements() != 1) {
      ptx->emit("add.u64", comp->addr, comp->addr, comp->offset);
    }
    ptx->emit(PTXInstr("st.global", comp->type), PTXAddr(comp->addr), src);
  }

  void GenerateBinaryOp(const char *op, Express::Op *instr, Compilation *comp) {
    CHECK_EQ(instr->arity(), 2);
    CHECK_EQ(instr->result->type, Express::TEMP);
    CHECK_EQ(instr->args[0]->type, Express::TEMP);
    switch (instr->args[1]->type) {
      case Express::TEMP:
        // op reg,reg,reg.
        comp->ptx->emit(PTXInstr(op, comp->type),
                        comp->reg[instr->dst],
                        comp->reg[instr->src],
                        comp->reg[instr->src2]);
        break;

      case Express::NUMBER:
        // op reg,reg,imm.
        if (IsFloat(comp->dtype)) {
          comp->ptx->emit(PTXInstr(op, comp->type),
                          comp->reg[instr->dst],
                          comp->reg[instr->src],
                          PTXFloat(Express::NumericFlt32(instr->args[1]->id)));
        } else {
          comp->ptx->emit(PTXInstr(op, comp->type),
                          comp->reg[instr->dst],
                          comp->reg[instr->src],
                          PTXImm(Express::NumericFlt32(instr->args[1]->id)));
        }
        break;

      default:
        LOG(FATAL) << "Unsupported: " << instr->AsInstruction();
    }
  }

  void GenerateUnaryOp(const char *op, Express::Op *instr, Compilation *comp) {
    CHECK_EQ(instr->arity(), 1);
    CHECK_EQ(instr->result->type, Express::TEMP);
    CHECK_NE(instr->dst, -1);
    switch (instr->args[0]->type) {
      case Express::TEMP:
        // op reg, reg.
        comp->ptx->emit(PTXInstr(op, comp->type),
                        comp->reg[instr->dst],
                        comp->reg[instr->src]);
        break;

      case Express::NUMBER:
        // op reg, imm.
        if (IsFloat(comp->dtype)) {
          comp->ptx->emit(PTXInstr(op, comp->type),
                          comp->reg[instr->dst],
                          PTXFloat(Express::NumericFlt32(instr->args[0]->id)));
        } else {
          comp->ptx->emit(PTXInstr(op, comp->type),
                          comp->reg[instr->dst],
                          PTXImm(Express::NumericFlt32(instr->args[0]->id)));
        }
        break;

      default:
        LOG(FATAL) << "Unsupported: " << instr->AsInstruction();
    }
  }

  static bool IsFloat(Type type) {
    return type == DT_FLOAT || type == DT_DOUBLE || type == DT_HALF;
  }

 private:
  const string name_;       // kernel name
  const string operation_;  // kernel operation
  int arity_;               // number of inputs
};

// CUDA-based argmax using reduction.
class CUDAArgMax : public CUDAKernel {
 public:
  string Name() override { return "CUDAArgMax"; }
  string Operation() override { return "ArgMax"; }

  bool Supports(Step *step) override {
    // Requires CUDA support.
    if (!CUDAKernel::Supports(step)) return false;

    // Check inputs and outputs.
    if (step->inputs().size() != 1) return false;
    if (step->outputs().size() != 1) return false;
    Tensor *x = step->input(0);
    Tensor *y = step->output(0);

    // Check type.
    if (x->type() != DT_FLOAT) return false;
    if (y->type() != DT_INT32) return false;
    if (y->elements() != 1) return false;

    return true;
  }

  void GeneratePTX(Step *step, PTXMacroAssembler *ptx) override {
    // Get input and output tensors.
    Tensor *x = step->input(0);
    Tensor *y = step->output(0);
    int size = x->elements();

    // Get device capabilities.
    CUDADevice *device = step->cell()->runtime()->Device();
    int max_block_size = device->max_threads_per_block();
    int warp_size = device->warp_size();

    // Compute the block size.
    int block_size = 1;
    while (block_size * 2 <= size && block_size < max_block_size) {
      block_size *= 2;
    }
    ptx->set_grid_dims(block_size);
    if (step->variant().empty()) {
      string variant = "B" + std::to_string(block_size);
      step->set_variant(variant);
    }

    // Declare shared memory for reduction.
    char str[128];
    sprintf(str, ".shared .f32 maxval_array[%d];\n", block_size);
    ptx->emit(str);
    sprintf(str, ".shared .u32 best_array[%d];\n", block_size);
    ptx->emit(str);
    ptx_decl(b64, maxval);
    ptx_decl(b64, best);
    ptx_emit(mov.u64, maxval, PTXLiteral("maxval_array"));
    ptx_emit(mov.u64, best, PTXLiteral("best_array"));

    // Get thread index.
    ptx_decl(b32, idx);
    ptx->GetThreadIndex(idx, 0);

    // Check bounds.
    ptx_decl(pred, outside);
    ptx_emit(setp.ge.u32, outside, idx, PTXImm(block_size));
    ptx_if(outside);
    ptx_jump(done);
    ptx_endif();

    // Reduce input array down to block size:
    //  m = x[idx];
    //  s = idx;
    //  while (s < size) {
    //    s += block_size;
    //    if (x[s] > m) {
    //      m = x[s];
    //      b = s;
    //    }
    //  }
    //  maxval[idx] = m;
    //  best[idx] = b;
    ptx_decl(b64, xptr);
    ptx->LoadTensorAddress(xptr, x);

    // Initially set m = x[idx] and b = idx.
    ptx_decl(b64, xiptr);
    ptx_emit(mad.wide.u32, xiptr, idx, PTXImm(sizeof(float)), xptr);
    ptx_decl(pred, larger);
    ptx_decl(f32, m);
    ptx_decl(u32, b);
    ptx_emit(ld.global.f32, m, PTXAddr(xiptr));
    ptx_emit(mov.u32, b, idx);

    if (block_size < size) {
      // Strided loop over inputs.
      ptx_decl(u32, s);
      ptx_emit(mov.u32, s, idx);
      ptx_label(loop1);

      // Next element in stride. Stop when reaching end of input.
      ptx_emit(add.u32, s, s, PTXImm(block_size));
      ptx_emit(setp.ge.u32, outside, s, PTXImm(size));
      ptx_if(outside);
      ptx_jump(done1);
      ptx_endif();

      // Get x[s].
      ptx_decl(b64, sptr);
      ptx_emit(mad.wide.u32, sptr, s, PTXImm(sizeof(float)), xptr);
      ptx_decl(f32, value);
      ptx_emit(ld.global.f32, value, PTXAddr(sptr));

      // Update max element if x[s] is larger than m.
      ptx_emit(setp.gt.f32, larger, value, m);
      ptx_if(larger);
      ptx_emit(mov.f32, m, value);
      ptx_emit(mov.u32, b, s);
      ptx_endif();

      ptx_jump(loop1);
      ptx_label(done1);
    }

    // Store max element into shared memory.
    ptx_decl(b64, mptr);
    ptx_emit(mad.wide.u32, mptr, idx, PTXImm(sizeof(float)), maxval);
    ptx_emit(st.shared.f32, PTXAddr(mptr), m);

    ptx_decl(b64, bptr);
    ptx_emit(mad.wide.u32, bptr, idx, PTXImm(sizeof(int)), best);
    ptx_emit(st.shared.u32, PTXAddr(bptr), b);

    // The input has now been reduced down to the block size and the largest
    // element in each block, together with its index, is now stored in shared
    // memory. The block is reduced in a number of steps that reduce the problem
    // in half. This is done until there is only one element left.
    ptx_decl(pred, completed);
    ptx_decl(f32, ms);
    ptx_decl(u32, bs);
    while (block_size > 1) {
      // Terminate threads that are no longer active in the reduction.
      ptx_emit(setp.ge.u32, completed, idx, PTXImm(block_size / 2));
      ptx_if(completed);
      ptx_jump(done);
      ptx_endif();

      // Synchronize threads. No synchronization is needed when all remaining
      // active threads are running in the same warp because instructions are
      // SIMD synchronous within a warp.
      if (block_size > warp_size) {
        ptx_emit(bar.sync, PTXImm(0));
      }
      block_size >>= 1;

      // Reduce block by comparing strided elements.
      //  if (maxval[idx + block_size] > maxval[idx]) {
      //    maxval[idx] = maxval[idx + block_size];
      //    best[idx] = best[idx + block_size];
      //  }
      ptx_emit(ld.shared.f32, m, PTXAddr(mptr));
      ptx_emit(ld.shared.f32, ms, PTXAddr(mptr, block_size * sizeof(float)));
      ptx_emit(setp.gt.f32, larger, ms, m);
      ptx_if(larger);
      ptx_emit(ld.shared.u32, bs, PTXAddr(bptr, block_size * sizeof(int)));
      ptx_emit(st.shared.f32, PTXAddr(mptr), ms);
      ptx_emit(st.shared.u32, PTXAddr(bptr), bs);
      ptx_endif();
    }

    // ArgMax is now in the first element.
    ptx_decl(u32, result);
    ptx_emit(ld.shared.u32, result, PTXAddr(best));
    ptx_decl(b64, yptr);
    ptx->LoadTensorAddress(yptr, y);
    ptx_emit(st.global.u32, PTXAddr(yptr), result);

    // Done.
    ptx_label(done);
    ptx_ret();
  }

  int64 Complexity(const Step *step) override {
    return 0;
  }
};

// Register CUDA arithmetic library.
void RegisterCUDAArithmeticLibrary(Library *library) {
  ptx_model.mov_reg_reg = true;
  ptx_model.mov_reg_imm = true;
  ptx_model.mov_reg_mem = true;
  ptx_model.mov_mem_reg = true;
  ptx_model.op_reg_reg = true;
  ptx_model.op_reg_imm = true;
  ptx_model.op_reg_reg_reg = true;
  ptx_model.op_reg_reg_imm = true;
  ptx_model.func_reg_reg = true;
  ptx_model.func_reg_imm = true;
  ptx_model.fm_reg_reg_reg = true;
  ptx_model.fm_reg_reg_imm = true;

  library->Register(new CUDACalculate("CUDAAdd", "Add", 2));
  library->Register(new CUDACalculate("CUDASub", "Sub", 2));
  library->Register(new CUDACalculate("CUDAMul", "Mul", 2));
  library->Register(new CUDACalculate("CUDADiv", "Div", 2));
  library->Register(new CUDACalculate("CUDAMax", "Maximum", 2));
  library->Register(new CUDACalculate("CUDAMin", "Minimum", 2));

  library->Register(new CUDACalculate("CUDALog", "Log", 1));
  library->Register(new CUDACalculate("CUDAExp", "Exp", 1));
  library->Register(new CUDACalculate("CUDASigmoid", "Sigmoid", 1));
  library->Register(new CUDACalculate("CUDATanh", "Tanh", 1));
  library->Register(new CUDACalculate("CUDACalculate", "Calculate"));

  library->Register(new CUDACalculate("CUDANeg", "Neg", 1));
  library->Register(new CUDACalculate("CUDAAbs", "Abs", 1));
  library->Register(new CUDACalculate("CUDARelu", "Relu", 1));
  library->Register(new CUDACalculate("CUDASoftsign", "Softsign", 1));
  library->Register(new CUDACalculate("CUDASoftplus", "Softplus", 1));
  library->Register(new CUDACalculate("CUDALogSigmoid", "LogSigmoid", 1));
  library->Register(new CUDACalculate("CUDAReciprocal", "Reciprocal", 1));
  library->Register(new CUDACalculate("CUDASquare", "Square", 1));

  library->Register(new CUDAArgMax());
}

}  // namespace myelin
}  // namespace sling

