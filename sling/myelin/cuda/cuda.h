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

#ifndef SLING_MYELIN_CUDA_CUDA_H_
#define SLING_MYELIN_CUDA_CUDA_H_

#include <iostream>
#include <string>
#include <vector>

#include "sling/base/logging.h"
#include "sling/base/types.h"
#include "sling/myelin/cuda/cuda-api.h"

namespace sling {
namespace myelin {

class CUDAModule;

// Pointer to data in device memory.
typedef uint64 DevicePtr;
#define DEVICE_NULL 0

// Check that CUDA call is successful.
#define CHECK_CUDA(op) CHECK_EQ((op), CUDA_SUCCESS)

// Check that CUBLAS call is successful.
#define CHECK_CUBLAS(op) CHECK_EQ((op), CUBLAS_STATUS_SUCCESS)

// CUDA driver interface.
class CUDA {
 public:
  // Check if CUDA is supported on computer and it has a GPU.
  static bool Supported();

  // Return the number of CUDA-enabled GPUs.
  static int Devices();

 private:
  // Initialize CUDA. This function should only be called once.
  static void Init();
};

// CUDA device.
class CUDADevice {
 public:
  // Initialize CUDA device.
  CUDADevice(int number, int flags = 0);
  ~CUDADevice();

  // Return device number.
  int number() const { return number_; }

  // Return handle for device.
  CUdevice handle() const { return handle_; }

  // Return context for device.
  CUcontext context() const { return context_; }

  // Return handle for CUBLAS Lt.
  cublasLtHandle_t lthandle() const { return lthandle_; }

  // Compile PTX code and return module. The module is owned by the device
  // object and is destroyed together with the device object.
  CUDAModule *Compile(const char *ptx);

  // Return compute capability for device.
  int capability() const { return capability_; }

  // Get device attributes.
  int GetAttribute(CUdevice_attribute attr) const {
    int value;
    CHECK_CUDA(cuDeviceGetAttribute(&value, attr, handle_));
    return value;
  }

  // Return number of multiprocessors on the device.
  int multiprocessors() const {
    return GetAttribute(CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT);
  }

  // Return GPU clock rate in Hz.
  int64 clock_rate() const {
    return 1000LL * GetAttribute(CU_DEVICE_ATTRIBUTE_CLOCK_RATE);
  }

  // Return GPU memory transfer rate in Hz.
  int64 memory_transfer_rate() const {
    return 1000LL * GetAttribute(CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE);
  }

  // Return global memory bus width in bits.
  int bus_width() const {
    return GetAttribute(CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH);
  }

  // Return L2 cache size.
  int l2_cache_size() const {
    return GetAttribute(CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE);
  }

  // Maximum number of threads in a block.
  int max_threads_per_block() const {
    return GetAttribute(CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK);
  }

  // Number of threads per warp.
  int warp_size() const {
    return GetAttribute(CU_DEVICE_ATTRIBUTE_WARP_SIZE);
  }

  // Maximum block dimensions supported by device.
  void max_block_dim(int dims[3]) const {
    dims[0] = GetAttribute(CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X);
    dims[1] = GetAttribute(CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y);
    dims[2] = GetAttribute(CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z);
  }

  // Maximum grid dimensions supported by device.
  void max_grid_dim(int dims[3]) const {
    dims[0] = GetAttribute(CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X);
    dims[1] = GetAttribute(CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y);
    dims[2] = GetAttribute(CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z);
  }

  // Return number of cores per processor.
  int CoresPerSM() const;

  // Return number of cores.
  int cores() const { return multiprocessors() * CoresPerSM(); }

  // Return device name.
  string Name() const;

  // Return total amount of global memory on device.
  size_t TotalMemory() const;

  // Return device information as text.
  string ToString() const;

 public:
  // Device number.
  int number_;

  // CUDA device handle.
  CUdevice handle_;

  // Context for device.
  CUcontext context_;

  // CUBLAS Lt handle.
  cublasLtHandle_t lthandle_;

  // Compute capabilities.
  int capability_;

  // List of modules owned by device.
  std::vector<CUDAModule *> modules_;
};

// CUDA module.
class CUDAModule {
 public:
  // Compile and initialize PTX module.
  CUDAModule(const char *ptx);
  ~CUDAModule();

  // Return module handle.
  CUmodule handle() const { return handle_; }

  // Get function handle.
  CUfunction function(const char *name);

 private:
  // CUDA module handle.
  CUmodule handle_;
};

// CUDA function.
class CUDAFunction {
 public:
  // Initialize CUDA kernel function.
  CUDAFunction(CUfunction handle) : handle_(handle) {}
  CUDAFunction(const CUDAModule &module, const char *name);

  // Return function handle.
  CUfunction handle() const { return handle_; }

  // Get function attributes.
  int GetAttribute(CUfunction_attribute attr) const {
    int value;
    CHECK_CUDA(cuFuncGetAttribute(&value, attr, handle_));
    return value;
  }

  // Return the maximum number of threads per block, beyond which a launch of
  // the function would fail. This number depends on both the function and the
  // device on which the function is currently loaded.
  int max_threads_per_block() const {
    return GetAttribute(CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK);
  }

  // Return the size in bytes of statically-allocated shared memory per block
  // required by this function. This does not include dynamically-allocated
  // shared memory requested by the user at runtime.
  int shared_size() const {
    return GetAttribute(CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES);
  }

  // Return the size in bytes of user-allocated constant memory required by this
  // function.
  int const_size() const {
    return GetAttribute(CU_FUNC_ATTRIBUTE_CONST_SIZE_BYTES);
  }

  // Return the size in bytes of local memory used by each thread of this
  // function.
  int local_size() const {
    return GetAttribute(CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES);
  }

  // Return the number of registers used by each thread of this function.
  int num_regs() const {
    return GetAttribute(CU_FUNC_ATTRIBUTE_NUM_REGS);
  }

  // Return the PTX virtual architecture version for which the function was
  // compiled. This value is the major PTX version * 10 + the minor PTX version.
  int ptx_version() const {
    return GetAttribute(CU_FUNC_ATTRIBUTE_PTX_VERSION);
  }

  // Return the binary architecture version for which the function was compiled.
  // This value is the major binary version * 10 + the minor binary version.
  int binary_version() const {
    return GetAttribute(CU_FUNC_ATTRIBUTE_BINARY_VERSION);
  }

 private:
  // CUDA function handle.
  CUfunction handle_;
};

// PTX assembler instruction.
class PTXInstr {
 public:
  PTXInstr(const char *op) : op_(op), type_(nullptr) {}
  PTXInstr(const char *op, const char *type) : op_(op), type_(type) {}

  const char *op() const { return op_; }
  const char *type() const { return type_; }

 private:
  const char *op_;
  const char *type_;
};

// PTX assembler instruction argument.
class PTXArg {
 public:
  virtual ~PTXArg() = default;
  virtual void Generate(string *code) const = 0;
};

// PTX literal argument.
class PTXLiteral : public PTXArg {
 public:
  PTXLiteral(const char *arg) : arg_(arg) {}

  void Generate(string *code) const override;

 private:
  const char *arg_;
};

// PTX label argument.
class PTXLabel : public PTXArg {
 public:
  PTXLabel(const char *name) : name_(name), index_(-1) {}
  PTXLabel(const char *name, int index) : name_(name), index_(index) {}

  void Generate(string *code) const override;

 private:
  const char *name_;
  int index_;
};

// PTX immediate argument.
class PTXImm : public PTXArg {
 public:
  PTXImm(int64 number) : number_(number) {}

  void Generate(string *code) const override;

 private:
  int64 number_;
};

// PTX 32-bit floating point number argument.
class PTXFloat : public PTXArg {
 public:
  PTXFloat(float number) : number_(number) {}

  void Generate(string *code) const override;

 private:
  float number_;
};

// PTX 64-bit floating point number argument.
class PTXDouble : public PTXArg {
 public:
  PTXDouble(double number) : number_(number) {}

  void Generate(string *code) const override;

 private:
  double number_;
};

// PTX constant argument.
class PTXConst : public PTXArg {
 public:
  enum Constant {ZERO, ONE, FALSE, TRUE};

  PTXConst(Constant constant, const char *type);

  void Generate(string *code) const override;

 private:
  const char *value_;
};

// PTX register argument.
class PTXReg : public PTXArg {
 public:
  PTXReg() : type_(nullptr), name_(nullptr), index_(-1) {}
  PTXReg(const char *type, const char *name)
      : type_(type), name_(name), index_(-1) {}
  PTXReg(const char *type, const char *name, int index)
      : type_(type), name_(name), index_(index) {}

  void Generate(string *code) const override;

  const char *type() const { return type_; }
  const char *name() const { return name_; }
  int index() const { return index_; }
  bool none() const { return name_ == nullptr; }

 private:
  const char *type_;  // register type
  const char *name_;  // register name
  int index_;         // index for register arrays
};

// PTX address indirection argument.
class PTXAddr : public PTXArg {
 public:
  PTXAddr(const PTXReg &reg) : reg_(reg), disp_(0) {}
  PTXAddr(const PTXReg &reg, int64 disp) : reg_(reg), disp_(disp) {}
  PTXAddr(int64 disp) : reg_(), disp_(disp) {}
  PTXAddr(DevicePtr addr) : reg_(), disp_(addr) {}
  PTXAddr(DevicePtr addr, PTXReg &ofs) : reg_(ofs), disp_(addr) {}

  void Generate(string *code) const override;

 private:
  const PTXReg reg_;
  int64 disp_;
};

// PTX assembler for generating code for CUDA kernels.
class PTXAssembler {
 public:
  // Initialize PTX assembler for generating code for function.
  PTXAssembler(const string &name) : name_(name) {}

  // Generate PTX code for function.
  void Generate(string *ptx);

  // Declare register.
  PTXReg reg(const char *type, const char *name,
             const char *source = nullptr, int line = -1) {
    registers_.emplace_back(type, name, SourceIndex(source), line);
    return registers_.back().reg;
  }

  PTXReg reg(const char *type, const char *name, int index,
             const char *source = nullptr, int line = -1) {
    registers_.emplace_back(type, name, index, SourceIndex(source), line);
    return registers_.back().reg;
  }

  // Declare parameter.
  PTXReg param(const char *type, const char *name,
               const char *source = nullptr, int line = -1) {
    parameters_.emplace_back(type, name, SourceIndex(source), line);
    return parameters_.back().reg;
  }

  // Emit instruction with no arguments.
  void emit(const PTXInstr &instr,
            const char *source = nullptr, int line = -1) {
    EmitLoc(source, line);
    EmitPredicate();
    EmitInstruction(instr);
    EmitLineEnd();
  }

  // Emit instruction with one argument.
  void emit(const PTXInstr &instr, const PTXArg &arg1,
            const char *source = nullptr, int line = -1) {
    EmitLoc(source, line);
    EmitPredicate();
    EmitInstruction(instr);
    EmitArg(arg1);
    EmitLineEnd();
  }

  // Emit instruction with two arguments.
  void emit(const PTXInstr &instr, const PTXArg &arg1, const PTXArg &arg2,
            const char *source = nullptr, int line = -1) {
    EmitLoc(source, line);
    EmitPredicate();
    EmitInstruction(instr);
    EmitArg(arg1);
    EmitComma();
    EmitArg(arg2);
    EmitLineEnd();
  }

  // Emit instruction with three arguments.
  void emit(const PTXInstr &instr, const PTXArg &arg1, const PTXArg &arg2,
            const PTXArg &arg3,
            const char *source = nullptr, int line = -1) {
    EmitLoc(source, line);
    EmitPredicate();
    EmitInstruction(instr);
    EmitArg(arg1);
    EmitComma();
    EmitArg(arg2);
    EmitComma();
    EmitArg(arg3);
    EmitLineEnd();
  }

  // Emit instruction with four arguments.
  void emit(const PTXInstr &instr, const PTXArg &arg1, const PTXArg &arg2,
            const PTXArg &arg3, const PTXArg &arg4,
            const char *source = nullptr, int line = -1) {
    EmitLoc(source, line);
    EmitPredicate();
    EmitInstruction(instr);
    EmitArg(arg1);
    EmitComma();
    EmitArg(arg2);
    EmitComma();
    EmitArg(arg3);
    EmitComma();
    EmitArg(arg4);
    EmitLineEnd();
  }

  // Declare label.
  void label(const char *name, int index = -1,
             const char *source = nullptr, int line = -1) {
    EmitLoc(source, line);
    EmitLabel(name, index);
  }

  // Set instruction predicate.
  void pred(const PTXReg &predicate, bool condition = true) {
    predicate_ = predicate;
    condition_ = condition;
  }

  // Negate predicate condition.
  void negate() {
    condition_ = !condition_;
  }

  // Clear instruction predicate.
  void clear() {
    predicate_ = PTXReg();
    condition_ = true;
  }

  // Declare constant for absolute address.
  PTXReg abs(DevicePtr ptr);

  // Emit printf call.
  void vprintf(const char *fmt, va_list args);
  void printf(const char *fmt, ...);

  // Emit custom PTX code.
  void emit(const char *snippet) { code_.append(snippet); }

  // CUDA SM target architecture.
  int target() const { return target_; }
  void set_target(int target) { target_ = target; }

  // Enable generation of source code line information in PTX code.
  void EnableSourceLineInfo() { generate_line_info_ = true; }

 private:
  // Return source index for source file name. If the source file is not in the
  // index it will be added.
  int SourceIndex(const char *source);

  // Parameter or variable declaration.
  struct Declaration {
    Declaration(const char *type, const char *name, int source, int line)
        : reg(type, name), source(source), line(line) {}
    Declaration(const char *type, const char *name, int index,
                int source, int line)
        : reg(type, name, index), source(source), line(line) {}
    PTXReg reg;
    int source;
    int line;
  };

  // Emit source location info.
  void EmitLoc(const char *source, int line);

  // Emit predicate.
  void EmitPredicate();

  // Emit instruction name. Underscores are replaced by periods.
  void EmitInstruction(const PTXInstr &instr);

  // Emit instruction argument.
  void EmitArg(const PTXArg &arg);

  // Emit label declaration.
  void EmitLabel(const char *name, int index);

  // Emit line termination with semicolon.
  void EmitLineEnd();

  // Emit a space character.
  void EmitSpace();

  // Emit a comma.
  void EmitComma();

  // Function name.
  string name_;

  // Target architecture.
  int target_ = 21;

  // Function parameters.
  std::vector<Declaration> parameters_;

  // Declared registers.
  std::vector<Declaration> registers_;

  // Absolute address constants.
  std::vector<DevicePtr> addresses_;

  // Current predicate.
  PTXReg predicate_;

  // Predicate condition.
  bool condition_ = true;

  // Whether source line information is generated in PTX code.
  bool generate_line_info_ = false;

  // Source files.
  std::vector<const char *> source_files_;

  // PTX code instruction buffer.
  string code_;

  // Number of printf calls.
  int num_printf_calls_ = 0;

  // Maximum number of printf arguments.
  int max_printf_args_ = 0;
};

// Utility macros for emitting PTX code.
#define ptx_decl(type, name) \
  PTXReg name = ptx->reg(#type, #name, __FILE__, __LINE__)
#define ptx_param(type, name) \
  PTXReg name = ptx->param(#type, #name, __FILE__, __LINE__)
#define ptx_emit(instr, ...) \
  ptx->emit(#instr, __VA_ARGS__, __FILE__, __LINE__)
#define ptx_label(name) \
  ptx->label(#name, -1, __FILE__, __LINE__)
#define ptx_ret() \
  ptx->emit("ret", __FILE__, __LINE__)

#define ptx_if(p) ptx->pred(p)
#define ptx_ifnot(p) ptx->pred(p, false)
#define ptx_else() ptx->negate()
#define ptx_endif() ptx->clear()
#define ptx_jump(l) ptx_emit(bra, PTXLabel(#l))

}  // namespace myelin
}  // namespace sling

#endif  // SLING_MYELIN_CUDA_CUDA_H_

