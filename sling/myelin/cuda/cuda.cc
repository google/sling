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

#include "sling/myelin/cuda/cuda.h"

#include <inttypes.h>
#include <stdio.h>
#include <stdarg.h>
#include <mutex>

#include "sling/base/logging.h"
#include "sling/myelin/cuda/cuda-api.h"

namespace sling {
namespace myelin {

// Flag to check that we only try to initialize the CUDA library once.
static std::once_flag cuda_initialized;

// Number of CUDA-enabled devices.
static int num_cuda_devices = 0;

// Use default CUDA context.
static const bool use_default_context = false;

// Initialize CUDA support. This function should only be called once.
void CUDA::Init() {
  // Load the CUDA driver API.
  if (!LoadCUDALibrary()) return;

  // Initialize CUDA driver library.
  CHECK_CUDA(cuInit(0));

  // Get the number of CUDA-enabled devices.
  CHECK_CUDA(cuDeviceGetCount(&num_cuda_devices));
}

bool CUDA::Supported() {
  std::call_once(cuda_initialized, []() { Init(); });
  return num_cuda_devices > 0;
}

int CUDA::Devices() {
  if (!Supported()) return 0;
  return num_cuda_devices;
}

CUDADevice::CUDADevice(int number, int flags) : number_(number) {
  // Check that CUDA is supported.
  CHECK(CUDA::Supported());

  // Check that device is valid.
  CHECK_LT(number, num_cuda_devices);

  // Get device handle.
  CHECK_CUDA(cuDeviceGet(&handle_, number));

  // Create/get context for device.
  if (use_default_context) {
    CHECK_CUDA(cuDevicePrimaryCtxRetain(&context_, handle_));
  } else {
    CHECK_CUDA(cuCtxCreate(&context_, flags, handle_));
  }

  // Get CUBLAS Lt handle.
  if (HasCuBLASLt()) {
    CHECK_CUBLAS(cublasLtCreate(&lthandle_));
  } else {
    lthandle_ = nullptr;
  }

  // Get compute capabilities.
  int minor, major;
  CHECK_CUDA(cuDeviceComputeCapability(&major, &minor, handle_));
  capability_ = major * 10 + minor;
}

CUDADevice::~CUDADevice() {
  for (auto *m : modules_) delete m;
  if (lthandle_) cublasLtDestroy(lthandle_);
  if (use_default_context) {
    cuDevicePrimaryCtxRelease(handle_);
  } else {
    cuCtxDestroy(context_);
  }
}

CUDAModule *CUDADevice::Compile(const char *ptx) {
  CUDAModule *module = new CUDAModule(ptx);
  modules_.push_back(module);
  return module;
}

int CUDADevice::CoresPerSM() const {
  switch (capability_) {
    case 20: return 32;   // Fermi Generation (SM 2.0) GF100 class
    case 21: return 48;   // Fermi Generation (SM 2.1) GF10x class
    case 30: return 192;  // Kepler Generation (SM 3.0) GK10x class
    case 32: return 192;  // Kepler Generation (SM 3.2) GK10x class
    case 35: return 192;  // Kepler Generation (SM 3.5) GK11x class
    case 37: return 192;  // Kepler Generation (SM 3.7) GK21x class
    case 50: return 128;  // Maxwell Generation (SM 5.0) GM10x class
    case 52: return 128;  // Maxwell Generation (SM 5.2) GM20x class
    case 53: return 128;  // Maxwell Generation (SM 5.3) GM20x class
    case 60: return 64;   // Pascal Generation (SM 6.0) GP100 class
    case 61: return 128;  // Pascal Generation (SM 6.1) GP10x class
    case 62: return 128;  // Pascal Generation (SM 6.2) GP10x class
    case 70: return 64;   // Volta Generation (SM 7.0) GV100 class
    case 72: return 64;   // Volta Generation (SM 7.2) GV10B class
    case 75: return 64;   // Turing Generation (SM 7.5) TU1xx class
    default: return 128;
  }
}

string CUDADevice::Name() const {
  // Get GPU device name.
  char name[256];
  CHECK_CUDA(cuDeviceGetName(name, sizeof(name), handle_));
  return name;
}

size_t CUDADevice::TotalMemory() const {
  // Get size of GPU global memory.
  size_t memory;
  CHECK_CUDA(cuDeviceTotalMem(&memory, handle_));
  return memory;
}

string CUDADevice::ToString() const {
  int version;
  CHECK_CUDA(cuDriverGetVersion(&version));
  string name = Name();
  int memory_mb = TotalMemory()  >> 20;
  int bandwidth_gbs = memory_transfer_rate() * (bus_width() / 8) / 1000000000;
  int clock_rate_mhz = clock_rate() / 1000000;
  int ram_xfer_rate_mhz = memory_transfer_rate() / 1000000;
  char str[256];
  snprintf(str, sizeof(str), "%s, SM %d.%d, %d MB RAM, "
           "%d cores @ %d MHz, "
           "%d GB/s bandwidth (%d-bits @ %d Mhz), "
           "%d KB L2 cache, "
           "CUDA v%d.%d",
           name.c_str(),
           capability_ / 10, capability_ % 10,
           memory_mb,
           cores(),
           clock_rate_mhz,
           bandwidth_gbs,
           bus_width(),
           ram_xfer_rate_mhz,
           l2_cache_size() >> 10,
           version / 1000, version % 1000);
  return str;
}

CUDAModule::CUDAModule(const char *ptx) {
  const static int buffer_size = 1024;
  const static int num_options = 5;
  char log[buffer_size];
  char errors[buffer_size];
  CUjit_option option[num_options];
  void *value[num_options];

  option[0] = CU_JIT_INFO_LOG_BUFFER;
  value[0] = log;
  memset(log, 0, buffer_size);

  option[1] = CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES;
  value[1] = reinterpret_cast<void *>(buffer_size);

  option[2] = CU_JIT_ERROR_LOG_BUFFER;
  value[2] = errors;
  memset(errors, 0, buffer_size);

  option[3] = CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES;
  value[3] = reinterpret_cast<void *>(buffer_size);

  option[4] = CU_JIT_FALLBACK_STRATEGY;
  value[4] = reinterpret_cast<void *>(CU_PREFER_PTX);

  CUresult res = cuModuleLoadDataEx(&handle_, ptx, num_options, option, value);
  if (res != CUDA_SUCCESS) {
    LOG(ERROR) << "PTX compile error " << res << ": " << errors;
    const char *line = ptx;
    int lineno = 1;
    for (;;) {
      const char *end = strchr(line, '\n');
      if (end == nullptr) end = line + strlen(line);
      LOG(INFO) << lineno++ << ": " << string(line, end - line);
      if (*end == 0) break;
      line = end + 1;
      if (lineno > 100) break;
    }
    LOG(FATAL) << "Error compiling PTX code";
  }
  if (strlen(log) > 0) {
    LOG(INFO) << log;
  }
}

CUDAModule::~CUDAModule() {
  cuModuleUnload(handle_);
}

CUfunction CUDAModule::function(const char *name) {
  CUfunction func;
  CHECK_CUDA(cuModuleGetFunction(&func, handle_, name));
  return func;
}

CUDAFunction::CUDAFunction(const CUDAModule &module, const char *name) {
  CHECK_CUDA(cuModuleGetFunction(&handle_, module.handle(), name));
}

void PTXLiteral::Generate(string *code) const {
  code->append(arg_);
}

void PTXLabel::Generate(string *code) const {
  code->append(name_);
  if (index_ != -1) code->append(std::to_string(index_));
}

void PTXImm::Generate(string *code) const {
  code->append(std::to_string(number_));
}

void PTXFloat::Generate(string *code) const {
  uint32 bits;
  memcpy(&bits, &number_, sizeof(float));
  char str[16];
  snprintf(str, sizeof(str), "0f%08x", bits);
  code->append(str);
}

void PTXDouble::Generate(string *code) const {
  uint64 bits;
  memcpy(&bits, &number_, sizeof(double));
  char str[32];
  snprintf(str, sizeof(str), "0d%016lx", bits);
  code->append(str);
}

PTXConst::PTXConst(Constant constant, const char *type) {
  char basetype = type[0];
  int width = atoi(type + 1);
  value_ = nullptr;
  switch (constant) {
    case ZERO:
    case FALSE:
      value_ = (basetype == 'f') ? "0.0" : "0";
      break;
    case ONE:
      value_ = (basetype == 'f') ? "1.0" : "1";
      break;
    case TRUE:
      switch (basetype) {
        case 'f':
          switch (width) {
            case 16: value_ = "0fFFFF"; break;
            case 32: value_ = "0fFFFFFFFF"; break;
            case 64: value_ = "0dFFFFFFFFFFFFFFFF"; break;
          }
          break;
        case 's':
        case 'u':
        case 'b':
          switch (width) {
            case 8: value_ = "0xFF"; break;
            case 16: value_ = "0xFFFF"; break;
            case 32: value_ = "0xFFFFFFFF"; break;
            case 64: value_ = "0xFFFFFFFFFFFFFFFF"; break;
          }
      }
      break;
  }
  CHECK(value_ != nullptr) << "Unknown CUDA type: " << type;
}

void PTXConst::Generate(string *code) const {
  code->append(value_);
}

void PTXReg::Generate(string *code) const {
  code->append(name_);
  if (index_ != -1) code->append(std::to_string(index_));
}

void PTXAddr::Generate(string *code) const {
  code->append("[");
  if (reg_.none()) {
    // Absolute address.
    code->append(std::to_string(static_cast<uint64>(disp_)));
  } else {
    // Register addressing with optional displacement.
    reg_.Generate(code);
    if (disp_ > 0) {
      code->append("+");
      code->append(std::to_string(disp_));
    } else if (disp_ < 0) {
      code->append("-");
      code->append(std::to_string(-disp_));
    }
  }
  code->append("]");
}

void PTXAssembler::Generate(string *ptx) {
  // Generate directives.
  ptx->clear();
  ptx->append(".version 5.0\n");
  ptx->append(".target sm_");
  ptx->append(std::to_string(target_));
  ptx->append("\n");
  ptx->append(".address_size 64\n");

  // Generate source file index.
  if (generate_line_info_) {
    for (int i = 0; i < source_files_.size(); ++i) {
      ptx->append(".file ");
      ptx->append(std::to_string(i));
      ptx->push_back(' ');
      ptx->push_back('"');
      ptx->append(source_files_[i]);
      ptx->push_back('"');
      ptx->push_back('\n');
    }
  }

  // Generate external references.
  if (num_printf_calls_ > 0) {
    ptx->append(".extern .func (.param.s32 status) vprintf ("
                ".param.b64 format, .param.b64 valist);\n");
  }

  // Generate entry point.
  ptx->append(".visible .entry ");
  ptx->append(name_);
  ptx->append("(");
  bool first = true;
  for (auto &p : parameters_) {
    if (!first) ptx->append(", ");
    ptx->append(".param .");
    ptx->append(p.reg.type());
    ptx->append(" ");
    ptx->append(p.reg.name());
    if (p.reg.index() != -1) {
      ptx->append(std::to_string(p.reg.index()));
    }
    first = false;
  }
  ptx->append(") {\n");

  // Generate register declarations.
  for (auto &r : registers_) {
    if (r.source != -1 && r.line != -1) {
      ptx->append(".loc ");
      ptx->append(std::to_string(r.source));
      ptx->push_back(' ');
      ptx->append(std::to_string(r.line));
      ptx->append(" 0\n");
    }
    ptx->append(".reg .");
    ptx->append(r.reg.type());
    ptx->append(" ");
    ptx->append(r.reg.name());
    if (r.reg.index() != -1) {
      ptx->append(std::to_string(r.reg.index()));
    }
    ptx->append(";\n");
  }
  for (int i = 0; i < addresses_.size(); ++i) {
    ptx->append(".const .b64 abs");
    ptx->append(std::to_string(i));
    ptx->append(" = ");
    char str[32];
    uint64_t addr = addresses_[i];
    snprintf(str, sizeof(str), "0x%" PRIx64, addr);
    ptx->append(str);
    ptx->append(";\n");
  }

  if (num_printf_calls_ > 0) {
    ptx->append(".param .b64 param0;\n");
    ptx->append(".param .b64 param1;\n");
    ptx->append(".reg .b64 fmtptr;\n");
    ptx->append(".reg .b64 vaptr;\n");
    if (max_printf_args_ > 0) {
      ptx->append(".local .align 8 .b8 argbuf[");
      ptx->append(std::to_string(max_printf_args_ * 8));
      ptx->append("];\n");
   }
  }

  // Add code instructions.
  ptx->append(code_);
  ptx->append("}\n");
}

void PTXAssembler::EmitLoc(const char *source, int line) {
  int fileno = SourceIndex(source);
  if (fileno != -1 && line != -1) {
    code_.append(".loc ");
    code_.append(std::to_string(fileno));
    code_.push_back(' ');
    code_.append(std::to_string(line));
    code_.append(" 0\n");
  }
}

void PTXAssembler::EmitPredicate() {
  if (predicate_.name() == nullptr) return;
  code_.push_back('@');
  if (!condition_) code_.push_back('!');
  predicate_.Generate(&code_);
  EmitSpace();
}

void PTXAssembler::EmitInstruction(const PTXInstr &instr) {
  for (const char *p = instr.op(); *p; ++p) {
    code_.push_back(*p == '_' ? '.' : *p);
  }
  if (instr.type() != nullptr) {
    code_.push_back('.');
    code_.append(instr.type());
  }
  EmitSpace();
}

void PTXAssembler::EmitArg(const PTXArg &arg) {
  arg.Generate(&code_);
}

void PTXAssembler::EmitLabel(const char *name, int index) {
  code_.append(name);
  if (index != -1) code_.append(std::to_string(index));
  code_.append(":\n");
}

void PTXAssembler::EmitLineEnd() {
  code_.push_back(';');
  code_.push_back('\n');
}

void PTXAssembler::EmitSpace() {
  code_.push_back(' ');
}

void PTXAssembler::EmitComma() {
  code_.push_back(',');
}

PTXReg PTXAssembler::abs(DevicePtr ptr) {
  int idx = -1;
  for (int i = 0; i < addresses_.size(); ++i) {
    if (addresses_[i] == ptr) {
      idx = i;
      break;
    }
  }
  if (idx == -1) {
    idx = addresses_.size();
    addresses_.push_back(ptr);
  }
  return PTXReg("b64", "abs", idx);
}

void PTXAssembler::vprintf(const char *fmt, va_list args) {
  // Generate format string.
  int idx = num_printf_calls_++;
  code_.append(".global .align 4 .b8 fmtstr");
  code_.append(std::to_string(idx));
  code_.append("[");
  code_.append(std::to_string(strlen(fmt) + 1));
  code_.append("]={");
  int num_args = 0;
  bool pct_seen = false;
  const char *p = fmt;
  while (*p) {
    code_.append(std::to_string(*p));
    code_.append(",");

    if (*p == '%') {
      pct_seen = !pct_seen;
    } else {
      if (pct_seen) num_args++;
      pct_seen = false;
    }

    p++;
  }
  code_.append("0};\n");
  if (num_args > max_printf_args_) max_printf_args_ = num_args;

  code_.append("cvta.global.u64 fmtptr,fmtstr");
  code_.append(std::to_string(idx));
  code_.append(";\n");
  code_.append("st.param.b64 [param0],fmtptr;\n");

  // Generate argument aray.
  if (num_args > 0) {
    for (int i = 0; i < num_args; ++i) {
      PTXReg *reg = va_arg(args, PTXReg *);
      code_.append("st.local.");
      code_.append(reg->type());
      code_.append(" [argbuf+");
      code_.append(std::to_string(i * 8));
      code_.append("],");
      reg->Generate(&code_);
      code_.append(";\n");
    }
    code_.append("cvta.local.u64 vaptr,argbuf;\n");
  } else {
    code_.append("cvta.global.u64 vaptr,0;\n");
  }
  code_.append("st.param.b64 [param1],vaptr;\n");

  // Call vprintf.
  code_.append("call.uni (_), vprintf, (param0, param1);\n");
}

void PTXAssembler::printf(const char *fmt, ...) {
 va_list args;
 va_start(args, fmt);
 vprintf(fmt, args);
 va_end(args);
}

int PTXAssembler::SourceIndex(const char *source) {
  if (source == nullptr) return -1;
  if (!generate_line_info_) return -1;
  for (int i = 0; i < source_files_.size(); ++i) {
    if (source == source_files_[i]) return i;
  }
  source_files_.push_back(source);
  return source_files_.size() - 1;
}

}  // namespace myelin
}  // namespace sling

