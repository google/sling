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

#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>

#include "sling/myelin/compiler.h"

#include "sling/base/flags.h"
#include "sling/base/logging.h"
#include "sling/base/perf.h"
#include "sling/file/file.h"
#include "sling/myelin/compute.h"
#include "sling/myelin/elf-linker.h"
#include "sling/myelin/flow.h"
#include "sling/myelin/graph.h"
#include "sling/myelin/profile.h"
#include "sling/myelin/cuda/cuda-runtime.h"
#include "sling/myelin/kernel/cuda.h"
#include "sling/myelin/kernel/dragnn.h"
#include "sling/myelin/kernel/mkl.h"
#include "sling/myelin/kernel/tensorflow.h"

DEFINE_string(cpu, "", "Enable/disable CPU features");
DEFINE_bool(gpu, false, "Run kernels on GPU");
DEFINE_bool(profile, false, "Profile neural network computations");
DEFINE_string(input_flow, "", "File for saving raw input flow");
DEFINE_string(final_flow, "", "File for saving final analyzed flow");
DEFINE_string(input_dot, "", "File for saving raw input flow as DOT file");
DEFINE_string(input_graph, "", "File for saving raw input flow as SVG file");
DEFINE_string(final_graph, "", "File for saving analyzed flow as SVG file");
DEFINE_string(final_dot, "", "File for saving analyzed flow as DOT file");
DEFINE_string(jit_code, "", "File for saving JIT generated code");
DEFINE_bool(dump_input_flow, false, "Dump raw input flow to log");
DEFINE_bool(dump_final_flow, false, "Dump final analyzed flow to log");
DEFINE_bool(dump_cells, false, "Dump cells after compilation");
DEFINE_bool(dump_code, false, "Dump generated assembly code");
DEFINE_bool(param_stats, false, "Dump model parameter statistics");
DEFINE_bool(check_flow_consistency, false, "Check that flow is consistent");
DEFINE_bool(dynamic_instance_allocation, false, "Dynamic instance allocation");
DEFINE_bool(mkl, false, "Use Intel Math Kernel Library");
DEFINE_bool(dragnn, false, "Use DRAGNN kernels");
DEFINE_bool(sync_steps, false, "Synchronize all compute steps");
DEFINE_bool(fast_math, false, "Fast approximate math ops");
DEFINE_bool(graph_all_vars, false, "Include all variables in DOT graph");
DEFINE_string(graph_layout, "", "DOT graph layout");
DEFINE_string(data_profile, "", "File name prefix for data instance diagrams");
DEFINE_bool(jit_debug, false, "Debug break in jit code");
DEFINE_int32(cuda_device, -1, "CUDA device number");
DEFINE_int32(cuda_context_flags, 0, "CUDA context flags");
DEFINE_int32(sparse_threshold, 64, "Minimum dimension size for sparse update");
DEFINE_bool(compile_only, false, "Stop after compilation");

namespace sling {
namespace myelin {

// CUDA runtime.
static myelin::CUDARuntime *cudart = nullptr;
static int cudart_refs = 0;

Compiler::Compiler() {
  // Register standard kernels.
  library_ = new Library();
  RegisterTensorflowLibrary(library_);

  // Parse CPU feature flags and enable/disable CPU features.
  if (!FLAGS_cpu.empty()) SetCPUFeatures(FLAGS_cpu);

  // Initialize CUDA runtime and register CUDA kernels in GPU mode.
  if (FLAGS_gpu) {
    if (cudart == nullptr) {
      cudart = new myelin::CUDARuntime();
      cudart->Connect(FLAGS_cuda_device, FLAGS_cuda_context_flags);
    }
    runtime_ = cudart;
    cudart_refs++;
    RegisterCUDALibrary(library_);
  }

  // Add extra kernels.
  if (FLAGS_dragnn) RegisterDragnnLibrary(library_);
  if (FLAGS_mkl) RegisterMKLLibrary(library_);
}

Compiler::~Compiler() {
  // Kernel library cannot be deallocated when profiling is enabled since the
  // profiler needs to be able to access the registered kernels.
  if (!FLAGS_profile) delete library_;

  if (--cudart_refs == 0) {
    delete cudart;
    cudart = nullptr;
  }
}

void Compiler::Compile(Flow *flow, Network *net) {
  // Optionally dump input flow.
  if (FLAGS_dump_input_flow) {
    LOG(INFO) << "Input flow:\n" << flow->ToString();
  }

  // Optionally save input flow.
  if (!FLAGS_input_flow.empty()) {
    flow->Save(FLAGS_input_flow);
  }

  // Optionally output DOT file for input.
  WriteGraph(*flow, FLAGS_input_dot, FLAGS_input_graph);

  // Analyze flow.
  flow->Analyze(*library_);

  // Optionally dump final flow.
  if (FLAGS_dump_final_flow) {
    LOG(INFO) << "Final flow:\n" << flow->ToString();
  }

  // Optionally save final flow.
  if (!FLAGS_final_flow.empty()) {
    flow->Save(FLAGS_final_flow);
  }

  // Optionally output graph for final flow.
  WriteGraph(*flow, FLAGS_final_dot, FLAGS_final_graph);

  // Optionally check flow consistency.
  if (FLAGS_check_flow_consistency) {
    CHECK(flow->IsConsistent());
  }

  // Register runtime.
  if (runtime_ != nullptr) net->set_runtime(runtime_);

  // Set FLOPs counter for measuring performance.
  if (perf_flopctr_ && net->options().flops_address == nullptr) {
    net->options().flops_address = Perf::flopptr();
  }

  // Optionally enable profiling.
  if (FLAGS_profile) {
    net->options().profiling = true;
    net->options().global_profiler = true;
  }

  // Compile flow to network.
  ElfLinker linker;
  if (!FLAGS_jit_code.empty() || FLAGS_dump_code) {
    net->set_linker(&linker);
  }
  if (FLAGS_dynamic_instance_allocation) {
    net->options().dynamic_allocation = true;
  }
  if (FLAGS_sync_steps) net->options().sync_steps = true;
  if (FLAGS_jit_debug) net->options().debug = true;
  if (FLAGS_fast_math) net->options().fast_math = true;
  net->options().sparse_threshold = FLAGS_sparse_threshold;

  CHECK(net->Compile(*flow, *library_));

  // Bind flow artifacts to network tensors, cells, and steps.
  net->Bind(flow);

  // Optionally dump cells to log.
  if (FLAGS_dump_cells) {
    for (Cell *cell : net->cells()) {
      LOG(INFO) << "Cell " << cell->name() << "\n" << cell->ToString();
    }
  }

  // Optionally output data layout diagrams.
  if (!FLAGS_data_profile.empty()) {
    for (Cell *cell : net->cells()) {
      DataProfile profile(cell);
      string filename = FLAGS_data_profile + cell->name() + ".svg";
      File::WriteContents(filename, profile.AsSVG());
    }
  }

  // Optionally output parameter statictics.
  if (FLAGS_param_stats) {
    int total = 0;
    for (Tensor *t : net->globals()) {
      if (t->IsScalar()) continue;
      if (t->type() != DT_FLOAT) continue;
      printf("%8d %s\n", t->elements(), t->name().c_str());
      total += t->elements();
    }
    printf("%8d TOTAL\n", total);
  }

  // Optionally output generated code to ELF file.
  if (!FLAGS_jit_code.empty() || FLAGS_dump_code) {
    // Link code.
    linker.Link();

    if (!FLAGS_jit_code.empty()) {
      // Write ELF object file.
      linker.Write(FLAGS_jit_code.c_str());
    } else {
      // Output code to temporary file.
      char tmpname[PATH_MAX];
      const char *tmpdir = getenv("TMPDIR");
      if (tmpdir == nullptr) tmpdir = "/tmp";
      strcpy(tmpname, tmpdir);
      strcat(tmpname, "/jitcode.XXXXXX");
      int fd = mkstemp(tmpname);

      // Write code to temporary file.
      linker.Write(tmpname);

      // Run objdump to output assembly.
      fflush(stdout);
      fflush(stderr);
      string cmd = "objdump -xrtdw -C -M intel --no-show-raw-insn ";
      cmd.append(tmpname);
      int rc = system(cmd.c_str());
      if (rc != 0) LOG(WARNING) << "Error dumping jit code";

      // Remove temporary file.
      close(fd);
      unlink(tmpname);
    }
  }

  // Stop after compilation if requested.
  if (FLAGS_compile_only) {
    LOG(INFO) << "Stop after compilation";
    exit(1);
  }
}

void Compiler::WriteGraph(const Flow &flow,
                          const string &dot,
                          const string &svg) {
  if (dot.empty() && svg.empty()) return;

  // Generate GraphViz DOT script.
  GraphOptions opts;
  if (FLAGS_graph_all_vars) opts.include_intermediates = true;
  if (!FLAGS_graph_layout.empty()) opts.direction = FLAGS_graph_layout.c_str();
  string graph = FlowToDotGraph(flow, opts);

  // Write DOT file.
  if (!dot.empty()) {
    CHECK(File::WriteContents(dot, graph));
  }

  // Produce SVG by piping DOT script through dot program.
  if (!svg.empty()) {
    string cmd = "dot -T svg -o " + svg;
    FILE *dotpgm = popen(cmd.c_str(), "w");
    if (dotpgm == nullptr) {
      LOG(WARNING) << "Error running dot program";
    } else {
      fwrite(graph.data(), graph.size(), 1, dotpgm);
      pclose(dotpgm);
    }
  }
}

void SetCPUFeatures(const string &features) {
  const char *p = features.c_str();

  if (*p != 0 && *p != '+' && *p != '-') {
    // Disable all features initially.
    jit::CPU::Disable(jit::SSE);
    jit::CPU::Disable(jit::SSE2);
    jit::CPU::Disable(jit::SSE3);
    jit::CPU::Disable(jit::SSE4_1);
    jit::CPU::Disable(jit::SSE4_2);
    jit::CPU::Disable(jit::AVX);
    jit::CPU::Disable(jit::AVX2);
    jit::CPU::Disable(jit::AVX512F);
    jit::CPU::Disable(jit::FMA3);
  }

  while (*p != 0) {
    bool enable = true;
    if (*p == '+') {
      enable = true;
      p++;
    }
    if (*p == '-') {
      enable = false;
      p++;
    }
    const char *q = p;
    while (*q != 0 && *q != '+' && *q != '-') q++;
    string name(p, q - p);
    jit::CpuFeature feature;
    if (name == "sse") {
      feature = jit::SSE;
    } else if (name == "sse2") {
      feature = jit::SSE2;
    } else if (name == "sse3") {
      feature = jit::SSE3;
    } else if (name == "sse4.1") {
      feature = jit::SSE4_1;
    } else if (name == "sse4.2") {
      feature = jit::SSE4_2;
    } else if (name == "avx") {
      feature = jit::AVX;
    } else if (name == "avx2") {
      feature = jit::AVX2;
    } else if (name == "avx512") {
      feature = jit::AVX512F;
    } else if (name == "fma3") {
      feature = jit::FMA3;
    } else {
      LOG(FATAL) << "Unknown CPU feature: " << name;
    }
    p = q;

    if (enable) {
      jit::CPU::Enable(feature);
    } else {
      jit::CPU::Disable(feature);
    }
  }
}

}  // namespace myelin
}  // namespace sling

