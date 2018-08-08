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

#include "sling/myelin/compiler.h"

#include "sling/base/flags.h"
#include "sling/base/logging.h"
#include "sling/base/perf.h"
#include "sling/myelin/compute.h"
#include "sling/myelin/elf-linker.h"
#include "sling/myelin/flow.h"
#include "sling/myelin/graph.h"
#include "sling/myelin/profile.h"
#include "sling/myelin/cuda/cuda-runtime.h"
#include "sling/myelin/kernel/tensorflow.h"
#include "sling/myelin/kernel/cuda.h"

DEFINE_string(cpu, "", "Enable/disable CPU features");
DEFINE_bool(gpu, false, "Run kernels on GPU");
DEFINE_bool(profile, false, "Profile neural network computations");
DEFINE_string(input_flow, "", "File for saving raw input flow");
DEFINE_string(final_flow, "", "File for saving final analyzed flow");
DEFINE_string(input_graph, "", "File for saving raw input flow as DOT file");
DEFINE_string(final_graph, "", "File for saving analyzed flow as DOT file");
DEFINE_string(jit_code, "", "File for saving JIT generated code");
DEFINE_bool(dump_input_flow, false, "Dump raw input flow to log");
DEFINE_bool(dump_final_flow, false, "Dump final analyzed flow to log");
DEFINE_bool(dump_cells, false, "Dump cells after compilation");
DEFINE_bool(check_flow_consistency, false, "Check that flow is consistent");
DEFINE_bool(dynamic_instance_allocation, false, "Dynamic instance allocation");

namespace sling {
namespace myelin {

// CUDA runtime.
static myelin::CUDARuntime cudart;

Compiler::Compiler() {
  // Register standard kernels.
  RegisterTensorflowLibrary(&library_);

  // Parse CPU feature flags and enable/disable CPU features.
  if (!FLAGS_cpu.empty()) {
    const char *p = FLAGS_cpu.c_str();
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

  // Initialize CUDA runtime and register CUDA kernels in GPU mode.
  if (FLAGS_gpu) {
    RegisterCUDALibrary(&library_);
    cudart.Connect();
    runtime_ = &cudart;
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
  if (!FLAGS_input_graph.empty()) {
    GraphOptions opts;
    FlowToDotGraphFile(*flow, opts, FLAGS_input_graph);
  }

  // Analyze flow.
  flow->Analyze(library_);

  // Optionally dump final flow.
  if (FLAGS_dump_final_flow) {
    LOG(INFO) << "Final flow:\n" << flow->ToString();
  }

  // Optionally save final flow.
  if (!FLAGS_final_flow.empty()) {
    flow->Save(FLAGS_final_flow);
  }

  // Optionally output DOT file for final flow.
  if (!FLAGS_final_graph.empty()) {
    GraphOptions opts;
    FlowToDotGraphFile(*flow, opts, FLAGS_final_graph);
  }

  // Optionally check flow consistency.
  if (FLAGS_check_flow_consistency) {
    CHECK(flow->IsConsistent());
  }

  // Register runtime.
  if (runtime_ != nullptr) net->set_runtime(runtime_);

  // Set FLOPs counter for measuring performance.
  if (net->options().flops_address == nullptr) {
    net->options().flops_address = Perf::flopptr();
  }

  // Optionally enable profiling.
  if (FLAGS_profile) {
    net->options().profiling = true;
    net->options().global_profiler = true;
  }

  // Compile flow to network.
  ElfLinker linker;
  if (!FLAGS_jit_code.empty()) {
    net->set_linker(&linker);
  }
  if (FLAGS_dynamic_instance_allocation) {
    net->options().dynamic_allocation = true;
  }
  CHECK(net->Compile(*flow, library_));

  // Optionally dump cells to log.
  if (FLAGS_dump_cells) {
    for (Cell *cell : net->cells()) {
      LOG(INFO) << "Cell " << cell->name() << "\n" << cell->ToString();
    }
  }

  // Optionally output generated code to ELF file.
  if (!FLAGS_jit_code.empty()) {
    linker.Link();
    linker.Write(FLAGS_jit_code.c_str());
  }
}

void LogProfile(const Network &net) {
  if (net.options().global_profiler) {
    string report;
    for (const Cell *cell : net.cells()) {
      Profile profile(cell->profile_summary());
      report.append("\n");
      report.append(profile.ASCIIReport());
    }
    LOG(INFO) << "Profiling report:\n" << report;
  }
}

}  // namespace myelin
}  // namespace sling

