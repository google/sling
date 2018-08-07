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

#include <iostream>
#include <string>

#include "sling/base/init.h"
#include "sling/base/flags.h"
#include "sling/base/logging.h"
#include "sling/base/types.h"
#include "sling/file/file.h"
#include "sling/myelin/compute.h"
#include "sling/myelin/elf-linker.h"
#include "sling/myelin/flow.h"
#include "sling/myelin/graph.h"
#include "sling/myelin/profile.h"
#include "sling/myelin/cuda/cuda-runtime.h"
#include "sling/myelin/kernel/cuda.h"
#include "sling/myelin/kernel/dragnn.h"
#include "sling/myelin/kernel/tensorflow.h"

using namespace sling::jit;

DEFINE_string(flow, "", "Myelin flow file");
DEFINE_bool(raw, false, "Do not analyze or compile flow");
DEFINE_bool(dump_flow, false, "Dump analyzed flow to stdout");
DEFINE_bool(dump_cell, false, "Dump network cell to stdout");
DEFINE_bool(tf, true, "Use Tensorflow kernel library");
DEFINE_bool(dragnn, true, "Use DRAGNN kernel library");
DEFINE_bool(check_consistency, false, "Check flow for consistency");
DEFINE_bool(profile, false, "Profile network");
DEFINE_bool(dynalloc, false, "Dynamic instance allocation");
DEFINE_string(cell, "", "Network cell name");
DEFINE_string(code, "", "Filename prefix for code");
DEFINE_string(graph, "", "DOT file name for flow");
DEFINE_bool(consts, true, "Include constants in DOT graph");
DEFINE_string(datagraph, "", "DOT file name prefix for data profile");
DEFINE_int32(batch, 1, "Batch size");
DEFINE_int32(codealign, 16, "Alignment of code objects");
DEFINE_string(o, "", "ELF object output file for generated code");
DEFINE_bool(gendata, false, "Output tensor data to ELF object file");
DEFINE_bool(genrwdata, false, "Allocate space for tensor data in object file");
DEFINE_bool(gpu, false, "Run kernels on GPU");
DEFINE_bool(argmax, false, "Use argmax for predictions");
DEFINE_bool(compile, true, "Compile flow");

DEFINE_bool(sse, CPU::Enabled(SSE), "SSE support");
DEFINE_bool(sse2, CPU::Enabled(SSE2), "SSE2 support");
DEFINE_bool(sse3, CPU::Enabled(SSE3), "SSE3 support");
DEFINE_bool(sse41, CPU::Enabled(SSE4_1), "SSE 4.1 support");
DEFINE_bool(avx, CPU::Enabled(AVX), "AVX support");
DEFINE_bool(avx2, CPU::Enabled(AVX2), "AVX2 support");
DEFINE_bool(avx512, CPU::Enabled(AVX512F), "AVX-512 support");
DEFINE_bool(fma3, CPU::Enabled(FMA3), "FMA3 support");

using namespace sling;
using namespace sling::myelin;

// CUDA runtime.
static myelin::CUDARuntime cudart;

// Stub for Dragnn initializer.
class FixedDragnnInitializer : public Kernel {
 public:
  string Name() override { return "WordInitializerDummy"; }
  string Operation() override { return "WordEmbeddingInitializer"; }

  bool Supports(Step *step) override { return true; }

  void Generate(Step *step, MacroAssembler *masm) override {}
};

// Type inference for Dragnn ops.
class FixedDragnnTyper : public Typer {
 public:
  string Name() override { return "FixedDragnnTyper"; }

  bool InferTypes(Flow *flow, Flow::Operation *op) override {
    if (op->type == "WordEmbeddingInitializer") {
      if (op->outdegree() == 1) {
        Flow::Variable *result = op->outputs[0];
        result->type = DT_INT32;
        result->shape.clear();
      }
    }

    return false;
  }
};

static void SetCPUFeature(CpuFeature feature, bool enable) {
  if (enable) {
    CPU::Enable(feature);
  } else {
    CPU::Disable(feature);
  }
}

int main(int argc, char *argv[]) {
  InitProgram(&argc, &argv);

  // Set up CPU features.
  SetCPUFeature(SSE, FLAGS_sse);
  SetCPUFeature(SSE2, FLAGS_sse2);
  SetCPUFeature(SSE3, FLAGS_sse3);
  SetCPUFeature(SSE4_1, FLAGS_sse41);
  SetCPUFeature(AVX, FLAGS_avx);
  SetCPUFeature(AVX2, FLAGS_avx2);
  SetCPUFeature(AVX512F, FLAGS_avx512);
  SetCPUFeature(FMA3, FLAGS_fma3);

  // Set up kernel library.
  Library library;
  if (FLAGS_tf) RegisterTensorflowLibrary(&library);
  if (FLAGS_dragnn) {
    RegisterDragnnLibrary(&library);
    library.Register(new FixedDragnnInitializer());
    library.RegisterTyper(new FixedDragnnTyper());
  }
  if (FLAGS_gpu) RegisterCUDALibrary(&library);

  // Load flow.
  Flow flow;
  LOG(INFO) << "Loading flow from " << FLAGS_flow;
  flow.set_batch_size(FLAGS_batch);
  CHECK(flow.Load(FLAGS_flow));

  if (FLAGS_argmax) {
    for (auto *func : flow.funcs()) {
      auto *output = flow.Var(func->name + "/output");
      if (output != nullptr) {
        auto *prediction = flow.AddVariable(func->name + "/prediction",
                                            DT_INT32, {1});
        flow.AddOperation(func, func->name + "/ArgMax", "ArgMax",
                          {output}, {prediction});
      }
    }
  }

  if (!FLAGS_raw) {
    // Analyze flow.
    LOG(INFO) << "Analyzing flow";
    flow.Analyze(library);
  }

  // Check flow consistency.
  if (FLAGS_check_consistency) {
    if (flow.IsConsistent()) {
      std::cout << "Flow is inconsistent!!!\n";
    } else {
      std::cout << "Flow is consistent\n";
    }
  }

  // Dump flow.
  if (FLAGS_dump_flow) {
    std::cout << flow.ToString();
  }

  // Output DOT graph. The file can be converted to SVG using GraphWiz dot:
  // dot /tmp/model.dot -Tsvg > model.svg
  if (!FLAGS_graph.empty()) {
    LOG(INFO) << "Writing flow graph to " << FLAGS_graph;
    GraphOptions opts;
    opts.include_constants = FLAGS_consts;
    FlowToDotGraphFile(flow, opts, FLAGS_graph);
  }

  if (FLAGS_compile && !FLAGS_raw) {
    // Compile model.
    LOG(INFO) << "Compiling flow";
    ElfLinker *linker = nullptr;
    Network network;
    if (FLAGS_gpu) {
      cudart.Connect();
      network.set_runtime(&cudart);
    }
    if (!FLAGS_o.empty()) {
      ElfLinker::Options linker_opts;
      if (FLAGS_gendata) linker_opts.generate_data = true;
      if (FLAGS_genrwdata) {
        linker_opts.generate_data = true;
        linker_opts.writeable_data = true;
      }
      linker_opts.code_align = FLAGS_codealign;
      linker = new ElfLinker(linker_opts);
      network.set_linker(linker);
    }
    if (FLAGS_profile) network.options().profiling = true;
    if (FLAGS_dynalloc) network.options().dynamic_allocation = true;
    if (!network.Compile(flow, library)) {
      std::cout << "Compilation of flow failed\n";
      return 1;
    }

    // Analyze cells.
    for (Cell *cell : network.cells()) {
      if (!FLAGS_cell.empty() && FLAGS_cell != cell->name()) continue;

      // Dump cell.
      if (FLAGS_dump_cell) {
        std::cout << cell->ToString();
      }

      // Dump data profile.
      if (!FLAGS_datagraph.empty()) {
        string svgfn = FLAGS_datagraph + cell->name() + ".svg";
        DataProfile data_profile(cell);
        LOG(INFO) << "Writing data profile for " << cell->name()
                  << " to " << svgfn;
        File::WriteContents(svgfn, data_profile.AsSVG());
      }

      // Dump generated code to file. The file can be viewed with objdump:
      // objdump -D -Mintel,x86-64 -bbinary -mi386 --no-show-raw-insn <binfn>
      if (!FLAGS_code.empty()) {
        string binfn = FLAGS_code + cell->name() + ".bin";
        LOG(INFO) << "Writing code for " << cell->name() << " to " << binfn;
        cell->WriteCodeToFile(binfn);
      }
    }

    // Write ELF object file. The file can be viewed with objdump:
    // objdump -xrtdw -M intel --no-show-raw-insn <ofn>
    if (linker != nullptr) {
      LOG(INFO) << "Write code and data to " << FLAGS_o;
      linker->Link();
      linker->Write(FLAGS_o.c_str());
    }
    delete linker;
  }

  return 0;
}

