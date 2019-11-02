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

#ifndef SLING_MYELIN_COMPILER_H_
#define SLING_MYELIN_COMPILER_H_

#include "sling/myelin/compute.h"
#include "sling/myelin/flow.h"

namespace sling {
namespace myelin {

// Myelin neural network JIT compiler for compiling a flow to a network.
class Compiler {
 public:
  // Initialize compiler.
  Compiler();
  ~Compiler();

  // Compile flow to network.
  void Compile(Flow *flow, Network *net);

  // Library with transformations and kernels for compilation.
  Library *library() { return library_; }

  // Custom runtime.
  Runtime *runtime() const { return runtime_; }
  void set_runtime(Runtime *runtime) { runtime_ = runtime; }

  // Perf module FLOP counter support.
  bool perf_flopctr() const { return perf_flopctr_; }
  void set_perf_flopctr(bool enabled) { perf_flopctr_ = enabled; }

 private:
  // Write graph to output as DOT and/or SVG file.
  static void WriteGraph(const Flow &flow,
                         const string &dot,
                         const string &svg);

  // Compiler library with kernels, transformations, etc.
  Library *library_ = nullptr;

  // Custom runtime for generated network.
  Runtime *runtime_ = nullptr;

  // Enable perf FLOP counter.
  bool perf_flopctr_ = true;
};

// Enable/disable CPU features for compiler.
void SetCPUFeatures(const string &features);

}  // namespace myelin
}  // namespace sling

#endif  // SLING_MYELIN_COMPILE_H_

