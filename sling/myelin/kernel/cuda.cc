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

#include "sling/myelin/compute.h"

namespace sling {
namespace myelin {

// cuda-matmul.cc
void RegisterCUDAMatMulLibrary(Library *library);

// cublas-matmul.cc
void RegisterCUBLASMatMulLibrary(Library *library);

// cuda-arithmetic.cc
void RegisterCUDAArithmeticLibrary(Library *library);

// cuda-array.cc
void RegisterCUDAArrayLibrary(Library *library);

// Register CUDA kernels.
void RegisterCUDALibrary(Library *library) {
  RegisterCUDAMatMulLibrary(library);
  RegisterCUBLASMatMulLibrary(library);
  RegisterCUDAArithmeticLibrary(library);
  RegisterCUDAArrayLibrary(library);
}

}  // namespace myelin
}  // namespace sling

