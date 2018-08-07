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

#include "sling/myelin/kernel/avx.h"

#include "sling/myelin/compute.h"

namespace sling {
namespace myelin {

// avx-math.cc
void RegisterAVXMath(Library *library);

// avx-matmul.cc
void RegisterAVXMatMul(Library *library);

// avx-operators.cc
void RegisterAVXOperators(Library *library);

// simd-matmul.cc
void RegisterSIMDMatMulLibrary(Library *library);

// Register AVX library.
void RegisterAVXLibrary(Library *library) {
  RegisterAVXMath(library);
  RegisterSIMDMatMulLibrary(library);
  RegisterAVXMatMul(library);
  RegisterAVXOperators(library);
}

}  // namespace myelin
}  // namespace sling

