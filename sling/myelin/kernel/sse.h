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

#ifndef SLING_MYELIN_KERNEL_SSE_H_
#define SLING_MYELIN_KERNEL_SSE_H_

#include "sling/myelin/compute.h"

namespace sling {
namespace myelin {

// SSE vectors.
typedef float FloatVec4[8] __attribute__ ((aligned (16)));
#define CONST4(x) {x, x, x, x}

// Register SSE library.
void RegisterSSELibrary(Library *library);

}  // namespace myelin
}  // namespace sling

#endif  // SLING_MYELIN_KERNEL_SSE_H_

