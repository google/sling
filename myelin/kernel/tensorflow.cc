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

#include "myelin/kernel/tensorflow.h"

#include "myelin/compute.h"
#include "myelin/kernel/arithmetic.h"
#include "myelin/kernel/avx.h"
#include "myelin/kernel/generic.h"
#include "myelin/kernel/sse.h"
#include "myelin/kernel/precompute.h"

namespace sling {
namespace myelin {

// Register Tensorflow library.
void RegisterTensorflowLibrary(Library *library) {
  RegisterArithmeticTransforms(library);
  RegisterGenericLibrary(library);
  RegisterSSELibrary(library);
  RegisterAVXLibrary(library);
  RegisterArithmeticLibrary(library);
  RegisterPrecomputeLibrary(library);
}

}  // namespace myelin
}  // namespace sling

