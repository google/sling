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

#include "sling/myelin/kernel/tensorflow.h"

#include <mutex>

#include "sling/myelin/compute.h"
#include "sling/myelin/kernel/arithmetic.h"
#include "sling/myelin/kernel/avx.h"
#include "sling/myelin/kernel/generic.h"
#include "sling/myelin/kernel/gradients.h"
#include "sling/myelin/kernel/sse.h"
#include "sling/myelin/kernel/precompute.h"

namespace sling {
namespace myelin {

static std::once_flag gradients_initialized;

// Register Tensorflow ops.
void RegisterTensorflowLibrary(Library *library) {
  RegisterArithmeticTransforms(library);
  RegisterGenericLibrary(library);
  RegisterSSELibrary(library);
  RegisterAVXLibrary(library);
  RegisterArithmeticLibrary(library);
  RegisterPrecomputeLibrary(library);
  RegisterGenericTransforms(library);

  std::call_once(gradients_initialized, RegisterStandardGradients);
}

}  // namespace myelin
}  // namespace sling

