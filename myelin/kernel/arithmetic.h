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

#ifndef MYELIN_KERNEL_ARITHMETIC_H_
#define MYELIN_KERNEL_ARITHMETIC_H_

#include "myelin/compute.h"

namespace sling {
namespace myelin {

// Register arithmetic library.
void RegisterArithmeticLibrary(Library *library);

// Register arithmetic transforms.
void RegisterArithmeticTransforms(Library *library);

}  // namespace myelin
}  // namespace sling

#endif  // MYELIN_KERNEL_ARITHMETIC_H_

