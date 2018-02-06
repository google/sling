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

#include "sling/base/init.h"
#include "sling/base/logging.h"

namespace sling {

// Class for initializing program modules.
class LibraryInitializer {
 public:
  LibraryInitializer() {
    InitSharedLibrary();
  };
};

// The initialization priority should be set higher than the priority of the
// module initializers in init.h.
static LibraryInitializer init __attribute__((init_priority(2000)));

}  // namespace sling

