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

#ifndef SLING_WEB_ENTITY_REF_H_
#define SLING_WEB_ENTITY_REF_H_

#include <string>

#include "sling/base/types.h"

namespace sling {

// Parse entity reference. Return -1 on errors.
int ParseEntityRef(const char *str, int len, int *consumed);
int ParseEntityRef(const string &str);

}  // namespace sling

#endif  // SLING_WEB_ENTITY_REF_H_

