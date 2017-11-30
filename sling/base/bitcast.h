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

#ifndef SLING_BASE_BITCAST_H_
#define SLING_BASE_BITCAST_H_

#include <string.h>

namespace sling {

// bit_cast<Dest,Source> is a template function that implements the equivalent
// of "*reinterpret_cast<Dest*>(&source)".
template <class Dest, class Source>
inline Dest bit_cast(const Source &source) {
  // Compile time assertion: sizeof(Dest) == sizeof(Source)
  // A compile error here means your Dest and Source have different sizes.
  typedef char VerifySizesAreEqual[sizeof(Dest) == sizeof(Source) ? 1 : -1];

  Dest dest;
  memcpy(&dest, &source, sizeof(dest));
  return dest;
}

}  // namespace sling

#endif  // SLING_BASE_BITCAST_H_

