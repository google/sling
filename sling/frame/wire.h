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

#ifndef SLING_FRAME_WIRE_H_
#define SLING_FRAME_WIRE_H_

namespace sling {

// Tag numbers for encoding objects in binary wire format. The wire type is
// three bits, but the special tags can be up to 64-3=61 bits.
enum WireType {
  WIRE_REF     = 0,  // reference to previous object (argument is refnum)
  WIRE_FRAME   = 1,  // frame (argument is the number of slots)
  WIRE_STRING  = 2,  // string (argument is the string length in bytes)
  WIRE_SYMBOL  = 3,  // unbound symbol (argument is the symbol name length)
  WIRE_LINK    = 4,  // bound symbol (argument is the symbol name length)
  WIRE_INTEGER = 5,  // integer (argument is the integer value)
  WIRE_FLOAT   = 6,  // floating-point number (argument is the float value)
  WIRE_SPECIAL = 7,  // special values
};

enum WireSpecial {
  WIRE_NIL      = 1,  // "nil" value
  WIRE_ID       = 2,  // "id" value
  WIRE_ISA      = 3,  // "isa" value
  WIRE_IS       = 4,  // "is" value
  WIRE_ARRAY    = 5,  // array, followed by array size and the arguments
  WIRE_INDEX    = 6,  // index value, followed by varint32 encoded integer
  WIRE_RESOLVE  = 7,  // resolve link, followed by slots and replacement index
};

// The binary marker (i.e. a nul character) is used for prefixing serialized
// SLING objects to indicate that they are binary encoded. The textual encoding
// will never contain a nul character. In binary encoding, a nul character is
// decoded as REF(0). This will never be the first tag in a binary encoding
// since initially there are no references to refer to.
enum EncodingMarker {
  WIRE_BINARY_MARKER = 0,
};

}  // namespace sling

#endif  // SLING_FRAME_WIRE_H_

