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

#ifndef SLING_FRAME_ENCODER_H_
#define SLING_FRAME_ENCODER_H_

#include <string>

#include "sling/base/macros.h"
#include "sling/frame/object.h"
#include "sling/frame/store.h"
#include "sling/stream/output.h"

namespace sling {

// The encoder encodes objects in binary format.
class Encoder {
 public:
  // Initializes encoder.
  Encoder(const Store *store, Output *output);

  // Encodes object to output.
  void Encode(const Object &object) { EncodeObject(object.handle()); }
  void Encode(Handle handle) { EncodeObject(handle); }

  // Encodes all frames in the symbol table of the store.
  void EncodeAll();

  // Configuration parameters.
  void set_shallow(bool shallow) { shallow_ = shallow; }
  void set_global(bool global) { global_ = global; }

 private:
  // Object encoding states.
  enum Status {
    UNRESOLVED,  // object has not been encoded in the output
    LINKED,      // only a link to the object has been encoded in the output
    ENCODED,     // object has been encoded in the output
  };

  // Reference to previously encoded or linked object.
  struct Reference {
    // Creates unresolved reference for new object.
    Reference() : status(UNRESOLVED), index(0) {}

    // Creates  resolved reference for pre-defined values.
    explicit Reference(int idx) : status(ENCODED), index(idx) {}

    Status status;  // reference status
    int index;      // reference number
  };

  // Encodes object for handle.
  void EncodeObject(Handle handle);

  // Encodes object link.
  void EncodeLink(Handle handle);

  // Encodes symbol.
  void EncodeSymbol(const SymbolDatum *symbol, int type);

  // Writes tag to output.
  void WriteTag(int tag, uint64 arg) {
    output_->WriteVarint64(tag | (arg << 3));
  }

  // Writes reference to output.
  void WriteReference(const Reference &ref);

  // Object store.
  const Store *store_;

  // Output for encoder.
  Output *output_;

  // Hash table mapping handles to object reference numbers for all objects that
  // have been encoded so far.
  HandleMap<Reference> references_;

  // Next available reference index.
  int next_index_ = 0;

  // Output frames with public ids by reference.
  bool shallow_ = true;

  // Output frames in the global store by value.
  bool global_;

  DISALLOW_IMPLICIT_CONSTRUCTORS(Encoder);
};

}  // namespace sling

#endif  // SLING_FRAME_ENCODER_H_
