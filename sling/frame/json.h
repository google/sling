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

#ifndef SLING_FRAME_JSON_H_
#define SLING_FRAME_JSON_H_

#include <string>

#include "sling/base/macros.h"
#include "sling/frame/object.h"
#include "sling/frame/store.h"
#include "sling/stream/output.h"

namespace sling {

// The JSON writer outputs objects in JSON format.
class JSONWriter {
 public:
  // Initializes writer with store and output.
  JSONWriter(const Store *store, Output *output)
      : store_(store), output_(output), global_(store->globals() == nullptr) {}

  // Write object on output.
  void Write(const Object &object);
  void Write(Handle handle);

  // Configuration parameters.
  void set_indent(int indent) { indent_ = indent; }
  void set_shallow(bool shallow) { shallow_ = shallow; }
  void set_global(bool global) { global_ = global; }
  void set_byref(bool byref) { byref_ = byref; }

 private:
  // Returns true if the output should be pretty-printed.
  bool pretty() const { return indent_ > 0; }

  // Writes character on output.
  void WriteChar(char ch) {
    output_->WriteChar(ch);
  }

  // Writes two character on output.
  void WriteChars(char ch1, char ch2) {
    output_->WriteChar(ch1);
    output_->WriteChar(ch2);
  }

  // Writes quoted string with escapes.
  void WriteString(const StringDatum *str);

  // Writes frame.
  void WriteFrame(const FrameDatum *frame);

  // Writes array.
  void WriteArray(const ArrayDatum *array);

  // Writes symbol.
  void WriteSymbol(const SymbolDatum *symbol);

  // Writes integer.
  void WriteInt(int number);

  // Writes floating-point number.
  void WriteFloat(float number);

  // Writes link.
  void WriteLink(Handle handle, bool reference);

  // Object store.
  const Store *store_;

  // Output for writer.
  Output *output_;

  // Amount by which output would be indented. Zero means no indentation.
  int indent_ = 0;

  // Current indentation (number of spaces).
  int current_indentation_ = 0;

  // Output frames with public ids by reference.
  bool shallow_ = true;

  // Output frames in the global store by value.
  bool global_;

  // Output anonymous frames by reference using index ids.
  bool byref_ = true;

  // Mapping of frames that have been output mapped to their ids.
  HandleMap<Handle> references_;

  // Next index reference.
  int next_index_ = 1;

  DISALLOW_IMPLICIT_CONSTRUCTORS(JSONWriter);
};

}  // namespace sling

#endif  // SLING_FRAME_JSON_H_

