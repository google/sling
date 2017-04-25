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

#ifndef FRAME_PRINTER_H_
#define FRAME_PRINTER_H_

#include <string>

#include "base/macros.h"
#include "frame/object.h"
#include "frame/store.h"
#include "stream/output.h"

namespace sling {

// The printer outputs objects in human-readable text format which can be read
// by the reader.
class Printer {
 public:
  // Intializes printer with store and output.
  Printer(const Store *store, Output *output)
      : store_(store), output_(output), global_(store->globals() == nullptr) {}

  // Prints object on output.
  void Print(const Object &object);

  // Prints handle value relative to a store.
  void Print(Handle handle) { Print(handle, false); }

  // Prints all frames in the symbol table of the store.
  void PrintAll();

  // Configuration parameters.
  void set_indent(int indent) { indent_ = indent; }
  void set_shallow(bool shallow) { shallow_ = shallow; }
  void set_global(bool global) { global_ = global; }
  void set_byref(bool byref) { byref_ = byref; }

 private:
  // Returns true if the output should be indented.
  bool indent() const { return indent_ > 0; }

  // Prints character on output.
  void WriteChar(char ch) {
    output_->WriteChar(ch);
  }

  // Prints two character on output.
  void WriteChars(char ch1, char ch2) {
    output_->WriteChar(ch1);
    output_->WriteChar(ch2);
  }

  // Prints quoted string with escapes.
  void PrintString(const StringDatum *str);

  // Prints object.
  void Print(Handle handle, bool reference);

  // Prints frame.
  void PrintFrame(const FrameDatum *frame);

  // Prints array.
  void PrintArray(const ArrayDatum *array);

  // Prints symbol.
  void PrintSymbol(const SymbolDatum *symbol, bool reference);

  // Prints integer.
  void PrintInt(int number);

  // Prints floating-point number.
  void PrintFloat(float number);

  // Prints link.
  void PrintLink(Handle handle);

  // Object store.
  const Store *store_;

  // Output for printer.
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

  // Mapping of frames that have been printed mapped to their ids.
  HandleMap<Handle> references_;

  // Next index reference.
  int next_index_ = 1;

  DISALLOW_IMPLICIT_CONSTRUCTORS(Printer);
};

}  // namespace sling

#endif  // FRAME_PRINTER_H_

