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

#ifndef SLING_FRAME_READER_H_
#define SLING_FRAME_READER_H_

#include <string>

#include "sling/base/macros.h"
#include "sling/frame/object.h"
#include "sling/frame/store.h"
#include "sling/frame/tokenizer.h"
#include "sling/stream/input.h"

namespace sling {

// The reader reads objects in text format and converts these to the internal
// object format.
class Reader : public Tokenizer {
 public:
  // Initialize reader with input where data is read from and object store where
  // parsed objects are stored.
  Reader(Store *store, Input *input)
      : Tokenizer(input), store_(store), stack_(store), references_(store) {}

  // Reads next object from input.
  Object Read();

  // Reads all objects from the input and returns the last value.
  Object ReadAll();

  // Reads next object from input and return a handle to it.
  Handle ReadObject();

  // In JSON-mode, string keys for frames are converted to names.
  bool json() const { return json_; }
  void set_json(bool json) { json_ = json; }

 protected:
  // Parses the next object from input.
  Handle ParseObject();

  // Parses frame from input.
  Handle ParseFrame();

  // Parses array from input.
  Handle ParseArray();

  // Parse id symbol from input.
  Handle ParseId();

  // Gets the current location in the handle stack.
  Word Mark() { return stack_.offset(stack_.end()); }

  // Pops elements off the stack.
  void Release(Word mark) { stack_.set_end(stack_.address(mark)); }

  // Push value onto stack.
  void Push(Handle h) { *stack_.push() = h; }

  // Pops value from stack.
  Handle Pop() { Handle h = *stack_.top(); stack_.pop(); return h; }

  // Returns indexed reference.
  Handle &Reference(int index) {
    if (index >= references_.size()) references_.resize(index + 1);
    return references_[index];
  }

  // Object store.
  Store *store() const { return store_; }

 private:
  // Object store for storing parsed objects.
  Store *store_;

  // Stack for storing intermediate objects while parsing.
  HandleSpace stack_;

  // Locally indexed frames.
  Handles references_;

  // In JSON mode, string keys for frames are converted to names.
  bool json_ = false;

  DISALLOW_IMPLICIT_CONSTRUCTORS(Reader);
};

}  // namespace sling

#endif  // SLING_FRAME_READER_H_

