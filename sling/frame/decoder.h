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

#ifndef SLING_FRAME_DECODER_H_
#define SLING_FRAME_DECODER_H_

#include <string>

#include "sling/base/macros.h"
#include "sling/frame/object.h"
#include "sling/frame/store.h"
#include "sling/stream/input.h"

namespace sling {

// The decoder decodes objects in binary format and loads them into a store.
class Decoder {
 public:
  // Initializes decoder with store where objects should be stored and input
  // where objects are read from.
  Decoder(Store *store, Input *input);

  // Decodes the next object from the input.
  Object Decode();

  // Decodes all objects from the input and returns the last value.
  Object DecodeAll();

  // Returns true when there are no more objects in the input.
  bool done() { return input_->done(); }

  // Decodes object from input and returns handle to it.
  Handle DecodeObject();

  // Skips frames in the input which are already in the store.
  void set_skip_known_frames(bool b) { skip_known_frames_ = b; }

 private:
  // Decodes frame from input.
  Handle DecodeFrame(int slots, int replace);

  // Decodes string from input.
  Handle DecodeString(int size);

  // Decodes array from input.
  Handle DecodeArray();

  // Decodes unbound symbol from input.
  Handle DecodeSymbol(int name_size);

  // Decodes bound symbol from input.
  Handle DecodeLink(int name_size);

  // Gets the current location in the stack.
  Word Mark() { return stack_.offset(stack_.end()); }

  // Pushes value onto stack.
  void Push(Handle h) { *stack_.push() = h; }

  // Replaces element on top of stack.
  void ReplaceTop(Handle h) {
    stack_.pop();
    *stack_.push() = h;
  }

  // Pops elements off the stack.
  void Release(Word mark) { stack_.set_end(stack_.address(mark)); }

  // Returns handle for reference.
  Handle Reference(uint32 index) {
    Handle *h = references_.base() + index;
    CHECK(h < references_.end());
    return *h;
  }

  // Object store for storing decoded objects.
  Store *store_;

  // Decoder input.
  Input *input_;

  // References to previously decoded objects.
  HandleSpace references_;

  // Stack for storing intermediate values while decoding objects.
  HandleSpace stack_;

  // Frames that already exist in the store can be skipped by the decoder.
  bool skip_known_frames_ = false;

  DISALLOW_IMPLICIT_CONSTRUCTORS(Decoder);
};

}  // namespace sling

#endif  // SLING_FRAME_DECODER_H_

