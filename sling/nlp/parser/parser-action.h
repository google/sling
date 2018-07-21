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

#ifndef SLING_NLP_PARSER_PARSER_ACTION_H_
#define SLING_NLP_PARSER_PARSER_ACTION_H_

#include <string>

#include "sling/base/types.h"
#include "sling/frame/store.h"
#include "sling/util/fingerprint.h"

namespace sling {
namespace nlp {

// Parser action for transitioning from one parser state to another.
struct ParserAction {
  // Transition type.
  enum Type : uint8 {
    // Skips the next input token. Only valid when not at the end of the input
    // buffer.
    SHIFT,

    // Signals that we have reach the end of the parse. This is only valid when
    // at the end of the input buffer. Multiple STOP actions can be added to
    // the transition sequence to make all sequences in a beam have the same
    // length.
    STOP,

    // Evokes frame of with type 'type' from the next 'length' tokens in the
    // input. The new frame will become the center of attention.
    EVOKE,

    // Makes a new mention of an existing frame. This frame will become the new
    // center of attention.
    REFER,

    // Adds slot to frame 'source' with name 'role' and value 'target'. The
    // source frame become the new center of attention.
    CONNECT,

    // Adds slot to frame 'source' with name 'role' and value 'type' and moves
    // frame to the center of attention.
    ASSIGN,

    // Create new frame with type 'type' and add a slot to it with name 'role'
    // and value 'target', where target is a frame in the attention buffer.
    // The new frame become the new center of attention.
    EMBED,

    // Create new frame with type 'type' and add a slot to an existing frame
    // 'source' in the attention buffer with 'role' set to the new frame.
    // The new frame become the new center of attention.
    ELABORATE,

    // Delegate to another member (specified by 'delegate') of the cascade.
    CASCADE,
  };

  // Number of action types.
  static const int kNumActionTypes = CASCADE + 1;

  // Type of the action.
  Type type;

  // Transition parameters.
  // Length of the evoked frame for EVOKE and REFER.
  uint8 length;

  // Source frame index for CONNECT, ASSIGN, ELABORATE.
  uint8 source;

  // Target frame index for CONNECT, EMBED, REFER.
  uint8 target;

  // Role argument for CONNECT, ASSIGN, EMBED, ELABORATE.
  Handle role;

  // Frame type for EVOKE, EMBED, ELABORATE, and value for ASSIGN.
  Handle label;

  // Index of the delegate for CASCADE actions.
  uint8 delegate;

  // Default constructor.
  ParserAction() { memset(this, 0, sizeof(struct ParserAction)); }

  // Copy constructor.
  ParserAction(const ParserAction &other) {
    memcpy(this, &other, sizeof(struct ParserAction));
  }

  // Other constructors.
  explicit ParserAction(Type t) : ParserAction() {
    type = t;
  }
  ParserAction(Type t, uint8 len) : ParserAction() {
    type = t;
    length = len;
  }

  // Checks for equality with 'other'.
  inline bool operator ==(const ParserAction &other) const {
    return memcmp(this, &other, sizeof(struct ParserAction)) == 0;
  }

  // Checks for inequality with 'other'.
  inline bool operator !=(const ParserAction &other) const {
    return !(*this == other);
  }

  // Returns the type name of the action.
  string TypeName() const;

  // Returns name of action type.
  static string TypeName(Type type);

  // Returns a human-readable representation of the action.
  string ToString(Store *store) const;

  // Returns a SHIFT action.
  static ParserAction Shift() {
    return ParserAction(ParserAction::SHIFT);
  }

  // Returns an EVOKE action.
  static ParserAction Evoke(uint8 length, Handle type) {
    ParserAction action(ParserAction::EVOKE, length);
    action.label = type;
    return action;
  }
} __attribute__ ((packed));

// Hasher for ParserAction.
struct ParserActionHash {
  size_t operator()(const ParserAction &action) const {
    return Fingerprint(
        reinterpret_cast<const char *>(&action),
        sizeof(struct ParserAction));
  }
};

}  // namespace nlp
}  // namespace sling

#endif  // SLING_NLP_PARSER_PARSER_ACTION_H_

