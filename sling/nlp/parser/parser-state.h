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

#ifndef SLING_NLP_PARSER_PARSER_STATE_H_
#define SLING_NLP_PARSER_PARSER_STATE_H_

#include <unordered_set>
#include <utility>
#include <vector>

#include "sling/frame/object.h"
#include "sling/frame/store.h"
#include "sling/nlp/document/document.h"
#include "sling/nlp/parser/parser-action.h"

namespace sling {
namespace nlp {

// Parser state that represents the state of the transition-based parser. When
// actions are applied to the parser state, new spans and frames are added to
// the document.
class ParserState {
 public:
  // Slot in attention buffer.
  struct AttentionSlot {
    AttentionSlot(Handle frame, int step, Span *span)
      : frame(frame), created(step), focused(step), span(span) {}

    Handle frame;    // evoked frame
    int created;     // step that created frame
    int focused;     // step that last brought frame to the center of attention
    Span *span;      // last span that evoked this frame
  };

  // Position pushed on to the mark stack.
  struct Marker {
    Marker(int token, int step) : token(token), step(step) {}

    int token;   // start token for the mark
    int step;    // parse step when the mark was made
  };

  // Initializes parse state.
  ParserState(Document *document, int begin, int end);

  // Returns the underlying document.
  Document *document() const { return document_; }

  // Returns first token to be parsed.
  int begin() const { return begin_; }

  // Returns end token, i.e. first token not to be parsed.
  int end() const { return end_; }

  // Returns current input token.
  int current() const { return current_; }

  // Returns the current parse step.
  int step() const { return step_; }

  // Mark stack.
  const std::vector<Marker> &marks() const { return marks_; }

  // Applies parser action to transition parser to new state. Caller should
  // ensure that 'action' is applicable using CanApply().
  void Apply(const ParserAction &action);

  // Returns the first type for a frame in the attention buffer. This will be
  // the type specified when the frame was created with EVOKE/EMBED/ELABORATE.
  Handle Type(int index) const;

  // Gets the handles of the k frames that are closest to the center of
  // attention in the order of attention. There might be less than k frames if
  // there are fewer elements in the attention buffer.
  void GetFocus(int k, Handles *center) const;

  // Return the position in the attention buffer of a frame or -1 if the
  // frame is not in the attention buffer. The search can be
  // limited to the top-k frames that are closest to the center of attention.
  int AttentionIndex(Handle frame, int k = -1) const;

  // The parse is done when we have performed the first STOP action.
  bool done() const { return done_; }

  // Returns slot in attention buffer. The center of attention has index 0.
  const AttentionSlot &Attention(int index) const {
    return attention_[attention_.size() - index - 1];
  }
  AttentionSlot &Attention(int index) {
    return attention_[attention_.size() - index - 1];
  }

  // Returns the size of the attention buffer.
  int AttentionSize() const { return attention_.size(); }

  // Returns whether 'action' can be applied to the state.
  bool CanApply(const ParserAction &action) const;

  // Returns a human-readable representation of the state.
  string DebugString() const;

  // Returns the underlying store.
  Store *store() const { return document_->store(); }

 private:
  // Applies individual actions, which are assumed to be applicable.
  void Shift();
  void Stop();
  void Mark();
  void Evoke(int length, Handle type);
  void Refer(int length, int frame);
  void Connect(int source, Handle role, int target);
  void Assign(int frame, Handle role, Handle value);
  void Embed(int frame, Handle role, Handle type);
  void Elaborate(int frame, Handle role, Handle type);

  // Adds frame to attention buffer, making it the new center of attention.
  void Add(Handle frame, Span *span);

  // Moves element in attention buffer to the center of attention.
  void Center(int index, Span *span);

  // Document for the parser state. New frames and mentions are added to this
  // document when
  Document *document_;

  // Token range to be parsed.
  int begin_;
  int end_;

  // Current input token.
  int current_;

  // Current parse step.
  int step_;

  // When we have performed the first STOP action, the parse is done.
  bool done_;

  // Attention buffer. This contains evoked frames in order of attention. The
  // last element is the center of attention.
  std::vector<AttentionSlot> attention_;

  // Stack with positions for marked tokens. The mark stack tracks the start of
  // each mention.
  std::vector<Marker> marks_;

  // (Source/Target frame handle, Frame type) for frames embedded or elaborated
  // at the current position. This is cleared once the position advances.
  std::vector<std::pair<Handle, Handle>> embed_;
  std::vector<std::pair<Handle, Handle>> elaborate_;

  // Maximum mark depth.
  static const int MAX_MARK_DEPTH = 5;
};

}  // namespace nlp
}  // namespace sling

#endif  // SLING_NLP_PARSER_PARSER_STATE_H_

