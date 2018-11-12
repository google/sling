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

// Parser state that represents the state of the transition-based parser.
class ParserState {
 public:
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

  // Applies parser action to transition parser to new state. Caller should
  // ensure that 'action' is applicable using CanApply().
  void Apply(const ParserAction &action);

  // Returns the first type for a frame in the attention buffer. This will be
  // the type specified when the frame was created with EVOKE/EMBED/ELABORATE.
  Handle type(int index) const;

  // Gets the handles of the k frames that are closest to the center of
  // attention in the order of attention. There might be less than k frames if
  // there are fewer elements in the attention buffer.
  void GetFocus(int k, Handles *center) const;

  // Return the position in the attention buffer of a frame or -1 if the
  // frame is not in the attention buffer. The search can be
  // limited to the top-k frames that are closest to the center of attention.
  int AttentionIndex(Handle handle, int k = -1) const;

  // Creates final set of frames that the parse has generated.
  void GetFrames(Handles *frames);

  // Adds frames and mentions that the parse has generated to the document.
  void AddParseToDocument(Document *document);

  // The parse is done when we have performed the first STOP action.
  bool done() const { return done_; }

  // Returns handle of the frame in attention buffer. The center of attention
  // has index 0.
  Handle Attention(int index) const {
    return attention_[attention_.size() - index - 1];
  }

  // Returns the size of the attention buffer.
  int AttentionSize() const { return attention_.size(); }

  // Returns whether 'action' can be applied to the state.
  bool CanApply(const ParserAction &action) const;

  // Returns a human readable representation of the state.
  string DebugString() const;

  // Returns the underlying store.
  Store *store() const { return document_->store(); }

  // Returns the start token of the first span (if any) that evokes the
  // frame at the specified attention buffer index, -1 otherwise.
  int FrameEvokeBegin(int attention_index) const;

  // Returns the end token of the first span (exclusive; if any) that evokes
  // the frame at the specified attention buffer index, -1 otherwise.
  int FrameEvokeEnd(int attention_index) const;

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
  void Add(Handle handle) { attention_.push_back(handle); }

  // Moves element in attention buffer to the center of attention.
  void Center(int index);

  // SLING document underlying the parser state.
  Document *document_;

  // Token range to be parsed.
  int begin_;
  int end_;

  // Current input token.
  int current_;

  // When we have performed the first STOP action, the parse is done.
  bool done_;

  // Attention buffer. This contains handles of frames in order of attention.
  Handles attention_;

  // Token index for marked tokens.
  std::vector<int> marks_;

  // (Source/Target frame handle, Frame type) for frames embedded or elaborated
  // at the current position. This is cleared once the position advances.
  std::vector<std::pair<Handle, Handle>> embed_;
  std::vector<std::pair<Handle, Handle>> elaborate_;
};

}  // namespace nlp
}  // namespace sling

#endif  // SLING_NLP_PARSER_PARSER_STATE_H_

