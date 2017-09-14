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

#ifndef NLP_PARSER_PARSER_STATE_H_
#define NLP_PARSER_PARSER_STATE_H_

#include <unordered_set>
#include <utility>
#include <vector>

#include "frame/object.h"
#include "frame/store.h"
#include "nlp/document/document.h"
#include "nlp/parser/parser-action.h"

namespace sling {
namespace nlp {

// Parser state that represents the state of the transition-based parser.
class ParserState {
 public:
  // Initializes parse state.
  ParserState(Store *store, int begin, int end);

  // Clones parse state.
  ParserState(const ParserState &other);

  // Returns first token to be parsed.
  int begin() const { return begin_; }

  // Returns end token, i.e. first token not to be parsed.
  int end() const { return end_; }

  // Returns current input token.
  int current() const { return current_; }

  // Applies parser action to transition parser to new state. Returns false if
  // the action is not valid in the current state.
  bool Apply(const ParserAction &action);

  // Returns the length of the longest span that can be evoked starting at the
  // current token. This length is capped by the smallest of:
  // - 'max_length'.
  // - the length of the remaining input.
  // - the end of any existing span that covers the current token.
  int MaxEvokeLength(int max_length) const;

  // Returns frame in frame buffer.
  Handle frame(int index) const { return frames_[index]; }

  // Returns the first type for a frame in the attention buffer. This will be
  // the type specified when the frame was created with EVOKE/EMBED/ELABORATE.
  Handle type(int index) const;

  // Gets the indices of the k frames that are closest to the center of
  // attention in the order of attention. There might be less than k frames if
  // there are fewer elements in the attention buffer.
  void GetFocus(int k, std::vector<int> *center) const;

  // Return the position in the attention buffer of a frame in the frame index
  // or -1 if the frame is not in the attention buffer. The search can be
  // limited to the top-k frames that are closest to the center of attention.
  int AttentionIndex(int index, int k = -1) const;

  // Creates final set of frames that the parse has generated.
  void GetFrames(Handles *frames);

  // Adds frames and mentions that the parse has generated to the document.
  void AddParseToDocument(Document *document);

  // The parse is done when we have performed the first STOP action.
  bool done() const { return done_; }

  // Returns element in attention buffer. The elements are numbered so the
  // frame at the center of attention has index 0.
  int Attention(int index) const {
    return attention_[attention_.size() - index - 1];
  }

  // Returns the size of the attention buffer.
  int AttentionSize() const { return attention_.size(); }

  // Returns whether 'action' can be applied to the state.
  bool CanApply(const ParserAction &action) const;

  // Returns a human readable representation of the state.
  string DebugString() const;

  // Returns the underlying store.
  Store *store() const { return store_; }

  // Returns the start token (if any, else -1) for frame with absolute index
  // of 'frame_index'.
  int FrameEvokeBegin(int frame_index) const {
    if (frame_index >= frame_to_mention_.size()) return -1;
    int m = frame_to_mention_[frame_index];
    return (m < 0) ? -1 : mentions_[m].begin;
  }

  // Returns the end token (if any, else -1) for frame with absolute index
  // of 'frame_index'.
  int FrameEvokeEnd(int frame_index) const {
    if (frame_index >= frame_to_mention_.size()) return -1;
    int m = frame_to_mention_[frame_index];
    return (m < 0) ? -1 : mentions_[m].end;
  }

  // Number of spans covering the current token.
  int NestingLevel() const { return nesting_.NestingLevel(); }

 private:
  // Applies individual actions, which are assumed to be applicable.
  void Shift();
  void Stop();
  void Evoke(int length, Handle type);
  void Refer(int length, int frame);
  void Connect(int source, Handle role, int target);
  void Assign(int frame, Handle role, Handle value);
  void Embed(int frame, Handle role, Handle type);
  void Elaborate(int frame, Handle role, Handle type);

  // Returns true if the frame at the given absolute index has the given type.
  bool FrameHasType(int index, Handle type) const {
    return Frame(store_, frame(index)).GetHandle(Handle::isa()) == type;
  }

  // Adds frame to attention buffer. The frame will become the new center of
  // attention.
  void Add(int frame) { attention_.push_back(frame); }

  // Moves element in attention buffer to the center of attention.
  void Center(int index);

  // Mention evoking a frame.
  struct Mention {
    Mention(int b, int e, int f) : begin(b), end(e), frame(f) {}

    // Phrase boundary (semi-open interval).
    int begin;
    int end;

    // Index of frame in the frame buffer that this phrase evokes.
    int frame;
  };

  // Stack for tracking span nesting. This is only populated when some spans
  // are currently open. Spans are stored ordered by nesting level, so the top
  // of the stack is the innermost nested span currently open.
  struct Nesting {
    // Constructors.
    explicit Nesting(int begin) { current = begin; }
    Nesting(const Nesting &n) : spans(n.spans), current(n.current) {}

    // (End position (exclusive), mention index).
    std::vector<std::pair<int, int>> spans;

    // Current input buffer position.
    int current = 0;

    // Moves the current position ahead and pops relevant nesting information.
    void Advance() {
      current++;
      while (!spans.empty()) {
        DCHECK_GE(spans.back().first, current);
        if (spans.back().first == current) {
          spans.pop_back();
        } else {
          break;
        }
      }
    }

    // Returns true if a span can be added with the current position as the
    // start and 'end' as the end.
    bool Valid(int end) const {
      // Span is valid if it doesn't go beyond the innermost span.
      return spans.empty() || (spans.back().first >= end);
    }

    // Returns the maximum end of a span that can start at the current position.
    int MaxEnd(int default_end) const {
      return spans.empty() ? default_end : spans.back().first;
    }

    // Adds a span to the nesting.
    void Add(int end, int index) {
      DCHECK(Valid(end));
      spans.emplace_back(end, index);
    }

    // Number of spans covering the current token.
    int NestingLevel() const { return spans.size(); }

    // Iterator that begins from the innermost mention.
    typedef std::vector<std::pair<int, int>>::const_reverse_iterator iterator;
    iterator begin() const { return spans.rbegin(); }
    iterator end() const { return spans.rend(); }
  };

  // SLING store for allocating frames.
  Store *store_;

  // Token range to be parsed.
  int begin_;
  int end_;

  // Current input token.
  int current_;

  // When we have performed the first STOP action, the parse is done.
  bool done_;

  // List of all evoked frames. References between frames are encoded using
  // index handles (@n) into this array. Frames are modified using copy-on-write
  // so it is safe to copy the handles when cloning a parse state.
  Handles frames_;

  // List of mentions evoking frames.
  std::vector<Mention> mentions_;

  // Absolute frame index -> Index of mention that evoked it (or -1).
  std::vector<int> frame_to_mention_;

  // Center of attention. This contains indices of frames in order of attention.
  // The last frame in the attention vector is the frame closest to the center
  // of attention.
  std::vector<int> attention_;

  // Span nesting information.
  Nesting nesting_;

  // (Source/Target frame index, Frame type) for frames embedded or elaborated
  // at the current position. This is cleared once the position advances.
  std::vector<std::pair<int, Handle>> embed_;
  std::vector<std::pair<int, Handle>> elaborate_;
};

}  // namespace nlp
}  // namespace sling

#endif  // NLP_PARSER_PARSER_STATE_H_

