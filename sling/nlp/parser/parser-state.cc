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

#include "sling/nlp/parser/parser-state.h"

#include "sling/base/logging.h"
#include "sling/frame/object.h"
#include "sling/frame/store.h"
#include "sling/nlp/document/document.h"
#include "sling/string/strcat.h"

namespace sling {
namespace nlp {

ParserState::ParserState(Store *store, int begin, int end)
    : store_(store),
      begin_(begin),
      end_(end),
      current_(begin),
      done_(false),
      frames_(store),
      nesting_(begin) {}

ParserState::ParserState(const ParserState &other)
    : store_(other.store_),
      begin_(other.begin_),
      end_(other.end_),
      current_(other.current_),
      done_(other.done_),
      frames_(other.store_),
      mentions_(other.mentions_),
      frame_to_mention_(other.frame_to_mention_),
      attention_(other.attention_),
      nesting_(other.nesting_) {
  frames_.assign(other.frames_.begin(), other.frames_.end());
}

int ParserState::MaxEvokeLength(int max_length) const {
  if (current_ == end_) return 0;
  int end = max_length + current_;
  if (end > end_) end = end_;

  return nesting_.MaxEnd(end) - current_;
}

string ParserState::DebugString() const {
  static const int kMaxAttention = 10;
  string s =
      StrCat("Begin:", begin_, " End:", end_, " Current:", current_,
             " Done: ", (done_ ? "Y" : "N"), " AttentionSize: ",
             attention_.size(), "\n");
  for (int i = 0; i < kMaxAttention; ++i) {
    if (i == attention_.size()) break;
    StrAppend(&s, "AttentionIndex: ", i, " FrameIndex: ", Attention(i),
              " FrameType:", store_->DebugString(type(i)), "\n");
  }
  if (attention_.size() > kMaxAttention) {
    StrAppend(&s, "..and ", (attention_.size() - kMaxAttention), " more.\n");
  }

  return s;
}

void ParserState::Apply(const ParserAction &action) {
  switch (action.type) {
    case ParserAction::SHIFT:
      Shift();
      break;

    case ParserAction::STOP:
      Stop();
      break;

    case ParserAction::EVOKE:
      Evoke(action.length, action.label);
      break;

    case ParserAction::REFER:
      Refer(action.length, action.target);
      break;

    case ParserAction::CONNECT:
      Connect(action.source, action.role, action.target);
      break;

    case ParserAction::ASSIGN:
      Assign(action.source, action.role, action.label);
      break;

    case ParserAction::EMBED:
      Embed(action.target, action.role, action.label);
      break;

    case ParserAction::ELABORATE:
      Elaborate(action.source, action.role, action.label);
      break;

    case ParserAction::CASCADE:
      LOG(FATAL) << "CASCADE action shouldn't reach ParserState";
      break;
  }
}

// Returns if 'frame' has a 'role' slot whose value is 'value'.
static bool SlotPresent(const Frame &frame, Handle role, Handle value) {
  for (const auto &slot : frame) {
    if (slot.name == role && slot.value == value) return true;
  }

  return false;
}

// Returns if 'frame' has a 'role' slot whose value is the index 'value'.
static bool SlotPresent(const Frame &frame, Handle role, int value) {
  return SlotPresent(frame, role, Handle::Index(value));
}

bool ParserState::CanApply(const ParserAction &action) const {
  switch (action.type) {
    case ParserAction::CASCADE:
      return false;

    case ParserAction::SHIFT:
      // Do not allow shifting past the end of the input buffer.
      return current_ < end_;

    case ParserAction::STOP:
      // Only allow stop if we are at the end of the input buffer.
      return current_ == end_;

    case ParserAction::EVOKE: {
      int length = action.length;

      // Check that phrase is inside the input buffer.
      int end = current_ + length;
      if (end > end_) return false;

      // Check that the proposed span doesn't cross any existing span.
      int max_end = nesting_.MaxEnd(end);
      if (max_end < end) return false;

      // See if we haven't already evoked the same span with the same type.
      if (max_end == end) {
        for (const auto &p : nesting_) {
          if (p.first > end) break;
          const auto &mention = mentions_[p.second];
          if (mention.begin < current_) break;
          if (FrameHasType(mention.frame, action.label)) return false;
        }
      }

      return true;
    }

    case ParserAction::REFER: {
      int length = action.length;
      int index = action.target;

      // Check that phrase is inside input buffer.
      int end = current_ + length;
      if (end > end_) return false;

      // Check that 'index' is valid.
      if (index >= attention_.size()) return false;

      // Check that the proposed span doesn't cross any existing span.
      int max_end = nesting_.MaxEnd(end);
      if (max_end < end) return false;

      // See if we haven't already referred to the same frame before.
      if (max_end == end) {
        int frame_index = Attention(index);
        for (const auto &p : nesting_) {
          if (p.first > end) break;
          const auto &mention = mentions_[p.second];
          if (mention.begin < current_) break;
          if (mention.frame == frame_index) return false;
        }
      }

      return true;
    }

    case ParserAction::ASSIGN: {
      // Check that we are not done.
      if (done_) return false;

      // Check that source is a valid frame.
      int source = action.source;
      if (source >= attention_.size()) return false;

      // Check that we haven't output this assignment in the past.
      Frame frame(store_, frames_[Attention(source)]);
      return !SlotPresent(frame, action.role, action.label);
    }

    case ParserAction::CONNECT: {
      // Check that we are not done.
      if (done_) return false;

      // Check that source and target are valid indices.
      int source = action.source;
      int target = action.target;
      if (source >= attention_.size()) return false;
      if (target >= attention_.size()) return false;

      // Check that we haven't output this connection before.
      Frame frame(store_, frames_[Attention(source)]);
      return !SlotPresent(frame, action.role, Attention(target));
    }

    case ParserAction::EMBED: {
      // Check that we are not done.
      if (done_) return false;

      // Check that target is a valid index into the attention buffer.
      int target = action.target;
      if (target >= attention_.size()) return false;

      // Check that we haven't embedded the same frame the same way.
      int target_index = Attention(target);
      for (const auto &e : embed_) {
        if (e.first == target_index && e.second == action.label) return false;
      }

      return true;
    }

    case ParserAction::ELABORATE: {
      // Check that we are not done.
      if (done_) return false;

      // Check that source is a valid index into the attention buffer.
      int source = action.source;
      if (source >= attention_.size()) return false;

      // Check that we haven't elaborated the same frame the same way.
      int source_index = Attention(source);
      for (const auto &e : elaborate_) {
        if (e.first == source_index && e.second == action.label) return false;
      }

      return true;
    }
  }

  return false;
}

void ParserState::Shift() {
  // Move to the next token in the input buffer.
  current_++;

  // Advance the nesting stack.
  nesting_.Advance();

  // Clear the states for EMBED and ELABORATE.
  embed_.clear();
  elaborate_.clear();
}

void ParserState::Stop() {
  done_ = true;
}

void ParserState::Evoke(int length, Handle type) {
  // Create new frame.
  Slot slot(Handle::isa(), type);
  Handle h = store_->AllocateFrame(&slot, &slot + 1);
  int frame = frames_.size();
  frames_.push_back(h);

  // Create new mention.
  int index = mentions_.size();
  mentions_.emplace_back(current_, current_ + length, frame);
  nesting_.Add(current_ + length, index);

  // Add new frame to the attention buffer.
  Add(frame);

  // Add frame->mention mapping.
  if (frame_to_mention_.size() < frame + 1) {
    frame_to_mention_.resize(frame + 1, -1);
  }
  frame_to_mention_[frame] = index;
}

void ParserState::Refer(int length, int index) {
  // Create new mention.
  mentions_.emplace_back(current_, current_ + length, Attention(index));
  Center(index);
}

void ParserState::Connect(int source, Handle role, int target) {
  // Create new frame with an additional role linking source to target. The role
  // value is encoded as an index into the frame buffer. The new frame replaces
  // the old frame in the frame buffer.
  int source_index = Attention(source);
  int target_index = Attention(target);
  Handle value = Handle::Index(target_index);
  frames_[source_index] = store_->Extend(frames_[source_index], role, value);

  // Move the source frame to the center of attention.
  Center(source);
}

void ParserState::Assign(int frame, Handle role, Handle value) {
  // Create new frame with an additional role. The new frame replaces the old
  // frame in the frame buffer.
  int index = Attention(frame);
  frames_[index] = store_->Extend(frames_[index], role, value);

  // Move the frame to the center of attention.
  Center(frame);
}

void ParserState::Embed(int frame, Handle role, Handle type) {
  // Create new frame with the specified type and add link to target frame.
  int target = Attention(frame);
  Slot slots[2];
  slots[0].name = Handle::isa();
  slots[0].value = type;
  slots[1].name = role;
  slots[1].value = Handle::Index(target);
  int index = frames_.size();
  frames_.push_back(store_->AllocateFrame(slots, slots + 2));
  embed_.emplace_back(target, type);

  // Add new frame to the attention buffer.
  Add(index);
}

void ParserState::Elaborate(int frame, Handle role, Handle type) {
  int source_index = Attention(frame);

  // Create new frame with the specified type.
  Slot slot(Handle::isa(), type);
  int target_index = frames_.size();
  frames_.push_back(store_->AllocateFrame(&slot, &slot + 1));

  // Add link to new frame from source frame.
  Handle value = Handle::Index(target_index);
  frames_[source_index] = store_->Extend(frames_[source_index], role, value);
  elaborate_.emplace_back(source_index, type);

  // Add new frame to the attention buffer.
  Add(target_index);
}

void ParserState::Center(int index) {
  if (index == 0) return;  // already the center of attention
  int frame = Attention(index);
  attention_.erase(attention_.end() - 1 - index);
  attention_.push_back(frame);
}

void ParserState::GetFocus(int k, std::vector<int> *center) const {
  center->clear();
  for (int i = attention_.size() - 1; i >= 0 && center->size() < k; --i) {
    center->push_back(attention_[i]);
  }
}

int ParserState::AttentionIndex(int index, int k) const {
  if (k < 0 || k > attention_.size()) k = attention_.size();
  for (int i = 0; i < k; ++i) {
    if (Attention(i) == index) return i;
  }
  return -1;
}

void ParserState::GetFrames(Handles *frames) {
  // Allocate new frames for all the frames in the frame buffer.
  frames->resize(frames_.size());
  for (int i = 0; i < frames_.size(); ++i) {
    // If the frame has any slots with index values we need to clone it and
    // update the indices to point to the final frames. Otherwise we can just
    // return the existing frame from the frame buffer.
    FrameDatum *frame = store_->GetFrame(frames_[i]);
    bool has_indices = false;
    for (Slot *slot = frame->begin(); slot < frame->end(); ++slot) {
      if (slot->value.IsIndex()) {
        has_indices = true;
        break;
      }
    }

    if (has_indices) {
      (*frames)[i] = store_->AllocateFrame(frame->slots());
    } else {
      (*frames)[i] = frames_[i];
    }
  }

  // Copy slots to the new frames translating indices to frame references.
  for (int i = 0; i < frames_.size(); ++i) {
    FrameDatum *source = store_->GetFrame(frames_[i]);
    FrameDatum *target = store_->GetFrame((*frames)[i]);
    if (target == source) continue;
    Slot *s = source->begin();
    Slot *end = source->end();
    Slot *t = target->begin();
    while (s < end) {
      t->name = s->name;
      t->value = s->value.IsIndex() ? (*frames)[s->value.AsIndex()] : s->value;
      s++;
      t++;
    }
  }
}

void ParserState::AddParseToDocument(Document *document) {
  CHECK(document->store() == store_);

  // Get frames generated by parse.
  Handles frames(store_);
  GetFrames(&frames);
  std::vector<bool> evoked(frames.size());

  // Add mentions to document document.
  for (Mention &m : mentions_) {
    Span *span = document->AddSpan(m.begin, m.end);
    if (span != nullptr) {
      span->Evoke(frames[m.frame]);
      evoked[m.frame] = true;
    }
  }

  // Add frames to document that are not evoked by a phrase as thematic frames.
  for (int i = 0; i < frames.size(); ++i) {
    if (!evoked[i]) document->AddTheme(frames[i]);
  }
}

Handle ParserState::type(int index) const {
  return store_->GetFrame(frames_[Attention(index)])->get(Handle::isa());
}

}  // namespace nlp
}  // namespace sling

