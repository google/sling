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

ParserState::ParserState(Document *document, int begin, int end)
    : document_(document),
      begin_(begin),
      end_(end),
      current_(begin),
      step_(0),
      done_(false) {}

string ParserState::DebugString() const {
  static const int MAX_ATTENTION = 10;
  string s =
      StrCat("Begin:", begin_, " End:", end_, " Current:", current_,
             " Done: ", (done_ ? "Y" : "N"), " AttentionSize: ",
             attention_.size(), "\n");
  for (int i = 0; i < attention_.size(); ++i) {
    if (i == MAX_ATTENTION) {
      StrAppend(&s, "..and ", (attention_.size() - MAX_ATTENTION), " more.\n");
      break;
    }
    StrAppend(&s, "AttentionIndex: ", i,
              " FrameType:", store()->DebugString(Type(i)), "\n");
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

    case ParserAction::MARK:
      Mark();
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
      LOG(FATAL) << "Cannot apply CASCADE action";
      break;
  }
  step_++;
}

bool ParserState::CanApply(const ParserAction &action) const {
  if (done_) return false;
  switch (action.type) {
    case ParserAction::CASCADE:
      // Do not allow cascading back to the main cascade.
      return action.delegate > 0;

    case ParserAction::SHIFT:
      // Do not allow shifting past the end of the input buffer.
      return current_ < end_;

    case ParserAction::STOP:
      // Only allow stop if we are at the end of the input buffer.
      return current_ == end_;

    case ParserAction::MARK:
      return current_ < end_ && marks_.size() < MAX_MARK_DEPTH;

    case ParserAction::EVOKE: {
      int begin, end;
      if (action.length == 0) {
        // EVOKE paired with MARK.
        if (marks_.empty()) return false;
        begin = marks_.back().token;
        end = current_ + 1;
      } else {
        // EVOKE with explicit length.
        begin = current_;
        end = current_ + action.length;
      }

      // Check that phrase is inside the input buffer.
      if (end > end_) return false;

      // Check for crossing spans.
      bool crossing = false;
      Span *enclosing = document_->EnclosingSpan(begin, end, &crossing);
      if (crossing) return false;

      // Check for duplicates.
      if (enclosing != nullptr &&
          enclosing->begin() == begin &&
          enclosing->end() == end) {
        return !enclosing->EvokesType(action.label);
      }
      return true;
    }

    case ParserAction::REFER: {
      int begin, end;
      if (action.length == 0) {
        // REFER paired with MARK.
        if (marks_.empty()) return false;
        begin = marks_.back().token;
        end = current_ + 1;
      } else {
        // REFER with explicit length.
        begin = current_;
        end = current_ + action.length;
      }

      // Check that phrase is inside the input buffer.
      if (end > end_) return false;

      // Check that antecedent index is valid.
      int index = action.target;
      if (index >= attention_.size()) return false;

      // Check for crossing spans.
      bool crossing = false;
      Span *enclosing = document_->EnclosingSpan(begin, end, &crossing);
      if (crossing) return false;

      // Check for duplicates.
      if (enclosing != nullptr &&
          enclosing->begin() == begin &&
          enclosing->end() == end) {
        Handle antecedent = Attention(index).frame;
        if (enclosing->Evokes(antecedent)) return false;
      }

      return true;
    }

    case ParserAction::ASSIGN: {
      // Check that source is a valid frame.
      int source = action.source;
      if (source >= attention_.size()) return false;

      // Check that we haven't output this assignment in the past.
      Frame frame(store(), Attention(source).frame);
      return !frame.Has(action.role, action.label);
    }

    case ParserAction::CONNECT: {
      // Check that source and target are valid indices.
      int source = action.source;
      int target = action.target;
      if (source >= attention_.size()) return false;
      if (target >= attention_.size()) return false;

      // Check that we haven't output this connection before.
      Frame frame(store(), Attention(source).frame);
      return !frame.Has(action.role, Attention(target).frame);
    }

    case ParserAction::EMBED: {
      // Check that target is a valid index into the attention buffer.
      if (action.target >= attention_.size()) return false;

      // Check that we haven't embedded the same frame the same way.
      Handle target = Attention(action.target).frame;
      for (const auto &e : embed_) {
        if (e.first == target && e.second == action.label) return false;
      }

      return true;
    }

    case ParserAction::ELABORATE: {
      // Check that source is a valid index into the attention buffer.
      if (action.source >= attention_.size()) return false;

      // Check that we haven't elaborated the same frame the same way.
      Handle source = Attention(action.source).frame;
      for (const auto &e : elaborate_) {
        if (e.first == source && e.second == action.label) return false;
      }

      return true;
    }
  }

  return false;
}

void ParserState::Shift() {
  // Move to the next token in the input buffer.
  current_++;

  // Clear the states for EMBED and ELABORATE.
  embed_.clear();
  elaborate_.clear();
}

void ParserState::Stop() {
  done_ = true;
}

void ParserState::Evoke(int length, Handle type) {
  // Create new frame.
  Handle frame;
  if (type.IsNil()) {
    // Allocate empty frame.
    frame = store()->AllocateFrame(nullptr, nullptr);
  } else {
    // Allocate frame with type.
    Slot slot(Handle::isa(), type);
    frame = store()->AllocateFrame(&slot, &slot + 1);
  }

  // Get or create a new mention.
  int begin, end;
  if (length == 0) {
    begin = marks_.back().token;
    end = current_ + 1;
    marks_.pop_back();
  } else {
    begin = current_;
    end = current_ + length;
  }
  Span *span = document_->AddSpan(begin, end);
  DCHECK(span != nullptr) << begin << " " << end;
  span->Evoke(frame);

  // Add new frame to the attention buffer.
  Add(frame, span);
}

void ParserState::Refer(int length, int index) {
  // Get or create a new mention.
  int begin, end;
  if (length == 0) {
    begin = marks_.back().token;
    end = current_ + 1;
    marks_.pop_back();
  } else {
    begin = current_;
    end = current_ + length;
  }
  Span *span = document_->AddSpan(begin, end);

  // Refer to an existing frame.
  Handle antecedent = Attention(index).frame;
  span->Evoke(antecedent);
  Center(index, span);
}

void ParserState::Mark() {
  marks_.emplace_back(current_, step_);
}

void ParserState::Connect(int source, Handle role, int target) {
  // Create new slot with the specified role linking source to target.
  Handle subject = Attention(source).frame;
  Handle object = Attention(target).frame;
  store()->Add(subject, role, object);

  // Move the source frame to the center of attention.
  Center(source, nullptr);
}

void ParserState::Assign(int frame, Handle role, Handle value) {
  // Create new slot in the source frame.
  Handle subject = Attention(frame).frame;
  store()->Add(subject, role, value);

  // Move the frame to the center of attention.
  Center(frame, nullptr);
}

void ParserState::Embed(int frame, Handle role, Handle type) {
  // Create new frame with the specified type and add link to target frame.
  Handle target = Attention(frame).frame;
  Slot slots[2];
  slots[0].name = Handle::isa();
  slots[0].value = type;
  slots[1].name = role;
  slots[1].value = target;
  Handle h = store()->AllocateFrame(slots, slots + 2);
  embed_.emplace_back(target, type);

  // Add new frame to the attention buffer.
  Add(h, nullptr);

  // Add new frame as a thematic frame to the document.
  document_->AddTheme(h);
}

void ParserState::Elaborate(int frame, Handle role, Handle type) {
  // Create new frame with the specified type.
  Handle source = Attention(frame).frame;
  Slot slot(Handle::isa(), type);
  Handle target = store()->AllocateFrame(&slot, &slot + 1);

  // Add new frame as a thematic frame to the document.
  document_->AddTheme(target);

  // Add link to new frame from source frame.
  store()->Add(source, role, target);
  elaborate_.emplace_back(Attention(frame).frame, type);

  // Add new frame to the attention buffer.
  Add(target, nullptr);
}

void ParserState::Add(Handle frame, Span *span) {
  attention_.emplace_back(frame, step_, span);
}

void ParserState::Center(int index, Span *span) {
  if (index == 0) {
    // Already center of attention.
    AttentionSlot &attention = Attention(index);
    attention.focused = step_;
    if (span != nullptr) attention.span = span;
  } else {
    // Move slot to the center of attention.
    AttentionSlot attention = Attention(index);
    attention.focused = step_;
    if (span != nullptr) attention.span = span;
    attention_.erase(attention_.end() - 1 - index);
    attention_.push_back(attention);
  }
}

void ParserState::GetFocus(int k, Handles *center) const {
  center->clear();
  for (int i = attention_.size() - 1; i >= 0 && center->size() < k; --i) {
    center->push_back(attention_[i].frame);
  }
}

int ParserState::AttentionIndex(Handle frame, int k) const {
  if (k < 0 || k > attention_.size()) k = attention_.size();
  for (int i = 0; i < k; ++i) {
    if (Attention(i).frame == frame) return i;
  }
  return -1;
}

Handle ParserState::Type(int index) const {
  return store()->GetFrame(Attention(index).frame)->get(Handle::isa());
}

}  // namespace nlp
}  // namespace sling

