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

#include "sling/nlp/document/document.h"

#include <algorithm>
#include <string>
#include <vector>

#include "sling/base/logging.h"
#include "sling/base/types.h"
#include "sling/frame/object.h"
#include "sling/frame/store.h"
#include "sling/nlp/document/fingerprinter.h"
#include "sling/nlp/document/token-breaks.h"

namespace sling {
namespace nlp {

void Span::Evoke(const Frame &frame) {
  mention_.Add(document_->n_evokes_, frame);
  document_->AddMention(frame.handle(), this);
}

void Span::Evoke(Handle frame) {
  mention_.Add(document_->n_evokes_, frame);
  document_->AddMention(frame, this);
}

Frame Span::Evoked(Handle type) const {
  Handle n_evokes = document_->n_evokes_.handle();
  for (const Slot &slot : mention_) {
    if (slot.name != n_evokes) continue;
    Frame frame(document_->store(), slot.value);
    if (frame.IsA(type)) return frame;
  }

  return Frame::nil();
}

Frame Span::Evoked(const Name &type) const {
  Handle n_evokes = document_->n_evokes_.handle();
  for (const Slot &slot : mention_) {
    if (slot.name != n_evokes) continue;
    Frame frame(document_->store(), slot.value);
    if (frame.IsA(type)) return frame;
  }

  return Frame::nil();
}

Frame Span::Evoked() const {
  Handle n_evokes = document_->n_evokes_.handle();
  for (const Slot &slot : mention_) {
    if (slot.name == n_evokes) {
      return Frame(document_->store(), slot.value);
    }
  }

  return Frame::nil();
}

bool Span::Evokes(Handle type) const {
  Handle n_evokes = document_->n_evokes_.handle();
  for (const Slot &slot : mention_) {
    if (slot.name != n_evokes) continue;
    Frame frame(document_->store(), slot.value);
    if (frame.IsA(type)) return true;
  }

  return false;
}

bool Span::Evokes(const Name &type) const {
  Handle n_evokes = document_->n_evokes_.handle();
  for (const Slot &slot : mention_) {
    if (slot.name != n_evokes) continue;
    Frame frame(document_->store(), slot.value);
    if (frame.IsA(type)) return true;
  }

  return false;
}

string Span::GetText() const {
  return document_->PhraseText(begin_, end_);
}

uint64 Span::Fingerprint() {
  // Check for cached fingerprint.
  if (fingerprint_ != 0) return fingerprint_;

  // Compute span fingerprint.
  uint64 fp = document_->PhraseFingerprint(begin_, end_);
  fingerprint_ = fp;
  return fp;
}

Document::Document(Store *store) : themes_(store) {
  // Bind names.
  CHECK(names_.Bind(store));

  // Build empty document.
  Builder builder(store);
  builder.AddIsA(n_document_);
  top_ = builder.Create();
}

Document::Document(const Frame &top) : top_(top), themes_(top.store()) {
  // Bind names.
  CHECK(names_.Bind(store()));

  // Add document frame if it is missing.
  if (!top_.valid()) {
    Builder builder(store());
    builder.AddIsA(n_document_);
    top_ = builder.Create();
  }

  // Get tokens.
  Array tokens = top_.Get(n_document_tokens_).AsArray();
  if (tokens.valid()) {
    // Initialize tokens.
    int num_tokens = tokens.length();
    tokens_.resize(num_tokens);
    for (int i = 0; i < num_tokens; ++i) {
      // Get token information from token frame.
      Handle h = tokens.get(i);
      FrameDatum *token = store()->GetFrame(h);
      Handle text = token->get(n_token_text_.handle());
      Handle start = token->get(n_token_start_.handle());
      Handle length = token->get(n_token_length_.handle());
      Handle brk = token->get(n_token_break_.handle());

      // Fill token from frame.
      Token &t = tokens_[i];
      t.document_ = this;
      t.handle_ = h;
      t.index_ = i;
      if (!start.IsNil() && !length.IsNil()) {
        t.begin_ = start.AsInt();
        t.end_ = t.begin_ + length.AsInt();
      } else {
        t.begin_ = -1;
        t.end_ = -1;
      }
      if (!text.IsNil()) {
        StringDatum *str = store()->GetString(text);
        t.text_.assign(str->data(), str->size());
      }
      if (!brk.IsNil()) {
        t.brk_ = static_cast<BreakType>(brk.AsInt());
      } else {
        t.brk_ = SPACE_BREAK;
      }
      t.fingerprint_ = Fingerprinter::Fingerprint(t.text_);
      t.span_ = nullptr;
    }
  }

  // Add themes and spans from document.
  FrameDatum *frame = store()->GetFrame(top_.handle());
  for (const Slot *slot = frame->begin(); slot < frame->end(); ++slot) {
    if (slot->name ==  n_mention_.handle()) {
      // Get token span.
      FrameDatum *mention = store()->GetFrame(slot->value);
      Handle start = mention->get(n_begin_.handle());
      Handle length = mention->get(n_length_.handle());
      int begin = start.AsInt();
      int end = length.IsNil() ? begin + 1 : begin + length.AsInt();

      // Add phrase span.
      Span *span = Insert(begin, end);
      CHECK(span != nullptr) << "Crossing span: " << begin << "," << end;
      span->mention_ = Frame(store(), mention->self);
      for (const Slot &s : span->mention_) {
        if (s.name == n_evokes_) AddMention(s.value, span);
      }
    } else if (slot->name == n_theme_.handle()) {
      // Add thematic frame.
      themes_.push_back(slot->value);
    }
  }
}

Document::~Document() {
  // Delete all spans. This also clears all references to the mention frames.
  for (auto *s : spans_) delete s;
}

void Document::Update() {
  // Build document frame.
  Builder builder(top_);
  builder.Delete(n_mention_);
  builder.Delete(n_theme_);

  // Update tokens.
  if (tokens_changed_) {
    Handles tokens(store());
    tokens.reserve(tokens_.size());
    for (int i = 0; i < tokens_.size(); ++i) {
      Token &t = tokens_[i];
      Builder token(store());
      token.AddIsA(n_token_);
      token.Add(n_token_index_, i);
      token.Add(n_token_text_, t.text_);
      if (t.begin_ != -1) {
        token.Add(n_token_start_, t.begin_);
        if (t.end_ != -1) {
          token.Add(n_token_length_, t.end_ - t.begin_);
        }
      }
      if (t.brk_ != SPACE_BREAK) {
        token.Add(n_token_break_, t.brk_);
      }
      tokens.push_back(token.Create().handle());
    }
    Array token_array(store(), tokens);
    builder.Set(n_document_tokens_, token_array);
    tokens_changed_ = false;
  }

  // Update mentions.
  for (Span *span : spans_) {
    if (span->deleted()) continue;
    builder.Add(n_mention_, span->mention_);
  }

  // Update thematic frames.
  for (Handle theme : themes_) {
    builder.Add(n_theme_, theme);
  }

  builder.Update();
}

void Document::SetText(Text text) {
  top_.Set(n_document_text_, text);
  tokens_.clear();
  tokens_changed_ = true;
}

void Document::AddToken(Text text, int begin, int end, BreakType brk) {
  // Expand token array.
  int index = tokens_.size();
  tokens_.resize(index + 1);

  // Fill new token.
  Token &t = tokens_[index];
  t.document_ = this;
  t.handle_ = Handle::nil();
  t.index_ = index;
  t.begin_ = begin;
  t.end_ = end;
  t.text_.assign(text.data(), text.size());
  t.brk_ = brk;
  t.fingerprint_ = Fingerprinter::Fingerprint(text);
  t.span_ = nullptr;
  tokens_changed_ = true;
}

Span *Document::AddSpan(int begin, int end, Handle type) {
  // Add new span for the phrase or get existing span.
  Span *span = Insert(begin, end);
  if (span == nullptr) return nullptr;

  if (span->mention_.IsNil()) {
    // Create phrase frame.
    int length = end - begin;
    Builder phrase(store());
    phrase.AddIsA(type);
    phrase.Add(n_begin_, begin);
    if (length != 1) phrase.Add(n_length_, length);
    span->mention_ = phrase.Create();
  } else {
    // Span already exists. Add the type to the phrase frame.
    if (!span->mention_.IsA(type)) {
      span->mention_.AddIsA(type);
    }
  }

  return span;
}

void Document::DeleteSpan(Span *span) {
  // Ignore if it has already been deleted.
  if (span->deleted()) return;

  // Remove span from span index.
  Remove(span);

  // Remove all evoked frames from mention table.
  for (const Slot &slot : span->mention_) {
    if (slot.name == n_evokes_) {
      RemoveMention(slot.value, span);
    }
  }

  // Clear the reference to the mention frame. This will mark the span as
  // deleted.
  span->mention_ = Frame::nil();
}

void Document::AddTheme(Handle handle) {
  themes_.push_back(handle);
}

void Document::RemoveTheme(Handle handle) {
  auto it = std::find(themes_.begin(), themes_.end(), handle);
  if (it != themes_.end()) themes_.erase(it);
}

void Document::AddMention(Handle handle, Span *span) {
  mentions_.emplace(handle, span);
}

void Document::RemoveMention(Handle handle, Span *span) {
  auto interval = mentions_.equal_range(handle);
  for (auto it = interval.first; it != interval.second; ++it) {
    if (it->second == span) {
      mentions_.erase(it);
      break;
    }
  }
}

uint64 Document::PhraseFingerprint(int begin, int end) {
  uint64 fp = 1;
  for (int t = begin; t < end; ++t) {
    uint64 word_fp = TokenFingerprint(t);
    if (word_fp == 1) continue;
    fp = Fingerprinter::Mix(word_fp, fp);
  }

  return fp;
}

string Document::PhraseText(int begin, int end) const {
  string phrase;
  for (int t = begin; t < end; ++t) {
    const Token &token = tokens_[t];
    if (t > begin && token.brk() != NO_BREAK) phrase.push_back(' ');
    phrase.append(token.text());
  }

  return phrase;
}

Span *Document::Insert(int begin, int end) {
  // Find smallest non-crossing enclosing span.
  Span *enclosing = nullptr;
  Span *prev = nullptr;
  for (int t = begin; t < end; ++t) {
    Span *s = tokens_[t].span_;

    // Skip if is the same as the leaf span for the previous token.
    if (s == prev) continue;
    prev = s;

    // Check nested spans up until the first enclosing span.
    while (s != nullptr) {
      int delta_b = begin - s->begin();
      int delta_e = end - s->end();

      // Check for matching span.
      if (delta_b == 0 && delta_e == 0) return s;

      // Check for crossing span.
      if (delta_b * delta_e > 0) return nullptr;

      // Check for enclosing span.
      if (delta_b >= 0 && delta_e <= 0) {
        if (enclosing == nullptr) enclosing = s;
        break;
      }

      s = s->parent_;
    }
  }

  // Add new span and update span tree.
  Span *span;
  if (enclosing == nullptr) {
    // There is no enclosing span, so this is a top-level span. Find the spans
    // enclosed by this span. These must be other top-level spans.
    Span *children = nullptr;
    Span *tail = nullptr;
    for (int t = begin; t < end; ++t) {
      // Find top-level span at position t.
      Span *s = tokens_[t].span_;
      if (s == nullptr) continue;
      while (s->parent_ != nullptr) s = s->parent_;

      // Skip if this is the same as for the previous token.
      if (s == tail) continue;

      // Insert span into child chain.
      if (children == nullptr) children = s;
      if (tail != nullptr) tail->sibling_ = s;
      tail = s;
    }
    if (tail != nullptr) tail->sibling_ = nullptr;

    // Add new span top-level span.
    span = new Span(this, spans_.size(), begin, end);
    spans_.push_back(span);

    // Add covered top-level spans as children.
    span->children_ = children;
  } else {
    // The new span is enclosed by a parent span. First, find the left and
    // right-most child spans enclosed by the new span. We also track the spans
    // (if any) to the immediate left and right of these child spans.
    Span *left, *right, *left_prev, *right_next;
    left = left_prev = right = right_next = nullptr;
    Span *s = enclosing->children_;
    while (s != nullptr) {
      if (s->begin() >= begin && s->end() <= end) {
        if (left == nullptr) left = s;  // first child contained in the new span
        right = s;  // last child contained in the new span
      } else if (s->begin() < begin) {
        left_prev = s;  // last child before the new span
      } else if (s->begin() >= end && right_next == nullptr) {
        right_next = s;  // first child after the new span
      }
      s = s->sibling_;
    }

    // Add new span.
    span = new Span(this, spans_.size(), begin, end);
    spans_.push_back(span);

    // Insert span into tree.
    span->parent_ = enclosing;
    span->children_ = left;
    if (left_prev != nullptr) {
      left_prev->sibling_ = span;
    } else {
      enclosing->children_ = span;
    }
    span->sibling_ = right_next;
    if (right != nullptr) {
      right->sibling_ = nullptr;
    }
  }

  // Update parent for enclosed spans.
  Span *child = span->children_;
  while (child != nullptr) {
    child->parent_ = span;
    child = child->sibling_;
  }

  // Update leaf pointers.
  Span *parent = span->parent_;
  for (int t = begin; t < end; ++t) {
    if (tokens_[t].span_ == parent) tokens_[t].span_ = span;
  }

  return span;
}

void Document::Remove(Span *span) {
  // Move leaf spans to parent.
  for (int t = span->begin(); t < span->end(); ++t) {
    if (tokens_[t].span_ == span) tokens_[t].span_ = span->parent_;
  }

  // Move parent pointers of children to grandparent.
  Span *parent = span->parent_;
  Span *child = span->children_;
  Span *last = nullptr;
  while (child != nullptr) {
    child->parent_ = parent;
    last = child;
    child = child->sibling_;
  }

  // Replace span with child spans in parent.
  if (parent != nullptr) {
    Span **next = &parent->children_;
    Span *child = parent->children_;
    while (child != span) {
      next = &child->sibling_;
      child = child->sibling_;
    }
    if (span->children_ != nullptr) {
      *next = span->children_;
      last->sibling_ = span->sibling_;
    } else {
      *next = span->sibling_;
    }
  }
}

Span *Document::GetSpan(int begin, int end) const {
  Span *span = tokens_[begin].span();
  while (span != nullptr) {
    if (span->begin() == begin && span->end() == end) return span;
    span = span->parent_;
  }
  return nullptr;
}

void Document::ClearAnnotations() {
  for (Token &t : tokens_) t.span_ = nullptr;
  for (Span *s : spans_) delete s;
  spans_.clear();
  mentions_.clear();
  themes_.clear();
}

}  // namespace nlp
}  // namespace sling

