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
#include "sling/nlp/document/token-properties.h"

namespace sling {
namespace nlp {

uint64 Token::Fingerprint() const {
  // Compute token fingerprint if not already done.
  if (fingerprint_ == 0) fingerprint_ = Fingerprinter::Fingerprint(word_);
  return fingerprint_;
}

CaseForm Token::Form() const {
  if (form_ == CASE_INVALID) {
    form_ = UTF8::Case(word_);

    // Case for first token in a sentence is indeterminate.
    if (initial() && form_ == CASE_TITLE) form_ = CASE_NONE;
  }
  return form_;
}

void Span::Evoke(const Frame &frame) {
  mention_.Add(document_->names_->n_evokes, frame);
}

void Span::Evoke(Handle frame) {
  mention_.Add(document_->names_->n_evokes, frame);
}

void Span::Replace(Handle existing, Handle replacement) {
  Handle n_evokes = document_->names_->n_evokes.handle();
  FrameDatum *mention = mention_.store()->GetFrame(mention_.handle());
  for (Slot *slot = mention->begin(); slot < mention->end(); ++slot) {
    if (slot->name == n_evokes && slot->value == existing) {
      slot->value = replacement;
      return;
    }
  }
}

Frame Span::Evoked(Handle type) const {
  Handle n_evokes = document_->names_->n_evokes.handle();
  for (const Slot &slot : mention_) {
    if (slot.name != n_evokes) continue;
    Frame frame(document_->store(), slot.value);
    if (frame.IsA(type)) return frame;
  }

  return Frame::nil();
}

Frame Span::Evoked(const Name &type) const {
  Handle n_evokes = document_->names_->n_evokes.handle();
  for (const Slot &slot : mention_) {
    if (slot.name != n_evokes) continue;
    Frame frame(document_->store(), slot.value);
    if (frame.IsA(type)) return frame;
  }

  return Frame::nil();
}

Frame Span::Evoked() const {
  Handle n_evokes = document_->names_->n_evokes.handle();
  for (const Slot &slot : mention_) {
    if (slot.name == n_evokes) {
      return Frame(document_->store(), slot.value);
    }
  }

  return Frame::nil();
}

Handle Span::evoked() const {
  Handle n_evokes = document_->names_->n_evokes.handle();
  for (const Slot &slot : mention_) {
    if (slot.name == n_evokes) return slot.value;
  }

  return Handle::nil();
}

void Span::AllEvoked(Handles *evoked) const {
  evoked->clear();
  if (mention_.valid()) {
    Handle n_evokes = document_->names_->n_evokes.handle();
    for (const Slot &slot : mention_) {
      if (slot.name == n_evokes) {
        evoked->push_back(slot.value);
      }
    }
  }
}

bool Span::Evokes(Handle frame) const {
  Handle n_evokes = document_->names_->n_evokes.handle();
  for (const Slot &slot : mention_) {
    if (slot.name == n_evokes && slot.value == frame) return true;
  }

  return false;
}

bool Span::EvokesType(Handle type) const {
  Handle n_evokes = document_->names_->n_evokes.handle();
  for (const Slot &slot : mention_) {
    if (slot.name != n_evokes) continue;
    Frame frame(document_->store(), slot.value);
    if (type.IsNil()) {
      if (frame.GetHandle(Handle::isa()).IsNil()) return true;
    } else {
      if (frame.IsA(type)) return true;
    }
  }

  return false;
}

bool Span::EvokesType(const Name &type) const {
  Handle n_evokes = document_->names_->n_evokes.handle();
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

uint64 Span::Fingerprint() const {
  // Check for cached fingerprint.
  if (fingerprint_ != 0) return fingerprint_;

  // Compute span fingerprint.
  uint64 fp = document_->PhraseFingerprint(begin_, end_);
  fingerprint_ = fp;
  return fp;
}

CaseForm Span::Form() const {
  if (form_ == CASE_INVALID) form_ = document_->PhraseForm(begin_, end_);
  return form_;
}

Document::Document(Store *store, const DocumentNames *names)
    : themes_(store), names_(names) {
  // Bind names.
  if (names_ == nullptr) {
    names_ = new DocumentNames(store);
  } else {
    names_->AddRef();
  }

  // Build empty document.
  Builder builder(store);
  builder.AddIsA(names_->n_document);
  top_ = builder.Create();
}

Document::Document(const Frame &top, const DocumentNames *names)
    : top_(top), themes_(top.store()), names_(names) {
  // Bind names.
  if (names_ == nullptr) {
    names_ = new DocumentNames(top.store());
  } else {
    names_->AddRef();
  }

  // Add document frame if it is missing.
  if (!top_.valid()) {
    Builder builder(store());
    builder.AddIsA(names_->n_document);
    top_ = builder.Create();
  }

  // Get document text.
  text_ = top_.GetString(names_->n_text);

  // Get tokens.
  Array tokens = top_.Get(names_->n_tokens).AsArray();
  if (tokens.valid()) {
    // Initialize tokens.
    int num_tokens = tokens.length();
    tokens_.resize(num_tokens);
    for (int i = 0; i < num_tokens; ++i) {
      // Get token information from token frame.
      Handle h = tokens.get(i);
      FrameDatum *token = store()->GetFrame(h);
      Handle word = token->get(names_->n_word.handle());
      Handle start = token->get(names_->n_start.handle());
      Handle size = token->get(names_->n_size.handle());
      Handle brk = token->get(names_->n_break.handle());
      Handle style = token->get(names_->n_style.handle());

      // Fill token from frame.
      Token &t = tokens_[i];
      t.document_ = this;
      t.handle_ = h;
      t.index_ = i;
      if (!start.IsNil()) {
        t.begin_ = start.AsInt();
        t.end_ = t.begin_ + (size.IsNil() ? 1 : size.AsInt());
      } else {
        t.begin_ = -1;
        t.end_ = -1;
      }
      if (!word.IsNil()) {
        StringDatum *str = store()->GetString(word);
        t.word_.assign(str->data(), str->size());
      } else if (t.begin_ != -1 && t.end_ != -1) {
        t.word_ = text_.substr(t.begin_, t.end_ - t.begin_);
      }
      if (!brk.IsNil()) {
        t.brk_ = static_cast<BreakType>(brk.AsInt());
      } else {
        t.brk_ = i == 0 ? NO_BREAK : SPACE_BREAK;
      }
      if (!style.IsNil()) {
        t.style_ = style.AsInt();
      } else {
        t.style_ = 0;
      }
      t.fingerprint_ = 0;
      t.form_ = CASE_INVALID;
      t.span_ = nullptr;
    }
  }

  // Add themes and spans from document.
  FrameDatum *frame = store()->GetFrame(top_.handle());
  for (const Slot *slot = frame->begin(); slot < frame->end(); ++slot) {
    if (slot->name ==  names_->n_mention.handle()) {
      // Get token span.
      FrameDatum *mention = store()->GetFrame(slot->value);
      Handle start = mention->get(names_->n_begin.handle());
      Handle length = mention->get(names_->n_length.handle());
      int begin = start.AsInt();
      int end = length.IsNil() ? begin + 1 : begin + length.AsInt();

      // Add phrase span.
      Span *span = Insert(begin, end);
      CHECK(span != nullptr) << "Crossing span: " << begin << "," << end;
      span->mention_ = Frame(store(), mention->self);
    } else if (slot->name == names_->n_theme.handle()) {
      // Add thematic frame.
      themes_.push_back(slot->value);
    }
  }
}

Document::Document(const Document &other, bool annotations)
    : text_(other.text_),
      tokens_(other.tokens_),
      themes_(other.store()),
      names_(other.names_) {
  // Make a copy of the document frame (except document id).
  names_->AddRef();
  Store *store = other.store();
  Builder builder(store);
  for (const Slot &s : other.top_) {
    if (s.name != Handle::id()) {
      builder.Add(s.name, s.value);
    }
  }
  top_ = builder.Create();

  // Update tokens.
  for (Token &token : tokens_) {
    token.document_ = this;
    token.span_ = nullptr;
  }

  if (annotations) {
    // Copy mention spans.
    for (const Span *s : other.spans_) {
      Span *span = Insert(s->begin_, s->end_);
      span->mention_ = Frame(store, store->Clone(s->mention_.handle()));
    }

    // Copy themes.
    for (Handle h : other.themes_) {
      themes_.push_back(h);
    }

    // Copy extra slots.
    if (other.extras_ != nullptr) {
      extras_ = new Slots(store);
      extras_->reserve(other.extras_->size());
      for (const Slot &s : *other.extras_) {
        extras_->emplace_back(s.name, s.value);
      }
    }
  }
}

Document::Document(const Document &other,
                   int begin, int end,
                   bool annotations)
    : themes_(other.store()), names_(other.names_) {
  // Copy tokens.
  names_->AddRef();
  Store *store = other.store();
  int length = end - begin;
  int text_begin = other.text().size();
  int text_end = 0;
  tokens_.resize(length);
  for (int i = 0; i < length; ++i) {
    const Token &o = other.tokens_[i + begin];
    Token &t = tokens_[i];
    t = o;
    t.document_ = this;
    t.index_ = i;
    t.span_ = nullptr;
    if (o.begin_ < text_begin) text_begin = o.begin_;
    if (o.end_ > text_end) text_end = o.end_;
  }
  if (length > 0) tokens_changed_ = true;

  // Copy text and adjust token positions.
  if (text_end > text_begin) {
    text_ = other.text_.substr(text_begin, text_end - text_begin);
    for (Token &t : tokens_) {
      t.begin_ -= text_begin;
      t.end_ -= text_begin;
    }
  }

  // Copy annotations.
  if (annotations) {
    for (const Span *s : other.spans_) {
      int b = s->begin_ - begin;
      int e = s->end_ - begin;
      if (b < 0 || e > length) continue;
      Span *span = Insert(b, e);
      span->mention_ = Frame(store, store->Clone(s->mention_.handle()));
    }
  }

  // Create document frame.
  Builder builder(store);
  builder.AddIsA(names_->n_document);
  if (!text_.empty()) builder.Add(names_->n_text, text_);
  top_ = builder.Create();
}

Document::~Document() {
  // Delete all spans. This also clears all references to the mention frames.
  for (auto *s : spans_) delete s;
  delete extras_;

  // Release names.
  names_->Release();
}

void Document::Update() {
  // Build document frame.
  Builder builder(top_);
  builder.Delete(names_->n_mention);
  builder.Delete(names_->n_theme);

  // Update tokens.
  if (tokens_changed_) {
    Handles tokens(store());
    tokens.reserve(tokens_.size());
    for (int i = 0; i < tokens_.size(); ++i) {
      Token &t = tokens_[i];
      Builder token(store());
      if (t.begin_ != -1 && t.end_ != -1 &&
          text_.compare(t.begin_, t.end_ - t.begin_, t.word_) != 0) {
        token.Add(names_->n_word, t.word_);
      }
      if (t.begin_ != -1) {
        token.Add(names_->n_start, t.begin_);
        if (t.end_ != -1 && t.end_ != t.begin_ + 1) {
          token.Add(names_->n_size, t.end_ - t.begin_);
        }
      }
      if (t.brk_ != (i == 0 ? NO_BREAK : SPACE_BREAK)) {
        token.Add(names_->n_break, t.brk_);
      }
      if (t.style_ != 0) {
        token.Add(names_->n_style, t.style_);
      }
      tokens.push_back(token.Create().handle());
    }
    Array token_array(store(), tokens);
    builder.Set(names_->n_tokens, token_array);
    tokens_changed_ = false;
  }

  // Update mentions.
  for (Span *span : spans_) {
    if (span->deleted()) continue;
    builder.Add(names_->n_mention, span->mention_);
  }

  // Update thematic frames.
  for (Handle theme : themes_) {
    builder.Add(names_->n_theme, theme);
  }

  // Add extra slots to document frame.
  if (extras_ != nullptr) {
    for (const Slot &s : *extras_) {
      builder.Add(s.name, s.value);
    }
  }

  builder.Update();
}

void Document::SetText(Handle text) {
  top_.Set(names_->n_text, text);
  text_ = String(store(), text).value();
  tokens_.clear();
  tokens_changed_ = true;
}

void Document::SetText(Text text) {
  top_.Set(names_->n_text, text);
  text_ = text.str();
  tokens_.clear();
  tokens_changed_ = true;
}

void Document::AddToken(Text word, int begin, int end,
                        BreakType brk, int style) {
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
  t.word_.assign(word.data(), word.size());
  t.brk_ = brk;
  t.style_ = style;
  t.fingerprint_ = 0;
  t.form_ = CASE_INVALID;
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
    if (type != Handle::nil()) phrase.AddIsA(type);
    phrase.Add(names_->n_begin, begin);
    if (length != 1) phrase.Add(names_->n_length, length);
    span->mention_ = phrase.Create();
  } else {
    // Span already exists. Add the type to the phrase frame.
    if (type != Handle::nil() && !span->mention_.IsA(type)) {
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

void Document::AddExtra(Handle name, Handle value) {
  if (extras_ == nullptr) extras_ = new Slots(store());
  extras_->emplace_back(name, value);
}

int Document::Locate(int position) const {
  int index = 0;
  int len = tokens_.size();
  while (len > 0) {
    int width = len / 2;
    int mid = index + width;
    if (tokens_[mid].begin() < position) {
      index = mid + 1;
      len -= width + 1;
    } else {
      len = width;
    }
  }
  return index;
}

uint64 Document::PhraseFingerprint(int begin, int end) const {
  uint64 fp = 1;
  for (int t = begin; t < end; ++t) {
    uint64 word_fp = TokenFingerprint(t);
    if (word_fp == 1) continue;
    fp = Fingerprinter::Mix(word_fp, fp);
  }

  return fp;
}

CaseForm Document::PhraseForm(int begin, int end) const {
  CaseForm form = CASE_INVALID;
  for (int t = begin; t < end; ++t) {
    if (token(t).skipped()) continue;
    CaseForm token_form = token(t).Form();
    if (form == CASE_INVALID) {
      form = token_form;
    } else if (form != token_form) {
      form = CASE_NONE;
    }
  }
  return form;
}

string Document::PhraseText(int begin, int end) const {
  string phrase;
  for (int t = begin; t < end; ++t) {
    const Token &token = tokens_[t];
    if (t > begin && token.brk() != NO_BREAK) phrase.push_back(' ');
    phrase.append(token.word());
  }

  return phrase;
}

Span *Document::EnclosingSpan(int begin, int end, bool *crossing) {
  Span *enclosing = nullptr;
  Span *prev = nullptr;
  *crossing = false;
  for (int t = begin; t < end; ++t) {
    Span *s = tokens_[t].span_;

    // Skip if it has the same leaf span as the previous token.
    if (s == prev) continue;
    prev = s;

    // Check nested spans up until the first enclosing span.
    while (s != nullptr) {
      int delta_b = begin - s->begin();
      int delta_e = end - s->end();

      // Check for matching span.
      if (delta_b == 0 && delta_e == 0) return s;

      // Check for crossing span.
      if (delta_b * delta_e > 0) {
        *crossing = true;
        return nullptr;
      }

      // Check for enclosing span.
      if (delta_b >= 0 && delta_e <= 0) {
        if (enclosing == nullptr) enclosing = s;
        break;
      }

      s = s->parent_;
    }
  }

  return enclosing;
}

Span *Document::Insert(int begin, int end) {
  // Find smallest non-crossing enclosing span.
  bool crossing = false;
  Span *enclosing = EnclosingSpan(begin, end, &crossing);
  if (crossing) return nullptr;

  // Check if span already exists.
  if (enclosing != nullptr && enclosing->begin() == begin &&
      enclosing->end() == end) {
    return enclosing;
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
    span = new Span(this, begin, end);
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
    span = new Span(this, begin, end);
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
  themes_.clear();
}

}  // namespace nlp
}  // namespace sling

