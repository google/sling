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

#ifndef SLING_NLP_DOCUMENT_DOCUMENT_H_
#define SLING_NLP_DOCUMENT_DOCUMENT_H_

#include <string>
#include <vector>
#include <unordered_map>

#include "sling/base/types.h"
#include "sling/frame/object.h"
#include "sling/frame/store.h"
#include "sling/nlp/document/token-properties.h"
#include "sling/string/text.h"
#include "sling/util/unicode.h"

namespace sling {
namespace nlp {

class Span;
class Document;

// Symbol names for documents.
struct DocumentNames : public SharedNames {
  DocumentNames(Store *store) { CHECK(Bind(store)); }

  Name n_document{*this, "document"};
  Name n_title{*this, "title"};
  Name n_url{*this, "url"};
  Name n_text{*this, "text"};
  Name n_tokens{*this, "tokens"};
  Name n_mention{*this, "mention"};
  Name n_theme{*this, "theme"};

  Name n_index{*this, "index"};
  Name n_start{*this, "start"};
  Name n_size{*this, "size"};
  Name n_word{*this, "word"};
  Name n_break{*this, "break"};
  Name n_style{*this, "style"};

  Name n_begin{*this, "begin"};
  Name n_length{*this, "length"};
  Name n_evokes{*this, "evokes"};
};

// A token represents a range of characters in the document text. A token is a
// word or any other kind of lexical unit like punctuation, number, etc.
class Token {
 public:
  // Document that the token belongs to.
  Document *document() const { return document_; }

  // Handle for token in the store.
  Handle handle() const { return handle_; }

  // Index of token in document.
  int index() const { return index_; }

  // Text span for token in document text The [begin;end[ is a semi-open byte
  // range of the UTF-8 encoded token in the document text, where begin is the
  // index of the first byte of the token and end is the first byte after the
  // token.
  int begin() const { return begin_; }
  int end() const { return end_; }
  int size() const { return end_ - begin_; }

  // Token word.
  const string &word() const { return word_; }

  // Break level before token.
  BreakType brk() const { return brk_; }

  // Token style change before token.
  int style() const { return style_; }

  // Lowest span covering the token.
  Span *span() const { return span_; }

  // Token fingerprint.
  uint64 Fingerprint() const;

  // Token case form.
  CaseForm Form() const;

  // Punctuation tokens etc. are skipped in phrase comparison.
  bool skipped() const { return Fingerprint() == 1; }

  // Check for initial token in a sentence.
  bool initial() const { return index_ == 0 || brk_ >= SENTENCE_BREAK; }

 private:
  Document *document_;          // document the token belongs to
  Handle handle_;               // handle for token in the store
  int index_;                   // index of token in document

  int begin_;                   // first byte position of token
  int end_;                     // first byte position after token

  string word_;                 // token word
  BreakType brk_;               // break level before token
  int style_;                   // token style change before token

  mutable uint64 fingerprint_;  // fingerprint for token text
  mutable CaseForm form_;       // case form for token

  Span *span_;                  // lowest span covering the token

  friend class Document;
};

// A span represents a range of tokens in the document. The token span is
// represented as a mention frame which can record features of the mention as
// well as other frames that are evoked by this mention.
class Span {
 public:
  Span(Document *document, int begin, int end)
      : document_(document), begin_(begin), end_(end) {}

  // Returns the document that that the span belongs to.
  Document *document() const { return document_; }

  // Returns the begin and end token. This is a half-open interval, so the
  // span covers the tokens in the range [begin;end[.
  int begin() const { return begin_; }
  int end() const { return end_; }

  // Returns the length of the spans in number of tokens.
  int length() const { return end_ - begin_; }

  // Returns text for span.
  string GetText() const;

  // Returns true if this spans contains the other span.
  bool Contains(Span *other) const {
    return begin_ <= other->begin_ && end_ >= other->end_;
  }

  // Returns true if this span contains the token.
  bool Contains(int token) const {
    return begin_ <= token && token < end_;
  }

  // Returns true if this spans is contained by the other span.
  bool ContainedBy(Span *other) const {
    return begin_ >= other->begin_ && end_ <= other->end_;
  }

  // Returns the mention frame for the token span.
  const Frame &mention() const { return mention_; }

  // Returns true if the span has been deleted from the document.
  bool deleted() const { return mention_.invalid(); }

  // Returns the enclosing parent span, or null is it is a top-level span.
  Span *parent() const { return parent_; }

  // Returns the left-most enclosed child span.
  Span *children() const { return children_; }

  // Returns the sibling span to the right. All the child spans of a parent are
  // linked together left-to-right through the sibling pointers.
  Span *sibling() const { return sibling_; }

  // Returns outer-most containing span.
  Span *outer() {
    Span *s = this;
    while (s->parent_ != nullptr) s = s->parent_;
    return s;
  }

  // Adds frame evocation to span.
  void Evoke(const Frame &frame);
  void Evoke(Handle frame);

  // Replaces evoked frame.
  void Replace(Handle existing, Handle replacement);
  void Replace(const Frame &existing, const Frame &replacement) {
    Replace(existing.handle(), replacement.handle());
  }

  // Returns (the first) evoked frame of a certain type.
  Frame Evoked(Handle type) const;
  Frame Evoked(const Name &type) const;

  // Returns the first evoked frame.
  Frame Evoked() const;
  Handle evoked() const;

  // Returns all evoked frames.
  void AllEvoked(Handles *evoked) const;

  // Checks if span evokes a certain frame.
  bool Evokes(Handle frame) const;
  bool Evokes(const Frame &frame) const { return Evokes(frame.handle()); }

  // Checks if span evokes a certain type of frame.
  bool EvokesType(Handle type) const;
  bool EvokesType(const Name &type) const;

  // Returns fingerprint for span phrase.
  uint64 Fingerprint() const;

  // Returns case form for span phrase.
  CaseForm Form() const;

  // Returns first/last token in span.
  inline const Token &first() const;
  inline const Token &last() const;

  // Check for initial span in a sentence.
  bool initial() const { return first().initial(); }

 private:
  // Document that span belongs to.
  Document *document_;

  // Tokens covered by span. The span covers tokens in the interval [begin;end[,
  // i.e. begin is inclusive and end is exclusive.
  int begin_;
  int end_;

  // Mention frame for span.
  Frame mention_;

  // Span indexing.
  Span *parent_ = nullptr;    // enclosing parent span
  Span *sibling_ = nullptr;   // first sibling to the right enclosed by parent
  Span *children_ = nullptr;  // left-most enclosed sub-span

  // Span fingerprint and case form. This is lazily initialized and cached.
  mutable uint64 fingerprint_ = 0;
  mutable CaseForm form_ = CASE_INVALID;

  friend class Document;
};

// A document wraps a frame that contains the token, span, and frame
// annotations for the document.
class Document {
 public:
  // Create empty document.
  explicit Document(Store *store, const DocumentNames *names = nullptr);

  // Initialize document from frame.
  explicit Document(const Frame &top, const DocumentNames *names = nullptr);

  // Copy constructor for making a shallow copy of the whole document.
  Document(const Document &other, bool annotations);
  Document(const Document &other) : Document(other, true) {}

  // Make a shallow copy of parts of the document. Only annotations within the
  // token range are copied.
  Document(const Document &other, int begin, int end, bool annotations);

  ~Document();

  // Return document frame.
  const Frame &top() const { return top_; }

  // Return store for document.
  Store *store() const { return top_.store(); }

  // Update the document frame.
  void Update();

  // Return the document text.
  const string &text() const { return text_; }

  // Return document title.
  Text title() const { return top_.GetText(names_->n_title); }

  // Return document url.
  Text url() const { return top_.GetText(names_->n_url); }

  // Set document text. This will delete all existing tokens.
  void SetText(Handle text);
  void SetText(const String &text) { SetText(text.handle()); }
  void SetText(Text text);

  // Add token to document.
  void AddToken(Text word,
                int begin = -1, int end = -1,
                BreakType brk = SPACE_BREAK,
                int style = 0);

  // Returns the small enclosing span for [begin, end). If no such span exists,
  // then returns nullptr. If a crossing span exists, then returns nullptr and
  // sets 'crossing' to true.
  Span *EnclosingSpan(int begin, int end, bool *crossing);

  // Add new span to the document. The span is initialized with a mention frame
  // for the span. If the span already exists, the type is added to the mention
  // and the existing span is returned. Spans can be nested but are not allowed
  // to cross, in which case null is returned.
  Span *AddSpan(int begin, int end, Handle type);
  Span *AddSpan(int begin, int end, const Name &type) {
    return AddSpan(begin, end, type.Lookup(store()));
  }
  Span *AddSpan(int begin, int end) {
    return AddSpan(begin, end, Handle::nil());
  }

  // Deletes span from the document.
  void DeleteSpan(Span *span);

  // Returns the number of spans in the document.
  int num_spans() const { return spans_.size(); }

  // Return span in document.
  Span *span(int index) const { return spans_[index]; }

  // Return all spans in document.
  const std::vector<Span *> spans() const { return spans_; }

  // Return the number of tokens in the document.
  int length() const { return tokens_.size(); }
  int num_tokens() const { return tokens_.size(); }  // deprecated

  // Return token in the document.
  const Token &token(int index) const { return tokens_[index]; }

  // Return document tokens.
  const std::vector<Token> &tokens() const { return tokens_; }

  // Locate token index containing text position.
  int Locate(int position) const;

  // Return fingerprint for token in document.
  uint64 TokenFingerprint(int token) const {
    return tokens_[token].Fingerprint();
  }

  // Returns the fingerprint for [begin, end).
  uint64 PhraseFingerprint(int begin, int end) const;

  // Returns case form for phrase [begin, end).
  CaseForm PhraseForm(int begin, int end) const;

  // Returns the phrase text for span.
  string PhraseText(int begin, int end) const;

  // Finds span for phrase, or null if there is no matching span.
  Span *GetSpan(int begin, int end) const;

  // Returns lowest span at token position or null if no spans are covering the
  // token.
  Span *GetSpanAt(int index) const { return tokens_[index].span(); }

  // Adds thematic frame to document.
  void AddTheme(Handle handle);
  void AddTheme(const Frame &frame) { AddTheme(frame.handle()); }

  // Removes thematic frame from document.
  void RemoveTheme(Handle handle);
  void RemoveTheme(const Frame &frame) { RemoveTheme(frame.handle()); }

  // Returns list of thematic frames for document.
  const Handles &themes() const { return themes_; }

  // Add extra slots to document frame.
  void AddExtra(Handle name, Handle value);
  void AddExtra(const Name &name, Handle value) {
    AddExtra(name.handle(), value);
  }
  void AddExtra(const Name &name, Text value) {
    AddExtra(name.handle(), store()->AllocateString(value));
  }

  // Clears annotations (mentions and themes) from document.
  void ClearAnnotations();

  // Document schema.
  const DocumentNames *names() const { return names_; }

 private:
  // Inserts the span in the span index. If the span already exists, the
  // existing span is returned. Returns null if the new span crosses an existing
  // span.
  Span *Insert(int begin, int end);

  // Removes the span from the span index.
  void Remove(Span *span);

  // Document frame.
  Frame top_;

  // Document text.
  string text_;

  // Document tokens.
  std::vector<Token> tokens_;

  // If the tokens have been changed the Update() method will update the tokens
  // in the document frame.
  bool tokens_changed_ = false;

  // Document mention spans.
  std::vector<Span *> spans_;

  // List of thematic frames. These are frames that are not evoked by any
  // particular phrase in the text.
  Handles themes_;

  // Additional slots that should be added to document.
  Slots *extras_ = nullptr;

  // Document symbol names.
  const DocumentNames *names_;

  friend class Span;
};

// Iteration over parts of a document based on break level.
class DocumentIterator {
 public:
  // Initialize iterator for iterating over document based on break level.
  DocumentIterator(const Document *document, BreakType brk, int skip = 0)
      : document_(document), brk_(brk), skip_(skip) {
    next();
  }

  // Check if there are more parts in the document
  bool more() const { return begin_ < document_->length(); }

  // Go to next document part.
  void next() {
    int n = document_->length();
    while (begin_ < n) {
      // Find start of next part.
      begin_ = end_++;
      if (begin_ >= n) break;
      while (end_ < n && document_->token(end_).brk() < brk_) end_++;

      // Stop unless document part should be skipped.
      if ((document_->token(begin_).style() & skip_) == 0) break;
    }
  }

  // Return the span for the current document part.
  int begin() const { return begin_; }
  int end() const { return end_; }

  // Return length of current document part.
  int length() const { return end_ - begin_; }

 private:
  // Document to iterate over.
  const Document *document_;

  // Break level for document parts.
  BreakType brk_;

  // Token style mask for skipping parts.
  int skip_;

  // Current document part.
  int begin_ = 0;
  int end_ = 0;
};

// Document sentence iterator. Example:
//
//   for (SentenceIterator s(document); s.more(); s.next()) {
//     LOG(INFO) << "Sentence: " << s.begin() << " to " << s.end();
//   }
//
class SentenceIterator : public DocumentIterator {
 public:
  SentenceIterator(const Document *document, int skip = 0)
      : DocumentIterator(document, SENTENCE_BREAK, skip) {}
};

inline const Token &Span::first() const { return document_->token(begin_); }
inline const Token &Span::last() const { return document_->token(end_ - 1); }

}  // namespace nlp
}  // namespace sling

#endif  // SLING_NLP_DOCUMENT_DOCUMENT_H_

