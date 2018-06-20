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
#include "sling/nlp/document/token-breaks.h"
#include "sling/string/text.h"

namespace sling {
namespace nlp {

class Span;
class Document;

// Symbol names for documents.
struct DocumentNames : public SharedNames {
  DocumentNames(Store *store) { CHECK(Bind(store)); }

  Name n_document{*this, "/s/document"};
  Name n_document_text{*this, "/s/document/text"};
  Name n_document_tokens{*this, "/s/document/tokens"};
  Name n_mention{*this, "/s/document/mention"};
  Name n_theme{*this, "/s/document/theme"};

  Name n_token{*this, "/s/token"};
  Name n_token_index{*this, "/s/token/index"};
  Name n_token_start{*this, "/s/token/start"};
  Name n_token_length{*this, "/s/token/length"};
  Name n_token_text{*this, "/s/token/text"};
  Name n_token_break{*this, "/s/token/break"};

  Name n_phrase{*this, "/s/phrase"};
  Name n_begin{*this, "/s/phrase/begin"};
  Name n_length{*this, "/s/phrase/length"};
  Name n_evokes{*this, "/s/phrase/evokes"};
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

  // Token text.
  const string &text() const { return text_; }

  // Break level before token.
  BreakType brk() const { return brk_; }

  // Lowest span covering the token.
  Span *span() const { return span_; }

  // Token fingerprint.
  uint64 fingerprint() const { return fingerprint_; }

 private:
  Document *document_;      // document the token belongs to
  Handle handle_;           // handle for token in the store
  int index_;               // index of token in document

  int begin_;               // first byte position of token
  int end_;                 // first byte position after token

  string text_;             // token text
  BreakType brk_;           // break level before token

  uint64 fingerprint_;      // fingerprint for token text
  Span *span_;              // lowest span covering the token

  friend class Document;
};

// A span represents a range of tokens in the document. The token span is
// represented as a mention frame which can record features of the mention as
// well as other frames that are evoked by this mention.
class Span {
 public:
  Span(Document *document, int index, int begin, int end)
      : document_(document), index_(index), begin_(begin), end_(end) {}

  // Returns the document that that the span belongs to.
  Document *document() const { return document_; }

  // Returns the index of the span in the document.
  int index() const { return index_; }

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

  // Adds frame evocation to span.
  void Evoke(const Frame &frame);
  void Evoke(Handle frame);

  // Returns (the first) evoked frame of a certain type.
  Frame Evoked(Handle type) const;
  Frame Evoked(const Name &type) const;

  // Returns the first evoked frame.
  Frame Evoked() const;

  // Returns all evoked frames.
  void AllEvoked(Handles *evoked) const;

  // Checks if span evokes a certain type of frame.
  bool Evokes(Handle type) const;
  bool Evokes(const Name &type) const;

  // Returns fingerprint for span phrase.
  uint64 Fingerprint();

 private:
  // Document that span belongs to.
  Document *document_;

  // Index of span in document.
  int index_;

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

  // Span fingerprint. This is lazily initialized and cached.
  uint64 fingerprint_ = 0;

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

  ~Document();

  // Return document frame.
  const Frame &top() const { return top_; }

  // Return store for document.
  Store *store() const { return top_.store(); }

  // Update the document frame.
  void Update();

  // Return the document text.
  string GetText() const {
    return top_.GetString(names_->n_document_text);
  }

  // Set document text. This will delete all existing tokens.
  void SetText(Text text);

  // Add token to document.
  void AddToken(Text text,
                int begin = -1, int end = -1,
                BreakType brk = SPACE_BREAK);

  // Add new span to the document. The span is initialized with a mention frame
  // for the span. If the span already exists, the type is added to the mention
  // and the existing span is returned. Spans can be nested but are not allowed
  // to cross, in which case null is returned.
  Span *AddSpan(int begin, int end, Handle type);
  Span *AddSpan(int begin, int end, const Name &type) {
    return AddSpan(begin, end, type.Lookup(store()));
  }
  Span *AddSpan(int begin, int end) {
    return AddSpan(begin, end, names_->n_phrase);
  }

  // Deletes span from the document.
  void DeleteSpan(Span *span);

  // Returns the number of spans in the document.
  int num_spans() const { return spans_.size(); }

  // Return span in document.
  Span *span(int index) const { return spans_[index]; }

  // Return the number of tokens in the document.
  int num_tokens() const { return tokens_.size(); }

  // Return token in the document.
  const Token &token(int index) const { return tokens_[index]; }

  // Return document tokens.
  const std::vector<Token> &tokens() const { return tokens_; }

  // Locate token index containing text position.
  int Locate(int position) const;

  // Return fingerprint for token in document.
  uint64 TokenFingerprint(int token) const {
    return tokens_[token].fingerprint();
  }

  // Returns the fingerprint for [begin, end).
  uint64 PhraseFingerprint(int begin, int end);

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

  // Types for mapping from frame to spans that evoke it.
  typedef std::unordered_multimap<Handle, Span *, HandleHash> MentionMap;
  typedef std::pair<MentionMap::const_iterator, MentionMap::const_iterator>
      ConstMentionIteratorPair;
  typedef std::pair<MentionMap::iterator, MentionMap::iterator>
      MentionIteratorPair;

  // Iterator adapters for mention ranges.
  class ConstMentionRange {
   public:
    explicit ConstMentionRange(const ConstMentionIteratorPair &interval)
        : interval_(interval) {}
    MentionMap::const_iterator begin() const { return interval_.first; }
    MentionMap::const_iterator end() const { return interval_.second; }

   private:
    ConstMentionIteratorPair interval_;
  };

  class MentionRange {
   public:
    explicit MentionRange(const MentionIteratorPair &interval)
        : interval_(interval) {}
    MentionMap::iterator begin() { return interval_.first; }
    MentionMap::iterator end() { return interval_.second; }

   private:
    MentionIteratorPair interval_;
  };

  // Iterates over all spans that evoke a frame, e.g.:
  //   for (const auto &it : document.EvokingSpans(h)) {
  //     Span *s = it.second;
  //   }
  ConstMentionRange EvokingSpans(Handle handle) const {
    return ConstMentionRange(mentions_.equal_range(handle));
  }
  ConstMentionRange EvokingSpans(const Frame &frame) const {
    return ConstMentionRange(mentions_.equal_range(frame.handle()));
  }

  MentionRange EvokingSpans(Handle handle) {
    return MentionRange(mentions_.equal_range(handle));
  }
  MentionRange EvokingSpans(const Frame &frame) {
    return MentionRange(mentions_.equal_range(frame.handle()));
  }

  // Returns the number of spans evoking a frame.
  int EvokingSpanCount(Handle handle) {
    return mentions_.count(handle);
  }
  int EvokingSpanCount(const Frame &frame) {
    return mentions_.count(frame.handle());
  }

  // Clears annotations (mentions and themes) from document.
  void ClearAnnotations();

 private:
  // Inserts the span in the span index. If the span already exists, the
  // existing span is returned. Returns null if the new span crosses an existing
  // span.
  Span *Insert(int begin, int end);

  // Removes the span from the span index.
  void Remove(Span *span);

  // Adds frame to mention mapping.
  void AddMention(Handle handle, Span *span);

  // Removes frame from mention mapping.
  void RemoveMention(Handle handle, Span *span);

  // Document frame.
  Frame top_;

  // Document tokens.
  std::vector<Token> tokens_;

  // If the tokens have been changed the Update() method will update the tokens
  // in the document frame.
  bool tokens_changed_ = false;

  // Span index. This contains all the spans in the document in index order,
  // including the deleted spans.
  std::vector<Span *> spans_;

  // List of thematic frames. These are frames that are not evoked by any
  // particular phrase in the text.
  Handles themes_;

  // Inverse mapping from frames to spans that can be used for looking up all
  // mentions of a frame. The handles are tracked by the mention frame in the
  // span.
  MentionMap mentions_;

  // Document symbol names.
  const DocumentNames *names_;

  friend class Span;
};

// Iteration over parts of a document based on break level.
class DocumentIterator {
 public:
  // Initialize iterator for iterating over document based on break level.
  DocumentIterator(const Document *document, BreakType brk)
      : document_(document), brk_(brk) {
    next();
  }

  // Check if there are more parts in the document
  bool more() const { return begin_ < document_->num_tokens(); }

  // Go to next document part.
  void next() {
    // Check if we are already done.
    int n = document_->num_tokens();
    if (begin_ >= n) return;

    // Find start of next part.
    begin_ = end_++;
    while (end_ < n && document_->token(end_).brk() < brk_) end_++;
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
  SentenceIterator(const Document *document)
      : DocumentIterator(document, SENTENCE_BREAK) {}
};

}  // namespace nlp
}  // namespace sling

#endif  // SLING_NLP_DOCUMENT_DOCUMENT_H_

