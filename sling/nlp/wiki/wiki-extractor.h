// Copyright 2018 Google Inc.
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

#ifndef SLING_NLP_WIKI_WIKI_EXTRACTOR_H_
#define SLING_NLP_WIKI_WIKI_EXTRACTOR_H_

#include "sling/base/logging.h"
#include "sling/nlp/wiki/wiki-parser.h"

namespace sling {
namespace nlp {

class WikiExtractor;

// Abstract class for collecting text and annotations from Wikipedia document.
class WikiSink {
 public:
  typedef WikiParser::Node Node;

  virtual ~WikiSink() = default;

  // Output content text.
  virtual void Content(const char *begin, const char *end) {}

  // Font change.
  virtual void Font(int font) {}

  // Word break.
  virtual void WordBreak() {}

  // Output link.
  virtual void Link(const Node &node,
                    WikiExtractor *extractor,
                    bool unanchored);

  // Output template.
  virtual void Template(const Node &node,
                        WikiExtractor *extractor,
                        bool unanchored);

  // Output category.
  virtual void Category(const Node &node,
                        WikiExtractor *extractor,
                        bool unanchored);
};

// Extract text and annotations from Wikipedia page. The text and annotations
// are extracted from the AST nodes of a parsed Wikipedia text which are output
// to sinks that can collect the text and annotations.
class WikiExtractor {
 public:
  typedef WikiParser::Node Node;

  // Initialize Wikipedia text extractor.
  WikiExtractor(const WikiParser &parser) : parser_(parser) {}

  // Extract text to sink by traversing the nodes in the AST.
  void Extract(WikiSink *sink);

  // Nesting of output sinks.
  void Enter(WikiSink *sink) {
    sinks_.push_back(sink);
  }
  void Leave(WikiSink *sink) {
    CHECK(!sinks_.empty());
    CHECK(sinks_.back() == sink);
    sinks_.pop_back();
  }
  WikiSink *sink() { return sinks_.back(); }

  // Extract text and annotations from node. This uses the handler for the node
  // type for extraction.
  void ExtractNode(const Node &node);

  // Extract text and annotations from all children of the parent node.
  void ExtractChildren(const Node &parent);

  // Extract annotations from skipped AST nodes.
  void ExtractSkip(const Node &node);

  // Handlers for extracting text and annotations for specific AST node types.
  // The can be overridden in subclasses to customize the extraction.
  void ExtractDocument(const Node &node);
  void ExtractArg(const Node &node);
  void ExtractAttr(const Node &node);
  void ExtractText(const Node &node);
  void ExtractFont(const Node &node);
  void ExtractTemplate(const Node &node);
  void ExtractLink(const Node &node);
  void ExtractImage(const Node &node);
  void ExtractCategory(const Node &node);
  void ExtractUrl(const Node &node);
  void ExtractComment(const Node &node);
  void ExtractTag(const Node &node);
  void ExtractBeginTag(const Node &node);
  void ExtractEndTag(const Node &node);
  void ExtractMath(const Node &node);
  void ExtractGallery(const Node &node);
  void ExtractReference(const Node &node);
  void ExtractNoWiki(const Node &node);
  void ExtractHeading(const Node &node);
  void ExtractIndent(const Node &node);
  void ExtractTerm(const Node &node);
  void ExtractListBegin(const Node &node);
  void ExtractListItem(const Node &node);
  void ExtractListEnd(const Node &node);
  void ExtractRuler(const Node &node);
  void ExtractSwitch(const Node &node);
  void ExtractTable(const Node &node);
  void ExtractTableCaption(const Node &node);
  void ExtractTableRow(const Node &node);
  void ExtractTableHeader(const Node &node);
  void ExtractTableCell(const Node &node);
  void ExtractTableBreak(const Node &node);

  // Get attribute value from child nodes.
  Text GetAttr(const Node &node, Text attrname);

  // Reset font back to normal.
  void ResetFont();

  // Emit text to sink.
  void Emit(const char *begin, const char *end) { sink()->Content(begin, end); }
  void Emit(const char *str) { Emit(str, str + strlen(str)); }
  void Emit(Text str) { Emit(str.data(), str.data() + str.size()); }
  void Emit(const Node &node) { Emit(node.begin, node.end); }

  // Return parser for extractor.
  const WikiParser &parser() const { return parser_; }

  // Skip tables for extraction.
  bool skip_tables() const { return skip_tables_; }
  void set_skip_tables(bool skip_tables) { skip_tables_ = skip_tables; }

 private:
  // Wiki text parser with AST.
  const WikiParser &parser_;

  // Stack of output sinks for collecting extracted text and annotations.
  std::vector<WikiSink *> sinks_;

  // Skip tables in extraction.
  bool skip_tables_ = false;
};

// Sink for collecting plain text extracted from Wikipedia page.
class WikiPlainTextSink : public WikiSink {
 public:
  void Content(const char *begin, const char *end) override;

  // Return extracted text.
  const string &text() const { return text_; }

 protected:
  // Extracted text.
  string text_;

  // Pending space break.
  bool space_break_ = false;
};

// Sink for collecting text with markup extracted from Wikipedia page.
class WikiTextSink : public WikiSink {
 public:
  void Content(const char *begin, const char *end) override;
  void Content(const char *str) { Content(str, str + strlen(str)); }
  void Content(Text str) { Content(str.data(), str.data() + str.size()); }
  void Font(int font) override;
  void WordBreak() override;

  // Return extracted text.
  const string &text() const { return text_; }

  // Return current text position.
  int position() const { return text_.size(); }

 protected:
  // Extracted text.
  string text_;

  // Number of pending line breaks.
  int line_breaks_ = 0;

  // Pending word break.
  bool word_break_ = false;

  // Current font.
  int font_ = 0;
};

}  // namespace nlp
}  // namespace sling

#endif  // SLING_NLP_WIKI_WIKI_EXTRACTOR_H_

