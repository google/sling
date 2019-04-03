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

#include "sling/nlp/wiki/wiki-extractor.h"

#include "sling/string/strcat.h"

namespace sling {
namespace nlp {

void WikiSink::Link(const Node &node,
                    WikiExtractor *extractor,
                    bool unanchored) {
  if (unanchored) return;
  WikiPlainTextSink plain;
  extractor->Enter(&plain);
  extractor->ExtractChildren(node);
  extractor->Leave(&plain);
  extractor->Emit(plain.text());
}

void WikiSink::Template(const Node &node,
                        WikiExtractor *extractor,
                        bool unanchored) {
  extractor->ExtractSkip(node);
}

void WikiSink::Category(const Node &node,
                        WikiExtractor *extractor,
                        bool unanchored) {
  extractor->ExtractSkip(node);
}

void WikiExtractor::Extract(WikiSink *sink) {
  Enter(sink);
  ExtractNode(parser_.node(0));
  Leave(sink);
}

void WikiExtractor::ExtractNode(const Node &node) {
  // Call handler for node type.
  switch (node.type) {
    case WikiParser::DOCUMENT: ExtractDocument(node); break;
    case WikiParser::ARG: ExtractArg(node); break;
    case WikiParser::ATTR: ExtractAttr(node); break;
    case WikiParser::TEXT: ExtractText(node); break;
    case WikiParser::FONT: ExtractFont(node); break;
    case WikiParser::TEMPLATE: ExtractTemplate(node); break;
    case WikiParser::LINK: ExtractLink(node); break;
    case WikiParser::IMAGE: ExtractImage(node); break;
    case WikiParser::CATEGORY: ExtractCategory(node); break;
    case WikiParser::URL: ExtractUrl(node); break;
    case WikiParser::COMMENT: ExtractComment(node); break;
    case WikiParser::TAG: ExtractTag(node); break;
    case WikiParser::BTAG: ExtractBeginTag(node); break;
    case WikiParser::ETAG: ExtractEndTag(node); break;
    case WikiParser::MATH: ExtractMath(node); break;
    case WikiParser::GALLERY: ExtractGallery(node); break;
    case WikiParser::REF: ExtractReference(node); break;
    case WikiParser::NOWIKI: ExtractNoWiki(node); break;
    case WikiParser::HEADING: ExtractHeading(node); break;
    case WikiParser::INDENT: ExtractIndent(node); break;
    case WikiParser::TERM: ExtractTerm(node); break;
    case WikiParser::UL: ExtractListItem(node); break;
    case WikiParser::OL: ExtractListItem(node); break;
    case WikiParser::HR: ExtractRuler(node); break;
    case WikiParser::SWITCH: ExtractSwitch(node); break;
    case WikiParser::TABLE: ExtractTable(node); break;
    case WikiParser::CAPTION: ExtractTableCaption(node); break;
    case WikiParser::ROW: ExtractTableRow(node); break;
    case WikiParser::HEADER: ExtractTableHeader(node); break;
    case WikiParser::CELL: ExtractTableCell(node); break;
    case WikiParser::BREAK: ExtractTableBreak(node); break;
  }
}

void WikiExtractor::ExtractChildren(const Node &parent) {
  int child = parent.first_child;
  while (child != -1) {
    const Node &node = parser_.node(child);
    if (node.type == WikiParser::UL || node.type == WikiParser::OL) {
      ExtractListBegin(node);
      while (child != -1 && parser_.node(child).type == node.type) {
        ExtractNode(parser_.node(child));
        child = parser_.node(child).next_sibling;
      }
      ExtractListEnd(node);
    } else {
      ExtractNode(node);
      child = node.next_sibling;
    }
  }
}

void WikiExtractor::ExtractDocument(const Node &node) {
  ExtractChildren(node);
}

void WikiExtractor::ExtractArg(const Node &node) {
  ExtractChildren(node);
}

void WikiExtractor::ExtractAttr(const Node &node) {
  ExtractSkip(node);
}

void WikiExtractor::ExtractText(const Node &node) {
  Emit(node);
}

void WikiExtractor::ExtractFont(const Node &node) {
  sink()->Font(node.param);
}

void WikiExtractor::ExtractTemplate(const Node &node) {
  sink()->Template(node, this, false);
}

void WikiExtractor::ExtractLink(const Node &node) {
  sink()->Link(node, this, false);
}

void WikiExtractor::ExtractImage(const Node &node) {
  ExtractSkip(node);
}

void WikiExtractor::ExtractCategory(const Node &node) {
  sink()->Category(node, this, false);
}

void WikiExtractor::ExtractUrl(const Node &node) {
  ExtractChildren(node);
}

void WikiExtractor::ExtractComment(const Node &node) {
  ExtractSkip(node);
}

void WikiExtractor::ExtractTag(const Node &node) {
  if (node.name() == "br") Emit("<br>");
}

void WikiExtractor::ExtractBeginTag(const Node &node) {
  if (node.name() == "br") Emit("<br>");
  if (node.name() == "blockquote") Emit("<blockquote>");
}

void WikiExtractor::ExtractEndTag(const Node &node) {
  if (node.name() == "blockquote") Emit("</blockquote>");
}

void WikiExtractor::ExtractMath(const Node &node) {
  ExtractSkip(node);
}

void WikiExtractor::ExtractGallery(const Node &node) {
  ExtractChildren(node);
}

void WikiExtractor::ExtractReference(const Node &node) {
  sink()->WordBreak();
}

void WikiExtractor::ExtractNoWiki(const Node &node) {
}

void WikiExtractor::ExtractHeading(const Node &node) {
  ResetFont();
  Emit("\n");
  Emit(StrCat("<h", node.param, ">"));
  ExtractChildren(node);
  Emit(StrCat("</h", node.param, ">"));
  Emit("\n");
}

void WikiExtractor::ExtractIndent(const Node &node) {
  ExtractChildren(node);
}

void WikiExtractor::ExtractTerm(const Node &node) {
  ExtractChildren(node);
}

void WikiExtractor::ExtractListBegin(const Node &node) {
  if (node.type == WikiParser::OL) Emit("<ol>");
  if (node.type == WikiParser::UL) Emit("<ul>");
}

void WikiExtractor::ExtractListItem(const Node &node) {
  Emit("<li>");
  ExtractChildren(node);
  Emit("</li>");
}

void WikiExtractor::ExtractListEnd(const Node &node) {
  if (node.type == WikiParser::OL) Emit("</ol>");
  if (node.type == WikiParser::UL) Emit("</ul>");
}

void WikiExtractor::ExtractRuler(const Node &node) {
  ExtractSkip(node);
}

void WikiExtractor::ExtractSwitch(const Node &node) {
  ExtractSkip(node);
}

void WikiExtractor::ExtractTable(const Node &node) {
  if (skip_tables_) {
    ExtractSkip(node);
  } else {
    Emit("<table border=1>");
    int child = node.first_child;
    while (child != -1) {
      const Node &n = parser_.node(child);
      if (n.type == WikiParser::ROW) {
        ExtractTableRow(n);
      } else {
        ExtractNode(n);
      }
      child = n.next_sibling;
    }
    Emit("</table>");
  }
}

void WikiExtractor::ExtractTableCaption(const Node &node) {
  ExtractSkip(node);
}

void WikiExtractor::ExtractTableRow(const Node &node) {
  Emit("<tr>");
  int child = node.first_child;
  while (child != -1) {
    const Node &n = parser_.node(child);
    if (n.type == WikiParser::HEADER) {
      ExtractTableHeader(n);
    } else if (n.type == WikiParser::CELL) {
      ExtractTableCell(n);
    } else {
      ExtractNode(n);
    }
    child = n.next_sibling;
  }
  Emit("</tr>");
}

void WikiExtractor::ExtractTableHeader(const Node &node) {
  Text colspan = GetAttr(node, "colspan");
  Text rowspan = GetAttr(node, "rowspan");
  if (!colspan.empty() && !rowspan.empty()) {
    Emit(StrCat("<th colspan=", colspan, " rowspan=" , rowspan, ">"));
  } else if (!colspan.empty()) {
    Emit(StrCat("<th colspan=", colspan, ">"));
  } else if (!rowspan.empty()) {
    Emit(StrCat("<th rowspan=", rowspan, ">"));
  } else {
    Emit("<th>");
  }
  ExtractChildren(node);
  Emit("</th>");
}

void WikiExtractor::ExtractTableCell(const Node &node) {
  Text colspan = GetAttr(node, "colspan");
  Text rowspan = GetAttr(node, "rowspan");
  if (!colspan.empty() && !rowspan.empty()) {
    Emit(StrCat("<td colspan=", colspan, " rowspan=" , rowspan, ">"));
  } else if (!colspan.empty()) {
    Emit(StrCat("<td colspan=", colspan, ">"));
  } else if (!rowspan.empty()) {
    Emit(StrCat("<td rowspan=", rowspan, ">"));
  } else {
    Emit("<td>");
  }
  ExtractChildren(node);
  Emit("</td>");
}

void WikiExtractor::ExtractTableBreak(const Node &node) {
  ExtractSkip(node);
}

void WikiExtractor::ExtractSkip(const Node &node) {
  int child = node.first_child;
  while (child != -1) {
    const Node &n = parser_.node(child);
    if (n.type == WikiParser::LINK) {
      sink()->Link(n, this, true);
    } else if (n.type == WikiParser::TEMPLATE) {
      sink()->Template(n, this, true);
    } else if (n.type == WikiParser::CATEGORY) {
      sink()->Category(n, this, true);
    } else {
      ExtractSkip(n);
    }
    child = n.next_sibling;
  }
}

Text WikiExtractor::GetAttr(const Node &node, Text attrname) {
  int child = node.first_child;
  while (child != -1) {
    const Node &n = parser_.node(child);
    if (n.type == WikiParser::ATTR && n.name() == attrname) {
      return n.text();
    }
    child = n.next_sibling;
  }
  return Text();
}

void WikiExtractor::ResetFont() {
  sink()->Font(0);
}

void WikiPlainTextSink::Content(const char *begin, const char *end) {
  if (begin != end && *begin == '<') return;
  for (const char *p = begin; p < end; ++p) {
    if (*p == ' ' || *p == '\n') {
      space_break_ = true;
    } else {
      if (space_break_) {
        text_.push_back(' ');
        space_break_ = false;
      }
      text_.push_back(*p);
    }
  }
}

void WikiTextSink::Content(const char *begin, const char *end) {
  for (const char *p = begin; p < end; ++p) {
    if (*p == '\n') {
      if (!text_.empty()) line_breaks_++;
      word_break_ = false;

      switch (font_) {
        case 2: text_.append("</em>"); break;
        case 3: text_.append("</b>"); break;
        case 5: text_.append("</em></b>"); break;
      }
      font_ = 0;
    } else if (*p != ' ' || line_breaks_ == 0) {
      if (line_breaks_ > 1) {
        text_.append("\n<p>");
        line_breaks_ = 0;
        word_break_ = false;
      } else if (line_breaks_ > 0) {
        text_.push_back('\n');
        line_breaks_ = 0;
        word_break_ = false;
      } else if (*p == ' ') {
        word_break_ = false;
      }
      if (word_break_) {
        text_.append("\xe2\x80\x8b");  // zero-width space
        word_break_ = false;
      }
      text_.push_back(*p);
    }
  }
}

void WikiTextSink::WordBreak() {
  word_break_ = true;
}

void WikiTextSink::Font(int font) {
  if (font_ == 5 && font == 2) {
    // Change from bold italic to bold.
    Content("</em>");
    font_ = 3;
  } else if (font_ == 5 && font == 3) {
    // Change from bold italic to italic.
    Content("</em></b><em>");
    font_ = 2;
  } else if (font_ != 0) {
    // Reset font.
    switch (font_) {
      case 2: Content("</em>"); break;
      case 3: Content("</b>"); break;
      case 4: Content("'"); Content("</b>"); break;
      case 5: Content("</em></b>"); break;
    }
    font_ = 0;
  } else {
    // Set font.
    switch (font) {
      case 2: Content("<em>"); break;
      case 3: Content("<b>"); break;
      case 4: Content("<b>"); Content("'"); font = 3; break;
      case 5: Content("<b><em>"); break;
    }
    font_ = font;
  }
}

}  // namespace nlp
}  // namespace sling

