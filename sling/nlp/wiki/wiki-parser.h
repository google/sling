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

#ifndef SLING_NLP_WIKI_WIKI_PARSER_H_
#define SLING_NLP_WIKI_WIKI_PARSER_H_

#include <string>
#include <vector>

#include "sling/base/types.h"
#include "sling/string/ctype.h"
#include "sling/string/text.h"

namespace sling {
namespace nlp {

// Parse wiki text and convert to abstract syntax tree (AST). The plain text as
// well as structured information can then be extracted from the AST.
class WikiParser {
 public:
  // Wiki AST node type.
  enum Type {
    DOCUMENT,    // top-level node
    ARG,         // argument for template, link, etc.
    ATTR,        // attribute

    // Inline elements.
    TEXT,        // plain text
    FONT,        // ''italics'', '''bold''', and '''''both'''''
    TEMPLATE,    // {{name | args... }}
    LINK,        // [[link | text]]
    IMAGE,       // [[File:link | text]]
    CATEGORY,    // [[Category:...]]
    URL,         // [url text]
    COMMENT,     // <!-- comment -->
    TAG,         // <tag/>
    BTAG,        // <tag attr=''>
    ETAG,        // </tag>
    MATH,        // <math>...</math>
    GALLERY,     // <gallery>...</gallery>
    REF,         // <ref>...</ref>
    NOWIKI,      // <nowiki>...</nowiki>

    // Elements that must be at the start of a line.
    HEADING,     // =h1= ==h2== ===h3===
    INDENT,      // : :: :::
    TERM,        // ; term
    UL,          // * ** *** ****
    OL,          // # ## ### ###
    HR,          // ----
    SWITCH,      // __SWITCH__

    // Tables.
    TABLE,       // {| |}
    CAPTION,     // |+
    ROW,         // |-
    HEADER,      // ! !!
    CELL,        // | ||
    BREAK,       // |- (outside table)
  };

  // Wiki AST node.
  struct Node {
    Node(Type t, int p) : type(t), param(p) {}

    Text text() const { return Text(begin, end - begin); }
    Text name() const { return Text(name_begin, name_end - name_begin); }
    bool named() const { return name_begin != nullptr && name_end != nullptr; }

    void CheckSpecialLink();

    // Node type with optional parameter.
    Type type;
    int param;

    // Links to children and sibling nodes.
    int first_child = -1;
    int last_child = -1;
    int prev_sibling = -1;
    int next_sibling = -1;

    // Input text span covered by the AST node.
    const char *begin = nullptr;
    const char *end = nullptr;

    // Name part of text span covered by the AST node.
    const char *name_begin = nullptr;
    const char *name_end = nullptr;
  };

  // Initialize parser with wiki text.
  WikiParser(const char *wikitext);

  // Parse wiki text.
  void Parse();

  // Print AST node and its children.
  void PrintAST(int index, int indent);

  // Return nodes.
  const std::vector<Node> &nodes() const { return nodes_; }

  // Return node.
  const Node &node(int index) const { return nodes_[index]; }

 private:
  // Parse input until stop mark is found.
  void ParseUntil(char stop);

  // Parse newline.
  void ParseNewLine();

  // Parse font change.
  void ParseFont();

  // Parse template start.
  void ParseTemplateBegin();

  // Parse template end.
  void ParseTemplateEnd();

  // Parse argument separator.
  void ParseArgument();

  // Parse link start.
  void ParseLinkBegin();

  // Parse link end.
  void ParseLinkEnd();

  // Parse url start.
  void ParseUrlBegin();

  // Parse url end.
  void ParseUrlEnd();

  // Parse tag (<...>) or comment (<!-- ... -->).
  void ParseTag();

  // Parse gallery (<gallery> ... </gallery>).
  void ParseGallery();

  // Parse heading start.
  void ParseHeadingBegin();

  // Parse heading end.
  void ParseHeadingEnd();

  // Parse list item (* or =).
  void ParseListItem();

  // Parse horizontal rule (----).
  void ParseHorizontalRule();

  // Parse behavior switch (__SWITCH__).
  void ParseSwitch();

  // Parse table start ({|).
  void ParseTableBegin();

  // Parse table caption (|+).
  void ParseTableCaption();

  // Parse table row (|-).
  void ParseTableRow();

  // Parse table header cell (! or !!).
  void ParseHeaderCell(bool first);

  // Parse table cell (| or ||).
  void ParseTableCell(bool first);

  // Parse table end (|}).
  void ParseTableEnd();

  // Parse break (|- outside table).
  void ParseBreak();

  // Parse HTML/XML attribute list. Return true if any attributes found.
  bool ParseAttributes(const char *delimiters);

  // Add child node to current AST node.
  int Add(Type type, int param = 0);

  // Set node name. This trims whitespace from the name.
  void SetName(int index, const char *begin, const char *end);

  // End current text block.
  void EndText();

  // Push new node top stack.
  int Push(Type type, int param = 0);

  // Pop top node from stack.
  int Pop();

  // Unwind stack.
  int UnwindUntil(int type);

  // Check if inside one element rather than another.
  bool Inside(Type type, Type another = DOCUMENT);
  bool Inside(Type type, Type another1, Type another2);

  // Check if current input matches string.
  bool Matches(const char *prefix);

  // Skip whitespace.
  void SkipWhitespace();

  // Check if a character is an XML name character.
  static bool IsNameChar(int c) {
    return ascii_isalnum(c) || c == ':' || c == '-' || c == '_' || c == '.';
  }

  // Current position in text.
  const char *ptr_ = nullptr;

  // Start of current text node.
  const char *txt_ = nullptr;

  // List of AST nodes on page.
  std::vector<Node> nodes_;

  // Current nesting of AST nodes. The stack contains indices into the AST
  // node array.
  std::vector<int> stack_;

 public:
  // Special template types.
  enum Special {
    TMPL_NORMAL,

    TMPL_DEFAULTSORT,
    TMPL_DISPLAYTITLE,
    TMPL_PAGENAME,
    TMPL_PAGENAMEE,
    TMPL_BASEPAGENAME,
    TMPL_BASEPAGENAMEE,
    TMPL_SUBPAGENAME,
    TMPL_SUBPAGENAMEE,
    TMPL_NAMESPACE,
    TMPL_NAMESPACEE,
    TMPL_FULLPAGENAME,
    TMPL_FULLPAGENAMEE,
    TMPL_TALKSPACE,
    TMPL_TALKSPACEE,
    TMPL_SUBJECTSPACE,
    TMPL_SUBJECTSPACEE,
    TMPL_ARTICLESPACE,
    TMPL_ARTICLESPACEE,
    TMPL_TALKPAGENAME,
    TMPL_TALKPAGENAMEE,
    TMPL_SUBJECTPAGENAME,
    TMPL_SUBJECTPAGENAMEE,
    TMPL_ARTICLEPAGENAME,
    TMPL_ARTICLEPAGENAMEE,
    TMPL_REVISIONID,
    TMPL_REVISIONDAY,
    TMPL_REVISIONDAY2,
    TMPL_REVISIONMONTH,
    TMPL_REVISIONYEAR,
    TMPL_REVISIONTIMESTAMP,
    TMPL_SITENAME,
    TMPL_SERVER,
    TMPL_SCRIPTPATH,
    TMPL_SERVERNAME,

    TMPL_CONTENTLANGUAGE,
    TMPL_DIRECTIONMARK,
    TMPL_CURRENTYEAR,

    TMPL_CURRENTMONTH,
    TMPL_CURRENTMONTH1,
    TMPL_CURRENTMONTHNAME,
    TMPL_CURRENTMONTHABBREV,
    TMPL_CURRENTDAY,
    TMPL_CURRENTDAY2,
    TMPL_CURRENTDOW,
    TMPL_CURRENTDAYNAME,
    TMPL_CURRENTTIME,
    TMPL_CURRENTHOUR,
    TMPL_CURRENTWEEK,
    TMPL_CURRENTTIMESTAMP,
    TMPL_CURRENTMONTHNAMEGEN,
    TMPL_LOCALYEAR,
    TMPL_LOCALMONTH,
    TMPL_LOCALMONTH1,
    TMPL_LOCALMONTHNAME,
    TMPL_LOCALMONTHNAMEGEN,
    TMPL_LOCALMONTHABBREV,
    TMPL_LOCALDAY,
    TMPL_LOCALDAY2,
    TMPL_LOCALDOW,
    TMPL_LOCALDAYNAME,
    TMPL_LOCALTIME,
    TMPL_LOCALHOUR,
    TMPL_LOCALWEEK,
    TMPL_LOCALTIMESTAMP,

    TMPL_FORMATNUM,
    TMPL_GRAMMAR,
    TMPL_PLURAL,

    TMPL_INT,
    TMPL_MSG,
    TMPL_MSGNW,
    TMPL_RAW,
    TMPL_SUBST,

    TMPL_EXPR,
    TMPL_IF,
    TMPL_IFEXPR,
    TMPL_IFEXIST,
    TMPL_IFEQ,
    TMPL_TAG,
    TMPL_RELATED,
    TMPL_TIME,
    TMPL_INVOKE,
    TMPL_SECTION,
    TMPL_SECTIONH,
    TMPL_PROPERTY,
    TMPL_DATEFORMAT,
    TMPL_FORMATDATE,
    TMPL_LIST,
    TMPL_STATEMENTS,
    TMPL_SWITCH,
  };
};

}  // namespace nlp
}  // namespace sling

#endif  // SLING_NLP_WIKI_WIKI_PARSER_H_

