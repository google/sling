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

#include "sling/nlp/wiki/wiki-parser.h"

#include <iostream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "sling/base/logging.h"
#include "sling/base/types.h"
#include "sling/string/ctype.h"
#include "sling/string/printf.h"
#include "sling/string/strcat.h"
#include "sling/string/text.h"

namespace sling {
namespace nlp {

namespace {

// AST node names.
static const char *node_names[] = {
  "DOCUMENT", "ARG", "ATTR",
  "TEXT", "FONT", "TEMPLATE", "LINK", "IMAGE", "CATEGORY", "URL",
  "COMMENT", "TAG", "BTAG", "ETAG", "MATH", "GALLERY", "REF", "NOWIKI",
  "HEADING", "INDENT", "TERM", "UL", "OL", "HR", "SWITCH",
  "TABLE", "CAPTION", "ROW", "HEADER", "CELL", "BREAK",
};

typedef std::unordered_map<string, WikiParser::Type> LinkPrefixMap;
typedef std::unordered_map<string, WikiParser::Special> TemplatePrefixMap;

// Link prefixes for images and categories.
static const LinkPrefixMap link_prefix = {
  {"Archivo",    WikiParser::IMAGE},       // es
  {"Bestand",    WikiParser::IMAGE},       // nl
  {"Categoría",  WikiParser::CATEGORY},    // es
  {"Categoria",  WikiParser::CATEGORY},    // it, pt, ca, la
  {"Catégorie",  WikiParser::CATEGORY},    // fr
  {"Categorie",  WikiParser::CATEGORY},    // nl, ro
  {"Category",   WikiParser::CATEGORY},    // en
  {"Datei",      WikiParser::IMAGE},       // de
  {"Datoteka",   WikiParser::IMAGE},       // sh
  {"Dosiero",    WikiParser::IMAGE},       // eo
  {"Fájl",       WikiParser::IMAGE},       // hu
  {"Fasciculus", WikiParser::IMAGE},       // la
  {"Ficheiro",   WikiParser::IMAGE},       // pt
  {"Fichier",    WikiParser::IMAGE},       // fr
  {"File",       WikiParser::IMAGE},       // en, el
  {"Fil",        WikiParser::IMAGE},       // da, no, sv
  {"Fișier",     WikiParser::IMAGE},       // ro
  {"Fitxategi",  WikiParser::IMAGE},       // eu
  {"Fitxer",     WikiParser::IMAGE},       // ca
  {"Image",      WikiParser::IMAGE},       // en
  {"Immagine",   WikiParser::IMAGE},       // it
  {"Kategória",  WikiParser::CATEGORY},    // hu
  {"Kategoria",  WikiParser::CATEGORY},    // pl, eu
  {"Kategorie",  WikiParser::CATEGORY},    // de, cs
  {"Kategorija", WikiParser::CATEGORY},    // sh
  {"Kategorio",  WikiParser::CATEGORY},    // eo
  {"Kategori",   WikiParser::CATEGORY},    // da, no, sv
  {"Luokka",     WikiParser::CATEGORY},    // fi
  {"Media",      WikiParser::IMAGE},       // en
  {"Plik",       WikiParser::IMAGE},       // pl
  {"Soubor",     WikiParser::IMAGE},       // cs
  {"Tiedosto",   WikiParser::IMAGE},       // fi
  {"Κατηγορία",  WikiParser::CATEGORY},    // el
  {"Датотека",   WikiParser::IMAGE},       // sr
  {"Категорија", WikiParser::CATEGORY},    // sr
  {"Категория",  WikiParser::CATEGORY},    // bg, ru
  {"Категорія",  WikiParser::CATEGORY},    // uk
  {"Файл",       WikiParser::IMAGE},       // bg, ru, uk
};

// Link prefixes for templates.
static const TemplatePrefixMap template_prefix = {
  {"DEFAULTSORT",         WikiParser::TMPL_DEFAULTSORT},
  {"DISPLAYTITLE",        WikiParser::TMPL_DISPLAYTITLE},
  {"PAGENAME",            WikiParser::TMPL_PAGENAME},
  {"PAGENAMEE",           WikiParser::TMPL_PAGENAMEE},
  {"BASEPAGENAME",        WikiParser::TMPL_BASEPAGENAME},
  {"BASEPAGENAMEE",       WikiParser::TMPL_BASEPAGENAMEE},
  {"SUBPAGENAME",         WikiParser::TMPL_SUBPAGENAME},
  {"SUBPAGENAMEE",        WikiParser::TMPL_SUBPAGENAMEE},
  {"NAMESPACE",           WikiParser::TMPL_NAMESPACE},
  {"NAMESPACEE",          WikiParser::TMPL_NAMESPACEE},
  {"FULLPAGENAME",        WikiParser::TMPL_FULLPAGENAME},
  {"FULLPAGENAMEE",       WikiParser::TMPL_FULLPAGENAMEE},
  {"TALKSPACE",           WikiParser::TMPL_TALKSPACE},
  {"TALKSPACEE",          WikiParser::TMPL_TALKSPACEE},
  {"SUBJECTSPACE",        WikiParser::TMPL_SUBJECTSPACE},
  {"SUBJECTSPACEE",       WikiParser::TMPL_SUBJECTSPACEE},
  {"ARTICLESPACE",        WikiParser::TMPL_ARTICLESPACE},
  {"ARTICLESPACEE",       WikiParser::TMPL_ARTICLESPACEE},
  {"TALKPAGENAME",        WikiParser::TMPL_TALKPAGENAME},
  {"TALKPAGENAMEE",       WikiParser::TMPL_TALKPAGENAMEE},
  {"SUBJECTPAGENAME",     WikiParser::TMPL_SUBJECTPAGENAME},
  {"SUBJECTPAGENAMEE",    WikiParser::TMPL_SUBJECTPAGENAMEE},
  {"ARTICLEPAGENAME",     WikiParser::TMPL_ARTICLEPAGENAME},
  {"ARTICLEPAGENAMEE",    WikiParser::TMPL_ARTICLEPAGENAMEE},
  {"REVISIONID",          WikiParser::TMPL_REVISIONID},
  {"REVISIONDAY",         WikiParser::TMPL_REVISIONDAY},
  {"REVISIONDAY2",        WikiParser::TMPL_REVISIONDAY2},
  {"REVISIONMONTH",       WikiParser::TMPL_REVISIONMONTH},
  {"REVISIONYEAR",        WikiParser::TMPL_REVISIONYEAR},
  {"REVISIONTIMESTAMP",   WikiParser::TMPL_REVISIONTIMESTAMP},
  {"SITENAME",            WikiParser::TMPL_SITENAME},
  {"SERVER",              WikiParser::TMPL_SERVER},
  {"SCRIPTPATH",          WikiParser::TMPL_SCRIPTPATH},
  {"SERVERNAME",          WikiParser::TMPL_SERVERNAME},
  {"CONTENTLANGUAGE",     WikiParser::TMPL_CONTENTLANGUAGE},
  {"DIRECTIONMARK",       WikiParser::TMPL_DIRECTIONMARK},
  {"DIRMARK",             WikiParser::TMPL_DIRECTIONMARK},

  {"CURRENTYEAR",         WikiParser::TMPL_CURRENTYEAR},
  {"CURRENTMONTH",        WikiParser::TMPL_CURRENTMONTH},
  {"CURRENTMONTH1",       WikiParser::TMPL_CURRENTMONTH1},
  {"CURRENTMONTHNAME",    WikiParser::TMPL_CURRENTMONTHNAME},
  {"CURRENTMONTHABBREV",  WikiParser::TMPL_CURRENTMONTHABBREV},
  {"CURRENTDAY",          WikiParser::TMPL_CURRENTDAY},
  {"CURRENTDAY2",         WikiParser::TMPL_CURRENTDAY2},
  {"CURRENTDOW",          WikiParser::TMPL_CURRENTDOW},
  {"CURRENTDAYNAME",      WikiParser::TMPL_CURRENTDAYNAME},
  {"CURRENTTIME",         WikiParser::TMPL_CURRENTTIME},
  {"CURRENTHOUR",         WikiParser::TMPL_CURRENTHOUR},
  {"CURRENTWEEK",         WikiParser::TMPL_CURRENTWEEK},
  {"CURRENTTIMESTAMP",    WikiParser::TMPL_CURRENTTIMESTAMP},
  {"CURRENTMONTHNAMEGEN", WikiParser::TMPL_CURRENTMONTHNAMEGEN},
  {"LOCALYEAR",           WikiParser::TMPL_LOCALYEAR},
  {"LOCALMONTH",          WikiParser::TMPL_LOCALMONTH},
  {"LOCALMONTH1",         WikiParser::TMPL_LOCALMONTH1},
  {"LOCALMONTHNAME",      WikiParser::TMPL_LOCALMONTHNAME},
  {"LOCALMONTHNAMEGEN",   WikiParser::TMPL_LOCALMONTHNAMEGEN},
  {"LOCALMONTHABBREV",    WikiParser::TMPL_LOCALMONTHABBREV},
  {"LOCALDAY",            WikiParser::TMPL_LOCALDAY},
  {"LOCALDAY2",           WikiParser::TMPL_LOCALDAY2},
  {"LOCALDOW",            WikiParser::TMPL_LOCALDOW},
  {"LOCALDAYNAME",        WikiParser::TMPL_LOCALDAYNAME},
  {"LOCALTIME",           WikiParser::TMPL_LOCALTIME},
  {"LOCALHOUR",           WikiParser::TMPL_LOCALHOUR},
  {"LOCALWEEK",           WikiParser::TMPL_LOCALWEEK},
  {"LOCALTIMESTAMP",      WikiParser::TMPL_LOCALTIMESTAMP},

  {"formatnum",           WikiParser::TMPL_FORMATNUM},
  {"Formatnum",           WikiParser::TMPL_FORMATNUM},
  {"grammar",             WikiParser::TMPL_GRAMMAR},
  {"plural",              WikiParser::TMPL_PLURAL},

  {"int",                 WikiParser::TMPL_INT},
  {"msg",                 WikiParser::TMPL_MSG},
  {"msgnw",               WikiParser::TMPL_MSGNW},
  {"raw",                 WikiParser::TMPL_RAW},
  {"subst",               WikiParser::TMPL_SUBST},

  {"#expr",               WikiParser::TMPL_EXPR},
  {"#EXPR",               WikiParser::TMPL_EXPR},
  {"#if",                 WikiParser::TMPL_IF},
  {"#ifexpr",             WikiParser::TMPL_IFEXPR},
  {"#ifexist",            WikiParser::TMPL_IFEXIST},
  {"#ifeq",               WikiParser::TMPL_IFEQ},
  {"#tag",                WikiParser::TMPL_TAG},
  {"#Tag",                WikiParser::TMPL_TAG},
  {"#related",            WikiParser::TMPL_RELATED},
  {"#time",               WikiParser::TMPL_TIME},
  {"#invoke",             WikiParser::TMPL_INVOKE},
  {"#section",            WikiParser::TMPL_SECTION},
  {"#section-h",          WikiParser::TMPL_SECTIONH},
  {"#property",           WikiParser::TMPL_PROPERTY},
  {"#Property",           WikiParser::TMPL_PROPERTY},
  {"#dateformat",         WikiParser::TMPL_DATEFORMAT},
  {"#formatdate",         WikiParser::TMPL_FORMATDATE},
  {"#list",               WikiParser::TMPL_LIST},
  {"#statements",         WikiParser::TMPL_STATEMENTS},
  {"#switch",             WikiParser::TMPL_SWITCH},
};

}  // namespace

WikiParser::WikiParser(const char *wikitext) {
  ptr_ = wikitext;
  txt_ = ptr_;
}

void WikiParser::Parse() {
  // Push top-level document element.
  Push(DOCUMENT);

  // Parse until end.
  ParseNewLine();
  ParseUntil(0);

  // End all remaining elements.
  UnwindUntil(DOCUMENT);
}

void WikiParser::ParseUntil(char stop) {
  while (*ptr_ != stop && *ptr_ != 0) {
    const char *before = ptr_;
    switch (*ptr_) {
      case '\n': ParseNewLine(); break;
      case '\'': ParseFont(); break;
      case '<':
        ParseTag();
        if (Inside(GALLERY)) ParseGallery();
        break;

      case '!':
        if (Inside(TABLE, TEMPLATE) && Matches("!!")) {
          ParseHeaderCell(false);
        } else {
          ptr_++;
        }
        break;

      case '|':
        if (Inside(TABLE, TEMPLATE) && Matches("||")) {
          ParseTableCell(false);
        } else {
          ParseArgument();
        }
        break;

      case '{':
        if (Matches("{{")) {
          ParseTemplateBegin();
        } else {
          ptr_++;
        }
        break;

      case '}':
        if (Matches("}}")) {
          ParseTemplateEnd();
        } else {
          ptr_++;
        }
        break;

      case '[':
        if (Matches("[[")) {
          ParseLinkBegin();
        } else {
          ParseUrlBegin();
        }
        break;

      case ']':
        if (Matches("]]")) {
          ParseLinkEnd();
        } else {
          ParseUrlEnd();
        }
        break;

      case '=':
        if (Inside(HEADING) && Matches("==")) {
          ParseHeadingEnd();
        } else {
          ptr_++;
        }
        break;

      case '_':
        if (Matches("__")) {
          ParseSwitch();
        } else {
          ptr_++;
        }
        break;

      default:
        ptr_++;
    }
    if (ptr_ == before) {
      LOG(WARNING) << "Parser stalled";
      break;
    }
  }
}

void WikiParser::ParseNewLine() {
  // Skip newlines and spaces.
  while (*ptr_ == '\n' || *ptr_ == ' ') ptr_++;

  // End all elements that cannot span newline.
  for (int i = stack_.size() - 1; i >= 0; --i) {
    int t = nodes_[stack_[i]].type;
    if (t == TEMPLATE) break;
    if (t >= HEADING && t <= SWITCH) {
      EndText();
      while (stack_.size() > i) Pop();
    }
  }

  // Parse image link at beginning of line inside gallery tag.
  if (Inside(GALLERY)) {
    if (Inside(LINK, GALLERY)) {
      UnwindUntil(LINK);
      ParseGallery();
      return;
    }
    if (Inside(IMAGE, GALLERY)) {
      UnwindUntil(IMAGE);
      ParseGallery();
      return;
    }
  }

  // Check for elements that can start line.
  switch (*ptr_) {
    case '=': ParseHeadingBegin(); return;

    case '*':
    case '#':
    case ';':
    case ':':
      ParseListItem();
      return;

    case '{':
      if (Matches("{|")) {
        ParseTableBegin();
        return;
      } else if (Matches("{{")) {
        ParseTemplateBegin();
        return;
      }
      break;

    case '|':
      if (Inside(TABLE, TEMPLATE)) {
        if (Matches("|+")) {
          ParseTableCaption();
          return;
        } else if (Matches("|-")) {
          ParseTableRow();
          return;
        } else if (Matches("|}")) {
          ParseTableEnd();
          return;
        } else {
          ParseTableCell(true);
          return;
        }
      } else if (Matches("|-")) {
        ParseBreak();
        return;
      }
      break;

    case '!':
      if (Inside(TABLE)) {
        ParseHeaderCell(true);
        return;
      }
      break;

    case '-':
      if (Matches("----")) {
        ParseHorizontalRule();
        return;
      }
      break;
  }

  if (txt_ == nullptr) txt_ = ptr_;
}

void WikiParser::ParseFont() {
  const char *p = ptr_;
  while (*p == '\'') p++;
  int n = p - ptr_;
  if (n > 1) {
    EndText();
    Add(FONT, n);
    txt_ = ptr_ = p;
  } else {
    ptr_++;
  }
}

void WikiParser::ParseTemplateBegin() {
  // Start template.
  int node = Push(TEMPLATE);
  ptr_ += 2;
  if (*ptr_ == ':') ptr_++;

  // Parse template name.
  const char *name = ptr_;
  while (*ptr_ != 0 && *ptr_ != '|' && *ptr_ != '}' &&
         *ptr_ != '{' && *ptr_ != '<') {
    ptr_++;
  }
  SetName(node, name, ptr_);
  txt_ = ptr_;
}

void WikiParser::ParseTemplateEnd() {
  if (!Inside(TEMPLATE)) {
    // Ignore unmatched }}.
    ptr_ += 2;
    return;
  }

  int node = UnwindUntil(TEMPLATE);
  ptr_ += 2;
  if (node != -1) {
    Node &n = nodes_[node];
    n.end = ptr_;

    // Check for special templates.
    int colon = n.name().find(':');
    if (colon != -1) {
      string prefix(n.name_begin, colon);
      auto f = template_prefix.find(prefix);
      if (f != template_prefix.end()) {
        n.param = f->second;
        n.name_begin += colon + 1;
      }
    } else {
      auto f = template_prefix.find(n.name().str());
      if (f != template_prefix.end()) {
        n.param = f->second;
      }
    }
  }
  txt_ = ptr_;
}

void WikiParser::ParseArgument() {
  // Allow | as verbatim text in URL nodes.
  if (Inside(URL, TEMPLATE, LINK)) {
    ptr_++;
    return;
  }

  // Terminate argument.
  EndText();
  if (Inside(ARG, TEMPLATE, LINK)) {
    UnwindUntil(ARG);
  }

  // Skip separator.
  ptr_++;
  SkipWhitespace();
  txt_ = ptr_;

  // Push new argument.
  int node = Push(ARG);

  // Try to parse argument name.
  const char *name = ptr_;
  const char *p = name;
  while (*p != 0 && *p != '\n' && *p != '=' && *p != '<' &&
         *p != ']' && *p != '|' && *p != '}' && *p != '{') {
    p++;
  }
  if (*p == '=') {
    SetName(node, name, p);
    ptr_ = p + 1;
  }

  SkipWhitespace();
  txt_ = ptr_;
}

void WikiParser::ParseLinkBegin() {
  // Start link.
  EndText();
  int node = Push(LINK);
  ptr_ += 2;

  // Parse link name.
  const char *name = ptr_;
  while (*ptr_ != 0 && *ptr_ != '|' && *ptr_ != ']' &&
         *ptr_ != '{' && *ptr_ != '<') {
    ptr_++;
  }
  SetName(node, name, ptr_);
  if (*ptr_ == ']') {
    txt_ = name;
  } else {
    txt_ = ptr_;
  }
}

void WikiParser::ParseLinkEnd() {
  if (!Inside(LINK)) {
    // Ignore unmatched ]].
    ptr_ += 2;
    return;
  }

  int node = UnwindUntil(LINK);
  ptr_ += 2;
  if (node != -1) {
    Node &n = nodes_[node];
    n.end = ptr_;
    n.CheckSpecialLink();
  }
  txt_ = ptr_;
}

void WikiParser::ParseUrlBegin() {
  // Check for valid url, i.e. it must start with protocol://.
  const char *p = ptr_ + 1;
  while (ascii_isalpha(*p)) p++;
  if (p[0] != ':' || p[1] != '/' || p[2] != '/') {
    ptr_++;
    return;
  }

  // Start URL node.
  EndText();
  int node = Push(URL);
  ptr_ += 1;

  // Parse url.
  const char *name = ptr_;
  while (*ptr_ != 0 && *ptr_ != ' ' && *ptr_ != '\n' && *ptr_ != ']') ptr_++;
  SetName(node, name, ptr_);

  while (*ptr_ == ' ') ptr_++;
  if (*ptr_ == ']') {
    txt_ = name;
  } else {
    txt_ = ptr_;
  }
}

void WikiParser::ParseUrlEnd() {
  if (!Inside(URL)) {
    // Ignore unmatched ].
    ptr_ += 1;
    return;
  }

  int node = UnwindUntil(URL);
  ptr_ += 1;
  if (node != -1) {
    Node &n = nodes_[node];
    n.end = ptr_;
  }
  txt_ = ptr_;
}

void WikiParser::ParseTag() {
  EndText();
  if (Matches("<!--")) {
    int node = Add(COMMENT);
    ptr_ += 4;
    while (*ptr_ != 0) {
      if (*ptr_ == '-' && Matches("-->")) break;
      ptr_++;
    }
    if (*ptr_ != 0) ptr_ += 3;
    nodes_[node].end = ptr_;
    txt_ = ptr_;
  } else if (Matches("<math>")) {
    int node = Add(MATH);
    ptr_ += 6;
    while (*ptr_ != 0) {
      if (*ptr_ == '<' && Matches("</math>")) break;
      ptr_++;
    }
    if (*ptr_ != 0) ptr_ += 7;
    nodes_[node].end = ptr_;
    txt_ = ptr_;
  } else if (Matches("<nowiki>")) {
    int node = Add(NOWIKI);
    ptr_ += 8;
    txt_ = ptr_;
    while (*ptr_ != 0) {
      if (*ptr_ == '<' && Matches("</nowiki>")) break;
      ptr_++;
    }
    if (*ptr_ != 0) ptr_ += 9;
    nodes_[node].end = ptr_;
    txt_ = ptr_;
  } else if (Matches("</ref>")) {
    ptr_ += 6;
    txt_ = ptr_;
    if (Inside(REF)) UnwindUntil(REF);
  } else if (Matches("</gallery>")) {
    ptr_ += 10;
    txt_ = ptr_;
    if (Inside(GALLERY)) UnwindUntil(GALLERY);
  } else {
    // Parse '<' (BTAG) or '</' (ETAG).
    const char *p = ptr_;
    Type type = BTAG;
    p += 1;
    if (*p == '/') {
      type = ETAG;
      p++;
    }

    // Parse tag name. Skip if tag name is empty or starts with a digit since
    // this is most likely an unescaped <.
    const char *tagname = p;
    while (IsNameChar(*p)) p++;
    if (p == tagname || ascii_isdigit(*tagname)) {
      txt_ = ptr_;
      ptr_++;
      return;
    }

    // Create tag node.
    int node = Push(type);
    SetName(node, tagname, p);
    ptr_ = p;

    // Parse attributes.
    ParseAttributes("/>");

    // Parse end of tag '>' (ETAG) or '/>' (TAG).
    while (*ptr_ == ' ') ptr_++;
    if (*ptr_ == '/') {
      type = TAG;
      ptr_++;
    }
    while (*ptr_ != 0 && *ptr_ != '>') ptr_++;
    if (*ptr_ != 0) ptr_ += 1;
    txt_ = ptr_;

    nodes_[node].end = ptr_;
    nodes_[node].type = type;

    if (type == BTAG && nodes_[node].name() == "ref") {
      nodes_[node].type = REF;
    } else if (type == BTAG && nodes_[node].name() == "gallery") {
      // The gallery tag encloses lines of image links.
      nodes_[node].type = GALLERY;
      SkipWhitespace();
      if (*ptr_ == '\n') {
        ptr_++;
        SkipWhitespace();
      }
      txt_ = ptr_;
    } else {
      UnwindUntil(type);
    }
  }
}

void WikiParser::ParseGallery() {
  // Parse link name at the start of a gallery line.
  if (*ptr_ == '<') return;
  int node = Push(LINK);
  const char *name = ptr_;
  while (*ptr_ != 0 && *ptr_ != '|' && *ptr_ != '<' && *ptr_ != '\n')  ptr_++;
  SetName(node, name, ptr_);
  txt_ = ptr_;
  nodes_[node].CheckSpecialLink();
}

void WikiParser::ParseHeadingBegin() {
  // Headings are always top-level.
  EndText();
  while (stack_.size() > 1) Pop();

  // Get heading level.
  const char *p = ptr_;
  while (*p == '=') p++;
  int level = p - ptr_;

  // Create heading node.
  Push(HEADING, level);
  txt_ = ptr_ = p;
}

void WikiParser::ParseHeadingEnd() {
  int node = UnwindUntil(HEADING);
  while (*ptr_ == '=') ptr_++;
  if (node != -1) nodes_[node].end = ptr_;
  txt_ = ptr_;
}

void WikiParser::ParseListItem() {
  // Get item level.
  const char *p = ptr_;
  while (*p == '*' || *p == '#' || *p == ':' || *p == ';') p++;

  // Start new item.
  Type type;
  switch (p[-1]) {
    case '*': type = UL; break;
    case '#': type = OL; break;
    case ':': type = INDENT; break;
    case ';': type = TERM; break;
    default:
      ptr_ = p;
      return;
  }

  Push(type, p - ptr_);
  ptr_ = p;
  SkipWhitespace();
  txt_ = ptr_;
}

void WikiParser::ParseHorizontalRule() {
  EndText();
  Add(HR);
  ptr_ += 4;
  SkipWhitespace();
}

void WikiParser::ParseSwitch() {
  const char *p = ptr_ + 2;
  while (*p != 0 && ascii_isupper(*p)) p++;
  if (p[0] == '_' && p[1] == '_') {
    p += 2;
    EndText();
    int node = Add(SWITCH);
    nodes_[node].end = p;
    txt_ = ptr_ = p;
  } else {
    ptr_++;
  }
}

void WikiParser::ParseTableBegin() {
  Push(TABLE);
  ptr_ += 2;
  if (!ParseAttributes("\n")) SkipWhitespace();
  txt_ = ptr_;
}

void WikiParser::ParseTableCaption() {
  EndText();
  if (Inside(CAPTION, TABLE)) UnwindUntil(CAPTION);
  if (Inside(ROW, TABLE)) UnwindUntil(ROW);

  Push(CAPTION);
  ptr_ += 2;
  SkipWhitespace();
  txt_ = ptr_;
}

void WikiParser::ParseTableRow() {
  EndText();
  if (Inside(CAPTION, TABLE)) UnwindUntil(CAPTION);
  if (Inside(ROW, TABLE)) UnwindUntil(ROW);

  Push(ROW);
  ptr_ += 2;
  if (*ptr_ == '-') ptr_++;
  if (!ParseAttributes("\n")) SkipWhitespace();
  txt_ = ptr_;
}

void WikiParser::ParseHeaderCell(bool first) {
  EndText();
  if (Inside(CAPTION, TABLE)) UnwindUntil(CAPTION);
  if (!Inside(ROW, TABLE)) Push(ROW);
  if (Inside(HEADER, ROW)) UnwindUntil(HEADER);

  Push(HEADER);
  ptr_ += first ? 1 : 2;
  if (ParseAttributes("!\n")) {
    if (*ptr_ == '!') ptr_++;
    if (*ptr_ == '!') ptr_++;
  }
  SkipWhitespace();
  txt_ = ptr_;
}

void WikiParser::ParseTableCell(bool first) {
  EndText();
  if (Inside(CAPTION, TABLE)) UnwindUntil(CAPTION);
  if (!Inside(ROW, TABLE)) Push(ROW);

  if (Inside(CELL, ROW)) {
    UnwindUntil(CELL);
  } else if (Inside(HEADER, ROW)) {
    UnwindUntil(HEADER);
  }
  Push(CELL);

  ptr_ += first ? 1 : 2;
  if (ParseAttributes("|\n")) {
    if (*ptr_ == '|') ptr_++;
    if (*ptr_ == '|') ptr_++;
  }
  SkipWhitespace();
  txt_ = ptr_;
}

void WikiParser::ParseTableEnd() {
  int node = UnwindUntil(TABLE);
  ptr_ += 2;
  if (node != -1) nodes_[node].end = ptr_;
  txt_ = ptr_;
}

void WikiParser::ParseBreak() {
  EndText();
  Add(BREAK);
  ptr_ += 2;
  SkipWhitespace();
}

bool WikiParser::ParseAttributes(const char *delimiters) {
  std::vector<std::pair<Text, Text>> attributes;
  const char *p = ptr_;
  while (1) {
    // Skip whitespace.
    while (*p == ' ') p++;

    // Check for delimiter.
    bool done = false;
    for (const char *d = delimiters; *d; ++d) {
      if (*p == *d) {
        done = true;
        break;
      }
    }
    if (done) break;

    // Try to parse attribute name.
    const char *name = p;
    while (IsNameChar(*p)) p++;
    if (p == name) break;
    int namelen = p - name;

    // Skip whitespace.
    while (*p == ' ') p++;

    // Check for '='.
    if (*p != '=') break;
    p++;

    // Skip whitespace.
    while (*p == ' ') p++;

    // Try to parse attribute value.
    const char *attr;
    int attrlen;
    if (*p == '"') {
      attr = ++p;
      while (*p != 0 && *p != '"' && *p != '\n') p++;
      if (*p != '"') return false;
      attrlen = p++ - attr;
    } else {
      attr = p;
      while (IsNameChar(*p)  || *p == '#'  || *p == '%' || *p == ';') p++;
      if (p == attr) return false;
      attrlen = p - attr;
    }

    // Add attribute to list.
    attributes.emplace_back(Text(name, namelen), Text(attr, attrlen));
  }

  // Bail out if no attributes found.
  if (attributes.empty()) return false;

  // Add attributes to current node.
  for (auto &a : attributes) {
    Node &node = nodes_[Add(ATTR)];
    node.begin = a.second.data();
    node.end = a.second.data() + a.second.size();
    node.name_begin = a.first.data();
    node.name_end = a.first.data() + a.first.size();
  }
  ptr_ = p;
  return true;
}

void WikiParser::PrintAST(int index, int indent) {
  std::cout << StringPrintf("%05d ", index);
  for (int i = 0; i < indent; ++i) std::cout << "  ";
  Node &node = nodes_[index];
  std::cout << node_names[node.type];
  if (node.param != 0) {
    std::cout << "(" << node.param << ")";
  }
  if (node.named()) {
    std::cout << "[" << node.name() << "]";
  }
  if (node.begin != nullptr && node.end != nullptr) {
    std::cout << ": ";
    if (node.end - node.begin > 4096) {
      std::cout << "<<<" << (node.end - node.begin) << " bytes>>>";
    } else {
      for (const char *p = node.begin; p < node.end; ++p) {
        if (*p == '\n') {
          std::cout << "⏎";
        } else if (*p == '<') {
          std::cout << "&lt;";
        } else if (*p == '>') {
          std::cout << "&gt;";
        } else if (*p == '&') {
          std::cout << "&amp;";
        } else {
          std::cout << *p;
        }
      }
    }
  }
  std::cout << "\n";
  int child = node.first_child;
  while (child != -1) {
    PrintAST(child, indent + 1);
    child = nodes_[child].next_sibling;
  }
}

int WikiParser::Add(Type type, int param) {
  // Create new node.
  int index = nodes_.size();
  nodes_.emplace_back(type, param);
  Node &node = nodes_.back();
  node.begin = ptr_;

  // Add node as child of the current top node.
  if (!stack_.empty()) {
    Node &parent = nodes_[stack_.back()];
    int before = parent.last_child;
    node.prev_sibling = before;
    if (before != -1) nodes_[before].next_sibling = index;
    if (parent.first_child == -1) parent.first_child = index;
    parent.last_child = index;
  }

  return index;
}

void WikiParser::SetName(int index, const char *begin, const char *end) {
  // Remove whitespace before.
  while (ascii_isspace(*begin) && begin < end) begin++;

  // Remove whitespace after name.
  while (end - 1 > begin && ascii_isspace(*(end - 1))) end--;

  // Set node name.
  nodes_[index].name_begin = begin;
  nodes_[index].name_end = end;
}

void WikiParser::EndText() {
  if (txt_ == nullptr || txt_ == ptr_) return;
  int index = Add(TEXT);
  Node &text = nodes_[index];
  text.begin = txt_;
  text.end = ptr_;
  txt_ = nullptr;
}

int WikiParser::Push(Type type, int param) {
  EndText();
  int index = Add(type, param);
  stack_.push_back(index);
  return index;
}

int WikiParser::Pop() {
  CHECK(!stack_.empty()) << ptr_;
  int top = stack_.back();
  nodes_[top].end = ptr_;
  stack_.pop_back();
  return top;
}

int WikiParser::UnwindUntil(int type) {
  EndText();
  while (!stack_.empty()) {
    if (nodes_[stack_.back()].type == type) {
      int node = stack_.back();
      Pop();
      return node;
    } else {
      Pop();
    }
  }
  return -1;
}

bool WikiParser::Inside(Type type, Type another) {
  for (int i = stack_.size() - 1; i >= 0; --i) {
    Type t = nodes_[stack_[i]].type;
    if (t == type) return true;
    if (t == another) return false;
  }
  return false;
}

bool WikiParser::Inside(Type type, Type another1, Type another2) {
  for (int i = stack_.size() - 1; i >= 0; --i) {
    Type t = nodes_[stack_[i]].type;
    if (t == type) return true;
    if (t == another1 || t == another2) return false;
  }
  return false;
}

bool WikiParser::Matches(const char *prefix) {
  return strncmp(ptr_, prefix, strlen(prefix)) == 0;
}

void WikiParser::SkipWhitespace() {
  while (*ptr_ == ' ') ptr_++;
}

void WikiParser::Node::CheckSpecialLink() {
  int colon = name().find(':');
  if (colon != -1) {
    string prefix(name_begin, colon);
    if (prefix.size() >= 1 && prefix[0] >= 'a' && prefix[0] <= 'z') {
      prefix[0] = prefix[0] - 'a' + 'A';
    }
    auto f = link_prefix.find(prefix);
    if (f != link_prefix.end()) {
      type = f->second;
      name_begin += colon + 1;
    }
  }
}

}  // namespace nlp
}  // namespace sling

