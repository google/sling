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

#include "sling/nlp/document/text-tokenizer.h"

#include <string>
#include <vector>
#include <unordered_map>

#include "sling/base/logging.h"
#include "sling/base/types.h"
#include "sling/string/ctype.h"
#include "sling/util/unicode.h"
#include "sling/web/entity-ref.h"

namespace sling {
namespace nlp {

static const int kMaxAscii = 128;

// Trie for searching for special token/suffix types.
class TrieNode {
 public:
  TrieNode();
  ~TrieNode();

  // Returns child for a character. Returns null if no child found.
  TrieNode *FindChild(char32 ch) const;

  // Returns child for a character. If it does not exist a new child node is
  // added and returned.
  TrieNode *AddChild(char32 ch);

  // Finds the longest matching token.
  const TrieNode *FindMatch(const TokenizerText &text, int start,
                            int *length) const;

  // Finds the longest matching token by traversing the text in reverse order.
  const TrieNode *FindReverseMatch(const TokenizerText &text, int start,
                                   int limit, int *length) const;

  // Returns the replacement value for node. This can only be returned if the
  // node has a value.
  const string &value() const { return *value_; }

  // Sets the replacement value for a node.
  void set_value(const string &value) {
    if (value_ != nullptr) {
      *value_ = value;
    } else {
      value_ = new string(value);
    }
  }

  // Returns true if the node has a replacement value.
  bool has_value() const { return value_ != nullptr; }

  // Returns/sets the flags for the node.
  TokenFlags flags() const { return flags_; }
  void set_flags(TokenFlags flags) { flags_ |= flags; }
  bool is(TokenFlags flags) const { return (flags_ & flags) != 0; }

  // Returns/sets the terminal status of this node.
  bool terminal() const { return terminal_; }
  void set_terminal(bool terminal) { terminal_ = terminal; }

 private:
  typedef std::unordered_map<char32, TrieNode *> TrieMap;

  // Replacement value for token. If it is defined, this is the value that
  // is used as a replacement for the matched token. This is only allocated in
  // case it is needed.
  string *value_;

  // Token flags for node.
  TokenFlags flags_;

  // A terminal node is a node that matches a token. If terminal is false this
  // is only an intermediate node.
  bool terminal_;

  // Children for node indexed by Unicode characters. The children for ASCII
  // characters (0-127) are places in the low children array, and the rest are
  // placed in the high children map. The maps are only allocated if the node
  // has any children.
  std::vector<TrieNode *> *low_children_;
  TrieMap *high_children_;
};

CharacterFlags::CharacterFlags(): low_flags_(kMaxAscii) {
}

void CharacterFlags::set(char32 ch, TokenFlags flags) {
  if (ch < kMaxAscii) {
    low_flags_[ch] = flags;
  } else {
    high_flags_[ch] = flags;
  }
}

TokenFlags CharacterFlags::get(char32 ch) const {
  if (ch < kMaxAscii) {
    return low_flags_[ch];
  } else {
    // Lookup flags in hash map.
    TokenFlags flags = 0;
    auto f = high_flags_.find(ch);
    if (f != high_flags_.end()) flags = f->second;

    // Add additional Unicode flags.
    if (Unicode::IsLetter(ch)) flags |= CHAR_LETTER;
    if (Unicode::IsUpper(ch)) flags |= CHAR_UPPER;
    if (Unicode::IsWhitespace(ch)) flags |= CHAR_SPACE;

    return flags;
  }
}

TokenizerText::TokenizerText(Text text, const CharacterFlags &char_flags) {
  // Keep reference to original text.
  source_ = text;

  // Initialize text element array with room for all characters. An extra
  // nul-termination element is added to the array. The final text might end
  // up being shorter because of escaped entities.
  elements_.resize(UTF8::Length(text.data(), text.size()) + 1);

  // Convert all characters from the UTF-8 encoded string to Unicode
  // characters. For each character we set the character flags.
  const char *start = text.data();
  const char *end = text.data() + text.size();
  const char *cur = text.data();
  int i = 0;
  int escapes = 0;
  while (cur < end) {
    Element &e = elements_[i];
    e.position = cur - start;
    e.node = nullptr;
    e.escapes = escapes;

    char32 c = UTF8::Decode(cur, end - cur);
    if (c == '&') {
      // Handle decoding of HTML entities like &amp; and &#39;.
      int consumed;
      char32 entity = ParseEntityRef(cur, end - cur, &consumed);
      if (entity >= 0) {
        escapes++;
        c = entity;
        cur += consumed;
      } else {
        cur = UTF8::Next(cur);
      }
    } else if (c == -1) {
      // Illegal UTF8 sequence; fall back on ASCII interpretation.
      c = *reinterpret_cast<const uint8 *>(cur++);
      escapes++;
      LOG(WARNING) << "Illegal UTF-8 string: " << text;
    } else {
      cur = UTF8::Next(cur);
    }

    e.ch = c;
    e.flags = char_flags.get(c);
    i++;
  }

  // Initialize the nul-termination element.
  length_ = i;
  Element &e = elements_[i];
  e.ch = 0;
  e.position = source_.size();
  e.flags = 0;
  e.node = nullptr;
  e.escapes = escapes;
}

void TokenizerText::GetText(int start, int end, string *result) const {
  // If start and end position has the same escape count the substring does not
  // contain any escaped entities. In this case we can just copy the data
  // directly from the source string. Otherwise we have to copy the characters
  // one at a time using the decoded character values.
  result->clear();
  if (elements_[start].escapes == elements_[end].escapes) {
    int from = elements_[start].position;
    int to = elements_[end].position;
    result->append(source_.data(), from, to - from);
  } else {
    for (int i = start; i < end; ++i) {
      UTF8::Encode(elements_[i].ch, result);
    }
  }
}

BreakType TokenizerText::BreakLevel(int index) const {
  int flags = elements_[index].flags;
  if (flags & TOKEN_PARA) {
    switch (flags & TOKEN_PARAM_MASK) {
      case 0: return PARAGRAPH_BREAK;
      case 1: return SECTION_BREAK;
      case 2: return CHAPTER_BREAK;
    }
  }
  if (flags & TOKEN_EOS) return SENTENCE_BREAK;
  if (flags & TOKEN_LINE) return LINE_BREAK;
  if (flags & TOKEN_NONBREAK) return NO_BREAK;
  if (flags & TOKEN_DISCARD) return SPACE_BREAK;

  return NO_BREAK;
}

int TokenizerText::Style(int index) const {
  TokenFlags flags = elements_[index].flags;
  if (flags & TOKEN_TAG) {
    TokenFlags style = flags & TOKEN_STYLE_MASK;
    if (style != 0) {
      return 1 << (style >> TOKEN_STYLE_SHIFT);
    }
  }
  return 0;
}

TrieNode::TrieNode() {
  value_ = nullptr;
  flags_ = 0;
  terminal_ = false;
  low_children_ = nullptr;
  high_children_ = nullptr;
}

TrieNode::~TrieNode() {
  delete value_;

  if (low_children_ != nullptr) {
    for (int i = 0; i < low_children_->size(); i++) {
      delete (*low_children_)[i];
    }
    delete low_children_;
  }

  if (high_children_ != nullptr) {
    for (const auto &it : *high_children_) delete it.second;
    delete high_children_;
  }
}

TrieNode *TrieNode::FindChild(char32 ch) const {
  if (ch < kMaxAscii) {
    if (low_children_ == nullptr) return nullptr;
    return (*low_children_)[ch];
  } else {
    if (high_children_ == nullptr) return nullptr;
    auto f = high_children_->find(ch);
    return f == high_children_->end() ? nullptr : f->second;
  }
}

TrieNode *TrieNode::AddChild(char32 ch) {
  TrieNode *node = nullptr;

  if (ch < kMaxAscii) {
    if (low_children_ != nullptr) {
      node = (*low_children_)[ch];
    }

    if (node == nullptr) {
      node = new TrieNode();
      if (low_children_ == nullptr) {
        low_children_ = new std::vector<TrieNode *>(kMaxAscii);
      }
      (*low_children_)[ch] = node;
    }
  } else {
    if (high_children_ != nullptr) {
      auto f = high_children_->find(ch);
      if (f != high_children_->end()) node = f->second;
    }

    if (node == nullptr) {
      node = new TrieNode();
      if (high_children_ == nullptr) high_children_ = new TrieMap();
      (*high_children_)[ch] = node;
    }
  }

  return node;
}

const TrieNode *TrieNode::FindMatch(const TokenizerText &text,
                                    int start,
                                    int *length) const {
  const TrieNode *matched_node = nullptr;
  int matched_length = 0;

  const TrieNode *node = this;
  int current = start;
  while (current < text.length()) {
    node = node->FindChild(text.lower(current));
    if (node == nullptr) break;
    current++;
    if (node->terminal()) {
      matched_node = node;
      matched_length = current - start;
    }
  }

  *length = matched_length;
  return matched_node;
}

const TrieNode *TrieNode::FindReverseMatch(const TokenizerText &text,
                                           int start, int limit,
                                           int *length) const {
  const TrieNode *matched_node = nullptr;
  int matched_length = 0;

  const TrieNode *node = this;
  int current = start;
  while (current >= limit) {
    node = node->FindChild(text.lower(current));
    if (node == nullptr) break;
    current--;
    if (node->terminal()) {
      matched_node = node;
      matched_length = start - current;
    }
  }

  *length = matched_length;
  return matched_node;
}


Tokenizer::Tokenizer() {
}

Tokenizer::~Tokenizer() {
  for (auto p : processors_) delete p;
}

void Tokenizer::SetCharacterFlags(char32 ch, TokenFlags flags) {
  char_flags_.add(ch, flags);
}

void Tokenizer::ClearCharacterFlags(char32 ch, TokenFlags flags) {
  char_flags_.clear(ch, flags);
}


void Tokenizer::InitPTB() {
  Add(new PTBTokenization());
}

void Tokenizer::InitLDC() {
  Add(new LDCTokenization());
}

void Tokenizer::Add(TokenProcessor *processor) {
  processor->Init(&char_flags_);
  processors_.push_back(processor);
}

void Tokenizer::Tokenize(Text text, const Callback &callback) const {
  // Initialize text by converting it to Unicode and setting character
  // classification flags.
  TokenizerText t(text, char_flags_);

  // Run token processors on text.
  for (auto p : processors_) p->Process(&t);

  // Generate tokens.
  Token token;
  int i = t.NextStart(0);
  bool in_quote = false;
  int bracket_level = 0;
  token.brk = NO_BREAK;
  token.style = 0;
  while (i < t.length()) {
    // Find start of next token.
    int j = t.NextStart(i + 1);

    if (t.is(i, TOKEN_DISCARD)) {
      // Update break level.
      BreakType brk = t.BreakLevel(i);
      if (brk > token.brk) token.brk = brk;

      // Update style.
      int style = t.Style(i);
      if (style != 0) {
        // In order to detect empty style changes, i.e. a BEGIN style followed
        // by an END style without emitting a token, the empty style change is
        // cancelled out.
        if (style & END_STYLE) {
          // Check for corresponding BEGIN style.
          if ((style >> 1) & token.style) {
            // Cancel style change.
            token.style &= ~(style >> 1);
            style = 0;
          }
        }
        token.style |= style;
      }
    } else {
      // Get token value.
      t.GetText(i, j, &token.text);

      // Track quotes and brackets.
      if (t.is(i, TOKEN_QUOTE)) {
        // Convert "double" quotes to ``Penn Treebank'' quotes.
        token.text = in_quote ? "''" : "``";
        in_quote = !in_quote;
      } else if (t.is(i, TOKEN_OPEN)) {
        bracket_level++;
      } else if (t.is(i, TOKEN_CLOSE)) {
        if (bracket_level > 0) bracket_level--;
      }

      if (t.node(i) != nullptr && t.node(i)->has_value()) {
        // Replacement token.
        token.text = t.node(i)->value();
      }

      // Emit token.
      token.begin = t.position(i);
      token.end = t.position(j);
      callback(token);
      token.brk = NO_BREAK;
      token.style = 0;
    }

    // Make a period followed by a lowercase letter conditional.
    if (t.at(i) == '.') {
      int k = j;
      while (k < t.length() && t.is(k, CHAR_SPACE)) k++;
      if (t.is(k, CHAR_LETTER) && !t.is(k, CHAR_UPPER)) {
        t.clear(i, TOKEN_EOS);
        t.set(i, TOKEN_CONDEOS);
      }
    }

    // Check for conditional end-of-sentence tokens. These must be followed by
    // an uppercase letter in order to be regarded as end-of-sentence markers.
    if (t.is(i, TOKEN_CONDEOS)) {
      int k = j;
      if (t.is(k, TOKEN_QUOTE | TOKEN_CLOSE)) k = t.NextStart(k + 1);
      while (k < t.length() && t.is(k, CHAR_SPACE)) k++;
      if (t.is(k, CHAR_UPPER)) t.set(i, TOKEN_EOS);
    }

    // Check for end of sentence. Do not break if the next token is also an
    // end of sentence to account for sentences like "Hi!!!".
    if (t.is(i, TOKEN_EOS | TOKEN_PARA) && !t.is(j, TOKEN_EOS)) {
      bool include_next_token = false;
      if (token.brk < SENTENCE_BREAK && !t.is(i, TOKEN_DISCARD)) {
        // If end-of-sentence punctuation is followed by a quote then the quote
        // is part of the sentence if we are inside a quotation.
        if (in_quote && t.is(j, TOKEN_QUOTE)) {
          include_next_token = true;
          in_quote = false;
        }

        // If end-of-sentence punctuation is followed by a closing bracket then
        // the bracket is part of the sentence if we are at bracket level 1.
        if (bracket_level == 1 && t.is(j, TOKEN_CLOSE)) {
          include_next_token = true;
          bracket_level = 0;
        }
      }

      // Brackets and quotes cannot span paragraph breaks.
      if (t.is(i, TOKEN_PARA)) {
        BreakType brk = t.BreakLevel(i);
        if (brk > token.brk) token.brk = brk;
        in_quote = false;
        bracket_level = 0;
        include_next_token = false;
      } else if (bracket_level == 0) {
        token.brk = SENTENCE_BREAK;
      }

      // End sentence if we are outside brackets.
      if (bracket_level == 0) {
        // Add trailing punctuation.
        if (include_next_token) {
          Token extra;
          int k = t.NextStart(j + 1);
          if (t.node(j) != nullptr && t.node(j)->has_value()) {
            extra.text = t.node(j)->value();
          } else if (t.is(j, TOKEN_QUOTE)) {
            extra.text = "''";
          } else {
            t.GetText(j, k, &extra.text);
          }

          extra.begin = t.position(j);
          extra.end = t.position(k);
          extra.brk = NO_BREAK;
          extra.style = 0;
          callback(extra);
          j = k;
        }
      }
    }

    // Move to next token.
    i = j;
  }
}

static const char *kBreakingTags[] = {
  "applet", "br", "caption", "/caption", "form", "frame",
  "h1", "/h1", "h2", "/h2", "h3", "/h3", "h4", "/h4", "h5",
  "/h5", "h6", "/h6", "hr", "li", "noframes", "/noframes", "ol",
  "/ol", "option", "/option", "p", "/p", "select", "/select",
  "table", "/table", "/title", "tr", "/tr", "ul", "/ul",
  "blockquote", "/blockquote", nullptr
};

#define TS(s) ((1ULL * s) << TOKEN_STYLE_SHIFT)

static const struct { const char *tag; uint64 flags; } kStyleTags[] = {
  {"<b>",   TS(STYLE_BOLD_BEGIN) | TOKEN_NONBREAK},
  {"</b>",  TS(STYLE_BOLD_END) | TOKEN_NONBREAK},
  {"<em>",  TS(STYLE_ITALIC_BEGIN) | TOKEN_NONBREAK},
  {"</em>", TS(STYLE_ITALIC_END) | TOKEN_NONBREAK},

  {"<h1>",  TS(STYLE_HEADING_BEGIN)},
  {"</h1>", TS(STYLE_HEADING_END)},
  {"<h2>",  TS(STYLE_HEADING_BEGIN)},
  {"</h2>", TS(STYLE_HEADING_END)},
  {"<h3>",  TS(STYLE_HEADING_BEGIN)},
  {"</h3>", TS(STYLE_HEADING_END)},
  {"<h4>",  TS(STYLE_HEADING_BEGIN)},
  {"</h4>", TS(STYLE_HEADING_END)},
  {"<h5>",  TS(STYLE_HEADING_BEGIN)},
  {"</h5>", TS(STYLE_HEADING_END)},

  {"<ul>",  TS(STYLE_ITEMIZE_BEGIN)},
  {"</ul>", TS(STYLE_ITEMIZE_END)},
  {"<ol>",  TS(STYLE_ITEMIZE_BEGIN)},
  {"</ol>", TS(STYLE_ITEMIZE_END)},
  {"<li>",  TS(STYLE_LISTITEM_BEGIN)},
  {"</li>", TS(STYLE_LISTITEM_END)},

  {"<blockquote>",  TS(STYLE_QUOTE_BEGIN)},
  {"</blockquote>", TS(STYLE_QUOTE_END)},

  {nullptr, 0}
};

#undef TS

static const char *kTokenSuffixes[] = {
  "'s", nullptr,
  "'m", nullptr,
  "'d", nullptr,
  "'ll", nullptr,
  "'re", nullptr,
  "'ve", nullptr,
  "n't", nullptr,
  "’s", "'s",
  "’m", "'m",
  "’d", "'d",
  "’ll", "'ll",
  "’re", "'re",
  "’ve", "'ve",
  "n’t", "n't",
  nullptr
};

static const char *kHyphenatedPrefixExceptions[] = {
  "e-", "a-", "u-", "x-", "anti-", "agro-", "be-", "bi-", "bio-", "co-",
  "counter-", "cross-", "cyber-", "de-", "eco-", "ex-", "extra-", "inter-",
  "intra-", "macro-", "mega-", "micro-", "mid-", "mini-", "multi-", "neo-",
  "non-", "over-", "pan-", "para-", "peri-", "post-", "pre-", "pro-",
  "pseudo-", "quasi-", "re-", "semi-", "sub-", "super-", "tri-", "ultra-",
  "un-", "uni-", "vice-",
  nullptr
};

static const char *kHyphenatedSuffixExceptions[] = {
  "-esque", "-fest", "-fold", "-gate", "-itis", "-less", "-most", "-rama",
  "-wise",
  nullptr
};

static const char *kHyphenationExceptions[] = {
  "mm-hm", "mm-mm", "o-kay", "uh-huh", "uh-oh",
  nullptr
};

static const char *kAbbreviations[] = {
  "a.", "abb.", "abg.", "abs.", "abt.", "ac.", "acad.", "acc.", "adm.",
  "admin.", "adopt.", "adr.", "ads.", "adv.", "af.", "ag.", "ala.", "alm.",
  "alt.", "amer.", "amex.", "ann.", "ans.", "ap.", "app.", "appl.", "approx.",
  "apr.", "apt.", "arch.", "ark.", "ariz.", "art.", "assoc.", "asst.", "atty.",
  "aufl.", "aug.", "auto.", "av.", "ave.", "avg.",

  "b.", "bc.", "bd.", "biochem.", "biol.", "bl.", "bldg.", "blvd.", "br.",
  "bros.", "bzw.",

  "c.", "cal.", "calif.", "capt.", "cds.", "ce.", "cert.", "cf.",
  "cfr.", "ch.", "chem.", "cir.", "circ.", "cit.", "cl.", "clin.", "cm.", "co.",
  "cod.", "col.", "coll.", "colo.", "comm.", "conf.", "conn.", "cons.", "cor.",
  "corp.", "cpl.", "cu.", "cz.",

  "d.", "dak.", "dc.", "dec.", "def.", "del.", "dem.", "den.", "dep.", "dept.",
  "dev.", "dez.", "dig.", "dipl.", "dir.", "dis.", "dist.", "div.", "dj.",
  "doc.", "dom.", "dott.", "dr.", "drs.", "ds.", "dt.", "durchg.", "dvs.",

  "e.", "ed.", "eds.", "eff.", "eg.", "ei.", "eks.", "el.", "em.", "en.",
  "ene.", "eng.", "engl.", "env.", "environ.", "eq.", "equiv.", "er.", "es.",
  "esp.", "esq.", "est.", "et.", "etc.", "ev.", "ex.", "exec.", "exp.",
  "ext.",

  "f.", "farbf.", "fax.", "feat.", "feb.", "fed.", "fest.", "ff.", "fides.",
  "fig.", "figs.", "fla.", "foy.", "fr.", "fri.", "ft.",

  "g.", "gal.", "gb.", "gen.", "gm.", "gmbh.", "gmt.", "gov.", "govt.",
  "gr.",

  "h.", "heb.", "hon.", "hp.", "hr.", "hrs.", "hum.", "hwy.",

  "i.", "ie.", "ii.", "iii.", "ill.", "im.", "inc.", "ind.", "inf.",
  "ing.", "inkl.", "inst.", "int.", "intl.", "ir.", "isa.", "iss.", "iv.",
  "ix.",

  "j.", "ja.",  "jan.",  "je.", "jg.", "jl.", "jr.", "jul.", "jun.",

  "k.", "kan.", "kans.", "kft.", "kg.", "kl.", "km.", "kr.", "kt.",

  "l.", "lab.",  "lap.", "lb.", "lbs.", "lett.", "lit.", "llc.", "lm.",
  "lo.", "loc.", "lt.", "ltd.",

  "m.", "mac.", "mag.", "maj.", "mar.", "mass.", "mat.", "math.",
  "matt.", "max.", "mb.", "med.", "mehr.", "mex.", "mfg.", "mg.", "mgr.",
  "mhz.", "mich.", "mil.", "mill.", "min.", "minn.", "mio.", "misc.", "miss.",
  "mix.",  "ml.", "mm.", "mod.", "mol.", "mont.", "mos.", "mr.", "mrs.", "ms.",

  "n.", "nac.", "nat.", "natl.", "nb.", "neb.", "nebr.", "neg.", "nev.", "no.",
  "nos.", "nov.", "np.", "nr.", "nt.", "nts.", "nucl.", "num.", "nutr.",

  "o.", "oct.", "okla.", "okt.", "om.", "ont.", "op.", "ord.", "oreg.", "os.",
  "oz.",

  "p.", "pag.", "par.", "para.", "pat.", "pcs.", "pct.", "pers.", "pg.",
  "pgs.", "ph.", "phil.", "php.", "phys.", "pic.", "plc.", "pol.", "pop.",
  "pos.", "pot.", "pp.", "pr.", "preg.", "pres.", "prev.", "priv.", "pro.",
  "proc.", "prof.", "prog.", "prov.", "ps.", "psa.", "pt.", "pub.", "publ.",
  "pvt.",

  "q.",

  "r.", "rd.", "re.", "rec.", "ref.", "reg.", "rel.", "rep.", "res.", "resp.",
  "rev.", "rm.", "rom.", "rs.", "rt.", "ru.", "rul.",

  "s.", "sa.",  "sat.", "sci.", "sec.", "sen.", "sens.", "sep.", "sept.",
  "ser.", "serv.", "sgt.", "sie.", "sig.", "sm.", "soc.", "sol.", "sp.", "spc.",
  "spec.", "sq.", "sr.", "ss.", "st.", "stat.", "std.", "ste.", "stk.", "str.",
  "sup.", "supp.",

  "t.", "tech.", "tel.", "temp.", "tenn.", "th.", "tim.", "tip.", "tj.", "tlf.",
  "tlg.", "tr.", "trans.", "treas.", "tsp.",

  "u.", "ud.", "ul.", "um.", "univ.", "ust.", "uu.",

  "v.", "var.", "ver.", "vert.", "vg.", "vgl.", "vii.", "viii.", "vol.",
  "vols.", "vor.", "vs.",

  "w.", "wis.", "wm.", "wyo.",

  "x.", "xi.", "xii.",

  "y.", "yr.", "yrs.",

  "z.", "zt.", "zu.", "zzgl.",

  // Weekdays.
  "mon.", "tue.", "wed.", "thu.", "fri.", "sat.", "sun.",

  // Compound abbreviations.
  "a.c.", "a.d.", "a.k.a.", "a.m.", "b.sc.", "c.e.", "cont'd.", "d.c.", "d.sc.",
  "dr.sc.", "e.g.", "f.a.o.", "g.m.b.h.", "i.b.m.", "i.e.", "l.a.", "m.a.",
  "m.b.a.", "m.d.", "m.sc.", "n.y.", "ph.d.", "p.m.", "p.r.", "u.k.", "u.n.",
  "u.s.a.", "u.s.s.r.", "u.s.",

  // Special words.
  "c++", "yahoo!", ".net", "google+",

  nullptr
};

StandardTokenization::StandardTokenization() {
  token_types_ = new TrieNode();
  suffix_types_ = new TrieNode();
}

StandardTokenization::~StandardTokenization() {
  delete token_types_;
  delete suffix_types_;
}

TrieNode *StandardTokenization::AddTokenType(const char *token,
                                             TokenFlags flags,
                                             const char *value) {
  TrieNode *node = token_types_;
  const char *p = token;
  const char *end = token + strlen(token);
  while (p < end) {
    int code = UTF8::Decode(p, end - p);
    node = node->AddChild(code);
    p = UTF8::Next(p);
  }

  if (value != nullptr) node->set_value(value);
  node->set_flags(flags);
  node->set_terminal(true);

  return node;
}

TrieNode *StandardTokenization::AddSuffixType(const char *token,
                                              const char *value) {
  std::vector<char32> ustr;
  const char *p = token;
  const char *end = token + strlen(token);
  while (p < end) {
    int code = UTF8::Decode(p, end - p);
    ustr.push_back(code);
    p = UTF8::Next(p);
  }

  TrieNode *node = suffix_types_;
  for (int i = ustr.size() - 1; i >= 0; --i) {
    node = node->AddChild(ustr[i]);
  }

  if (value != nullptr) node->set_value(value);
  node->set_terminal(true);

  return node;
}

void StandardTokenization::Init(CharacterFlags *char_flags) {
  // Setup character classifications.
  for (int c = 0; c < kMaxAscii; ++c) {
    TokenFlags flags = 0;
    if (ascii_isspace(c)) flags |= CHAR_SPACE;
    if (ascii_isalpha(c)) flags |= CHAR_LETTER;
    if (ascii_isupper(c)) flags |= CHAR_UPPER;
    if (ascii_ispunct(c)) flags |= CHAR_PUNCT;
    if (ascii_isdigit(c)) flags |= CHAR_DIGIT;
    char_flags->add(c, flags);
  }

  char_flags->add('\'', WORD_PUNCT);
  char_flags->add('+', NUMBER_START);
  char_flags->add(',', NUMBER_PUNCT);
  char_flags->add('.', NUMBER_START | NUMBER_PUNCT | WORD_PUNCT);
  char_flags->add('/', NUMBER_PUNCT);
  char_flags->add(':', NUMBER_PUNCT);
  char_flags->add('&', WORD_PUNCT);
  char_flags->add('<', TAG_START);
  char_flags->add('>', TAG_END);
  char_flags->add('@', WORD_PUNCT);
  char_flags->add('_', CHAR_LETTER);
  char_flags->add('`', WORD_PUNCT);
  char_flags->add(0x2019, WORD_PUNCT);  // ’ (single quote).
  char_flags->add('@', HASHTAG_START);  // for @handle
  char_flags->add('#', HASHTAG_START);  // for #tags
  char_flags->add(0x200B, TOKEN_DISCARD);  // zero-width space

  // Space tokens.
  AddTokenType(" ", TOKEN_DISCARD);
  AddTokenType("\xc2\xa0", TOKEN_DISCARD, " ");  // non-breaking space
  AddTokenType("\t", TOKEN_EOS | TOKEN_PARA | TOKEN_DISCARD);
  AddTokenType("\r", TOKEN_LINE | TOKEN_DISCARD);
  AddTokenType("\n", TOKEN_LINE | TOKEN_DISCARD);
  AddTokenType("\n\n", TOKEN_EOS | TOKEN_PARA | TOKEN_DISCARD);
  AddTokenType("\n\r\n", TOKEN_EOS | TOKEN_PARA | TOKEN_DISCARD);

  // Synthetic tags for section and chapter breaks.
  AddTokenType("<section>", TOKEN_EOS | TOKEN_PARA | TOKEN_DISCARD | 1);
  AddTokenType("<chapter>", TOKEN_EOS | TOKEN_PARA | TOKEN_DISCARD | 2);

  // Punctuation tokens.
  AddTokenType("@", 0);
  AddTokenType("#", 0);
  AddTokenType("&", 0);
  AddTokenType("$", 0);
  AddTokenType("%", 0);
  AddTokenType("/", 0);
  AddTokenType(".", TOKEN_EOS);
  AddTokenType(",", 0);
  AddTokenType("!", TOKEN_CONDEOS);
  AddTokenType("?", TOKEN_CONDEOS);
  AddTokenType(";", 0);
  AddTokenType(":", 0);
  AddTokenType("|", TOKEN_EOS | TOKEN_DISCARD);
  AddTokenType(" * ", TOKEN_EOS | TOKEN_PARA | TOKEN_DISCARD);  // ASCII bullet
  AddTokenType("·", TOKEN_EOS | TOKEN_PARA | TOKEN_DISCARD);  // middle dot
  AddTokenType("...", TOKEN_CONDEOS);
  AddTokenType("…", TOKEN_CONDEOS, "...");
  AddTokenType("&", 0, "&");
  AddTokenType(". . .", TOKEN_CONDEOS, "...");
  AddTokenType("--", 0);
  AddTokenType("---", 0, "--");
  AddTokenType("‒", 0, "--");  // U+2012 figure dash
  AddTokenType("–", 0, "--");  // U+2013 en dash
  AddTokenType("—", 0, "--");  // U+2014 em dash
  AddTokenType("−", 0, "--");  // U+2212 minus sign
  AddTokenType("\"", TOKEN_QUOTE);
  AddTokenType("＂", TOKEN_QUOTE);
  AddTokenType("，", 0, ",");
  AddTokenType("．", TOKEN_EOS, ".");
  AddTokenType("！", TOKEN_EOS, "!");
  AddTokenType("？", TOKEN_EOS, "?");
  AddTokenType("：", TOKEN_CONDEOS, ":");
  AddTokenType("；", TOKEN_CONDEOS, ";");
  AddTokenType("＆", 0, "&");

  // Bracketing tokens.
  AddTokenType("(", TOKEN_OPEN);
  AddTokenType(")", TOKEN_CLOSE);

  AddTokenType("[", TOKEN_OPEN);
  AddTokenType("]", TOKEN_CLOSE);

  AddTokenType("{", TOKEN_OPEN);
  AddTokenType("}", TOKEN_CLOSE);

  AddTokenType("``", TOKEN_OPEN);
  AddTokenType("''", TOKEN_CLOSE);

  AddTokenType("„", TOKEN_OPEN, "``");
  AddTokenType("”", TOKEN_CLOSE, "''");
  AddTokenType("“", TOKEN_CLOSE, "''");

  AddTokenType("‘", TOKEN_OPEN, "``");
  AddTokenType("‚", TOKEN_OPEN, "``");
  AddTokenType("’", TOKEN_CLOSE, "''");

  AddTokenType("“", TOKEN_OPEN, "``");
  AddTokenType("”", TOKEN_CLOSE, "''");

  AddTokenType("»", TOKEN_OPEN, "``");
  AddTokenType("«", TOKEN_CLOSE, "''");

  AddTokenType("›", TOKEN_OPEN, "``");
  AddTokenType("‹", TOKEN_CLOSE, "''");

  // URL tokens.
  TokenFlags url_flags = TOKEN_URL;
  if (discard_urls_) url_flags |= TOKEN_DISCARD;
  AddTokenType("http:", url_flags);
  AddTokenType("https:", url_flags);
  AddTokenType("ftp:", url_flags);
  AddTokenType("mailto:", url_flags);
  AddTokenType("www.", url_flags);

  // Emoticon tokens.
  AddTokenType(":-)", TOKEN_CONDEOS);
  AddTokenType(":-(", TOKEN_CONDEOS);
  AddTokenType(";-)", TOKEN_CONDEOS);
  AddTokenType(":-/", TOKEN_CONDEOS);
  AddTokenType(":-D", TOKEN_CONDEOS);
  AddTokenType(":-O", TOKEN_CONDEOS);
  AddTokenType(":-|", TOKEN_CONDEOS);

  // Words that should be split into two tokens.
  AddTokenType("cannot", TOKEN_WORD | TOKEN_SPLIT | 3);
  AddTokenType("d'ye", TOKEN_WORD | TOKEN_SPLIT | 2);
  AddTokenType("gimme", TOKEN_WORD | TOKEN_SPLIT | 3);
  AddTokenType("gonna", TOKEN_WORD | TOKEN_SPLIT | 3);
  AddTokenType("gotta", TOKEN_WORD | TOKEN_SPLIT | 3);
  AddTokenType("lemme", TOKEN_WORD | TOKEN_SPLIT | 3);
  AddTokenType("more'n", TOKEN_WORD | TOKEN_SPLIT | 4);
  AddTokenType("'tis", TOKEN_WORD | TOKEN_SPLIT | 2);
  AddTokenType("'twas", TOKEN_WORD | TOKEN_SPLIT | 2);
  AddTokenType("wanna", TOKEN_WORD | TOKEN_SPLIT | 3);

  // Clitics splitting.
  AddTokenType("d'", 0);
  AddTokenType("l'", 0);
  AddTokenType("all'", 0);
  AddTokenType("dell'", 0);
  AddTokenType("dall'", 0);
  AddTokenType("nell'", 0);
  AddTokenType("sull'", 0);

  // Breaking tag tokens.
  const char **tag = kBreakingTags;
  while (*tag) {
    // Add both "<tag>" and "<tag " as special tokens.
    string tag_closed;
    tag_closed.append("<");
    tag_closed.append(*tag);
    tag_closed.append(">");

    string tag_open;
    tag_open.append("<");
    tag_open.append(*tag);
    tag_open.append(" ");

    AddTokenType(tag_closed.c_str(),
                 TOKEN_EOS | TOKEN_PARA | TOKEN_DISCARD | TOKEN_TAG);
    AddTokenType(tag_open.c_str(),
                 TOKEN_EOS | TOKEN_PARA | TOKEN_DISCARD | TOKEN_TAG);
    tag++;
  }

  // Styling tags.
  const auto *style = kStyleTags;
  while (style->tag != nullptr) {
    AddTokenType(style->tag, TOKEN_TAG | TOKEN_DISCARD | style->flags);
    style++;
  }

  // Abbreviations.
  const char **abbrev = kAbbreviations;
  while (*abbrev) {
    AddTokenType(*abbrev, TOKEN_WORD);
    abbrev++;
  }

  // Suffixes that should be separate tokens. These are also added as special
  // token types to allow matching these as standalone tokens.
  const char **suffixes = kTokenSuffixes;
  while (*suffixes) {
    const char *suffix = *suffixes++;
    const char *replacement = *suffixes++;
    AddSuffixType(suffix, replacement);
    AddTokenType(suffix, TOKEN_WORD, replacement);
  }
}

void StandardTokenization::Process(TokenizerText *t) {
  int i = 0;
  while (i < t->length()) {
    // Check for number tokens. Numbers start with a digit or a number
    // punctuation start character (like . , + -) followed by a digit.
    // The rest of the number token consists of digits, letters, and number
    // punctuation characters (like . , / -). The number punctuation character
    // must be preceded by a digit or letter and cannot be the last character
    // in the token. Notice that it is important to check for number tokens
    // before special tokens because '.' can be both a decimal point and a
    // period.
    if (t->is(i, CHAR_DIGIT) ||
        (t->is(i, NUMBER_START) &&
         !t->is(i, CHAR_HYPHEN) &&
         t->is(i + 1, CHAR_DIGIT))) {
      bool prev_was_punct = !t->is(i, CHAR_DIGIT);
      int j = i + 1;
      while (j < t->length()) {
        if (t->is(j, CHAR_DIGIT | CHAR_LETTER)) {
          prev_was_punct = false;
          j++;
        } else if (t->is(j, NUMBER_PUNCT)) {
          if (prev_was_punct) break;
          prev_was_punct = true;
          j++;
        } else {
          break;
        }
      }

      // A number cannot end with a punctuation character.
      if (t->is(j - 1, NUMBER_PUNCT)) j--;

      // If the next character is a dash it is marked as a hyphen to prevent
      // it from being considered as a number sign.
      if (t->at(j) == '-') t->set(j, CHAR_HYPHEN);

      // Mark number token.
      t->set(i, TOKEN_START);
      i = j;
      continue;
    }

    // Check for special tokens. The text is matched against the token trie and
    // the longest match is found.
    TokenFlags tag_flags = 0;
    int length;
    const TrieNode *node = token_types_->FindMatch(*t, i, &length);
    if (length > 0) {
      int j = i + length;
      bool match = true;

      // If token is an URL, match rest of URL.
      if (node->is(TOKEN_URL)) {
        // Move forward until space character found.
        while (j < t->length() && !t->is(j, CHAR_SPACE)) j++;

        // The URL cannot end with punctuation characters.
        while (j > i + 1 && t->is(j - 1, CHAR_PUNCT)) j--;
      }

      // Word-like tokens cannot be followed by letters or digits.
      if (node->is(TOKEN_WORD)) {
        match = !t->is(j, CHAR_LETTER | CHAR_DIGIT);
      }

      // If token is a tag, it is handled below.
      if (node->is(TOKEN_TAG)) {
        tag_flags = node->flags();
        match = false;
      }

      if (match) {
        if (node->is(TOKEN_PREFIX)) {
          // This is a LDC hyphenated prefix that should not break on the
          // hyphen. Set the suffix flag for the next character to suppress
          // token breaks for the next word. Ignore the prefix if there is a
          // hyphen before it.
          if (i == 0 || !t->is(i - 1, CHAR_HYPHEN)) {
            t->set(i, TOKEN_START);
            t->set(j, TOKEN_SUFFIX);
            i = j;
            continue;
          }
        } else if (node->is(TOKEN_SUFFIX)) {
          // This is a LDC hyphenated suffix that should not break on the
          // hyphen. This cannot be followed by dash, letters or digits.
          if (t->is(i, CHAR_HYPHEN) &&
              !t->is(j, CHAR_DIGIT | CHAR_LETTER) &&
              t->at(j) != '-') {
            i = j;
            continue;
          }
        } else {
          // Mark start of token.
          t->set(i, TOKEN_START | node->flags());
          t->set_node(i, node);

          // Split token if needed.
          if (node->is(TOKEN_SPLIT)) {
            int split_pos = i + (node->flags() & TOKEN_PARAM_MASK);
            if (split_pos < t->length()) t->set(split_pos, TOKEN_START);
          }

          // Move to first position after token.
          i = j;
          continue;
        }
      }
    }

    // Check for word-like tokens. A words starts with a letter followed by
    // letters, digits, and word punctuation (like - '). Word punctuation must
    // be preceded by a letter or digit. The word token is then checked for
    // special suffixes that are marked as separate tokens.
    if (t->is(i, CHAR_LETTER)) {
      int j = i + 1;
      bool prev_was_punct = false;
      while (j < t->length()) {
        if (t->is(j, CHAR_LETTER) || t->is(j, CHAR_DIGIT)) {
          prev_was_punct = false;
          j++;
        } else if (t->is(j, WORD_PUNCT)) {
          if (prev_was_punct) break;
          prev_was_punct = true;
          j++;
        } else {
          break;
        }
      }

      // A word cannot end with a punctuation character, except for initials
      // like in "J.K. Rowling".
      if (t->is(j - 1, WORD_PUNCT)) {
        if (j < 3 ||
            t->at(j - 1) != '.' ||
            t->at(j - 3) != '.' ||
            !t->is(j - 2, CHAR_UPPER)) {
          j--;
        }
      }

      // If there is a period followed by a comma the period is considered part
      // of the word, i.e. abbreviation.
      if (t->at(j) == '.' && t->at(j + 1) == ',') j++;

      // Check for special suffix.
      int suffix_length;
      const TrieNode *suffix;
      suffix = suffix_types_->FindReverseMatch(*t, j - 1, i, &suffix_length);
      if (suffix_length != 0) {
        // Mark suffix as separate token.
        int suffix_start = j - suffix_length;
        t->set(suffix_start, TOKEN_START);
        t->set(suffix_start, suffix->flags());
        t->set_node(suffix_start, suffix);
      }

      // If the next character is a dash it is considered a hyphen.
      if (t->at(j) == '-') t->set(j, CHAR_HYPHEN);

      // Mark start of word token unless this was preceded by a prefix.
      if (!t->is(i, TOKEN_SUFFIX)) t->set(i, TOKEN_START);
      i = j;
      continue;
    }

    // Check for hash #tag or @handle.
    if (t->is(i, HASHTAG_START)) {
      int j = i + 1;
      bool has_letter = false;
      while (j < t->length()) {
        if (t->is(j, CHAR_LETTER)) has_letter = true;

        if (t->is(j, CHAR_DIGIT | CHAR_LETTER)) {
          j++;
        } else {
          break;
        }
      }

      // Mark #tag or @handle token.
      // A #tag and @handle must have at least one letter.
      if (has_letter) {
        t->set(i, TOKEN_START);
        i = j;
        continue;
      }
    }

    // Remove tags. The tag token is marked as a discarded token. The tags
    // cannot exceed the maximum configured length.
    if (t->is(i, TAG_START)) {
      int j = i + 1;
      while (j < t->length()) {
        if (t->is(j, TAG_END)) break;
        if (j - i - 1 == max_tag_token_length_) break;
        j++;
      }

      if (t->is(j, TAG_END)) {
        // Tag found. Mark it as a discarded token.
        t->set(i, TOKEN_START | TOKEN_DISCARD | tag_flags);
        i = j + 1;
        continue;
      }
    }

    // Discard other whitespace characters.
    if (t->is(i, CHAR_SPACE)) {
      t->set(i, TOKEN_START | TOKEN_DISCARD);
      i++;
      continue;
    }

    // Unknown token, mark next character as a separate token.
    t->set(i, TOKEN_START);
    i++;
  }
}

void PTBTokenization::Init(CharacterFlags *char_flags) {
  StandardTokenization::Init(char_flags);

  // Allow hyphens in numbers and words.
  char_flags->add('-', NUMBER_START | NUMBER_PUNCT | WORD_PUNCT);
}

void LDCTokenization::Init(CharacterFlags *char_flags) {
  StandardTokenization::Init(char_flags);

  // Allow dash to start a negative number (i.e. dash as a sign).
  char_flags->add('-', NUMBER_START);

  // Non-breaking space.
  char_flags->add(0x200B, TOKEN_DISCARD | TOKEN_NONBREAK);

  // Exceptions for prefixes with hyphens.
  const char **exception = kHyphenatedPrefixExceptions;
  while (*exception) {
    AddTokenType(*exception, TOKEN_PREFIX);
    exception++;
  }

  // Exceptions for suffixes with hyphens.
  exception = kHyphenatedSuffixExceptions;
  while (*exception) {
    AddTokenType(*exception, TOKEN_SUFFIX);
    exception++;
  }

  // Exceptions for hyphenated words.
  exception = kHyphenationExceptions;
  while (*exception) {
    AddTokenType(*exception, TOKEN_WORD);
    exception++;
  }

  // Special LDC tokens.
  AddTokenType("(...)", 0);
}

}  // namespace nlp
}  // namespace sling

