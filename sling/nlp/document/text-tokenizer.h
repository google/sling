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

#ifndef SLING_NLP_DOCUMENT_TEXT_TOKENIZER_H_
#define SLING_NLP_DOCUMENT_TEXT_TOKENIZER_H_

#include <functional>
#include <string>
#include <unordered_map>
#include <vector>

#include "sling/base/macros.h"
#include "sling/base/types.h"
#include "sling/nlp/document/token-breaks.h"
#include "sling/string/text.h"
#include "sling/util/unicode.h"

namespace sling {
namespace nlp {

class TrieNode;

// Flags for tokens.
enum TokenFlagValues {
  // The four lowest bits are reserved for a numerical parameter.
  TOKEN_PARAM_MASK = (1 << 4) - 1,

  // Token level flags.
  TOKEN_LINE     = (1 << 4),   // End-of-line token.
  TOKEN_EOS      = (1 << 5),   // End-of-sentence token.
  TOKEN_PARA     = (1 << 6),   // End-of-paragraph token.
  TOKEN_DISCARD  = (1 << 7),   // Discardable token (like whitespace).
  TOKEN_CONDEOS  = (1 << 8),   // Conditional end-of-sentence token.
  TOKEN_QUOTE    = (1 << 9),   // Quote token.
  TOKEN_OPEN     = (1 << 10),  // Opening bracket token.
  TOKEN_CLOSE    = (1 << 11),  // Closing bracket token.
  TOKEN_URL      = (1 << 12),  // URL-like token.
  TOKEN_TAG      = (1 << 13),  // Tag-like token.
  TOKEN_WORD     = (1 << 14),  // Word-like token.
  TOKEN_SPLIT    = (1 << 15),  // Split token.
  TOKEN_PREFIX   = (1 << 16),  // Prefix exception.
  TOKEN_SUFFIX   = (1 << 17),  // Suffix exception.

  // Character level flags.
  TOKEN_START    = (1 << 18),  // Start of token marker.
  CHAR_LETTER    = (1 << 19),  // Letter character.
  CHAR_DIGIT     = (1 << 20),  // Digit character.
  CHAR_UPPER     = (1 << 21),  // Uppercase letter.
  CHAR_SPACE     = (1 << 22),  // Whitespace character.
  CHAR_PUNCT     = (1 << 23),  // Punctuation character.
  CHAR_HYPHEN    = (1 << 24),  // Hyphen dash character.
  NUMBER_START   = (1 << 25),  // Character valid as first character in number.
  NUMBER_PUNCT   = (1 << 26),  // Character allowed inside number.
  WORD_PUNCT     = (1 << 27),  // Character allowed inside word.
  TAG_START      = (1 << 28),  // Tag start character.
  TAG_END        = (1 << 29),  // Tag end character.
  HASHTAG_START  = (1 << 30),  // Character indicating start of hash-tag.
};

typedef int32 TokenFlags;

// Mapping from characters values to character flags. The lookup table is split
// in two. The low ASCII characters (0-127) are mapped through an array and the
// rest of the characters are mapped though a hash table.
class CharacterFlags {
 public:
  CharacterFlags();

  // Sets the flags for a character value.
  void set(char32 ch, TokenFlags flags);

  // Adds the flags for a character value.
  void add(char32 ch, TokenFlags flags) { set(ch, get(ch) | flags); }

  // Clears the flags for a character value.
  void clear(char32 ch, TokenFlags flags) { set(ch, get(ch) & ~flags); }

  // Returns the flags for a character value.
  TokenFlags get(char32 ch) const;

 private:
  std::vector<TokenFlags> low_flags_;
  std::unordered_map<char32, TokenFlags> high_flags_;
};

// Unicode representation of text with extra information about each Unicode
// character. Each character has a set of token/character flags and an optional
// reference to the token from the trie that matches the token. The text has
// a nul-termination element.
class TokenizerText {
 public:
  // Initializes elements from a text string.
  TokenizerText(Text text, const CharacterFlags &char_flags);

  // Returns a substring of the text in UTF-8 encoded format.
  void GetText(int start, int end, string *result) const;

  // Returns the next element that starts a new token.
  int NextStart(int index) const {
    while (index < length_ && !is(index, TOKEN_START)) index++;
    return index;
  }

  // Returns the break level for a token.
  BreakType BreakLevel(int index) const;

  // Returns the number of characters in the text.
  int length() const { return length_; }

  // Returns true if an element has a flag set.
  bool is(int index, TokenFlags flags) const {
    return (elements_[index].flags & flags) != 0;
  }

  // Sets a flag for an element.
  void set(int index, TokenFlags flags) { elements_[index].flags |= flags; }

  // Returns the Unicode character at some position in the text.
  char32 at(int index) const { return elements_[index].ch; }

  // Returns character at some position in the text in lowercase.
  char32 lower(int index) const {
    return Unicode::ToLower(at(index));
  }

  // Sets/gets the token node for an element.
  const TrieNode *node(int index) const { return elements_[index].node; }
  void set_node(int index, const TrieNode *node) {
    elements_[index].node = node;
  }

  // Returns the position of a character in the source text.
  int position(int index) const { return elements_[index].position; }

 private:
  // Per-character information for text.
  struct Element {
    // Unicode character.
    char32 ch;

    // Position of character in source text.
    int position;

    // Token and character flags.
    TokenFlags flags;

    // Token node reference.
    const TrieNode *node;

    // Count of escaped entities so far in the text. This is used for quickly
    // determining if a range in the text contains any escaped entities.
    int escapes;
  };

  // Source text.
  Text source_;

  // Length of text (excluding the nul-termination).
  int length_;

  // One element for each character in the text (plus nul-termination).
  std::vector<Element> elements_;
};

// Tokenization processor.
class TokenProcessor {
 public:
  virtual ~TokenProcessor() = default;
  virtual void Init(CharacterFlags *char_flags) = 0;
  virtual void Process(TokenizerText *t) = 0;
};

// Tokenizer for breaking text into tokens and sentences.
class Tokenizer {
 public:
  // Tokens generated by tokenizer.
  struct Token {
    string text;
    BreakType brk;
    int begin;
    int end;
  };

  // Callback for collecting the generated tokens.
  typedef std::function<void(const Token &token)> Callback;

  Tokenizer();
  ~Tokenizer();

  // Initializes PTB tokenizer.
  void InitPTB();

  // Initializes LDC tokenizer.
  void InitLDC();

  // Adds tokenization processor to tokenizer. The tokenizer takes ownership
  // of the tokenization processor.
  void Add(TokenProcessor *processor);

  // Tokenizes text into sentences with tokens.
  void Tokenize(Text text, const Callback &callback) const;

  // Sets/clears character classification flags for character.
  void SetCharacterFlags(char32 ch, TokenFlags flags);
  void ClearCharacterFlags(char32 ch, TokenFlags flags);

 private:
  // Tokenization processors.
  std::vector<TokenProcessor *> processors_;

  // Character classification table.
  CharacterFlags char_flags_;

  DISALLOW_COPY_AND_ASSIGN(Tokenizer);
};

// Standard tokenization.
class StandardTokenization : public TokenProcessor {
 public:
  StandardTokenization();
  ~StandardTokenization() override;

  // Initialize common tokenizer.
  void Init(CharacterFlags *char_flags) override;

  // Adds new token type. The (optional) value is a replacement value for the
  // token.
  TrieNode *AddTokenType(const char *token, TokenFlags flags,
                         const char *value);
  TrieNode *AddTokenType(const char *token, TokenFlags flags) {
    return AddTokenType(token, flags, nullptr);
  }

  // Adds suffix type.
  TrieNode *AddSuffixType(const char *token, const char *value);

  // Break text into tokens.
  void Process(TokenizerText *t) override;

 protected:
  // Trie with special token types.
  TrieNode *token_types_;

  // Trie with special word suffixes that are considered separate tokens. The
  // suffixes are encoded in reverse order.
  TrieNode *suffix_types_;

  // Maximum length of a tag token, e.g. token of the form <...>.
  int max_tag_token_length_ = 20;

  // Discard URL-like tokens.
  bool discard_urls_ = true;
};

// Classic PTB (Penn Treebank) tokenization, which does not split on hyphens.
class PTBTokenization : public StandardTokenization {
 public:
  // Initialize PTB tokenization.
  void Init(CharacterFlags *char_flags) override;
};

// LDC (Linguistic Data Consortium) tokenization, which splits on hyphens,
// except for special prefixes and suffixes.
class LDCTokenization : public StandardTokenization {
 public:
  // Initialize LDC tokenization.
  void Init(CharacterFlags *char_flags) override;
};

}  // namespace nlp
}  // namespace sling

#endif  // SLING_NLP_DOCUMENT_TEXT_TOKENIZER_H_

