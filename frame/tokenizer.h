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

#ifndef FRAME_TOKENIZER_H_
#define FRAME_TOKENIZER_H_

#include <string>

#include "base/macros.h"
#include "stream/input.h"

namespace sling {

class Tokenizer {
 public:
  // Token types in the range 0-255 are used for single-character tokens.
  enum TokenType {
    ERROR = 256,
    END,

    // Literal types.
    STRING_TOKEN,
    INTEGER_TOKEN,
    FLOAT_TOKEN,
    SYMBOL_TOKEN,
    LITERAL_TOKEN,
    NUMERIC_TOKEN,
    INDEX_TOKEN,
    INDEX_REF_TOKEN,
    CHARACTER_TOKEN,

    // Multi-character operators.
    LTE_TOKEN,             // <=
    GTE_TOKEN,             // >=
    EQ_TOKEN,              // ==
    NE_TOKEN,              // !=
    EQ_STRICT_TOKEN,       // ===
    NE_STRICT_TOKEN,       // !==
    INC_TOKEN,             // ++
    DEC_TOKEN,             // --
    SHL_TOKEN,             // <<
    SAR_TOKEN,             // >>
    SHR_TOKEN,             // >>>
    AND_TOKEN,             // &&
    OR_TOKEN,              // ||
    ASSIGN_ADD_TOKEN,      // +=
    ASSIGN_SUB_TOKEN,      // -=
    ASSIGN_MUL_TOKEN,      // *=
    ASSIGN_MOD_TOKEN,      // %=
    ASSIGN_SHL_TOKEN,      // <<=
    ASSIGN_SAR_TOKEN,      // >>=
    ASSIGN_SHR_TOKEN,      // >>>=
    ASSIGN_BIT_AND_TOKEN,  // &=
    ASSIGN_BIT_OR_TOKEN,   // |=
    ASSIGN_BIT_XOR_TOKEN,  // ^=
    ASSIGN_DIV_TOKEN,      // /=

    // Reserved keywords.
    CONST_TOKEN,
    ELSE_TOKEN,
    FOR_TOKEN,
    FUNC_TOKEN,
    IF_TOKEN,
    IN_TOKEN,
    ISA_TOKEN,
    PRIVATE_TOKEN,
    RETURN_TOKEN,
    SELF_TOKEN,
    STATIC_TOKEN,
    THIS_TOKEN,
    VAR_TOKEN,
    WHILE_TOKEN,

    // Literal keywords.
    NULL_TOKEN,
    TRUE_TOKEN,
    FALSE_TOKEN,
  };

  // Initializes tokenizer with input.
  explicit Tokenizer(Input *input);

  // Checks if all input has been read.
  bool done() const { return token_ == END; }

  // Returns true if errors were found while parsing input.
  bool error() const { return token_ == ERROR; }

  // Returns current line and column.
  int line() const { return line_; }
  int column() const { return column_; }

  // Returns last error message.
  const string &error_message() const { return error_message_; }

  // Reads the next input token.
  int NextToken();

  // Returns current input token.
  int token() const { return token_; }

  // Returns token token.
  const string &token_text() const { return token_text_; }

  // Records error at current input position.
  void SetError(const string &error_message);

  // Records error and returns the ERROR token.
  int Error(const string &error_message);

  // Return error message with position information.
  string GetErrorMessage(const string &filename) const;

  // Enables and disables function parsing mode.
  bool func_mode() const { return func_mode_; }
  void set_func_mode(bool func_mode) { func_mode_ =  func_mode; }

 private:
  // Gets the next input character.
  void NextChar();

  // Sets current token and returns it.
  int Token(int token) { token_ = token; return token; }

  // Consumes current input character and returns token.
  int Select(int token) { NextChar(); return Token(token); }

  // Consumes current input character and return tokens dependent on the next
  // input character.
  int Select(char next, int then, int otherwise);

  // Parses string from input.
  int ParseString();

  // Parses number from input.
  int ParseNumber(bool negative, bool fractional);

  // Parses a sequence of digits. Returns the number of digits parsed.
  int ParseDigits();

  // Parses a sequence of Unicode hex characters and appends these as UTF-8 to
  // the token buffer. Returns false on error.
  bool ParseUnicode(int digits);

  // Parses symbol name from input. Returns false if symbol is invalid.
  bool ParseName(int first);

  // Looks up keyword for the token in the token buffer. If this matches a
  // reserved keyword, it returns the keyword token number. Otherwise it is
  // treated as a symbol token.
  int LookupKeyword();

  // Adds character to token buffer.
  void Append(char ch) { token_text_.push_back(ch); }

  // Input for reader.
  Input *input_;

  // Current input character or -1 if end of input has been reached.
  int current_;

  // Current position in input.
  int line_;
  int column_;

  // Last read token type. This is either a single-character token or one of the
  // values from the TokenType enumeration.
  int token_;

  // Text for last read token. This contains string, symbol, and number tokens.
  string token_text_;

  // Last error message.
  string error_message_;

  // In function parsing mode, the tokenization is slightly different to allow
  // for the richer syntax:
  //  - Single quote is used for character constants instead of literal symbols.
  //  - Symbol names can only contain alpha-numeric characters and underscores.
  //  - Semicolon cannot be used for comments.
  //  - Number signs are separate tokens.
  //  - Reserved keywords have separate token types.
  bool func_mode_ = false;

  DISALLOW_IMPLICIT_CONSTRUCTORS(Tokenizer);
};

}  // namespace sling

#endif  // FRAME_TOKENIZER_H_

