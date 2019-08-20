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

#ifndef SLING_FRAME_TOKENIZER_H_
#define SLING_FRAME_TOKENIZER_H_

#include <string>

#include "sling/stream/input.h"
#include "sling/frame/scanner.h"

namespace sling {

class Tokenizer : public Scanner {
 public:
  // Token types.
  enum TokenType {
    // Literal types.
    STRING_TOKEN = FIRST_AVAILABLE_TOKEN_TYPE,
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

  // Reads the next input token.
  int NextToken();

  // Enables and disables function parsing mode.
  bool func_mode() const { return func_mode_; }
  void set_func_mode(bool func_mode) { func_mode_ =  func_mode; }

 private:
  // Parses string from input.
  int ParseString();

  // Parses number from input.
  int ParseNumber(bool negative, bool fractional);

  // Parses symbol name from input. Returns false if symbol is invalid.
  bool ParseName(int first);

  // Looks up keyword for the token in the token buffer. If this matches a
  // reserved keyword, it returns the keyword token number. Otherwise it is
  // treated as a symbol token.
  int LookupKeyword();

  // In function parsing mode, the tokenization is slightly different to allow
  // for the richer syntax:
  //  - Single quote is used for character constants instead of literal symbols.
  //  - Symbol names can only contain alpha-numeric characters and underscores.
  //  - Semicolon cannot be used for comments.
  //  - Number signs are separate tokens.
  //  - Reserved keywords have separate token types.
  bool func_mode_ = false;
};

}  // namespace sling

#endif  // SLING_FRAME_TOKENIZER_H_

