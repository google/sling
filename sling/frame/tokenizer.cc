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

#include "sling/frame/tokenizer.h"

#include <string>

#include "sling/string/ctype.h"
#include "sling/string/numbers.h"
#include "sling/string/strcat.h"

namespace sling {

Tokenizer::Tokenizer(Input *input) : Scanner(input) {
  NextToken();
}

int Tokenizer::NextToken() {
  // Clear token text buffer.
  token_text_.clear();

  // Keep reading until we either read a token or reach the end of the input.
  for (;;) {
    // Skip whitespace.
    while (current_ != -1 && ascii_isspace(current_)) NextChar();

    // Parse next token (or comment).
    switch (current_) {
      case -1:  // end of input
        return Token(END);

      case '"':  // string
        return ParseString();

      case '0': case '1': case '2': case '3': case '4':
      case '5': case '6': case '7': case '8': case '9':  // number
        return ParseNumber(false, false);

      case '.':  // fractional number or dot
        NextChar();
        if (ascii_isdigit(current_)) return ParseNumber(false, true);
        return Token('.');

      case '#':  // numeric symbol name
        NextChar();
        if (current_ == '@') {
          NextChar();
          if (ParseDigits() > 0) return Token(INDEX_REF_TOKEN);
          return Error("Invalid index reference");
        } else if (ParseDigits() > 0) {
          return Token(NUMERIC_TOKEN);
        }
        return Token('#');

      case '@':  // index number
        NextChar();
        if (ParseDigits() > 0) return Token(INDEX_TOKEN);
        return Token('@');

      case ';':  // comment or semicolon
        NextChar();
        if (!func_mode_) {
          // Parse single-line comment starting with ";".
          while (current_ != -1 && current_ != '\n') NextChar();
          continue;
        }
        return Token(';');

      case '\\':  // escaped symbol name
        NextChar();
        if (!ParseName('\\')) return Error("Invalid symbol name");
        return Token(SYMBOL_TOKEN);

      case '{': case '}':  // frame delimiter
      case '[': case ']':  // array delimiter
      case '(': case ')':  // parentheses
      case ',': case '~': case '?': case ':':  // operator: , ~ ? :
        return Select(current_);

      case '\'':  // literal symbol or character constant
        if (func_mode_) {
          // Parse character constant.
          if (ParseString() == ERROR) return ERROR;

          // Character constants must be exactly one character long.
          if (token_text_.size() != 1) {
            return Error("Invalid character constant");
          }
          return Token(CHARACTER_TOKEN);
        } else {
          // Parse literal symbol.
          NextChar();
          int first = current_;
          NextChar();
          if (!ParseName(first)) return Error("Invalid symbol name");
          return Token(LITERAL_TOKEN);
        }

      case '<':  // operator: < <= << <<=
        NextChar();
        if (current_ == '=') return Select(LTE_TOKEN);
        if (current_ == '<') return Select('=', ASSIGN_SHL_TOKEN, SHL_TOKEN);
        return Token('<');

      case '>':  // operator: > >= >> >>= >>> >>>=
        NextChar();
        if (current_ == '=') return Select(GTE_TOKEN);
        if (current_ == '>') {
          NextChar();
          if (current_ == '=') return Select(ASSIGN_SAR_TOKEN);
          if (current_ == '>') return Select('=', ASSIGN_SHR_TOKEN, SHR_TOKEN);
          return Token(SAR_TOKEN);
        }
        return Token('>');

      case '=':  // operator: = == ===
        NextChar();
        if (current_ == '=') return Select('=', EQ_STRICT_TOKEN, EQ_TOKEN);
        return Token('=');

      case '!':  // operator: ! != !==
        NextChar();
        if (current_ == '=') return Select('=', NE_STRICT_TOKEN, NE_TOKEN);
        return Token('!');

      case '+':  // operator: + ++ +=
        NextChar();
        if (current_ == '+') return Select(INC_TOKEN);
        if (current_ == '=') return Select(ASSIGN_ADD_TOKEN);
        return Token('+');

      case '-':  // negative number or operator: - -- -=
        NextChar();
        if (!func_mode_) return ParseNumber(true, false);
        if (current_ == '-') return Select(DEC_TOKEN);
        if (current_ == '=') return Select(ASSIGN_SUB_TOKEN);
        return Token('-');

      case '*':  // operator: * *=
        return Select('=', ASSIGN_MUL_TOKEN, '*');

      case '%':  // operator: % %=
        return Select('=', ASSIGN_MOD_TOKEN, '%');

      case '/':  // comment, symbol name, or operator: / // /* /
        NextChar();
        if (current_ == '/') {
          // Parse single-line comment starting with "//".
          while (current_ != -1 && current_ != '\n') NextChar();
          continue;
        }
        if (current_ == '*') {
          // Parse multi-line comment starting with "/*".
          NextChar();
          for (;;) {
            if (current_ == -1) return Error("Unterminated comment");
            int prev = current_;
            NextChar();
            if (prev == '*' && current_ == '/') {
              NextChar();
              break;
            }
          }
          continue;
        }
        if (current_ == '=') return Select(ASSIGN_DIV_TOKEN);
        if (!func_mode_ || ascii_isalnum(current_) || current_ == '_') {
          if (!ParseName('/')) return Error("Invalid symbol name");
          return Token(SYMBOL_TOKEN);
        }
        return Token('/');

      case '&':  // operator: & && &=
        NextChar();
        if (current_ == '&') return Select(AND_TOKEN);
        if (current_ == '=') return Select(ASSIGN_BIT_AND_TOKEN);
        return Token('&');

      case '|':  // operator: | || |=
        NextChar();
        if (current_ == '|') return Select(OR_TOKEN);
        if (current_ == '=') return Select(ASSIGN_BIT_OR_TOKEN);
        return Token('|');

      case '^':  // operator: ^ ^=
        return Select('=', ASSIGN_BIT_XOR_TOKEN, '^');

      default:  // keyword, symbol name, or single-character token
        if (ascii_isalpha(current_) || current_ == '_') {
          int first = current_;
          NextChar();
          if (!ParseName(first)) return Error("Invalid name");
          return Token(LookupKeyword());
        }
        return Select(current_);
    }
  }
}

int Tokenizer::ParseString() {
  // Skip start quote character.
  int delimiter = current_;
  NextChar();

  // Read until end of string.
  while (current_ != delimiter) {
    // Check for unterminated string.
    if (current_ == -1 || current_ == '\n') {
      return Error("Unterminated string");
    } else if (current_ == '\\') {
      // Handle escape characters.
      NextChar();
      int ch;
      switch (current_) {
        case 'a': Append('\a'); NextChar(); break;
        case 'b': Append('\b'); NextChar(); break;
        case 'f': Append('\f'); NextChar(); break;
        case 'n': Append('\n'); NextChar(); break;
        case 'r': Append('\r'); NextChar(); break;
        case 't': Append('\t'); NextChar(); break;
        case 'v': Append('\v'); NextChar(); break;
        case 'x':
          // Parse hex escape (\x00).
          NextChar();
          ch = HexToDigit(current_);
          NextChar();
          ch = (ch << 4) + HexToDigit(current_);
          NextChar();
          if (ch < 0) return Error("Invalid hex escape in string");
          Append(ch);
          break;
        case 'u':
          // Parse unicode hex escape (\u0000).
          NextChar();
          if (!ParseUnicode(4)) {
            return Error("Invalid Unicode escape in string");
          }
          break;
        case 'U':
          // Parse unicode hex escape (\U00000000).
          NextChar();
          if (!ParseUnicode(8)) {
            return Error("Invalid Unicode escape in string");
          }
          break;
        default:
          // Just escape the next character.
          Append(current_);
          NextChar();
      }
    } else {
      // Add character to string.
      Append(current_);
      NextChar();
    }
  }

  // Skip end quote character.
  NextChar();
  return Token(STRING_TOKEN);
}

int Tokenizer::ParseNumber(bool negative, bool fractional) {
  // Add sign for negative numbers.
  if (negative) Append('-');

  // Parse integral part.
  int integral_digits = fractional ? 0 : ParseDigits();

  // Check for decimal part.
  if (current_ == '.') {
    fractional = true;
    NextChar();
  }
  if (fractional) Append('.');

  if (fractional || current_ == 'e' || current_ == 'E') {
    // Parse decimal part.
    int decimal_digits = ParseDigits();

    if (current_ == 'e' || current_ == 'E') {
      // Parse exponent.
      Append('e');
      NextChar();
      if (current_ == '-' || current_ == '+') {
        Append(current_);
        NextChar();
      }
      int exponent_digits = ParseDigits();
      if (exponent_digits == 0) {
        return Error("Missing exponent in number");
      }
    }

    if (integral_digits == 0 && decimal_digits == 0) {
      return Error("Invalid floating-point number");
    }
    return Token(FLOAT_TOKEN);
  } else {
    if (integral_digits == 0) return Error("Invalid integer number");

    // Parse hexadecimal number, if the number starts with 0x.
    if (integral_digits == 1 && current_ == 'x' && token_text_ == "0") {
      Append('x');
      NextChar();
      int hex_digits = 0;
      while (current_ != -1 && ascii_isxdigit(current_)) {
        Append(current_);
        NextChar();
        hex_digits++;
      }
      if (hex_digits == 0) return Error("Invalid hex number");
    }
    return Token(INTEGER_TOKEN);
  }
}

bool Tokenizer::ParseName(int first) {
  // Save first character in symbol name.
  int ch;
  if (first == '\\') {
    if (current_ == -1) return false;
    Append(current_);
    NextChar();
  } else if (first == '%') {
    ch = HexToDigit(current_);
    NextChar();
    ch = (ch << 4) + HexToDigit(current_);
    NextChar();
    if (ch < 0) return Error("Invalid hex escape in symbol");
    Append(ch);
  } else {
    Append(first);
  }

  // Parse remaining part of symbol name.
  for (;;) {
    switch (current_) {
      case -1:
        return true;

      case '_':
      case '/':
        Append(current_);
        NextChar();
        break;

      case '-':
      case '.':
      case '!':
        if (func_mode_) return true;
        Append(current_);
        NextChar();
        break;

      case '\\':
        NextChar();
        if (current_ == -1) return false;
        Append(current_);
        NextChar();
        break;

      case '%':
        NextChar();
        ch = HexToDigit(current_);
        NextChar();
        ch = (ch << 4) + HexToDigit(current_);
        NextChar();
        if (ch < 0) return Error("Invalid hex escape in symbol");
        Append(ch);
        break;

      default:
        if (!ascii_isalnum(current_)) return true;
        Append(current_);
        NextChar();
    }
  }
}

int Tokenizer::LookupKeyword() {
  const char *name = token_text_.data();
  int first = *name;
  if (func_mode_) {
    switch (token_text_.size()) {
      case 2:
        if (first == 'i') {
          if (memcmp(name, "if", 2) == 0) return IF_TOKEN;
          if (memcmp(name, "in", 2) == 0) return IN_TOKEN;
        }
        break;

      case 3:
        if (first == 'f' && memcmp(name, "for", 3) == 0) return FOR_TOKEN;
        if (first == 'i' && memcmp(name, "isa", 3) == 0) return ISA_TOKEN;
        if (first == 'n' && memcmp(name, "nil", 3) == 0) return NULL_TOKEN;
        if (first == 'v' && memcmp(name, "var", 3) == 0) return VAR_TOKEN;
        break;

      case 4:
        if (first == 'e' && memcmp(name, "else", 4) == 0) return ELSE_TOKEN;
        if (first == 'f' && memcmp(name, "func", 4) == 0) return FUNC_TOKEN;
        if (first == 'n' && memcmp(name, "null", 4) == 0) return NULL_TOKEN;
        if (first == 's' && memcmp(name, "self", 4) == 0) return SELF_TOKEN;
        if (first == 't') {
          if (memcmp(name, "this", 4) == 0) return THIS_TOKEN;
          if (memcmp(name, "true", 4) == 0) return TRUE_TOKEN;
        }
        break;

      case 5:
        if (first == 'c' && memcmp(name, "const", 5) == 0) return CONST_TOKEN;
        if (first == 'f' && memcmp(name, "false", 5) == 0) return FALSE_TOKEN;
        if (first == 'w' && memcmp(name, "while", 5) == 0) return WHILE_TOKEN;
        break;

      case 6:
        if (first == 'r' && memcmp(name, "return", 6) == 0) return RETURN_TOKEN;
        if (first == 's' && memcmp(name, "static", 6) == 0) return STATIC_TOKEN;
        break;

      case 7:
        if (first == 'p' && memcmp(name, "private", 7) == 0) {
          return PRIVATE_TOKEN;
        }
        break;

      case 8:
        if (first == 'f' && memcmp(name, "function", 8) == 0) return FUNC_TOKEN;
        break;
    }
  } else {
    switch (token_text_.size()) {
      case 3:
        if (first == 'n' && memcmp(name, "nil", 3) == 0) return NULL_TOKEN;
        break;

      case 4:
        if (first == 'n' && memcmp(name, "null", 4) == 0) return NULL_TOKEN;
        if (first == 't' && memcmp(name, "true", 4) == 0) return TRUE_TOKEN;
        break;

      case 5:
        if (first == 'f' && memcmp(name, "false", 5) == 0) return FALSE_TOKEN;
        break;
    }
  }

  return SYMBOL_TOKEN;
}

}  // namespace sling

