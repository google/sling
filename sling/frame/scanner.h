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

#ifndef SLING_FRAME_SCANNER_H_
#define SLING_FRAME_SCANNER_H_

#include <string>

#include "sling/stream/input.h"

namespace sling {

// Base scanner chunking the input into tokens.
class Scanner {
 public:
  // Token types in the range 0-255 are used for single-character tokens.
  enum ScannerTokenType {
    ERROR = 256,
    END,
    FIRST_AVAILABLE_TOKEN_TYPE
  };

  // Initializes scanner with input.
  explicit Scanner(Input *input);

  // Checks if all input has been read.
  bool done() const { return token_ == END; }

  // Returns true if errors were found while parsing input.
  bool error() const { return token_ == ERROR; }

  // Returns current line and column.
  int line() const { return line_; }
  int column() const { return column_; }

  // Returns last error message.
  const string &error_message() const { return error_message_; }

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

 protected:
  // Converts hexadecimal character to digit value.
  static int HexToDigit(int ch);

  // Gets the next input character.
  void NextChar();

  // Sets current token and returns it.
  int Token(int token) { token_ = token; return token; }

  // Consumes current input character and returns token.
  int Select(int token) { NextChar(); return Token(token); }

  // Consumes current input character and return tokens dependent on the next
  // input character.
  int Select(char next, int then, int otherwise);

  // Parses a sequence of digits. Returns the number of digits parsed.
  int ParseDigits();

  // Parses a sequence of Unicode hex characters and appends these as UTF-8 to
  // the token buffer. Returns false on error.
  bool ParseUnicode(int digits);

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

  // Text for last read token. This contains string, name, and number tokens.
  string token_text_;

  // Last error message.
  string error_message_;
};

}  // namespace sling

#endif  // SLING_FRAME_SCANNER_H_

