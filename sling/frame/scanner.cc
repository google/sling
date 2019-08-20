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

#include "sling/frame/scanner.h"

#include <string>

#include "sling/string/ctype.h"
#include "sling/string/strcat.h"

namespace sling {

int Scanner::HexToDigit(int ch) {
  if (ch >= '0' && ch <= '9') return ch - '0';
  if (ch >= 'A' && ch <= 'F') return ch - 'A' + 10;
  if (ch >= 'a' && ch <= 'f') return ch - 'a' + 10;
  return -1;
}

Scanner::Scanner(Input *input) : input_(input) {
  current_ = 0;
  line_ = 1;
  column_ = 1;
  NextChar();
}

void Scanner::NextChar() {
  char ch;

  if (current_ != -1 && input_->Next(&ch)) {
    current_ = static_cast<unsigned char>(ch);
  } else {
    current_ = -1;
  }
  if (current_ == '\n') {
    line_++;
    column_ = 0;
  } else {
    column_++;
  }
}

int Scanner::Select(char next, int then, int otherwise) {
  NextChar();
  if (current_ == next) {
    NextChar();
    return Token(then);
  } else {
    return Token(otherwise);
  }
}


int Scanner::ParseDigits() {
  int digits = 0;
  while (current_ != -1 && ascii_isdigit(current_)) {
    Append(current_);
    NextChar();
    digits++;
  }
  return digits;
}

bool Scanner::ParseUnicode(int digits) {
  // Parse Unicode code point.
  uint32 code = 0;
  for (int i = 0; i < digits; ++i) {
    int digit = HexToDigit(current_);
    if (digit < 0) return false;
    code = (code << 4) + digit;
    if (code > 0x10ffff) return false;
    NextChar();
  }

  // Convert code point to UTF-8.
  if (code <= 0x7f) {
    // One character sequence.
    Append(code);
  } else if (code <= 0x7ff) {
    // Two character sequence.
    Append(0xc0 | (code >> 6));
    Append(0x80 | (code & 0x3f));
  } else if (code <= 0xffff) {
    // Three character sequence.
    Append(0xe0 | (code >> 12));
    Append(0x80 | ((code >> 6) & 0x3f));
    Append(0x80 | (code & 0x3f));
  } else {
    // Four character sequence.
    Append(0xf0 | (code >> 18));
    Append(0x80 | ((code >> 12) & 0x3f));
    Append(0x80 | ((code >> 6) & 0x3f));
    Append(0x80 | (code & 0x3f));
  }

  return true;
}

void Scanner::SetError(const string &error_message) {
  if (error_message_.empty()) error_message_ = error_message;
  token_ = ERROR;
}

int Scanner::Error(const string &error_message) {
  SetError(error_message);
  return ERROR;
}

string Scanner::GetErrorMessage(const string &filename) const {
  return StrCat(filename, ":", line_, ":", column_, ": ", error_message_);
}

}  // namespace sling

