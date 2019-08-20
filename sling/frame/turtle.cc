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

#include "sling/frame/turtle.h"

#include <string>

#include "sling/string/ctype.h"
#include "sling/string/numbers.h"

namespace sling {

TurtleTokenizer::TurtleTokenizer(Input *input) : Scanner(input) {
  NextToken();
}

int TurtleTokenizer::NextToken() {
  // Clear token text buffer.
  token_text_.clear();

  // Keep reading until we either read a token or reach the end of the input.
  for (;;) {
    // Skip whitespace.
    while (current_ != -1 && ascii_isspace(current_)) NextChar();

    // Parse next token (or comment).
    switch (current_) {
      case -1:
        return Token(END);

      case '"': case '\'':
        return ParseString();

      case '0': case '1': case '2': case '3': case '4':
      case '5': case '6': case '7': case '8': case '9':
      case '+': case '-':
        return ParseNumber();

      case '#':
        // Parse comment.
        NextChar();
        while (current_ != -1 && current_ != '\n') NextChar();
        continue;

      case '<':
        return ParseURI();

      case '=':
        return Select('>', IMPLIES_TOKEN, '=');

      case '^':
        return Select('^', TYPE_TOKEN, '^');

      case '.':
        return Select(current_);

      default: {
        prefix_.clear();
        bool colon = false;
        for (;;) {
          if (current_ == -1) {
            break;
          } else if (current_ == ':') {
            // First colon ends the name prefix.
            if (colon) {
              Append(':');
            } else {
              prefix_ = token_text_;
              token_text_.clear();
              colon = true;
            }
            NextChar();
          } else if (current_ == '\\') {
            // Parse character escape (\c).
            NextChar();
            if (current_ == -1) return Error("invalid escape sequence in name");
            Append(current_);
            NextChar();
          } else if (current_ == '%') {
            // Parse hex escape (%00).
            NextChar();
            int ch = HexToDigit(current_);
            NextChar();
            ch = (ch << 4) + HexToDigit(current_);
            NextChar();
            if (ch < 0) return Error("invalid hex escape in name");
            Append(ch);
          } else if (current_ >= 128 || ascii_isalnum(current_) ||
                    current_ == '_' || current_ == '.' || current_ == '-') {
            Append(current_);
            NextChar();
          } else {
            break;
          }
        }

        if (!colon && token_text_.empty() && prefix_.empty()) {
          // Single-character token.
          return Select(current_);
        } else if (colon) {
          // Prefixed name.
          return Token(NAME_TOKEN);
        } else {
          // Name or reserved word.
          return Token(LookupKeyword());
        }
      }
    }
  }
}

int TurtleTokenizer::ParseURI() {
  // Skip start delimiter.
  NextChar();

  // Parse URI.
  while (current_ != '>') {
    if (current_ <= ' ') {
      return Error("Unterminated URI");
    } else if (current_ == '\\') {
      NextChar();
      if (current_ == 'u') {
        // Parse unicode hex escape (\u0000).
        NextChar();
        if (!ParseUnicode(4)) {
          return Error("Invalid Unicode escape in URI");
        }
      } else if (current_ == 'U') {
        // Parse unicode hex escape (\U00000000).
        NextChar();
        if (!ParseUnicode(8)) {
          return Error("Invalid Unicode escape in URI");
        }
      } else {
        return Error("Invalid URI");
      }
    } else {
      // Add character to URI.
      Append(current_);
      NextChar();
    }
  }

  NextChar();
  return Token(URI_TOKEN);
}

int TurtleTokenizer::ParseString() {
  // Skip start delimiter(s).
  int delimiter = current_;
  int delimiters = 0;
  while (delimiters < 3 && current_ == delimiter) {
    NextChar();
    delimiters++;
  }
  bool multi_line = false;
  if (delimiters == 3) {
    // Multi-line string.
    multi_line = true;
  } else if (delimiters == 2) {
    // Empty string.
    return Token(STRING_TOKEN);
  }

  // Read rest of string.
  bool done = false;
  delimiters = 0;
  while (!done) {
    // Check for unterminated string.
    if (current_ == -1 || (!multi_line && current_ == '\n')) {
      return Error("Unterminated string");
    }

    // Check for delimiters.
    if (current_ == delimiter) {
      if (multi_line) {
        if (++delimiters == 3) {
          // End of multi-line string. Remove two previous delimiters.
          token_text_.resize(token_text_.size() - 2);
          NextChar();
          done = true;
        } else {
          Append(current_);
          NextChar();
        }
      } else {
        // End of string.
        NextChar();
        done = true;
      }
    } else {
      delimiters = 0;
      if (current_ == '\\') {
        // Handle escape characters.
        NextChar();
        switch (current_) {
          case 'b': Append('\b'); NextChar(); break;
          case 'f': Append('\f'); NextChar(); break;
          case 'n': Append('\n'); NextChar(); break;
          case 'r': Append('\r'); NextChar(); break;
          case 't': Append('\t'); NextChar(); break;
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
  }

  return Token(STRING_TOKEN);
}

int TurtleTokenizer::ParseNumber() {
  // Parse sign.
  int sign = 0;
  if (current_ == '+' || current_ == '-') {
    sign = current_;
    Append(current_);
    NextChar();
    if (!ascii_isdigit(current_) &&
        current_ != '.' &&
        current_ != 'e' &&
        current_ != 'E') {
      return Token(sign);
    }
  }

  // Parse integral part.
  int integral_digits = ParseDigits();

  // Parse decimal part.
  int decimal_digits = 0;
  if (current_ == '.') {
    Append('.');
    NextChar();
    decimal_digits = ParseDigits();
    if (!sign && integral_digits == 0 && decimal_digits == 0) {
      return Token('.');
    }
  }

  // Parse exponent.
  int exponent_digits = 0;
  if (current_ == 'e' || current_ == 'E') {
    Append('e');
    NextChar();
    if (current_ == '-' || current_ == '+') {
      Append(current_);
      NextChar();
    }
    exponent_digits = ParseDigits();
    if (exponent_digits == 0) {
      return Error("Missing exponent in number");
    }
  }

  // Determine number type.
  if (exponent_digits != 0) {
    return Token(FLOAT_TOKEN);
  } else if (decimal_digits != 0) {
    return Token(DECIMAL_TOKEN);
  } else if (integral_digits != 0) {
    return Token(INTEGER_TOKEN);
  } else {
    return Error("Invalid number");
  }
}

int TurtleTokenizer::LookupKeyword() {
  const char *name = token_text_.data();
  int first = *name;
  switch (token_text_.size()) {
    case 1:
      if (first == 'a') return A_TOKEN;
      break;

    case 4:
      if (first == 't' && memcmp(name, "true", 4) == 0) return TRUE_TOKEN;
      if (first == 'b' && memcmp(name, "base", 4) == 0) return BASE_TOKEN;
      if (first == 'B' && memcmp(name, "BASE", 4) == 0) return BASE_TOKEN;
      break;

    case 5:
      if (first == 'f' && memcmp(name, "false", 5) == 0) return FALSE_TOKEN;
      break;

    case 6:
      if (first == 'p' && memcmp(name, "prefix", 6) == 0) return PREFIX_TOKEN;
      if (first == 'P' && memcmp(name, "PREFIX", 6) == 0) return PREFIX_TOKEN;
      break;
  }

  return NAME_TOKEN;
}

TurtleParser::TurtleParser(Store *store, Input *input)
    : TurtleTokenizer(input),
      store_(store),  stack_(store), references_(store) {
  // Add dummy for reference index zero.
  references_.push_back(Handle::nil());
}

Object TurtleParser::ReadAll() {
  Handle handle = Handle::nil();
  while (!done() && !error()) {
    handle = ReadObject();
  }
  return Object(store_, handle);
}

Object TurtleParser::Read() {
  return Object(store_, ReadObject());
}

Handle TurtleParser::ReadObject() {
  // Parse directives.
  while (token() == '@' || token() == PREFIX_TOKEN || token() == BASE_TOKEN) {
    if (!ParseDirective()) return Handle::error();
  }
  if (token() == END) return Handle::nil();

  // Put elements on the stack while parsing.
  Word mark = Mark();

  // Parse subject.
  Handle subject = Handle::nil();
  switch (token()) {
    case NAME_TOKEN:
    case URI_TOKEN:
      // Parse subject id.
      subject = ParseIdentifier(true);
      break;
    case '[':
      // Parse blank node subject.
      subject = ParseBlankNode();
      break;
    case '(':
      // Parse collection subject.
      subject = ParseCollection();
      break;
    default:
      SetError("Invalid subject");
      subject = Handle::error();
  }
  if (subject.IsError()) return Handle::error();

  // Resolve subject.
  int ref = 0;
  Handle handle = Handle::nil();
  if (subject.IsIndex()) {
    // Subject is a local name.
    ref = subject.AsIndex();
    handle = references_[ref];

    // Copy any existing slots.
    if (!handle.IsNil()) {
      FrameDatum *frame = store_->GetFrame(handle);
      PushFrame(frame);
    }
  } else if (subject.IsRef()) {
    Datum *datum = store_->Deref(subject);
    if (datum->IsSymbol()) {
      SymbolDatum *symbol = datum->AsSymbol();
      if (symbol->bound()) {
        // Copy any existing slots.
        handle = symbol->value;
        FrameDatum *frame = store_->GetFrame(handle);
        PushFrame(frame);
      } else {
        // Add id slot for unbound symbol.
        Push(Handle::id());
        Push(subject);
      }
    } else if (datum->IsFrame()) {
      // Copy any existing slots from blank node subject.
      handle = subject;
      FrameDatum *frame = store_->GetFrame(handle);
      PushFrame(frame);
    } else if (datum->IsArray()) {
      // Represent collection subject with is: slot.
      Push(Handle::is());
      Push(subject);
    } else {
      SetError("Unexpected subject type");
      return Handle::error();
    }
  } else {
    SetError("Unexpected subject");
    return Handle::error();
  }

  // Parse predicate object list for subject.
  if (!ParsePredicateObjectList()) return Handle::error();

  // Skip terminating period.
  if (token() != '.') {
    SetError("Missing '.' in triple");
    return Handle::error();
  }
  NextToken();

  // A frame with a local name could have gotten a proxy frame if it has been
  // referenced.
  if (ref != 0) handle = references_[ref];

  // Create new frame from slots.
  Slot *begin =  reinterpret_cast<Slot *>(stack_.address(mark));
  Slot *end =  reinterpret_cast<Slot *>(stack_.end());
  handle = store_->AllocateFrame(begin, end, handle);

  // Update reference.
  if (ref != 0) references_[ref] = handle;

  // Remove slots from stack.
  Release(mark);

  // Return handle to parsed frame.
  return handle;
}

Handle TurtleParser::ParseBlankNode() {
  // Skip '['.
  NextToken();

  // Put elements on the stack while parsing.
  Word mark = Mark();

  // Parse predicate object list.
  if (!ParsePredicateObjectList()) return Handle::error();

  // Skip ']'.
  if (token() != ']') {
    SetError("Missing ']' in predicate object list");
    return Handle::error();
  }
  NextToken();

  // Create new frame from slots.
  Slot *begin =  reinterpret_cast<Slot *>(stack_.address(mark));
  Slot *end =  reinterpret_cast<Slot *>(stack_.end());
  Handle handle = store_->AllocateFrame(begin, end);

  // Remove slots from stack.
  Release(mark);

  // Return handle to parsed frame.
  return handle;
}

bool TurtleParser::ParsePredicateObjectList() {
  while (token() != ']') {
    // Check for end of input.
    if (token() == END) {
      SetError("Syntax error");
      return false;
    }

    // Parse predicate.
    Handle predicate = ParsePredicate();
    if (predicate.IsError()) return false;

    // Parse object list.
    for (;;) {
      // Parse object value.
      Handle object = ParseValue();
      if (object.IsError()) return false;

      // Push predicate/object slot onto stack.
      Push(predicate);
      Push(object);

      // Multiple values are separted by comma.
      if (token() == ',') {
        NextToken();
      } else {
        break;
      }
    }

    // Multiple predicates are separted by semicolon.
    if (token() == ';') {
      NextToken();
    } else {
      break;
    }
  }

  return true;
}

Handle TurtleParser::ParseCollection() {
  // Skip '('.
  NextToken();

  // Put elements on the stack while parsing.
  Word mark = Mark();

  // Parse object list.
  while (token() != ')') {
    // Parse object value.
    Handle object = ParseValue();
    if (object.IsError()) return Handle::error();

    // Push value slot onto stack.
    Push(object);
  }

  // Skip ')'.
  NextToken();

  // Create new array.
  Handle *begin =  stack_.address(mark);
  Handle *end =  stack_.end();
  Handle handle = store_->AllocateArray(begin, end);

  // Remove slots from stack.
  Release(mark);

  // Return handle to parsed frame.
  return handle;
}

Handle TurtleParser::ParseIdentifier(bool subject) {
  if (token() == NAME_TOKEN) {
    if (prefix() == "_") {
      // Blank node label.
      int &label = locals_[token_text()];
      NextToken();
      if (subject) {
        if (label == 0) {
          // Add dummy reference for new subject label.
          label = references_.size();
          references_.push_back(Handle::nil());
        }
        return Handle::Index(label);
      } else {
        if (label == 0) {
          // Add empty frame for unknown label.
          label = references_.size();
          references_.push_back(store_->AllocateFrame(0));
        }
        return references_[label];
      }
    } else {
      // Look up prefix.
      auto f = namespaces_.find(prefix());
      if (f == namespaces_.end()) {
        SetError("Unknown namespace prefix");
        return Handle::error();
      }

      // Look up name in store.
      string fullname = f->second + token_text();
      NextToken();
      if (subject) {
        return store_->Symbol(fullname);
      } else {
        return store_->Lookup(fullname);
      }
    }
  } else if (token() == URI_TOKEN) {
    // Look up URI in store.
    Handle h;
    if (base_.empty() || !IsRelativeURI(token_text())) {
      if (subject) {
        h = store_->Symbol(token_text());
      } else {
        h = store_->Lookup(token_text());
      }
    } else {
      // Prepend base URI for relative URIs.
      string absolute = base_ + token_text();
      if (subject) {
        h = store_->Symbol(absolute);
      } else {
        h = store_->Lookup(absolute);
      }
    }
    NextToken();
    return h;
  } else {
    SetError("Invalid identifier");
    return Handle::error();
  }
}

Handle TurtleParser::ParsePredicate() {
  if (token() == A_TOKEN) {
    NextToken();
    return Handle::isa();
  } else if (token() == IMPLIES_TOKEN) {
    NextToken();
    return Handle::is();
  } else if (token() == '=') {
    NextToken();
    return Handle::id();
  } else {
    return ParseIdentifier(false);
  }
}

Handle TurtleParser::ParseValue() {
  Handle handle;
  switch (token()) {
    case STRING_TOKEN:
      handle = store_->AllocateString(token_text());
      NextToken();
      if (token() == '@') {
        // Add language annotation to string value.
        String str(store_, handle);
        NextToken();
        if (token() != NAME_TOKEN) {
          SetError("Invalid language for string");
          handle = Handle::error();
        } else {
          Slot slots[2];
          slots[0].name = Handle::is();
          slots[0].value = str.handle();
          slots[1].name = store_->Lookup("lang");
          slots[1].value = store_->Lookup("/lang/" + token_text());
          handle = store_->AllocateFrame(slots, slots + 2);
          NextToken();
        }
      } else if (token() == TYPE_TOKEN) {
        // Add type annotation to string value.
        String str(store_, handle);
        NextToken();
        Handle type = ParseIdentifier(false);
        if (type.IsError()) {
          handle = Handle::error();
        } else {
          Slot slots[2];
          slots[0].name = Handle::is();
          slots[0].value = str.handle();
          slots[1].name = Handle::isa();
          slots[1].value = type;
          handle = store_->AllocateFrame(slots, slots + 2);
        }
      }
      break;

    case INTEGER_TOKEN: {
      int32 value;
      float fvalue;
      if (safe_strto32(token_text(), &value)) {
        if (value >= Handle::kMinInt && value <= Handle::kMaxInt) {
          handle = Handle::Integer(value);
        } else {
          handle = Handle::Float(value);
        }
        NextToken();
      } else if (safe_strtof(token_text(), &fvalue)) {
        handle = Handle::Float(fvalue);
        NextToken();
      } else {
        SetError("Invalid number token");
        handle = Handle::error();
      }
      break;
    }

    case DECIMAL_TOKEN:
    case FLOAT_TOKEN: {
      float value;
      if (safe_strtof(token_text(), &value)) {
        handle = Handle::Float(value);
        NextToken();
      } else {
        SetError("Invalid floating point token");
        handle = Handle::error();
      }
      break;
    }

    case TRUE_TOKEN:
      handle = Handle::Bool(true);
      NextToken();
      break;

    case FALSE_TOKEN:
      handle = Handle::Bool(false);
      NextToken();
      break;

    case '[':
      handle = ParseBlankNode();
      break;

    case '(':
      handle = ParseCollection();
      break;

    default:
      handle = ParseIdentifier(false);
  }

  return handle;
}

bool TurtleParser::ParseDirective() {
  bool legacy = true;
  if (token() == '@') {
    NextToken();
    legacy = false;
  }
  if (token() == PREFIX_TOKEN) {
    // Parse prefix directive.
    NextToken();
    if (token() != NAME_TOKEN || !token_text().empty()) {
      SetError("Namespace prefix expected");
      return false;
    }
    string ns = prefix();
    NextToken();

    // Parse namespace URI.
    if (token() != URI_TOKEN) {
      SetError("URI expected in prefix directive");
      return false;
    }

    // Update namespace mapping.
    namespaces_[ns] = token_text();
    NextToken();

    // Parse trailing period.
    if (!legacy) {
      if (token() != '.') {
        SetError("Missing period in prefix directive");
        return false;
      }
      NextToken();
    }
    return true;
  } else if (token() == BASE_TOKEN) {
    // Parse base directive.
    NextToken();
    if (token() != URI_TOKEN) {
      SetError("URI expected in base directive");
      return false;
    }

    // Update base URI.
    base_ = token_text();
    NextToken();

    // Parse trailing period.
    if (!legacy) {
      if (token() != '.') {
        SetError("Missing period in prefix directive");
        return false;
      }
      NextToken();
    }
    return true;
  } else {
    SetError("Unknown directive");
    return false;
  }
}

bool TurtleParser::IsRelativeURI(const string &uri) {
  const char *p = uri.data();
  const char *end = p + uri.size();

  // Skip protocol, e.g. 'http:'.
  while (p < end && ascii_isalnum(*p)) p++;
  if (p < end && *p == ':') p++;

  // Check for '//' in absolute URI.
  if (p + 1 < end) return true;
  if (p[0] == '/' && p[1] == '/') return false;
  return true;
}

}  // namespace sling
