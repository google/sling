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

#include "sling/web/xml-parser.h"

#include <string.h>
#include <string>

#include "sling/base/logging.h"
#include "sling/base/types.h"
#include "sling/string/ctype.h"
#include "sling/util/unicode.h"
#include "sling/web/entity-ref.h"

namespace sling {

const char *XMLElement::Get(const char *name, const char *defval) const {
  for (const XMLAttribute &attr : attrs) {
    if (strcmp(attr.name, name) == 0) return attr.value;
  }
  return defval;
}

const char *XMLElement::Get(const char *name) const {
  for (const XMLAttribute &attr : attrs) {
    if (strcmp(attr.name, name) == 0) return attr.value;
  }
  return nullptr;
}

XMLParser::XMLParser() {
  buffer_ = bufptr_ = static_cast<char *>(malloc(4096));
  bufend_ = buffer_ + 4096;
}

XMLParser:: ~XMLParser() {
  free(buffer_);
}

void XMLParser::Add(char ch) {
  // Add character to buffer.
  if (bufptr_ < bufend_) {
    *bufptr_++ = ch;
    return;
  }

  // No more room in buffer, expand it.
  int buflen = bufend_ - buffer_;
  int newlen = buflen * 2;
  char *newbuf = static_cast<char *>(realloc(buffer_, newlen));

  // If the buffer has been moved to a new location, adjust pointers.
  if (newbuf != buffer_) {
    ssize_t offset = newbuf - buffer_;
    bufptr_ += offset;
    if (txtptr_) txtptr_ += offset;
    if (element_.name) element_.name += offset;

    for (XMLAttribute &attr : element_.attrs) {
      if (attr.name) attr.name += offset;
      if (attr.value) attr.value += offset;
    }

    for (char *&t : stack_) t += offset;

    buffer_ = newbuf;
  }

  bufend_ = buffer_ + newlen;

  // Add character to new buffer.
  *bufptr_++ = ch;
}

void XMLParser::AddText(char ch) {
  if (!txtptr_) txtptr_ = bufptr_;
  Add(ch);
}

void XMLParser::AddString(const char *text) {
  while (*text) Add(*text++);
}

bool XMLParser::IsNameChar(int ch) {
  if (ch < 0) return false;
  return ascii_isalnum(ch) || ch == ':' || ch == '-' || ch == '_' || ch == '.';
}

int XMLParser::ReadChar() {
  uint8 ch;
  if (!input_->Next(reinterpret_cast<char *>(&ch))) return -1;
  if (ch == '\n') line_++;
  return ch;
}

int XMLParser::SkipWhitespace(int ch) {
  while (ch >= 0 && ascii_isspace(ch)) ch = ReadChar();
  return ch;
}

bool XMLParser::Error(const char *message) {
  LOG(ERROR) << "XML parse error line " << line_ << ": " << message;
  return false;
};

void XMLParser::Init(Input *input) {
  input_ = input;
  bufptr_ = buffer_;
  txtptr_ = nullptr;
  element_.Clear();
  stack_.clear();
}

bool XMLParser::Parse(Input *input) {
  int ch;
  string entref;

  // Initialize document state.
  Init(input);
  if (!StartDocument()) return false;

  // Process XML input.
  ch = ReadChar();
  while (ch >= 0) {
    if (ch != '<') {
      // Process text.
      if (ch == '&') {
        // Process entity reference.
        entref.clear();
        entref.push_back('&');
        ch = ReadChar();
        while (ch >= 0 && ch != ';' && ch != '<' && ch != '&') {
          entref.push_back(ch);
          ch = ReadChar();
        }
        entref.push_back(';');

        int code = ParseEntityRef(entref);
        if (code < 0) {
          AddText('&');
          AddString(entref.data());
        } else {
          char utf[UTF8::MAXLEN + 1];
          int utflen = UTF8::Encode(code, utf);
          for (int i = 0; i < utflen; ++i) AddText(utf[i]);
          ch = ReadChar();
        }
      } else {
        // Add text to heap buffer.
        AddText(ch);

        // Read next char.
        ch = ReadChar();
      }
    } else {
      bool pi = false;
      bool endtag = false;
      bool single = false;

      // XML markup (tag, pi, or comment), flush text before processing markup.
      if (txtptr_) {
        Add(0);
        if (!Text(txtptr_)) return false;
        bufptr_ = txtptr_;
        txtptr_ = nullptr;
      }

      ch = ReadChar();
      if (ch == '!') {
        // Comment.
        ch = ReadChar();
        if (ch == '-') {
          ch = ReadChar();
          if (ch == '-') {
            // Start of comment.
            ch = ReadChar();

            // Parse until -->
            int dashes = 0;
            while (true) {
              Add(ch);
              if (ch == '-') {
                dashes++;
              } else if (ch == '>') {
                if (dashes >= 2) {
                  // End of comment found, output comment.
                  bufptr_ -= 3;
                  AddText(0);
                  if (!Comment(txtptr_)) return false;
                  bufptr_ = txtptr_;
                  txtptr_ = nullptr;
                  ch = ReadChar();
                  break;
                } else {
                  dashes = 0;
                }
              } else if (ch < 0) {
                return Error("comment not terminated");
              } else {
                dashes = 0;
              }
              ch = ReadChar();
            }
          } else {
            AddText('<');
            AddText('!');
          }
        } else {
          AddText('<');
        }
      } else {
        if (ch == '/') {
          // End tag.
          endtag = true;
          ch = ReadChar();
        } else if (ch == '?') {
          // Processing instruction.
          pi = true;
          ch = ReadChar();
        }

        // Skip whitespace before tag name.
        ch = SkipWhitespace(ch);

        // Read tag name.
        element_.name = bufptr_;
        while (IsNameChar(ch)) {
          Add(ch);
          ch = ReadChar();
        }
        Add(0);
        if (!*element_.name) return Error("element name missing");

        // Skip whitespace after tag name.
        ch = SkipWhitespace(ch);

        // Read attributes.
        while (IsNameChar(ch)) {
          // Read attribute name.
          int n = element_.attrs.size();
          element_.attrs.emplace_back(bufptr_, nullptr);
          while (IsNameChar(ch)) {
            Add(ch);
            ch = ReadChar();
          }
          Add(0);
          if (!*element_.attrs[n].name) {
            return Error("attribute name missing");
          }

          // Skip whitespace after attribute name.
          ch = SkipWhitespace(ch);

          // Expect '=' after attribute name.
          if (ch != '=') return Error("'=' expected after attribute name");
          ch = ReadChar();

          // Skip whitespace after '='.
          ch = SkipWhitespace(ch);

          // Expect '"' after '='.
          if (ch != '"') return Error("'\"' expected before attribute value");
          ch = ReadChar();

          // Read attribute value.
          element_.attrs[n].value = bufptr_;
          while (ch != '"') {
            if (ch < 0) return Error("string not terminated");

            if (ch == '&') {
              // Process entity reference.
              entref.clear();
              entref.push_back('&');
              ch = ReadChar();
              while (ch >= 0 && ch != ';' && ch != '"') {
                entref.push_back(ch);
                ch = ReadChar();
              }
              entref.push_back(';');

              int code = ParseEntityRef(entref);
              if (code < 0) {
                AddString(entref.data());
              } else {
                char utf[UTF8::MAXLEN + 1];
                int utflen = UTF8::Encode(code, utf);
                utf[utflen] = 0;
                AddString(utf);
                ch = ReadChar();
              }
            } else {
              Add(ch);
              ch = ReadChar();
            }
          }
          Add(0);

          // Skip '"' at end of attribute value.
          ch = ReadChar();

          // Skip whitespace after attribute value.
          ch = SkipWhitespace(ch);
        }

        // Process end of tag after attributes.
        if (pi) {
          // Expect processing instruction tag to end with '?>'.
          if (ch != '?') {
            return Error("'?' expected to terminate processing instruction");
          }
          ch = ReadChar();
        } else if (ch == '/') {
          // Single tag, i.e. <tag/>.
          single = true;
          ch = ReadChar();
        }

        // Skip whitespace before '>'.
        ch = SkipWhitespace(ch);

        // Expect tag to end with '>'.
        if (ch != '>') return Error("'>' expected to terminate element");
        ch = ReadChar();

        // Process tag.
        if (pi) {
          // Output processing instruction.
          if (!ProcessingInstruction(element_)) return false;

          // Clear tag from heap buffer.
          bufptr_ = element_.name;
          element_.Clear();
        } else if (endtag) {
          // Attributes not allowed on end tags.
          if (!element_.attrs.empty()) {
            return Error("attributes not allowed in endtag");
          }

          // Check tag end tag matches start tag.
          if (stack_.empty()) return Error("unmatched end tag");
          char *starttag = stack_.back();
          stack_.pop_back();
          if (strcmp(element_.name, starttag) != 0) {
            return Error("end tag name does not match start tag");
          }

          // Output end element.
          if (!EndElement(element_.name)) return false;

          // Clear tag from heap buffer.
          bufptr_ = starttag;
          element_.Clear();
        } else if (single) {
          // Output start and end element.
          if (!StartElement(element_)) return false;
          if (!EndElement(element_.name)) return false;

          // Clear tag from heap buffer.
          bufptr_ = element_.name;
          element_.Clear();
        } else {
          // Add start tag to stack.
          stack_.push_back(element_.name);

          // Output start element.
          if (!StartElement(element_)) return false;

          // Clear attributes from heap buffer, but keep element name used by
          // stack.
          if (!element_.attrs.empty()) bufptr_ = element_.attrs[0].name;
          element_.Clear();
        }
      }
    }
  }

  // Check for unmatched tags.
  if (!stack_.empty()) return Error("end tag missing");

  // Mark end of document.
  if (!EndDocument()) return false;

  return true;
}

bool XMLParser::StartDocument() {
  return true;
}

bool XMLParser::EndDocument() {
  return true;
}

bool XMLParser::StartElement(const XMLElement &element) {
  return true;
}

bool XMLParser::EndElement(const char *name) {
  return true;
}

bool XMLParser::Text(const char *str) {
  return true;
}

bool XMLParser::Comment(const char *str) {
  return true;
}

bool XMLParser::ProcessingInstruction(const XMLElement &element) {
  return true;
}

}  // namespace sling

