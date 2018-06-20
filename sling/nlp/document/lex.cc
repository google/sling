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

#include "sling/nlp/document/lex.h"

#include <string>
#include <vector>

#include "sling/base/types.h"
#include "sling/frame/serialization.h"
#include "sling/nlp/document/document.h"
#include "sling/string/ctype.h"

namespace sling {
namespace nlp {

bool DocumentLexer::Lex(Document *document, Text lex) const {
  // Extract the plain text and frame source from the LEX-encoded text and keep
  // track of mention boundaries.
  string text;
  string source;
  std::vector<Markable> markables;
  std::vector<int> stack;
  std::vector<int> themes;
  int current_object = -1;
  int frame_level = 0;
  bool in_annotation = false;
  for (char c : lex) {
    if (frame_level > 0) {
      // Inside frame. Parse until outer '}' found.
      source.push_back(c);
      if (c == '{') {
        frame_level++;
      } else if (c == '}') {
        if (--frame_level == 0) {
          if (!in_annotation) {
            // Add frame as theme.
            themes.push_back(current_object);
          }
        }
      }
    } else {
      switch (c) {
        case '[':
          // Start new span.
          stack.push_back(markables.size());
          markables.emplace_back(text.size());
          break;
        case '|':
          // Start span annotation.
          if (in_annotation) return false;
          if (!stack.empty()) {
            // Enclose evoked frames in a list.
            in_annotation = true;
            current_object++;
            source.push_back('[');
            markables[stack.back()].object = current_object;
          } else {
            text.push_back(c);
          }
          break;
        case ']':
          // End current span.
          if (stack.empty()) return false;
          if (in_annotation) {
            source.push_back(']');
            in_annotation = false;
          }
          markables[stack.back()].end = text.size();
          stack.pop_back();
          break;
        case '{':
          if (stack.empty()) {
            // Start new thematic frame.
            current_object++;
          }
          source.push_back(c);
          frame_level++;
          break;
        default:
          if (in_annotation) {
            // Add character to frame source.
            source.push_back(c);
          } else {
            // Add character to plain text.
            text.push_back(c);
          }
      }
    }
  }

  if (!stack.empty()) return false;

  // Trim whitespace.
  int begin = 0;
  while (begin < text.size() && ascii_isspace(text[begin])) begin++;
  int end = text.size();
  while (end > begin + 1 && ascii_isspace(text[end - 1])) end--;
  if (begin > 0 || end < text.size()) {
    text = text.substr(begin, end - begin);
  }
  if (begin > 0) {
    for (Markable &m : markables) {
      m.begin -= begin;
      m.end -= begin;
    }
    end -= begin;
  }

  // Tokenize plain text and add tokens to document.
  tokenizer_->Tokenize(document, text);

  // Parse frames.
  Store *store = document->store();
  Handles objects(store);
  StringReader input(store, source);
  Reader *reader = input.reader();
  while (!reader->done()) {
    objects.push_back(reader->ReadObject());
    if (reader->error()) return false;
  }
  if (objects.size() != current_object + 1) return false;

  // Add mentions to document.
  for (auto &m : markables) {
    int begin = document->Locate(m.begin);
    int end = document->Locate(m.end);
    Span *span = document->AddSpan(begin, end);
    if (m.object != -1) {
      Array evoked(store, objects[m.object]);
      for (int i = 0; i < evoked.length(); ++i) {
        span->Evoke(evoked.get(i));
      }
    }
  }

  // Add thematic frames. Do not add frames that are evoked by spans.
  for (int theme : themes) {
    Handle frame = objects[theme];
    if (document->EvokingSpanCount(frame) == 0) {
      document->AddTheme(frame);
    }
  }

  // Update underlying document frame.
  document->Update();

  return true;
}

string ToLex(const Document &document) {
  // Set up frame printer for output.
  string lex;
  StringOutputStream stream(&lex);
  Output output(&stream);
  Printer printer(document.store(), &output);

  // Output all tokens with mentions and evoked frames.
  Handles evoked(document.store());
  for (const Token &token : document.tokens()) {
    // Add token break.
    if (token.index() > 0) {
      switch (token.brk()) {
        case NO_BREAK: break;
        case SPACE_BREAK: output.WriteChar(' '); break;
        case LINE_BREAK: output.WriteChar('\n'); break;
        case SENTENCE_BREAK: output.Write("  ", 2); break;
        case PARAGRAPH_BREAK:
        case SECTION_BREAK:
        case CHAPTER_BREAK:
          output.Write("\n\n", 2);
          break;
      }
    }

    // Add span open brackets.
    Span *span = document.GetSpanAt(token.index());
    for (Span *s = span; s != nullptr; s = s->parent()) {
      if (s->begin() == token.index()) output.WriteChar('[');
    }

    // Add token text.
    output.Write(token.text());

    // Add span close brackets.
    for (Span *s = span; s != nullptr; s = s->parent()) {
      if (s->end() == token.index() + 1) {
        bool first = true;
        s->AllEvoked(&evoked);
        for (Handle frame : evoked) {
          if (first) output.WriteChar('|');
          first = false;
          printer.Print(frame);
        }
        output.WriteChar(']');
      }
    }
  }

  // Output themes.
  for (Handle frame : document.themes()) {
    printer.Print(frame);
  }

  output.Flush();
  return lex;
}

}  // namespace nlp
}  // namespace sling

