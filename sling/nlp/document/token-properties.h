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

#ifndef SLING_NLP_DOCUMENT_TOKEN_PROPERTIES_H_
#define SLING_NLP_DOCUMENT_TOKEN_PROPERTIES_H_

namespace sling {
namespace nlp {

// Token break types.
enum BreakType {
  NO_BREAK         = 0,
  SPACE_BREAK      = 1,
  LINE_BREAK       = 2,
  SENTENCE_BREAK   = 3,
  PARAGRAPH_BREAK  = 4,
  SECTION_BREAK    = 5,
  CHAPTER_BREAK    = 6,
};

// Token style attributes.
enum TokenStyles {
  // No formatting (default).
  STYLE_NORMAL           = 0,

  // Bold typeface.
  STYLE_BOLD_BEGIN       = 1,
  STYLE_BOLD_END         = 2,

  // Italic typeface.
  STYLE_ITALIC_BEGIN     = 3,
  STYLE_ITALIC_END       = 4,

  // Section headings.
  STYLE_HEADING_BEGIN    = 5,
  STYLE_HEADING_END      = 6,

  // Itemized lists.
  STYLE_ITEMIZE_BEGIN    = 7,
  STYLE_ITEMIZE_END      = 8,

  // List items.
  STYLE_LISTITEM_BEGIN   = 9,
  STYLE_LISTITEM_END     = 10,

  // Block quotes.
  STYLE_QUOTE_BEGIN      = 11,
  STYLE_QUOTE_END        = 12,
};

// Token style flags.
enum TokenStyleFlags {
  // Flag masks for styles.
  BOLD_BEGIN       = (1 << STYLE_BOLD_BEGIN),
  BOLD_END         = (1 << STYLE_BOLD_END),
  ITALIC_BEGIN     = (1 << STYLE_ITALIC_BEGIN),
  ITALIC_END       = (1 << STYLE_ITALIC_END),
  HEADING_BEGIN    = (1 << STYLE_HEADING_BEGIN),
  HEADING_END      = (1 << STYLE_HEADING_END),
  ITEMIZE_BEGIN    = (1 << STYLE_ITEMIZE_BEGIN),
  ITEMIZE_END      = (1 << STYLE_ITEMIZE_END),
  LISTITEM_BEGIN   = (1 << STYLE_LISTITEM_BEGIN),
  LISTITEM_END     = (1 << STYLE_LISTITEM_END),
  QUOTE_BEGIN      = (1 << STYLE_QUOTE_BEGIN),
  QUOTE_END        = (1 << STYLE_QUOTE_END),

  // Mask for begin and end styles.
  BEGIN_STYLE = BOLD_BEGIN |
                ITALIC_BEGIN |
                HEADING_BEGIN |
                ITEMIZE_BEGIN |
                LISTITEM_BEGIN |
                QUOTE_BEGIN,
  END_STYLE   = BOLD_END |
                ITALIC_END |
                HEADING_END |
                ITEMIZE_END |
                LISTITEM_END |
                QUOTE_END,
};

}  // namespace nlp
}  // namespace sling

#endif  // SLING_NLP_DOCUMENT_TOKEN_PROPERTIES_H_

