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

#include "sling/util/unicode.h"

#include <string>

#include "sling/base/types.h"

namespace sling {

// Unicode category and conversion tables defined in unicodetab.cc.
extern uint8 unicode_cat_tab[unicode_tab_size];
extern uint16 unicode_upper_tab[unicode_tab_size];
extern uint16 unicode_lower_tab[unicode_tab_size];
extern uint16 unicode_normalize_tab[unicode_tab_size];
const int unicode_tab_mask = ~(unicode_tab_size - 1);

// UTF8 character length based on lead byte.
const uint8 utf8_skip_tab[256] = {
  1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
  1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
  1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
  1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
  1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
  1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
  2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,
  3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,4,4,4,4,4,4,4,4,5,5,5,5,6,6,1,1,
};

int Unicode::Category(int c) {
  if (c & unicode_tab_mask) return CHARCAT_UNASSIGNED;
  return (unicode_cat_tab[c] & CHARCAT_MASK);
}

bool Unicode::IsLower(int c) {
  if (c & unicode_tab_mask) return false;
  return (unicode_cat_tab[c] & CHARCAT_MASK) == CHARCAT_LOWERCASE_LETTER;
}

bool Unicode::IsUpper(int c) {
  if (c & unicode_tab_mask) return false;
  return (unicode_cat_tab[c] & CHARCAT_MASK) == CHARCAT_UPPERCASE_LETTER;
}

bool Unicode::IsTitle(int c) {
  if (c & unicode_tab_mask) return false;
  return (unicode_cat_tab[c] & CHARCAT_MASK) == CHARCAT_TITLECASE_LETTER;
}

bool Unicode::IsDigit(int c) {
  if (c & unicode_tab_mask) return false;
  return (unicode_cat_tab[c] & CHARCAT_MASK) == CHARCAT_DECIMAL_DIGIT_NUMBER;
}

bool Unicode::IsDefined(int c) {
  if (c & unicode_tab_mask) return false;
  return (unicode_cat_tab[c] & CHARCAT_MASK) != CHARCAT_UNASSIGNED;
}

bool Unicode::IsLetter(int c) {
  static const int letter_mask =
    (1 << CHARCAT_UPPERCASE_LETTER) |
    (1 << CHARCAT_LOWERCASE_LETTER) |
    (1 << CHARCAT_TITLECASE_LETTER) |
    (1 << CHARCAT_MODIFIER_LETTER) |
    (1 << CHARCAT_OTHER_LETTER);

  if (c & unicode_tab_mask) return false;
  int category = unicode_cat_tab[c] & CHARCAT_MASK;
  return ((1 << category) & letter_mask) != 0;
}

bool Unicode::IsLetterOrDigit(int c) {
  static const int letter_digit_mask =
    (1 << CHARCAT_UPPERCASE_LETTER) |
    (1 << CHARCAT_LOWERCASE_LETTER) |
    (1 << CHARCAT_TITLECASE_LETTER) |
    (1 << CHARCAT_MODIFIER_LETTER) |
    (1 << CHARCAT_OTHER_LETTER) |
    (1 << CHARCAT_DECIMAL_DIGIT_NUMBER);

  if (c & unicode_tab_mask) return false;
  int category = unicode_cat_tab[c] & CHARCAT_MASK;
  return ((1 << category) & letter_digit_mask) != 0;
}

bool Unicode::IsSpace(int c)  {
  static const int space_mask =
    (1 << CHARCAT_SPACE_SEPARATOR) |
    (1 << CHARCAT_LINE_SEPARATOR) |
    (1 << CHARCAT_PARAGRAPH_SEPARATOR);

  if (c & unicode_tab_mask) return false;
  int category = unicode_cat_tab[c] & CHARCAT_MASK;
  return ((1 << category) & space_mask) != 0;
}

bool Unicode::IsWhitespace(int c) {
  if (c & unicode_tab_mask) return false;
  return (unicode_cat_tab[c] & CHARBIT_WHITESPACE) != 0;
}

bool Unicode::IsPunctuation(int c) {
  static const int punctuation_mask =
    (1 << CHARCAT_DASH_PUNCTUATION) |
    (1 << CHARCAT_START_PUNCTUATION) |
    (1 << CHARCAT_END_PUNCTUATION) |
    (1 << CHARCAT_CONNECTOR_PUNCTUATION) |
    (1 << CHARCAT_OTHER_PUNCTUATION) |
    (1 << CHARCAT_INITIAL_QUOTE_PUNCTUATION) |
    (1 << CHARCAT_FINAL_QUOTE_PUNCTUATION) |
    (1 << CHARCAT_MODIFIER_SYMBOL) |
    (1 << CHARCAT_OTHER_SYMBOL);

  if (c & unicode_tab_mask) return false;
  int category = unicode_cat_tab[c] & CHARCAT_MASK;
  return ((1 << category) & punctuation_mask) != 0;
}

int Unicode::ToLower(int c) {
  if (c & unicode_tab_mask) return c;
  return unicode_lower_tab[c];
}

int Unicode::ToUpper(int c) {
  if (c & unicode_tab_mask) return c;
  return unicode_upper_tab[c];
}

int Unicode::Normalize(int c) {
  if (c & unicode_tab_mask) return c;
  return unicode_normalize_tab[c];
}

int Unicode::Normalize(int c, NormalizationFlags flags) {
  if (c & unicode_tab_mask) return c;
  if (flags & NORMALIZE_LETTERS) {
    c = unicode_normalize_tab[c];
  }
  if (flags & NORMALIZE_DIGITS) {
    if (IsDigit(c)) c = '9';
  }
  if (flags & NORMALIZE_PUNCTUATION) {
    if (IsPunctuation(c)) c = 0;
  }
  return c;
}

int UTF8::Length(const char *s, int len) {
  const char *end = s + len;
  int n = 0;
  while (s < end) {
    if ((*s++ & 0xc0) != 0x80) n++;
  }
  return n;
}

const char *UTF8::Previous(const char *s, const char *limit) {
  while (s > limit) {
    if ((*--s & 0xc0) != 0x80) break;
  }
  return s;
}

char *UTF8::Previous(char *s, char *limit) {
  while (s > limit) {
    if ((*--s & 0xc0) != 0x80) break;
  }
  return s;
}

bool UTF8::Valid(const char *s, int len) {
  const uint8 *p = reinterpret_cast<const uint8 *>(s);
  const uint8 *end = p + len;
  while (p < end) {
    int c = *p++;
    if ((c & 0x80) == 0) continue;
    int len;
    if ((c & 0xe0) == 0xc0) {
      len = 1;
    } else if ((c & 0xf0) == 0xe0) {
      len = 2;
    } else if ((c & 0xf8) == 0xf0) {
      len = 3;
    } else {
      return false;
    }

    const uint8 *next = p + len;
    if (next > end) return false;
    while (p < next) {
      if ((*p & 0xc0) != 0x80) return false;
      p++;
    }
  }
  return true;
}

int UTF8::Decode(const char *s, int len) {
  // No more data.
  if (len <= 0) return -1;

  // One character sequence (7-bit value).
  int c0 = *reinterpret_cast<const uint8 *>(s);
  if (c0 < 0x80) return c0;
  if (len <= 1) return -1;

  // Two character sequence (11-bit value).
  int c1 = *reinterpret_cast<const uint8 *>(s + 1) ^ 0x80;
  if (c1 & 0xc0) return -1;
  if (c0 < 0xe0) {
    if (c0 < 0xc0) return -1;
    int code = ((c0 << 6) | c1) & 0x07ff;
    if (code <= 0x7f) return -1;
    return code;
  }
  if (len <= 2) return -1;

  // Three character sequence (16-bit value).
  int c2 = *reinterpret_cast<const uint8 *>(s + 2) ^ 0x80;
  if (c2 & 0xc0) return -1;
  if (c0 < 0xf0) {
    int code = ((((c0 << 6) | c1) << 6) | c2) & 0xffff;
    if (code <= 0x07ff) return -1;
    return code;
  }
  if (len <= 3) return -1;

  // Four character sequence (21-bit value).
  int c3 = *reinterpret_cast<const uint8 *>(s + 3) ^ 0x80;
  if (c3 & 0xc0) return -1;
  if (c0 < 0xf8) {
    int code = ((((((c0 << 6) | c1) << 6) | c2) << 6) | c3) & 0x001fffff;
    if (code <= 0xffff) return -1;
    return code;
  }

  return -1;
}

int UTF8::Decode(const char *s) {
  // One character sequence (7-bit value).
  int c0 = *reinterpret_cast<const uint8 *>(s);
  if (c0 < 0x80) return c0;

  // Two character sequence (11-bit value).
  int c1 = *reinterpret_cast<const uint8 *>(s + 1) ^ 0x80;
  if (c1 & 0xc0) return -1;
  if (c0 < 0xe0) {
    if (c0 < 0xc0) return -1;
    int code = ((c0 << 6) | c1) & 0x07ff;
    if (code <= 0x7f) return -1;
    return code;
  }

  // Three character sequence (16-bit value).
  int c2 = *reinterpret_cast<const uint8 *>(s + 2) ^ 0x80;
  if (c2 & 0xc0) return -1;
  if (c0 < 0xf0) {
    int code = ((((c0 << 6) | c1) << 6) | c2) & 0xffff;
    if (code <= 0x07ff) return -1;
    return code;
  }

  // Four character sequence (21-bit value).
  int c3 = *reinterpret_cast<const uint8 *>(s + 3) ^ 0x80;
  if (c3 & 0xc0) return -1;
  if (c0 < 0xf8) {
    int code = ((((((c0 << 6) | c1) << 6) | c2) << 6) | c3) & 0x001fffff;
    if (code <= 0xffff) return -1;
    return code;
  }

  return -1;
}

int UTF8::Encode(int code, char *s) {
  uint32 c = code;

  // One character sequence.
  if (c <= 0x7f) {
    s[0] = c;
    return 1;
  }

  // Two character sequence.
  if (c <= 0x7ff) {
    s[0] = 0xc0 | (c >> 6);
    s[1] = 0x80 | (c & 0x3f);
    return 2;
  }

  // Three character sequence.
  if (c <= 0xffff) {
    s[0] = 0xe0 | (c >> 12);
    s[1] = 0x80 | ((c >> 6) & 0x3f);
    s[2] = 0x80 | (c & 0x3f);
    return 3;
  }

  // Four character sequence.
  s[0] = 0xf0 | (c >> 18);
  s[1] = 0x80 | ((c >> 12) & 0x3f);
  s[2] = 0x80 | ((c >> 6) & 0x3f);
  s[3] = 0x80 | (c & 0x3f);
  return 4;
}

int UTF8::Encode(int code, string *str) {
  uint32 c = code;

  // One character sequence.
  if (c <= 0x7f) {
    str->push_back(c);
    return 1;
  }

  // Two character sequence.
  if (c <= 0x7ff) {
    str->push_back(0xc0 | (c >> 6));
    str->push_back(0x80 | (c & 0x3f));
    return 2;
  }

  // Three character sequence.
  if (c <= 0xffff) {
    str->push_back(0xe0 | (c >> 12));
    str->push_back(0x80 | ((c >> 6) & 0x3f));
    str->push_back(0x80 | (c & 0x3f));
    return 3;
  }

  // Four character sequence.
  str->push_back(0xf0 | (c >> 18));
  str->push_back(0x80 | ((c >> 12) & 0x3f));
  str->push_back(0x80 | ((c >> 6) & 0x3f));
  str->push_back(0x80 | (c & 0x3f));
  return 4;
}

void UTF8::Uppercase(const char *s, int len, string *result) {
  // Clear output string.
  result->clear();
  result->reserve(len);

  // Try fast conversion where all characters are below 128. All characters
  // below 128 are converted to one byte codes.
  const char *end = s + len;
  while (s < end) {
    uint8 c = *reinterpret_cast<const uint8 *>(s);
    if (c & 0x80) break;
    result->push_back(unicode_upper_tab[c]);
    s++;
  }

  // Handle any remaining part of the string which can contain multi-byte
  // characters.
  while (s < end) {
    int code = Decode(s);
    int upper = Unicode::ToUpper(code);
    Encode(upper, result);
    s = Next(s);
  }
}

void UTF8::Lowercase(const char *s, int len, string *result) {
  // Clear output string.
  result->clear();
  result->reserve(len);

  // Try fast conversion where all characters are below 128. All characters
  // below 128 are converted to one byte codes.
  const char *end = s + len;
  while (s < end) {
    uint8 c = *reinterpret_cast<const uint8 *>(s);
    if (c & 0x80) break;
    result->push_back(unicode_lower_tab[c]);
    s++;
  }

  // Handle any remaining part of the string which can contain multi-byte
  // characters.
  while (s < end) {
    int code = Decode(s);
    int lower = Unicode::ToLower(code);
    Encode(lower, result);
    s = Next(s);
  }
}

void UTF8::Normalize(const char *s, int len, NormalizationFlags flags,
                     string *normalized) {
  // Clear output string.
  normalized->clear();

  // Try fast conversion where all characters are below 128. All characters
  // below 128 are normalized to one byte codes.
  const char *end = s + len;
  while (s < end) {
    uint8 c = *reinterpret_cast<const uint8 *>(s);
    if (c & 0x80) break;
    int ch = Unicode::Normalize(c, flags);
    if (ch > 0) normalized->push_back(ch);
    s++;
  }

  // Handle any remaining part of the string which can contain multi-byte
  // characters.
  while (s < end) {
    int ch = Unicode::Normalize(Decode(s), flags);
    if (ch > 0) Encode(ch, normalized);
    s = Next(s);
  }
}

void UTF8::ToTitleCase(const string &str, string *titlecased) {
  titlecased->clear();
  if (str.empty()) return;
  const char *s = str.data();
  int initial = Decode(s);
  if (Unicode::IsLower(initial)) {
    // Convert first character to upper case.
    Encode(Unicode::ToUpper(initial), titlecased);

    // Copy rest of string.
    titlecased->append(str, CharLen(s), -1);
  } else {
    // Just copy string.
    titlecased->assign(str);
  }
}

bool UTF8::IsPunctuation(const char *s, int len) {
  // Try fast check where all characters are below 128.
  const char *end = s + len;
  while (s < end) {
    uint8 c = *reinterpret_cast<const uint8 *>(s);
    if (c & 0x80) break;
    if (!Unicode::IsPunctuation(c)) return false;
    s++;
  }

  // Handle any remaining part of the string which can contain multi-byte
  // characters.
  while (s < end) {
    int code = Unicode::Normalize(Decode(s));
    if (!Unicode::IsPunctuation(code)) return false;
    s = Next(s);
  }

  return true;
}

}  // namespace sling

