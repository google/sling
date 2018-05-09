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

#ifndef SLING_UTIL_UNICODE_H
#define SLING_UTIL_UNICODE_H

#include <string.h>
#include <string>

#include "sling/base/types.h"

namespace sling {

extern const uint8 utf8_skip_tab[];
const int unicode_tab_size = 1 << 16;

// Unicode character categories.
enum UnicodeCategory {
  CHARCAT_UNASSIGNED                = 0,   // Cn
  CHARCAT_UPPERCASE_LETTER          = 1,   // Lu
  CHARCAT_LOWERCASE_LETTER          = 2,   // Ll
  CHARCAT_TITLECASE_LETTER          = 3,   // Lt
  CHARCAT_MODIFIER_LETTER           = 4,   // Lm
  CHARCAT_OTHER_LETTER              = 5,   // Lo
  CHARCAT_NON_SPACING_MARK          = 6,   // Mn
  CHARCAT_ENCLOSING_MARK            = 7,   // Me
  CHARCAT_COMBINING_SPACING_MARK    = 8,   // Mc
  CHARCAT_DECIMAL_DIGIT_NUMBER      = 9,   // Nd
  CHARCAT_LETTER_NUMBER             = 10,  // Nl
  CHARCAT_OTHER_NUMBER              = 11,  // No
  CHARCAT_SPACE_SEPARATOR           = 12,  // Zs
  CHARCAT_LINE_SEPARATOR            = 13,  // Zl
  CHARCAT_PARAGRAPH_SEPARATOR       = 14,  // Zp
  CHARCAT_CONTROL                   = 15,  // Cc
  CHARCAT_FORMAT                    = 16,  // Cf
  CHARCAT_CASED_LETTER              = 17,  // Lc
  CHARCAT_PRIVATE_USE               = 18,  // Co
  CHARCAT_SURROGATE                 = 19,  // Cs
  CHARCAT_DASH_PUNCTUATION          = 20,  // Pd
  CHARCAT_START_PUNCTUATION         = 21,  // Ps
  CHARCAT_END_PUNCTUATION           = 22,  // Pe
  CHARCAT_CONNECTOR_PUNCTUATION     = 23,  // Pc
  CHARCAT_OTHER_PUNCTUATION         = 24,  // Po
  CHARCAT_MATH_SYMBOL               = 25,  // Sm
  CHARCAT_CURRENCY_SYMBOL           = 26,  // Sc
  CHARCAT_MODIFIER_SYMBOL           = 27,  // Sk
  CHARCAT_OTHER_SYMBOL              = 28,  // So
  CHARCAT_INITIAL_QUOTE_PUNCTUATION = 29,  // Pi
  CHARCAT_FINAL_QUOTE_PUNCTUATION   = 30,  // Pf

  CHARCAT_MASK                      = 0x1F,
  CHARBIT_WHITESPACE                = 0x80,
};

// String normalization flags.
enum NormalizationFlags {
  NORMALIZE_LETTERS     = 0x01,  // lowercase and remove diacritics
  NORMALIZE_DIGITS      = 0x02,  // replace all digits with 9
  NORMALIZE_PUNCTUATION = 0x04,  // remove punctuation
};

class Unicode {
 public:
   // Return Unicode category for code point.
   static int Category(int c);

   // Check if code point is lower case.
   static bool IsLower(int c);

   // Check if code point is upper case.
   static bool IsUpper(int c);

   // Check if code point is title case.
   static bool IsTitle(int c);

   // Check if code point is a digit.
   static bool IsDigit(int c);

   // Check if code point is defined.
   static bool IsDefined(int c);

   // Check if code point is a letter.
   static bool IsLetter(int c);

   // Check if code point is a letter or digit.
   static bool IsLetterOrDigit(int c);

   // Check if code point is a space.
   static bool IsSpace(int c);

   // Check if code point is whitespace.
   static bool IsWhitespace(int c);

   // Check if code point is punctuation.
   static bool IsPunctuation(int c);

   // Convert code point to lower case.
   static int ToLower(int c);

   // Convert code point to upper case.
   static int ToUpper(int c);

   // Normalize code point to by lowercasing and removing punctuation and
   // diacritics. Return zero for code points that should be removed.
   static int Normalize(int c);
   static int Normalize(int c, NormalizationFlags flags);
};

class UTF8 {
 public:
  // Maximum length of UTF8 encoded code point.
  static const int MAXLEN = 4;

  // Return the length of the UTF8 string in characters.
  static int Length(const char *s, int len);
  static int Length(const char *s) { return Length(s, strlen(s)); }
  static int Length(const string &s) { return Length(s.data(), s.size()); }

  // Return the length of the next UTF8 character.
  static int CharLen(const char *s) {
    return utf8_skip_tab[*reinterpret_cast<const uint8 *>(s)];
  }

  // Return pointer to next UTF8 character in string.
  static const char *Next(const char *s) { return s + CharLen(s); }
  static char *Next(char *s) { return s + CharLen(s); }

  // Return pointer to previous UTF8 character in string.
  static const char *Previous(const char *s, const char *limit = nullptr);
  static char *Previous(char *s, char *limit = nullptr);;

  // Check if string is structurally valid.
  static bool Valid(const char *s, int len);
  static bool Valid(const string &s) {
    return Valid(s.data(), s.length());
  }

  // Return next UTF8 code point in string. Returns -1 on errors.
  static int Decode(const char *s, int len);
  static int Decode(const char *s);

  // Encode one Unicode code point to at most UTF8::MAXLEN bytes and return
  // the number of bytes generated.
  static int Encode(int code, char *s);

  // Encode one Unicode point and append it to string.
  static int Encode(int code, string *str);

  // Lowercase UTF8 encoded string.
  static void Lowercase(const char *s, int len, string *result);
  static void Lowercase(const string &str, string *result) {
    Lowercase(str.data(), str.size(), result);
  }

  // Uppercase UTF8 encoded string.
  static void Uppercase(const char *s, int len, string *result);
  static void Uppercase(const string &str, string *result) {
    Uppercase(str.data(), str.size(), result);
  }

  // Normalize UTF8 encoded string for matching.
  static void Normalize(const char *s, int len, NormalizationFlags flags,
                        string *normalized);
  static void Normalize(const string &str, NormalizationFlags flags,
                        string *normalized) {
    Normalize(str.data(), str.size(), flags, normalized);
  }
  static void Normalize(const char *s, int len, string *normalized) {
    Normalize(s, len, NORMALIZE_LETTERS, normalized);
  }
  static void Normalize(const string &str, string *normalized) {
    Normalize(str.data(), str.size(), normalized);
  }

  // Convert string to title case, i.e. make the first letter uppercase.
  static void ToTitleCase(const string &str, string *titlecased);

  // Check if all characters are punctuation characters.
  static bool IsPunctuation(const char *s, int len);
  static bool IsPunctuation(const string &str) {
    return IsPunctuation(str.data(), str.size());
  }
};

}  // namespace sling

#endif  // SLING_UTIL_UNICODE_H

