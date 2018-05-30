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

#include "sling/nlp/document/features.h"

#include "sling/base/types.h"
#include "sling/nlp/document/document.h"
#include "sling/util/unicode.h"

namespace sling {
namespace nlp {

void DocumentFeatures::Extract(const Document &document, int begin, int end) {
  if (end == -1) end = document.num_tokens();
  int length = end - begin;
  features_.resize(length);
  bool extract_prefixes = lexicon_->prefixes().size() != 0;
  bool extract_suffixes = lexicon_->suffixes().size() != 0;
  int oov = lexicon_->oov();
  bool in_quote = false;
  for (int i = 0; i < length; ++i) {
    string word = document.token(begin + i).text();
    TokenFeatures &f = features_[i];
    f.hyphen = NO_HYPHEN;
    f.quote = NO_QUOTE;

    // Look up word in lexicon.
    bool changed = false;
    f.word = lexicon_->LookupWord(word, &changed);

    // Look up longest prefix.
    if (extract_prefixes) {
      if (f.word != oov && !changed) {
        f.prefix = lexicon_->prefix(f.word);
      } else {
        f.prefix = lexicon_->prefixes().GetLongestAffix(word);
      }
    }

    // Look up longest suffix.
    if (extract_suffixes) {
      if (f.word != oov && !changed) {
        f.suffix = lexicon_->suffix(f.word);
      } else {
        f.suffix = lexicon_->suffixes().GetLongestAffix(word);
      }
    }

    // Categorize token.
    bool has_upper = false;
    bool has_lower = false;
    bool has_punctuation = false;
    bool all_punctuation = true;
    bool has_digit = false;
    bool all_digit = true;
    const char *p = word.data();
    const char *end = p + word.size();
    while (p < end) {
      int code = UTF8::Decode(p);
      int cat = Unicode::Category(code);

      // Hyphenation.
      if (cat == CHARCAT_DASH_PUNCTUATION) {
        f.hyphen = HAS_HYPHEN;
      }

      // Capitalization.
      if (Unicode::IsUpper(code)) has_upper = true;
      if (Unicode::IsLower(code)) has_lower = true;

      // Punctuation.
      bool is_punct = Unicode::IsPunctuation(code);
      all_punctuation &= is_punct;
      has_punctuation |= is_punct;

      // Quotes.
      switch (cat) {
        case CHARCAT_INITIAL_QUOTE_PUNCTUATION:
          f.quote = OPEN_QUOTE;
          break;
        case CHARCAT_FINAL_QUOTE_PUNCTUATION:
          f.quote = CLOSE_QUOTE;
          break;
        case CHARCAT_OTHER_PUNCTUATION:
          if (code == '\'' || code == '"') f.quote = UNKNOWN_QUOTE;
          break;
        case CHARCAT_MODIFIER_SYMBOL:
          if (code == '`') f.quote = UNKNOWN_QUOTE;
          break;
      }

      // Digits.
      bool is_digit = Unicode::IsDigit(code);
      all_digit &= is_digit;
      has_digit |= is_digit;

      p = UTF8::Next(p);
    }

    // Compute word capitalization.
    if (!has_upper && has_lower) {
      f.capitalization = LOWERCASE;
    } else if (has_upper && !has_lower) {
      f.capitalization = UPPERCASE;
    } else if (!has_upper && !has_lower) {
      f.capitalization = NON_ALPHABETIC;
    } else if (i == 0 || document.token(i).brk() >= SENTENCE_BREAK) {
      f.capitalization = INITIAL;
    } else {
      f.capitalization = CAPITALIZED;
    }

    // Compute punctuation feature.
    if (all_punctuation) {
      f.punctuation = ALL_PUNCTUATION;
    } else if (has_punctuation) {
      f.punctuation = SOME_PUNCTUATION;
    } else {
      f.punctuation = NO_PUNCTUATION;
    }

    // Compute quote feature.
    if (f.quote != NO_QUOTE) {
      // Penn Treebank open and close quotes are multi-character.
      if (word == "``") f.quote = OPEN_QUOTE;
      if (word == "''") f.quote = CLOSE_QUOTE;
      if (f.quote == UNKNOWN_QUOTE) {
        f.quote = in_quote ? CLOSE_QUOTE : OPEN_QUOTE;
        in_quote = !in_quote;
      }
    }

    // Compute digit feature.
    if (all_digit) {
      f.digit = ALL_DIGIT;
    } else if (has_digit) {
      f.digit = SOME_DIGIT;
    } else {
      f.digit = NO_DIGIT;
    }
  }
}

}  // namespace nlp
}  // namespace sling

