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

#include "sling/nlp/document/lexicon.h"

#include "sling/base/types.h"
#include "sling/nlp/document/affix.h"
#include "sling/stream/memory.h"
#include "sling/util/vocabulary.h"

namespace sling {
namespace nlp {

void WordShape::Extract(const string &word) {
  quote = NO_QUOTE;
  hyphen = NO_HYPHEN;
  bool has_upper = false;
  bool has_lower = false;
  bool has_punctuation = false;
  bool all_punctuation = !word.empty();
  bool has_digit = false;
  bool all_digit = !word.empty();
  const char *p = word.data();
  const char *end = p + word.size();
  while (p < end) {
    int code = UTF8::Decode(p);
    int cat = Unicode::Category(code);

    // Hyphenation.
    if (cat == CHARCAT_DASH_PUNCTUATION) hyphen = HAS_HYPHEN;

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
        quote = OPEN_QUOTE;
        break;
      case CHARCAT_FINAL_QUOTE_PUNCTUATION:
        quote = CLOSE_QUOTE;
        break;
      case CHARCAT_OTHER_PUNCTUATION:
        if (code == '\'' || code == '"') quote = UNKNOWN_QUOTE;
        break;
      case CHARCAT_MODIFIER_SYMBOL:
        if (code == '`') quote = UNKNOWN_QUOTE;
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
    capitalization = LOWERCASE;
  } else if (has_upper && !has_lower) {
    capitalization = UPPERCASE;
  } else if (!has_upper && !has_lower) {
    capitalization = NON_ALPHABETIC;
  } else {
    capitalization = CAPITALIZED;
  }

  // Compute punctuation feature.
  if (all_punctuation) {
    punctuation = ALL_PUNCTUATION;
  } else if (has_punctuation) {
    punctuation = SOME_PUNCTUATION;
  } else {
    punctuation = NO_PUNCTUATION;
  }

  // Compute quote feature.
  if (quote != NO_QUOTE) {
    // Penn Treebank open and close quotes are multi-character.
    if (word == "``") quote = OPEN_QUOTE;
    if (word == "''") quote = CLOSE_QUOTE;
  }

  // Compute digit feature.
  if (all_digit) {
    digit = ALL_DIGIT;
  } else if (has_digit) {
    digit = SOME_DIGIT;
  } else {
    digit = NO_DIGIT;
  }
}

void Lexicon::InitWords(Vocabulary::Iterator *words) {
  // Initialize mapping from words to ids.
  vocabulary_.Init(words);

  // Initialize mapping from ids to words.
  words_.resize(words->Size());
  int index = 0;
  words->Reset();
  Text word;
  while (words->Next(&word, nullptr)) {
    words_[index].word.assign(word.data(), word.size());
    index++;
  }
}

void Lexicon::WriteVocabulary(string *buffer, char terminator) const {
  for (const Entry &e : words_) {
    buffer->append(e.word);
    buffer->push_back(terminator);
  }
}

void Lexicon::InitPrefixes(const char *data, size_t size) {
  // Read prefixes.
  ArrayInputStream stream(data, size);
  prefixes_.Read(&stream);

  // Pre-compute the longest prefix for all words in lexicon.
  for (Entry &entry : words_) {
    entry.prefix = prefixes_.GetLongestAffix(entry.word);
  }
}

void Lexicon::InitSuffixes(const char *data, size_t size) {
  // Read suffixes.
  ArrayInputStream stream(data, size);
  suffixes_.Read(&stream);

  // Pre-compute the longest suffix for all words in lexicon.
  for (Entry &entry : words_) {
    entry.suffix = suffixes_.GetLongestAffix(entry.word);
  }
}

void Lexicon::WritePrefixes(string *buffer) const {
  StringOutputStream stream(buffer);
  prefixes_.Write(&stream);
}

void Lexicon::WriteSuffixes(string *buffer) const {
  StringOutputStream stream(buffer);
  suffixes_.Write(&stream);
}

void Lexicon::BuildPrefixes(int max_prefix) {
  prefixes_.Reset(max_prefix);
  prefixes_.AddAffixesForWord("");
  if (max_prefix > 0) {
    for (Entry &entry : words_) {
      entry.prefix = prefixes_.AddAffixesForWord(entry.word);
    }
  }
}

void Lexicon::BuildSuffixes(int max_suffix) {
  suffixes_.Reset(max_suffix);
  suffixes_.AddAffixesForWord("");
  if (max_suffix > 0) {
    for (Entry &entry : words_) {
      entry.suffix = suffixes_.AddAffixesForWord(entry.word);
    }
  }
}

void Lexicon::PrecomputeShapes() {
  for (Entry &entry : words_) {
    entry.shape.Extract(entry.word);
  }
}

int Lexicon::Lookup(const string &word,
                    Affix **prefix, Affix **suffix,
                    WordShape *shape) const {
  // Normalize word.
  string normalized;
  UTF8::Normalize(word, normalization_, &normalized);

  // Look up word in vocabulary.
  int id = vocabulary_.Lookup(normalized);

  // Return pre-computed information from the lexicon for known words.
  if (id != -1) {
    const Entry &entry = words_[id];
    *prefix = entry.prefix;
    *suffix = entry.suffix;
    *shape = entry.shape;

    // Due to normalization of the words in the lexicon, some of the
    // pre-computed features need to be re-computed.
    if (normalization_ & NORMALIZE_CASE) {
      // Re-compute capitalization for case-normalized lexicon.
      bool has_upper = false;
      bool has_lower = false;
      const char *p = word.data();
      const char *end = p + word.size();
      while (p < end) {
        int code = UTF8::Decode(p);
        if (Unicode::IsUpper(code)) has_upper = true;
        if (Unicode::IsLower(code)) has_lower = true;
        p = UTF8::Next(p);
      }

      if (!has_upper && has_lower) {
        shape->capitalization = WordShape::LOWERCASE;
      } else if (has_upper && !has_lower) {
        shape->capitalization = WordShape::UPPERCASE;
      } else if (!has_upper && !has_lower) {
        shape->capitalization = WordShape::NON_ALPHABETIC;
      } else {
        shape->capitalization = WordShape::CAPITALIZED;
      }
    }
    if (normalization_ & NORMALIZE_PUNCTUATION) {
      // Re-compute punctuation for punctuation-normalized lexicon.
      bool has_punctuation = false;
      bool all_punctuation = !word.empty();
      const char *p = word.data();
      const char *end = p + word.size();
      while (p < end) {
        int code = UTF8::Decode(p);
        bool is_punct = Unicode::IsPunctuation(code);
        all_punctuation &= is_punct;
        has_punctuation |= is_punct;
        p = UTF8::Next(p);
      }

      if (all_punctuation) {
        shape->punctuation = WordShape::ALL_PUNCTUATION;
      } else if (has_punctuation) {
        shape->punctuation = WordShape::SOME_PUNCTUATION;
      } else {
        shape->punctuation = WordShape::NO_PUNCTUATION;
      }
    }

    return id;
  }

  // Compute affixes and shape features on-the-fly for unknown words.
  if (prefixes_.max_length() > 0) {
    *prefix = prefixes_.GetLongestAffix(normalized);
  } else {
    *prefix = nullptr;
  }
  if (suffixes_.max_length() > 0) {
    *suffix = suffixes_.GetLongestAffix(normalized);
  } else {
    *suffix = nullptr;
  }
  shape->Extract(word);
  return oov_;
}

int Lexicon::Lookup(const string &word) const {
  // Look up word in vocabulary.
  int id = vocabulary_.Lookup(word);
  return id == -1 ? oov_ : id;
}

}  // namespace nlp
}  // namespace sling

