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

#ifndef SLING_NLP_SILVER_MENTIONS_H_
#define SLING_NLP_SILVER_MENTIONS_H_

#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>

#include "sling/base/types.h"
#include "sling/frame/store.h"
#include "sling/nlp/document/document.h"
#include "sling/nlp/document/phrase-tokenizer.h"
#include "sling/nlp/kb/facts.h"
#include "sling/nlp/kb/phrase-table.h"
#include "sling/nlp/kb/resolver.h"
#include "sling/nlp/silver/chart.h"
#include "sling/nlp/silver/idf.h"

namespace sling {
namespace nlp {

// Span categorization flags.
enum SpanFlags {
  SPAN_EMPHASIS          = (1 << 0),
  SPAN_NUMBER            = (1 << 1),
  SPAN_NATURAL_NUMBER    = (1 << 2),
  SPAN_UNIT              = (1 << 3),
  SPAN_UNIT_OF_AMOUNT    = (1 << 4),
  SPAN_CURRENCY          = (1 << 5),
  SPAN_MEASURE           = (1 << 6),
  SPAN_GEO               = (1 << 7),
  SPAN_YEAR              = (1 << 8),
  SPAN_YEAR_BC           = (1 << 9),
  SPAN_MONTH             = (1 << 10),
  SPAN_WEEKDAY           = (1 << 11),
  SPAN_CALENDAR_MONTH    = (1 << 12),
  SPAN_CALENDAR_DAY      = (1 << 13),
  SPAN_DAY_OF_YEAR       = (1 << 14),
  SPAN_DECADE            = (1 << 15),
  SPAN_CENTURY           = (1 << 16),
  SPAN_DATE              = (1 << 17),
  SPAN_FAMILY_NAME       = (1 << 18),
  SPAN_GIVEN_NAME        = (1 << 19),
  SPAN_INITIALS          = (1 << 20),
  SPAN_DASH              = (1 << 21),
  SPAN_SUFFIX            = (1 << 22),
  SPAN_ABBREVIATED       = (1 << 23),
  SPAN_ABBREVIATION      = (1 << 24),
  SPAN_PERSON            = (1 << 25),
  SPAN_ART               = (1 << 26),
  SPAN_PREDICATE         = (1 << 27),
};

// Span markers.
extern Handle kItalicMarker;
extern Handle kBoldMarker;
extern Handle kPersonMarker;
extern Handle kRedlinkMarker;
extern Handle kAbbreviatedMarker;
extern Handle kAbbreviationMarker;

// Populate chart with phrase matches. It looks up all spans (up to the maximum
// span length) in the alias table and adds the matches to the chart. Spans
// cannot start or end on a stop word.
class SpanPopulator {
 public:
  void Annotate(const PhraseTable *aliases, SpanChart *chart);

  // Add stop word.
  void AddStopWord(Text word);

  // Add forbidden phrase.
  void Forbid(Text phrase);

 private:
  // Check if token is a stop word.
  bool Discard(const Token &token) const;

  // Fingerprints for stop words.
  std::unordered_set<uint64> stop_words_;

  // Fingerprints for forbidden phrases.
  std::unordered_set<uint64> forbidden_;

  // Phrase tokenizer for forbidden phrases.
  PhraseTokenizer tokenizer_;
};

// Import existing spans in the underlying document into the span chart. Dates,
// measures, and geo coordinates are imported directly into the chart. Link
// spans are only imported if the span text is an alias for the linked entity
// in the alias table.
class SpanImporter {
 public:
  // Initialize span importer.
  void Init(Store *store, bool detailed);

  // Import spans from document.
  void Annotate(const PhraseTable *aliases, SpanChart *chart);

 private:
  // Detailed annotations.
  bool detailed_ = true;

  // Symbols.
  Names names_;
  Name n_time_{names_, "/w/time"};
  Name n_quantity_{names_, "/w/quantity"};
  Name n_geo_{names_, "/w/geo"};
};

// Prune common words from chart. Common words are classified as words with
// IDF (Inverse Document Frequency) below the threshold. Only lowercase words
// are pruned from the span chart.
class CommonWordPruner {
 public:
  // Prune common words using IDF dictionary.
  void Annotate(const IDFTable &dictionary, SpanChart *chart);

 private:
  // IDF threshold for pruning single token spans.
  static constexpr float idf_threshold = 3.5;
};

// Add emphasized phrases in the text as span candidates.
class EmphasisAnnotator {
 public:
  // Add bold and italic phrases from the document to the chart.
  void Annotate(SpanChart *chart);

 private:
  // Maximum length of an emphasized phrase.
  static constexpr int max_length = 20;
};

// Associate the first bolded mention in the first sentence with the topic
// entity of the article.
class IntroAnnotator {
 public:
  // Initialize intro annotator.
  void Init(Store *store);

  // Resolve first bolded mention to the topic entity.
  void Annotate(SpanChart *chart);

 private:
  // Symbols.
  Names names_;
  Name n_page_item_{names_, "/wp/page/item"};

  // Maximum offset of intro phrase.
  static constexpr int max_offset = 5;

  // Maximum length of intro phrase.
  static constexpr int max_length = 20;
};

// Adds span flags based on taxonomy to the matched spans in the chart. All the
// possible matches for a span are classified and their types are used for
// adding flags to the matching span. The type-based span flags are used by
// down-stream annotators for identifying possible span matches.
class SpanTaxonomy {
 public:
  ~SpanTaxonomy();

  // Initialize span taxonomy.
  void Init(Store *store);

  // Annotate spans in the chart with type-based flags.
  void Annotate(const PhraseTable *aliases, SpanChart *chart);

 private:
  // Classify item according to taxonomy and return flags for item.
  int Classify(const Frame &item);

  // Fact catalog for constructing taxonomy.
  FactCatalog catalog_;

  // Taxonomy for classifying items.
  Taxonomy *taxonomy_ = nullptr;

  // Mapping from type to span flags.
  HandleMap<int> type_flags_;

  // Minimum length for work of art span.
  static constexpr int min_art_length = 4;
};

// Annotate person name spans. A person name consists of the following parts
// in order:
//  1) Given names separated by spaces or dashes (mandatory).
//  2) Nick names in quotes.
//  3) Single-letter initials.
//  4) Notability particle.
//  5) Family names separated by spaces or dashes.
//  6) Suffix like Jr. or Sr.
class PersonNameAnnotator {
 public:
  // Annotate person name spans.
  void Annotate(SpanChart *chart);

 private:
  // Check if span covers golden annotation.
  bool Covered(SpanChart *chart, int begin, int end);

  // Notability particles.
  static std::unordered_set<string> particles;
};

// Score annotated spans based on case patterns. Adds a penalty if a span starts
// with a lower case token and ends with an upper case token.
class CaseScorer {
 public:
  // Score spans based on case.
  void Annotate(SpanChart *chart);

 private:
  // Case pattern penalties.
  static constexpr float lower_upper_penalty = 1.1;
  static constexpr float upper_lower_penalty = 0.1;
};

// Annotate numbers. Both standard notation (comma as decimal separator and
// period as thousand separator) and imperial notation (period as decimal
// separator and comma as thousand separator) are supported. Imperial notation
// is used for English, and standard notation is used for other languages. The
// annotator can fall back on the secondary format if the number does not match
// the primary format. Integers that match years in the current calendar are
// annotated as years.
class NumberAnnotator {
 public:
  // Number formats.
  enum Format {STANDARD, IMPERIAL, NORWEGIAN};

  // Initialize number annotator.
  void Init(Store *store);

  // Annotate chart with number spans.
  void Annotate(SpanChart *chart);

 private:
  // Try to parse number using the specified thousand, decimal, and milli
  // separators. Returns number (integer or float) if the number could be
  // parsed. Otherwise, nil is returned.
  static Handle ParseNumber(Text str, char tsep, char dsep, char msep);

  // Try to parse number using the specified number format.
  static Handle ParseNumber(Text str, Format format);

  // Symbols.
  Names names_;
  Name n_lang_{names_, "lang"};
  Name n_english_{names_, "/lang/en"};
  Name n_time_{names_, "/w/time"};
};

// Annotate spelled numbers.
class SpelledNumberAnnotator {
 public:
  // Initialize spelled number annotator.
  void Init(Store *store);

  // Annotate chart with spelled number spans.
  void Annotate(const PhraseTable *aliases, SpanChart *chart);

 private:
  // Symbols.
  Names names_;
  Name n_numeric_value_{names_, "P1181"};
};

// Annotate scaled number, i.e. a number followed by thousand, million,
// billion, or trillion, according to the alias table.
class NumberScaleAnnotator {
 public:
  // Initialize annotator.
  void Init(Store *store);

  // Annotate scaled numbers.
  void Annotate(const PhraseTable *aliases, SpanChart *chart);

 private:
  // Mapping from item for scale to scalar.
  HandleMap<float> scalars_;

  // Symbols.
  Names names_;
  Name n_numeric_value_{names_, "P1181"};
};

// Annotate measures in the document. A measure is a number followed by a
// unit. For currencies, the unit can preceed the number.
class MeasureAnnotator {
 public:
  // Initialize measure annotator.
  void Init(Store *store, bool detailed);

  // Annotate measure spans.
  void Annotate(const PhraseTable *aliases, SpanChart *chart);

 private:
  // Check if item is a unit.
  bool IsUnit(const Frame &item);

  // Add quantity with amount and unit to chart.
  void AddQuantity(SpanChart *chart, int begin, int end,
                   Handle amount, Handle unit);

  // Set of types for units.
  HandleSet units_;

  // Detailed annotations.
  bool detailed_ = true;

  // Symbols.
  Names names_;
  Name n_instance_of_{names_, "P31"};
  Name n_quantity_{names_, "/w/quantity"};
  Name n_amount_{names_, "/w/amount"};
  Name n_unit_{names_, "/w/unit"};
};

// Annotate dates in the document. This annotates full dates (day, month, year)
// as well as months (month and year), years, decades, and centuries. Relative
// dates are not annotated.
class DateAnnotator {
 public:
  // Initialize date annotator.
  void Init(Store *store, bool detailed);

  // Annotate date spans.
  void Annotate(const PhraseTable *aliases, SpanChart *chart);

 private:
  // Try to find year at position in chart. Returns the year if found (and
  // set the end) or 0 if no year was found.
  int GetYear(const PhraseTable *aliases, Store *store, SpanChart *chart,
              int pos, int *end);

  // Find item for phrase with a certain type.
  Handle FindMatch(const PhraseTable *aliases,
                   const SpanChart::Item &span,
                   const Name &type,
                   Store *store);

  // Add date annotation to chart.
  void AddDate(SpanChart *chart, int begin, int end, const Date &date);

  // Calendar for date computations.
  Calendar calendar_;

  // Detailed annotations.
  bool detailed_ = true;

  // Symbols.
  Names names_;
  Name n_instance_of_{names_, "P31"};
  Name n_point_in_time_{names_, "P585"};
  Name n_time_{names_, "/w/time"};
  Name n_calendar_day_{names_, "Q47150325"};
  Name n_calendar_month_{names_, "Q47018478"};
  Name n_day_of_year_{names_, "Q14795564"};
  Name n_month_{names_, "Q47018901"};
  Name n_year_{names_, "Q577"};
  Name n_year_bc_{names_, "Q29964144"};
  Name n_decade_{names_, "Q39911"};
  Name n_century_{names_, "Q578"};
};

// Annotate abbreviated phrases and their abbreviations.
class AbbreviationAnnotator {
 public:
  typedef std::unordered_map<uint64, uint64> Abbreviations;

  // Initialize annotator.
  void Init();

  // Annotate abbreviations.
  void Annotate(SpanChart *chart, Abbreviations *abbreviations);

 private:
  // Convert word to sequence of Unicode code points.
  static std::vector<int> Letters(Text word);

  // Fingerprints for words that can be skipped in abbreviated phrases.
  std::unordered_set<uint64> skipwords_;
};

// Span annotator for annotating a (pre-annotated) document with mention
// annotations based on a knowledge base and an alias table. This runs the
// annotators above on each sentence to create an annotated output document
// with mention spans.
class SpanAnnotator {
 public:
  // Resources for initializing span annotator.
  struct Resources {
    string kb;             // knowledge base with entities and metadata
    string dictionary;     // dictionary table with IDF scores for words
    string language;       // language for documents

    // Phrase table with phrase to entity mapping
    const PhraseTable *aliases = nullptr;

    bool resolve = false;  // resolve spans to entities in knowledge base
    bool detailed = true;  // annotate frames attributes
  };

  // Initialize annotator.
  void Init(Store *commons, const Resources &resources);

  // Add stop words.
  void AddStopWords(const std::vector<string> &words);

  // Run annotators on document and add annotations to output document.
  void Annotate(const Document &document, Document *output);

 private:
  // Check if item is a person.
  bool IsPerson(const Frame &item) const;

  // Add person name parts as local mentions to context.
  void AddNameParts(const Document &document, int begin, int end,
                    ResolverContext *context,
                    Handle entity, int count);

  // Phrase table with aliases.
  const PhraseTable *aliases_;

  // Dictionary with IDF scores.
  IDFTable dictionary_;

  // Resolve spans to entities in the knowledge base.
  bool resolve_ = false;

  // Detailed annotations.
  bool detailed_ = true;

  // Entity resolver.
  EntityResolver resolver_;

  // Annotators.
  SpanPopulator populator_;
  SpanImporter importer_;
  SpanTaxonomy taxonomy_;
  PersonNameAnnotator persons_;
  NumberAnnotator numbers_;
  SpelledNumberAnnotator spelled_;
  NumberScaleAnnotator scales_;
  MeasureAnnotator measures_;
  DateAnnotator dates_;
  CommonWordPruner pruner_;
  CaseScorer case_;
  EmphasisAnnotator emphasis_;
  IntroAnnotator intro_;
  AbbreviationAnnotator abbreviated_;

  // Symbols.
  Names names_;
  Name n_quantity_{names_, "/w/quantity"};
  Name n_amount_{names_, "/w/amount"};
  Name n_instance_of_{names_, "P31"};
  Name n_human_{names_, "Q5"};
  Name n_person_{names_, "Q215627"};
  Name n_fictional_human_{names_, "Q15632617"};
  Name n_page_item_{names_, "/wp/page/item"};
  Name n_name_{names_, "name"};

  // Maximum phrase length.
  static constexpr int max_phrase_length = 10;
};

}  // namespace nlp
}  // namespace sling

#endif  // SLING_NLP_SILVER_MENTIONS_H_
