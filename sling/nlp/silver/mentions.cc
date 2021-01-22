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

#include "sling/nlp/silver/mentions.h"

#include "sling/base/flags.h"
#include "sling/frame/object.h"
#include "sling/frame/serialization.h"
#include "sling/nlp/document/annotator.h"
#include "sling/nlp/document/fingerprinter.h"
#include "sling/nlp/kb/calendar.h"
#include "sling/util/unicode.h"

DEFINE_int32(resolver_trace, 0, "Trace resolver scores");
DEFINE_bool(dump_chart, false, "Dump span chart");

namespace sling {
namespace nlp {

Handle kItalicMarker = Handle::Index(1);
Handle kBoldMarker = Handle::Index(2);
Handle kPersonMarker = Handle::Index(3);
Handle kRedlinkMarker = Handle::Index(4);
Handle kAbbreviatedMarker = Handle::Index(5);
Handle kAbbreviationMarker = Handle::Index(6);

// Markers are implemented as index handles.
static bool IsMarker(Handle h) {
  return h.IsIndex();
}

void SpanPopulator::Annotate(const PhraseTable *aliases,
                             SpanChart *chart) {
  // Spans cannot start or end on stop words.
  int begin = chart->begin();
  int end = chart->end();
  std::vector<bool> skip(chart->size());
  for (int i = 0; i < chart->size(); ++i) {
    skip[i] = Discard(chart->token(i));
  }

  // Find all matching spans up to the maximum length.
  for (int b = begin; b < end; ++b) {
    // Span cannot start on a skipped token.
    if (skip[b - begin]) continue;

    for (int e = b + 1; e <= std::min(b + chart->maxlen(), end); ++e) {
      // Span cannot end on a skipped token. This does not apply to upper case
      // tokens.
      if (skip[e - begin - 1]) {
        CaseForm form = chart->token(e - begin - 1).Form();
        if (form != CASE_TITLE && form != CASE_UPPER) continue;
      }

      // Check if phrase has been forbidden.
      uint64 fp = chart->document()->PhraseFingerprint(b, e);
      if (forbidden_.count(fp) > 0) continue;

      // Find matches in phrase table.
      SpanChart::Item &span = chart->item(b - begin, e - begin);
      span.matches = aliases->Find(fp);

      // Set the span cost to one if there are any matches.
      if (span.matches != nullptr) {
        span.cost = 1.0;
      }
    }
  }
}

void SpanPopulator::AddStopWord(Text word) {
  uint64 fp = Fingerprinter::Fingerprint(word);
  stop_words_.insert(fp);
}

void SpanPopulator::Forbid(Text phrase) {
  uint64 fp = tokenizer_.Fingerprint(phrase);
  forbidden_.insert(fp);
}

bool SpanPopulator::Discard(const Token &token) const {
  return stop_words_.count(token.Fingerprint()) > 0;
}

void SpanImporter::Init(Store *store, bool detailed) {
  detailed_ = detailed;
  CHECK(names_.Bind(store));
}

void SpanImporter::Annotate(const PhraseTable *aliases, SpanChart *chart) {
  // Find all mention spans covered by the chart.
  const Document *document = chart->document();
  Handles matches(document->store());
  int begin = chart->begin();
  int end = chart->end();
  for (Span *span : document->spans()) {
    // Skip spans outside the chart.
    if (span->begin() < begin || span->end() > end) continue;
    if (span->length() == 0) continue;

    // Get evoked frame for span.
    Frame evoked = span->Evoked();
    if (evoked.invalid()) {
      // Red link. Add as span candidate.
      chart->Add(span->begin(), span->end(), kRedlinkMarker, 0);
      continue;
    }

    // Check for special annotations.
    int flags = 0;
    if (evoked.IsA(n_time_)) flags |= SPAN_DATE;
    if (evoked.IsA(n_quantity_)) flags |= SPAN_MEASURE;
    if (evoked.IsA(n_geo_)) flags |= SPAN_GEO;
    if (!detailed_ && flags != 0) {
      // Only output simple frame with type.
      Builder b(document->store());
      if (flags & SPAN_DATE) b.AddIsA(n_time_);
      if (flags & SPAN_MEASURE) b.AddIsA(n_quantity_);
      if (flags & SPAN_GEO) b.AddIsA(n_geo_);
      evoked = b.Create();
    }

    auto &item = chart->item(span->begin() - begin, span->end() - begin);
    if (flags != 0) {
      // Clear any other matches for span for special annotations.
      item.matches = nullptr;
    } else {
      // Check that phrase is an alias for the annotated entity.
      aliases->Lookup(span->Fingerprint(), &matches);
      bool found = false;
      for (Handle h : matches) {
        if (h == evoked.handle()) found = true;
      }
      if (found) {
        // Match found for annotation, clear any other matches.
        item.matches = nullptr;
      } else {
        // No match found for annotation, skip it,
        continue;
      }
    }

    chart->Add(span->begin(), span->end(), evoked.handle(), flags);
  }
}

void CommonWordPruner::Annotate(const IDFTable &dictionary, SpanChart *chart) {
  for (int t = 0; t < chart->size(); ++t) {
    // Get chart item for single token.
    auto &item = chart->item(t);
    if (item.matches == nullptr) continue;
    const Token &token = chart->token(t);

    // Keep predicates.
    if (item.is(SPAN_PREDICATE)) continue;

    // Check case form.
    CaseForm form =  UTF8::Case(token.word());
    bool common = (form == CASE_LOWER);
    if (token.initial() && form == CASE_TITLE) common = true;

    // Prune lower-case tokens with low IDF scores.
    if (common) {
      float idf = dictionary.GetIDF(token.Fingerprint());
      if (idf < idf_threshold) {
        item.matches = nullptr;
      }
    }
  }
}

void EmphasisAnnotator::Annotate(SpanChart *chart) {
  int offset = chart->begin();
  for (int b = 0; b < chart->size(); ++b) {
    // Mark italic span.
    if (chart->token(b).style() & ITALIC_BEGIN) {
      // Find end of italic phrase.
      int e = b + 1;
      while (e < chart->size()) {
        if (chart->token(e).style() & ITALIC_END) break;
        e++;
      }

      // Only annotate italic phrase if length is below the threshold.
      auto &item = chart->item(b, e);
      if (e - b <= max_length && !item.matched()) {
        chart->Add(b + offset, e + offset, kItalicMarker, SPAN_EMPHASIS);
      }
    }

    // Mark bold span.
    if (chart->token(b).style() & BOLD_BEGIN) {
      // Find end of bold phrase.
      int e = b + 1;
      while (e < chart->size()) {
        if (chart->token(e).style() & BOLD_END) break;
        e++;
      }

      // Only annotate bold phrase if length is below the threshold.
      auto &item = chart->item(b, e);
      if (e - b <= max_length && !item.matched()) {
        chart->Add(b + offset, e + offset, kBoldMarker, SPAN_EMPHASIS);
      }
    }
  }
};

void IntroAnnotator::Init(Store *store) {
  CHECK(names_.Bind(store));
}

void IntroAnnotator::Annotate(SpanChart *chart) {
  // Only look for intro phrase in the first sentence.
  if (chart->begin() != 0) return;

  // Get topic item.
  Handle topic = chart->document()->top().GetHandle(n_page_item_);
  if (topic.IsNil()) return;

  // Find begining of bolded span.
  int begin = -1;
  int blimit = chart->size() < max_offset ? chart->size() : max_offset;
  for (int b = 0; b < blimit; ++b) {
    if (chart->token(b).style() & BOLD_BEGIN) {
      begin = b;
      break;
    }
  }
  if (begin == -1) return;

  // Find end of bolded span.
  int end = -1;
  for (int e = begin + 1; e < chart->size() && e - begin < max_length; ++e) {
    if (chart->token(e).style() & BOLD_END) {
      end = e;
      break;
    }
  }
  if (end == -1) return;

  // Annotate bolded intro span with topic item.
  chart->item(begin, end).aux = topic;
}

SpanTaxonomy::~SpanTaxonomy() {
  delete taxonomy_;
}

void SpanTaxonomy::Init(Store *store) {
  // Taxonomy used for classifying spans in the chart.
  static std::pair<const char *, int> span_taxonomy[] = {
    {"Q47150325",  SPAN_CALENDAR_DAY},     // calendar day of a given year
    {"Q47018478",  SPAN_CALENDAR_MONTH},   // calendar month of a given year
    {"Q14795564",  SPAN_DAY_OF_YEAR},      // date of periodic occurrence
    {"Q41825",     SPAN_WEEKDAY},          // day of the week
    {"Q47018901",  SPAN_MONTH},            // calendar month
    {"Q577",       SPAN_YEAR},             // year
    {"Q29964144",  SPAN_YEAR_BC},          // year BC
    {"Q39911",     SPAN_DECADE},           // decade
    {"Q578",       SPAN_CENTURY},          // century
    {"Q21199",     SPAN_NATURAL_NUMBER},   // natural number
    {"Q2360980",   SPAN_UNIT_OF_AMOUNT},   // unit of amount
    {"Q8142",      SPAN_CURRENCY},         // currency
    {"Q47574",     SPAN_UNIT},             // unit of measurement
    {"Q27084",     SPAN_UNIT},             // parts-per notation
    {"Q101352",    SPAN_FAMILY_NAME},      // family name
    {"Q202444",    SPAN_GIVEN_NAME},       // given name
    {"Q19838177",  SPAN_SUFFIX},           // suffix for person name
    {"Q215627",    SPAN_PERSON},           // person
    {"Q15632617",  SPAN_PERSON},           // fictional human
    {"Q12737077",  SPAN_PREDICATE},        // occupation
    {"Q4164871",   SPAN_PREDICATE},        // position
    {"Q828803",    SPAN_PREDICATE},        // job title
    {"Q11862829",  SPAN_PREDICATE},        // academic discipline
    {"Q20202269",  SPAN_PREDICATE},        // music term
    {"Q11032",     0},                     // newspaper
    {"Q35127",     0},                     // website
    {"Q5398426",   SPAN_ART},              // television series (brand but art)
    {"Q167270",    0},                     // trademark
    {"Q215380",    SPAN_ART},              // band
    {"Q17537576",  SPAN_ART},              // creative work
    {"Q47461344",  SPAN_ART},              // written work
    {"Q838948",    SPAN_ART},              // work of art
    {"Q3331189",   SPAN_ART},              // version, edition, or translation
    {nullptr, 0},
  };

  // Build taxonomy as well as mapping from types to span flags.
  std::vector<Text> types;
  for (auto *type = span_taxonomy; type->first != nullptr; ++type) {
    const char *name = type->first;
    int flags = type->second;
    Handle t = store->LookupExisting(name);
    if (t.IsNil()) {
      LOG(WARNING) << "Ignoring unknown type in taxonomy: " << name;
      continue;
    }
    type_flags_[t] = flags;
    types.push_back(name);
  }

  catalog_.Init(store);
  taxonomy_ = new Taxonomy(&catalog_, types);
}

void SpanTaxonomy::Annotate(const PhraseTable *aliases, SpanChart *chart) {
  const Document *document = chart->document();
  Store *store = document->store();
  PhraseTable::MatchList matchlist;

  // Run over all spans in the chart.
  for (int b = 0; b < chart->size(); ++b) {
    int end = std::min(b + chart->maxlen(), chart->size());
    for (int e = b + 1; e <= end; ++e) {
      SpanChart::Item &span = chart->item(b, e);

      if (span.aux.IsRef() && !span.aux.IsNil()) {
        // Classify span based on single resolved item.
        Frame item(store, span.aux);
        int flags = Classify(item);
        span.flags |= flags;
      } else if (span.matches != nullptr) {
        // Get the matches for the span from the alias table.
        aliases->GetMatches(span.matches, &matchlist);
        CaseForm form = document->PhraseForm(b + chart->begin(),
                                             e + chart->begin());
        bool matches = false;
        for (const auto &match : matchlist) {
          // Skip unknown items.
          if (match.item.IsNil()) {
            LOG(WARNING) << "Unknown item: " << match.id;
            continue;
          }

          // Skip candidate if case forms conflict.
          if (match.form != CASE_NONE &&
              form != CASE_NONE &&
              match.form != form) {
            continue;
          }

          // Classify item.
          Frame item(store, match.item);
          int flags = Classify(item);
          span.flags |= flags;

          // Titles of works of art can add a lot of spurious matches to the
          // chart because almost any word or short phrase can be the title
          // of a song, book, painting, etc. These are only included if the
          // length of the matching phrase is above a threshold and not all
          // lower case.
          if (flags & SPAN_ART) {
            if (form != CASE_LOWER && e - b > min_art_length) matches = true;
          } else {
            matches = true;
          }
        }

        // Discard matches if all candidates have conflicts.
        if (!matches) {
          span.matches = nullptr;
        }
      }
    }
  }
}

int SpanTaxonomy::Classify(const Frame &item) {
  Handle type = taxonomy_->Classify(item);
  if (type.IsNil()) return 0;
  auto f = type_flags_.find(type);
  if (f == type_flags_.end()) return 0;
  return f->second;
}

std::unordered_set<string> PersonNameAnnotator::particles = {
  "de", "du", "di", "dos", "von", "van", "bin", "ibn",
};

void PersonNameAnnotator::Annotate(SpanChart *chart) {
  // Mark initials and dashes.
  int size = chart->size();
  for (int i = 0; i < size; ++i) {
    const Token &token = chart->token(i);
    const string &word = token.word();
    if (UTF8::IsInitials(word)) {
      chart->item(i).flags |= SPAN_INITIALS;
    }
    if (UTF8::IsDash(word) && token.brk() == NO_BREAK) {
      chart->item(i).flags |= SPAN_DASH;
    }
  }

  // Find sequences of given names, nick name, initials, family names, and
  // suffix.
  int b = 0;
  while (b < size) {
    int e = b;

    // Parse given name(s).
    int given_names = 0;
    while (e < size && chart->item(e).is(SPAN_GIVEN_NAME)) {
      given_names++;
      e++;
    }

    // Parse dash followed by given name(s).
    if (e < size && given_names > 0 && chart->item(e).is(SPAN_DASH)) {
      int de = e + 1;
      while (de < size && chart->item(de).is(SPAN_GIVEN_NAME)) {
        given_names++;
        de++;
      }
      if (de > e + 1) e = de;
    }

    // Parse nickname, e.g. Eugene ``Buzz'' Aldrin.
    if (e < size && given_names > 0 && chart->token(e).word() == "``") {
      int q = e + 1;
      while (q < size && chart->token(q).word() != "''") q++;

      // No more than two nicknames.
      if (q < size && q - e < 4) {
        e = q + 1;
      }
    }

    // Parse initials. Initials count as given names, e.g. J. K. Rowling.
    while (e < size && chart->item(e).is(SPAN_INITIALS)) {
      given_names++;
      e++;
    }

    // Parse notability particle.
    if (e < size && particles.count(chart->token(e).word()) > 0) e++;

    // Parse dash followed by family names.
    int family_names = 0;
    if (e < size && given_names > 0 && chart->item(e).is(SPAN_DASH)) {
      int de = e + 1;
      while (de < size && chart->item(de).is(SPAN_FAMILY_NAME)) de++;
      if (de > e + 1) e = de;
    }

    // Parse family name(s).
    while (e < size && chart->item(e).is(SPAN_FAMILY_NAME)) {
      family_names++;
      e++;
    }

    // Parse dash followed by family names.
    if (e < size && family_names > 0 && chart->item(e).is(SPAN_DASH)) {
      int de = e + 1;
      while (de < size && chart->item(de).is(SPAN_FAMILY_NAME)) de++;
      if (de > e + 1) e = de;
    }

    // Parse suffix, e.g. Jr.
    if (e < size && chart->item(e).is(SPAN_SUFFIX)) e++;

    // Mark span if person name found except if it covers a golden span.
    if (given_names > 0 && !Covered(chart, b, e)) {
      if (!chart->item(b, e).matched()) {
        chart->Add(b + chart->begin(), e + chart->begin(), kPersonMarker);
      }
      b = e;
    } else {
      b++;
    }
  }
}

bool PersonNameAnnotator::Covered(SpanChart *chart, int begin, int end) {
  for (int b = begin; b < end; ++b) {
    for (int e = b + 1; e <= end; ++e) {
      SpanChart::Item &span = chart->item(b, e);
      if (!span.aux.IsNil()) return true;
    }
  }

  return false;
};

void CaseScorer::Annotate(SpanChart *chart) {
  // Run over all spans with more than one token in the chart.
  for (int b = 0; b < chart->size(); ++b) {
    CaseForm first = chart->token(b).Form();
    int end = std::min(b + chart->maxlen(), chart->size());
    for (int e = b + 2; e <= end; ++e) {
      SpanChart::Item &span = chart->item(b, e);

      // Do not score resolved spans.
      if (!span.aux.IsNil()) continue;

      // Apply penalty for camel case.
      CaseForm last = chart->token(e - 1).Form();
      if (first == CASE_LOWER && last == CASE_TITLE) {
        span.cost += lower_upper_penalty;
      } else if (first == CASE_TITLE && last == CASE_LOWER) {
        span.cost += upper_lower_penalty;
      }
    }
  }
}

void NumberAnnotator::Init(Store *store) {
  CHECK(names_.Bind(store));
}

void NumberAnnotator::Annotate(SpanChart *chart) {
  // Get document language.
  const Document *document = chart->document();
  Handle lang = document->top().GetHandle(n_lang_);
  if (lang.IsNil()) lang = n_english_.handle();
  Format format = (lang == n_english_ ? IMPERIAL : STANDARD);

  for (int t = chart->begin(); t < chart->end(); ++t) {
    const string &word = document->token(t).word();

    // Check if token contains digits.
    bool has_digits = false;
    bool all_digits = true;
    for (char c : word) {
      if (c >= '0' && c <= '9') {
        has_digits = true;
      } else {
        all_digits = false;
      }
    }
    if (!has_digits) continue;

    // Try to parse token as a number.
    Handle number = ParseNumber(word, format);
    if (!number.IsNil()) {
      // Numbers between 1582 and 2038 are considered years.
      int flags = SPAN_NUMBER;
      if (word.size() == 4 && all_digits && number.IsInt()) {
        int value = number.AsInt();
        if (value >= 1582 && value <= 2038) {
          Builder b(document->store());
          b.AddIsA(n_time_);
          b.AddIs(number);
          number = b.Create().handle();
          flags = SPAN_DATE;
        }
      }
      chart->Add(t, t + 1, number, flags);
    }
  }
}

Handle NumberAnnotator::ParseNumber(Text str, char tsep, char dsep, char msep) {
  const char *p = str.data();
  const char *end = p + str.size();
  if (p == end) return Handle::nil();

  // Parse sign.
  double scale = 1.0;
  if (*p == '-') {
    scale = -1.0;
    p++;
  } else if (*p == '+') {
    p++;
  }

  // Parse integer part. The thousand groups must be two or three digits except
  // for the last, which should always be three digits. Indian numbers use two
  // digit groups for lakhs and crores.
  double value = 0.0;
  const char *group = nullptr;
  while (p < end) {
    if (*p >= '0' && *p <= '9') {
      value = value * 10.0 + (*p++ - '0');
    } else if (*p == tsep) {
      if (group != nullptr && p - group != 3 && p - group != 2) {
        return Handle::nil();
      }
      group = p + 1;
      p++;
    } else if (*p == dsep) {
      break;
    } else {
      return Handle::nil();
    }
  }
  if (group != nullptr && p - group != 3) return Handle::nil();

  // Parse decimal part.
  bool decimal = false;
  if (p < end && *p == dsep) {
    decimal = true;
    p++;
    group = nullptr;
    while (p < end) {
      if (*p >= '0' && *p <= '9') {
        value = value * 10.0 + (*p++ - '0');
        scale /= 10.0;
      } else if (*p == msep) {
        if (group != nullptr && p - group != 3) return Handle::nil();
        group = p + 1;
        p++;
      } else {
        return Handle::nil();
      }
    }
    if (group != nullptr && p - group != 3) return Handle::nil();
  }
  if (p != end) return Handle::nil();

  // Compute number.
  value *= scale;
  if (decimal || value < Handle::kMinInt || value > Handle::kMaxInt) {
    return Handle::Float(value);
  } else {
    return Handle::Integer(value);
  }
}

Handle NumberAnnotator::ParseNumber(Text str, Format format) {
  Handle number = Handle::nil();
  switch (format) {
    case STANDARD:
      number = ParseNumber(str, '.', ',', 0);
      if (number.IsNil()) number = ParseNumber(str, ',', '.', 0);
      break;
    case IMPERIAL:
      number = ParseNumber(str, ',', '.', 0);
      if (number.IsNil()) number = ParseNumber(str, '.', ',', 0);
      break;
    case NORWEGIAN:
      number = ParseNumber(str, ' ', '.', ' ');
      if (number.IsNil()) number = ParseNumber(str, '.', ',', 0);
      break;
  }
  return number;
}

void SpelledNumberAnnotator::Init(Store *store) {
  CHECK(names_.Bind(store));
}

void SpelledNumberAnnotator::Annotate(const PhraseTable *aliases,
                                      SpanChart *chart) {
  const Document *document = chart->document();
  Store *store = document->store();
  Handles matches(document->store());
  for (int b = 0; b < chart->size(); ++b) {
    int end = std::min(b + chart->maxlen(), chart->size());
    for (int e = end; e > b; --e) {
      // Only consider number spans that have not already been annotated.
      SpanChart::Item &span = chart->item(b, e);
      if (span.is(SPAN_NUMBER) || span.is(SPAN_DATE)) continue;
      if (!span.is(SPAN_NATURAL_NUMBER)) continue;

      // Find matching numeric value.
      aliases->GetMatches(span.matches, &matches);
      Handle value = Handle::nil();
      for (Handle item : matches) {
        if (item.IsNil()) continue;
        Frame f(store, item);
        value = f.GetHandle(n_numeric_value_);
        if (!value.IsNil()) {
          value = store->Resolve(value);
          break;
        }
      }

      // Add spelled number annotation.
      if (value.IsNumber()) {
        chart->Add(b + chart->begin(), e + chart->begin(), value, SPAN_NUMBER);
      }
    }
  }
}

void NumberScaleAnnotator::Init(Store *store) {
  static std::pair<const char *, float> scalars[] = {
    {"Q43016", 1e3},      // thousand
    {"Q38526", 1e6},      // million
    {"Q16021", 1e9},      // billion
    {"Q862978", 1e12},    // trillion
    {nullptr, 0},
  };

  CHECK(names_.Bind(store));
  for (auto *scale = scalars; scale->first != nullptr; ++scale) {
    const char *qid = scale->first;
    float scalar = scale->second;
    scalars_[store->LookupExisting(qid)] = scalar;
  }
}

void NumberScaleAnnotator::Annotate(const PhraseTable *aliases,
                                    SpanChart *chart) {
  const Document *document = chart->document();
  Store *store = document->store();
  for (int b = 0; b < chart->size(); ++b) {
    int end = std::min(b + chart->maxlen(), chart->size());
    for (int e = end; e > b; --e) {
      // Only consider number spans.
      SpanChart::Item &span = chart->item(b, e);

      // Get scalar.
      float scale = 0;
      if (span.is(SPAN_NATURAL_NUMBER)) {
        Handles matches(store);
        aliases->GetMatches(span.matches, &matches);
        for (Handle item : matches) {
          if (item.IsNil()) continue;
          auto f = scalars_.find(item);
          if (f != scalars_.end()) {
            scale = f->second;
            break;
          }
        }
      } else if (span.is(SPAN_UNIT_OF_AMOUNT)) {
        Handles matches(store);
        aliases->GetMatches(span.matches, &matches);
        for (Handle item : matches) {
          if (item.IsNil()) continue;
          Frame unit(store, item);
          Handle value = unit.GetHandle(n_numeric_value_);
          if (value.IsInt()) {
            scale = value.AsInt();
            break;
          }
        }
      }
      if (scale == 0) continue;

      // Find number to the left.
      int left_end = b;
      int left_begin = std::max(0, left_end - chart->maxlen());
      Handle number = Handle::nil();
      int start;
      for (int left = left_begin; left < left_end; ++left) {
        SpanChart::Item &number_span = chart->item(left, left_end);
        if (!number_span.is(SPAN_NUMBER)) continue;
        if (number_span.aux.IsNumber()) {
          start = left;
          number = number_span.aux;
          break;
        }
      }

      // Add scaled number annotation.
      if (number.IsInt()) {
        float value = number.AsInt() * scale;
        chart->Add(start + chart->begin(), e + chart->begin(),
                   Handle::Float(value), SPAN_NUMBER);
      } else if (number.IsFloat()) {
        float value = number.AsFloat() * scale;
        chart->Add(start + chart->begin(), e + chart->begin(),
                   Handle::Float(value), SPAN_NUMBER);
      }
    }
  }
}

void MeasureAnnotator::Init(Store *store, bool detailed) {
  static const char *unit_types[] = {
    "Q10387685",   // unit of density
    "Q10387689",   // unit of power
    "Q1302471",    // unit of volume
    "Q1371562",    // unit of area
    "Q15222637",   // unit of speed
    "Q15976022",   // unit of force
    "Q16604158",   // unit of charge
    "Q1790144",    // unit of time
    "Q1978718",    // unit of length
    "Q2916980",    // unit of energy
    "Q3647172",    // unit of mass
    "Q8142",       // currency
    "Q756202",     // reserve currency
    "Q27084",      // parts-per notation, e.g. percentage
    nullptr,
  };

  detailed_ = detailed;
  CHECK(names_.Bind(store));
  for (const char **type = unit_types; *type != nullptr; ++type) {
    units_.insert(store->Lookup(*type));
  }
}

void MeasureAnnotator::Annotate(const PhraseTable *aliases, SpanChart *chart) {
  const Document *document = chart->document();
  Store *store = document->store();
  PhraseTable::MatchList matches;
  for (int b = 0; b < chart->size(); ++b) {
    int end = std::min(b + chart->maxlen(), chart->size());
    for (int e = end; e > b; --e) {
      // Only consider unit and currency spans.
      SpanChart::Item &span = chart->item(b, e);
      if (!span.is(SPAN_UNIT) && !span.is(SPAN_CURRENCY)) continue;

      // Get unit.
      Handle unit = Handle::nil();
      if (span.aux.IsNil()) {
        PhraseTable::MatchList matches;
        aliases->GetMatches(span.matches, &matches);
        for (auto &match : matches) {
          if (match.item.IsNil()) continue;
          if (!match.reliable) continue;
          Frame item(store, match.item);
          if (IsUnit(item)) {
            unit = match.item;
            break;
          }
        }
      } else if (store->IsFrame(span.aux)) {
        Frame item(store, span.aux);
        if (IsUnit(item)) {
          unit = span.aux;
        }
      }
      if (unit.IsNil()) continue;

      // Find number to the left.
      int left_end = b;
      if (left_end > 0 && UTF8::IsDash(chart->token(left_end - 1).word())) {
        // Allow dash between number and unit.
        left_end--;
      }
      int left_begin = std::max(0, left_end - chart->maxlen());
      Handle number = Handle::nil();
      int start;
      for (int left = left_begin; left < left_end; ++left) {
        SpanChart::Item &number_span = chart->item(left, left_end);
        if (!number_span.is(SPAN_NUMBER)) continue;
        if (number_span.aux.IsNumber()) {
          start = left;
          number = number_span.aux;
          break;
        }
      }

      // Add quantity annotation.
      if (!number.IsNil()) {
        AddQuantity(chart, start, e, number, unit);
        break;
      }

      // Find number to the right of currency (e.g. USD 100).
      if (span.is(SPAN_CURRENCY)) {
        int right_begin = e;
        int right_end = std::min(right_begin + chart->maxlen(), chart->size());
        Handle number = Handle::nil();
        int end;
        for (int right = right_end; right > right_begin; --right) {
          SpanChart::Item &number_span = chart->item(right_begin, right);
          if (!number_span.is(SPAN_NUMBER)) continue;
          if (number_span.aux.IsNumber()) {
            end = right;
            number = number_span.aux;
            break;
          }
        }

        // Add quantity annotation for amount.
        if (!number.IsNil()) {
          AddQuantity(chart, b, end, number, unit);
        }
      }
    }
  }
}

bool MeasureAnnotator::IsUnit(const Frame &item) {
  for (const Slot &s : item) {
    if (s.name == n_instance_of_) {
      Handle type = item.store()->Resolve(s.value);
      if (units_.count(type) > 0) return true;
    }
  }
  return false;
}

void MeasureAnnotator::AddQuantity(SpanChart *chart, int begin, int end,
                                   Handle amount, Handle unit) {
  // Make quantity frame.
  Store *store = chart->document()->store();
  Builder builder(store);
  builder.AddIsA(n_quantity_);
  if (detailed_) {
    builder.Add(n_amount_, amount);
    builder.Add(n_unit_, unit);
  }
  Handle h = builder.Create().handle();

  // Add quantity annotation to chart.
  chart->Add(begin + chart->begin(), end + chart->begin(), h, SPAN_MEASURE);
}

void DateAnnotator::Init(Store *store, bool detailed) {
  CHECK(names_.Bind(store));
  calendar_.Init(store);
  detailed_ = detailed;
}

void DateAnnotator::AddDate(SpanChart *chart, int begin, int end,
                            const Date &date) {
  // Make date annotation frame.
  Store *store = chart->document()->store();
  Builder builder(store);
  builder.AddIsA(n_time_);
  if (detailed_) {
    builder.AddIs(date.AsHandle(store));
  }
  Handle h = builder.Create().handle();

  // Add date annotation to chart.
  chart->Add(begin + chart->begin(), end + chart->begin(), h, SPAN_DATE);
}

Handle DateAnnotator::FindMatch(const PhraseTable *aliases,
                                const SpanChart::Item &span,
                                const Name &type,
                                Store *store) {
  // Return resolved item if present.
  if (span.aux.IsRef() && !span.aux.IsNil()) return span.aux;

  // Get matches from alias table.
  Handles matches(store);
  aliases->GetMatches(span.matches, &matches);

  // Find first match with the specified type.
  for (Handle h : matches) {
    if (h.IsNil()) continue;
    Frame item(store, h);
    for (const Slot &s : item) {
      if (s.name == n_instance_of_ && store->Resolve(s.value) == type) {
        return h;
      }
    }
  }

  // No match found.
  return Handle::nil();
}

int DateAnnotator::GetYear(const PhraseTable *aliases,
                           Store *store, SpanChart *chart,
                           int pos, int *end) {
  // Skip date delimiters.
  if (pos == chart->size()) return 0;
  const string &word = chart->token(pos).word();
  if (word == "," || word == "de" || word == "del") pos++;

  // Try to find year annotation at position.
  for (int e = std::min(pos + chart->maxlen(), chart->size()); e > pos; --e) {
    SpanChart::Item &span = chart->item(pos, e);

    // Find matching year.
    Handle year = Handle::nil();
    if (span.is(SPAN_YEAR)) {
      if (span.aux.IsInt() && span.aux.AsInt() > 0) {
        year = span.aux;
      } else {
        year = FindMatch(aliases, span, n_year_, store);
      }
    } else if (span.is(SPAN_YEAR_BC)) {
      year = FindMatch(aliases, span, n_year_bc_, store);
    }

    // Get year from match. Year 0 exists (Q23104) but is not a valid year.
    if (!year.IsNil()) {
      Date date(Object(store, store->Resolve(year)));
      if (date.precision == Date::YEAR && date.year != 0) {
        *end = e;
        return date.year;
      }
    }
  }

  // No year found.
  return 0;
}

void DateAnnotator::Annotate(const PhraseTable *aliases, SpanChart *chart) {
  const Document *document = chart->document();
  Store *store = document->store();
  for (int b = 0; b < chart->size(); ++b) {
    int end = std::min(b + chart->maxlen(), chart->size());
    for (int e = end; e > b; --e) {
      SpanChart::Item &span = chart->item(b, e);
      Date date;
      if (span.is(SPAN_CALENDAR_DAY)) {
        // Date with year, month and day.
        Handle h = FindMatch(aliases, span, n_calendar_day_, store);
        if (!h.IsNil()) {
          Frame item(store, h);
          date.ParseFromFrame(item);
          if (date.precision == Date::DAY) {
            AddDate(chart, b, e, date);
            b = e;
            break;
          }
        }
      } else if (span.is(SPAN_CALENDAR_MONTH)) {
        // Date with month and year.
        Handle h = FindMatch(aliases, span, n_calendar_month_, store);
        if (!h.IsNil()) {
          Frame item(store, h);
          date.ParseFromFrame(item);
          if (date.precision == Date::MONTH) {
            AddDate(chart, b, e, date);
            b = e;
            break;
          }
        }
      } else if (span.is(SPAN_DAY_OF_YEAR)) {
        // Day of year with day and month.
        Handle h = FindMatch(aliases, span, n_day_of_year_, store);
        if (calendar_.GetDayAndMonth(h, &date)) {
          int year = GetYear(aliases, store, chart, e, &e);
          if (year != 0) {
            date.year = year;
            date.precision = Date::DAY;
            AddDate(chart, b, e, date);
            b = e;
            break;
          }
        }
      } else if (span.is(SPAN_MONTH)) {
        // Month.
        Handle h = FindMatch(aliases, span, n_month_, store);
        if (calendar_.GetMonth(h, &date)) {
          int year = GetYear(aliases, store, chart, e, &e);
          if (year != 0) {
            date.year = year;
            date.precision = Date::MONTH;
            AddDate(chart, b, e, date);
            b = e;
            break;
          }
        }
      } else if (span.is(SPAN_YEAR) && !span.is(SPAN_NUMBER)) {
        // Year.
        Handle h = FindMatch(aliases, span, n_year_, store);
        date.ParseFromFrame(Frame(store, h));
        if (date.precision == Date::YEAR) {
          AddDate(chart, b, e, date);
          b = e;
          break;
        }
      } else if (span.is(SPAN_DECADE)) {
        // Decade.
        Handle h = FindMatch(aliases, span, n_decade_, store);
        date.ParseFromFrame(Frame(store, h));
        if (date.precision == Date::DECADE) {
          AddDate(chart, b, e, date);
          b = e;
          break;
        }
      } else if (span.is(SPAN_CENTURY)) {
        // Century.
        Handle h = FindMatch(aliases, span, n_century_, store);
        date.ParseFromFrame(Frame(store, h));
        if (date.precision == Date::CENTURY) {
          AddDate(chart, b, e, date);
          b = e;
          break;
        }
      }
    }
  }
}

void AbbreviationAnnotator::Init() {
  static const char *skipwords[] = {
    "-", "&",
    "of", "for", "and", "the", "in",
    "ltd", "inc", "corp", "llc", "gmbh",
    nullptr,
  };

  for (const char **word = skipwords; *word != nullptr; ++word) {
    skipwords_.insert(Fingerprinter::Fingerprint(*word));
  }
}

void AbbreviationAnnotator::Annotate(SpanChart *chart,
                                     Abbreviations *abbreviations) {
  // Find abbreviations in parentheses.
  for (int b = 0; b < chart->size() - 3; ++b) {
    // Check for '(' ABBREV ')' sequence.
    if (chart->token(b + 1).word() != "(") continue;
    if (chart->token(b + 2).Form() != CASE_UPPER) continue;
    if (chart->token(b + 3).word() != ")") continue;

    // Get letters in abbreviation.
    const string &abbreviation = chart->token(b + 2).word();
    std::vector<int> letters = Letters(abbreviation);
    if (letters.size() < 2) continue;

    // Try to match phrase to the left of the abbreviation.
    int l = letters.size() - 1;
    int i = b;
    while (l >= 0 && i >= 0) {
      const Token &t = chart->token(i);
      const string &word = t.word();
      int letter = letters[l];

      // Try to match first letter.
      int initial = Unicode::ToUpper(UTF8::Decode(word.data(), word.size()));
      if (initial == letter) {
        l--;
        i--;
        continue;
      }

      // Check for skipped word.
      if (skipwords_.count(t.Fingerprint()) > 0) {
        // Skip word.
        i--;
        continue;
      }

      // The phrase may contain abbreviations itself, e.g. US Air Force (USAF).
      if (t.Form() == CASE_UPPER) {
        std::vector<int> sub_letters = Letters(word);
        int letter_start = l + 1 - sub_letters.size();
        if (letter_start >= 0) {
          bool match = true;
          for (int j = 0; j < sub_letters.size(); ++j) {
            if (sub_letters[j] != letters[letter_start + j]) {
              match = false;
              break;
            }
          }
          if (match) {
            // Sub-abbreviation match.
            l -= sub_letters.size();
            i--;
            continue;
          }
        }
      }

      // No abbreviation match.
      break;
    }

    // Mark abbreviated phrase and abbreviation if a match was found.
    if (l < 0) {
      int offset = chart->begin();
      int phrase_begin = i + 1;
      int phrase_end = b + 1;
      int abbrev_begin = b + 2;
      int abbrev_end = b + 3;

      // Mark abbreviated phrase.
      auto &phrase_item = chart->item(phrase_begin, phrase_end);
      phrase_item.flags |= SPAN_ABBREVIATED;
      if (!phrase_item.matched()) {
        chart->Add(phrase_begin + offset, phrase_end + offset,
                   kAbbreviatedMarker);
      }

      // Mark abbreviation.
      auto &abbrev_item = chart->item(abbrev_begin, abbrev_end);
      abbrev_item.flags |= SPAN_ABBREVIATION;
      if (!abbrev_item.matched()) {
        chart->Add(abbrev_begin + offset, abbrev_end + offset,
                   kAbbreviationMarker);
      }

      // Add abbreviation to abbreviation mapping.
      uint64 phrase_fp = chart->fingerprint(phrase_begin, phrase_end);
      uint64 abbrev_fp = chart->fingerprint(abbrev_begin, abbrev_end);
      (*abbreviations)[phrase_fp] = abbrev_fp;
    }
  }
}

std::vector<int> AbbreviationAnnotator::Letters(Text word) {
  std::vector<int> letters;
  const char *s = word.data();
  const char *end = s + word.size();
  while (s < end) {
    int code = UTF8::Decode(s);
    if (Unicode::IsLetter(code)) letters.push_back(code);
    s = UTF8::Next(s);
  }
  return letters;
}

void SpanAnnotator::Init(Store *commons, const Resources &resources) {
  // Load resources.
  CHECK(names_.Bind(commons));
  aliases_ = resources.aliases;
  if (!resources.kb.empty()) {
    LoadStore(resources.kb, commons);
  }
  if (!resources.dictionary.empty()) {
    dictionary_.Load(resources.dictionary);
  }

  // Get stop words and forbidden phrases for language.
  if (!resources.language.empty()) {
    Frame lang(commons, "/lang/" + resources.language);

    Handle sw = lang.GetHandle("/lang/wikilang/stop_words");
    if (!sw.IsNil()) {
      Array stopwords(commons, sw);
      for (int i = 0; i < stopwords.length(); ++i) {
        String word(commons, stopwords.get(i));
        populator_.AddStopWord(word.value());
      }
    }

    Handle bl = lang.GetHandle("/lang/wikilang/forbidden_phrases");
    if (!bl.IsNil()) {
      Array forbidden(commons, bl);
      for (int i = 0; i < forbidden.length(); ++i) {
        String word(commons, forbidden.get(i));
        populator_.Forbid(word.value());
      }
    }
  }

  // Initialize annotators.
  detailed_ = resources.detailed;
  importer_.Init(commons, detailed_);
  taxonomy_.Init(commons);
  numbers_.Init(commons);
  spelled_.Init(commons);
  scales_.Init(commons);
  measures_.Init(commons, detailed_);
  dates_.Init(commons, detailed_);
  intro_.Init(commons);
  abbreviated_.Init();

  // Initialize entity resolver.
  resolve_ = resources.resolve;
  resolver_.Init(commons, aliases_);
}

void SpanAnnotator::AddStopWords(const std::vector<string> &words) {
  for (const string &word : words) {
    populator_.AddStopWord(word);
  }
}

void SpanAnnotator::Annotate(const Document &document, Document *output) {
  // Initialize entity resolver context.
  Store *store = output->store();
  ResolverContext context(store, &resolver_);
  if (resolve_) {
    // Add focus topic for document to entity resolver context.
    Handle topic = document.top().GetHandle(n_page_item_);
    if (!topic.IsNil()) context.AddEntity(topic);
  }

  // Run annotators on each non-heading sentence in the input document.
  for (SentenceIterator s(&document, HEADING_BEGIN); s.more(); s.next()) {
    // Make chart for sentence.
    AbbreviationAnnotator::Abbreviations abbreviations;
    SpanChart chart(&document, s.begin(), s.end(), max_phrase_length);

    // Run annotators.
    populator_.Annotate(aliases_, &chart);
    importer_.Annotate(aliases_, &chart);
    emphasis_.Annotate(&chart);
    intro_.Annotate(&chart);
    taxonomy_.Annotate(aliases_, &chart);
    persons_.Annotate(&chart);
    numbers_.Annotate(&chart);
    spelled_.Annotate(aliases_, &chart);
    scales_.Annotate(aliases_, &chart);
    dates_.Annotate(aliases_, &chart);
    measures_.Annotate(aliases_, &chart);
    abbreviated_.Annotate(&chart, &abbreviations);
    pruner_.Annotate(dictionary_, &chart);
    case_.Annotate(&chart);

    // Compute best span covering.
    chart.Solve();

    // Optionally dump chart.
    if (FLAGS_dump_chart) {
      for (int b = 0; b < chart.size(); ++b) {
        for (int e = b + 1; e <= chart.size(); ++e) {
          auto &item= chart.item(b, e);
          if (item.split != -1) {
            LOG(INFO) << chart.phrase(b, b + item.split) << "|"
                      << chart.phrase(b + item.split, e)
                      << ": " << item.cost;
          } else {
            LOG(INFO) << chart.phrase(b, e) << ": " << item.cost;
          }
        }
      }
    }

    // Extract best span covering to the output document.
    chart.Extract([&](int begin, int end, const SpanChart::Item &item) {
      bool resolve_span = resolve_;
      Span *span = nullptr;
      if (!item.aux.IsNil()) {
        // Add span annotation for auxiliary item.
        span = output->AddSpan(begin, end);
        if (!item.aux.IsNumber()) {
          // Span has already been resolved.
          if (store->IsPublic(item.aux)) {
            span->Evoke(Builder(store).AddIs(item.aux).Create());
          } else {
            span->Evoke(item.aux);
          }

          // Add annotated entity to resolver context model.
          resolve_span = false;
          if (resolve_) {
            context.AddEntity(item.aux);

            // Add first and last names to mention model for persons.
            if (item.is(SPAN_PERSON) && IsPerson(Frame(store, item.aux))) {
              int count = context.GetPopularity(item.aux);
              AddNameParts(document, begin, end, &context, item.aux, count);
            }
          }
        } else if (!IsMarker(item.aux)) {
          // Output stand-alone quantity annotation.
          Builder b(store);
          b.AddIsA(n_quantity_);
          if (detailed_) {
            b.Add(n_amount_, item.aux);
          }
          span->Evoke(b.Create());
          resolve_span = false;
        }
      } else if (item.matches != nullptr) {
        // Add span annotation for match.
        span = output->AddSpan(begin, end);
      }

      // Resolve span to entity in knowledge base.
      bool resolved = false;
      if (resolve_span && item.matched()) {
        // Resolve mentioned entity.
        int k = FLAGS_resolver_trace == 0 ? 1 : FLAGS_resolver_trace;
        ResolverContext::Candidates candidates(k);
        context.Score(span->Fingerprint(), span->Form(), &candidates);

        // Optionally, output resolver scores.
        if (FLAGS_resolver_trace > 0) {
          candidates.sort();
          LOG(INFO) << "Mention: " << span->GetText();
          for (const auto &candidate : candidates) {
            Frame item(store, candidate.entity);
            LOG(INFO) << "  " << candidate.score << " " << item.Id()
                      << " " << item.GetString("name")
                      << " (" << item.GetString("description") << ")"
                      << (candidate.local ? " (local)" : "");
          }
        }

        if (!candidates.empty()) {
          // Evoke resolved entity for span.
          auto &winner = candidates.front();
          Handle entity = winner.entity;
          span->Evoke(Builder(store).AddIs(entity).Create());

          // Add resolved entity to context model.
          context.AddEntity(entity);
          resolved = true;

          // Add first and last names to mention model for persons.
          if (item.is(SPAN_PERSON) && IsPerson(Frame(store, entity))) {
            AddNameParts(document, begin, end, &context, entity, winner.count);
          }

          // Add abbreviation as local mention for resolved abbreviated phrase.
          if (item.is(SPAN_ABBREVIATED)) {
            auto f = abbreviations.find(span->Fingerprint());
            if (f != abbreviations.end()) {
              context.AddMention(f->second, CASE_UPPER, entity, winner.count);
            }
          }
        }
      }

      // Mark unresolved person name spans.
      if (!resolved && item.aux == kPersonMarker) {
        Builder b(output->store());
        b.AddIsA(n_person_);
        Frame person = b.Create();
        span->Evoke(person);

        if (resolve_) {
          // Add person to context model.
          context.AddEntity(person.handle());

          // Add first and last name mentions.
          AddNameParts(document, begin, end, &context, person.handle(), 1);
        }
      }
    });
  }

  // Update output document.
  output->Update();
}

bool SpanAnnotator::IsPerson(const Frame &item) const {
  for (const Slot &s : item) {
    if ((s.name == n_instance_of_ || s.name == Handle::isa()) &&
        (s.value == n_human_ ||
         s.value ==  n_person_ ||
         s.value == n_fictional_human_)) {
      return true;
    }
  }
  return false;
}

void SpanAnnotator::AddNameParts(const Document &document, int begin, int end,
                                 ResolverContext *context,
                                 Handle entity, int count) {
  // Add suffixes, i.e. last names.
  for (int b = begin + 1; b < end; b++) {
    if (document.token(b).skipped()) continue;
    uint64 fp = document.PhraseFingerprint(b, end);
    CaseForm form = document.PhraseForm(b, end);
    context->AddMention(fp, form, entity, count);
  }

  // Add prefixes, i.e. first names.
  for (int e = end - 1; e > begin; e--) {
    if (document.token(e - 1).skipped()) continue;
    uint64 fp = document.PhraseFingerprint(begin, e);
    CaseForm form = document.PhraseForm(begin, e);
    context->AddMention(fp, form, entity, count);
  }
}

// Add span annotations to documents and resolve entity mentions.
class MentionAnnotator : public Annotator {
 public:
  void Init(task::Task *task, Store *commons) override {
    // Initialize span annotator.
    SpanAnnotator::Resources resources;
    resources.dictionary = task->GetInputFile("dictionary");
    resources.resolve = task->Get("resolve", false);
    resources.detailed = task->Get("detailed", true);
    resources.language = task->Get("language", "en");

    string alias_file = task->GetInputFile("aliases");
    resources.aliases = PhraseTable::Acquire(task, commons, alias_file);
    CHECK(resources.aliases != nullptr);

    annotator_.Init(commons, resources);
  }

  void Annotate(Document *document) override {
    // Make a copy of the input document and clear annotations for output.
    Document original(*document);
    document->ClearAnnotations();

    // Annotate document.
    annotator_.Annotate(original, document);
  }

 private:
  // Span annotator for labeling documents.
  SpanAnnotator annotator_;
};

REGISTER_ANNOTATOR("mentions", MentionAnnotator);

}  // namespace nlp
}  // namespace sling

