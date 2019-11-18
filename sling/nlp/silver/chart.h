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

#ifndef SLING_NLP_NER_CHART_H_
#define SLING_NLP_NER_CHART_H_

#include <functional>
#include <unordered_set>
#include <vector>

#include "sling/base/types.h"
#include "sling/frame/store.h"
#include "sling/nlp/document/document.h"
#include "sling/nlp/kb/phrase-table.h"

namespace sling {
namespace nlp {

// Span chart for sentence in document. This represents all the phrase matches
// up to a maximum length.
class SpanChart {
 public:
  // Chart item.
  struct Item {
    // Check span flag.
    bool is(int flag) const { return flags & flag; }

    // Check if span has matches.
    bool matched() const { return !aux.IsNil() || matches != nullptr; }

    // Phrase matches in phrase table.
    const PhraseTable::Phrase *matches = nullptr;

    // Auxiliary match from annotators.
    Handle aux = Handle::nil();

    // Span cost.
    float cost = 0.0;

    // Optimal split point for item.
    int split = -1;

    // Span flags.
    int flags = 0;
  };

  // Initialize empty span chart for (part of) document.
  SpanChart(const Document *document, int begin, int end, int maxlen);

  // Add auxiliary match to chart.
  void Add(int begin, int end, Handle match, int flags = 0);

  // Compute non-overlapping span covering with minimum cost.
  void Solve();

  // Extract best span covering.
  typedef std::function<void(int begin, int end, const Item &item)> Extractor;
  void Extract(const Extractor &extractor);
  void Extract(Document *output);

  // Return item for token span (0 <= begin < size, 0 < end <= size).
  Item &item(int begin, int end) {
    DCHECK_GE(begin, 0);
    DCHECK_LT(begin, size_);
    DCHECK_GT(end, begin);
    DCHECK_LE(end, size_);
    return items_[begin * size_ + end - 1];
  }

  // Return item for single-token span.
  Item &item(int index) { return item(index, index + 1); }

  // Return chart size.
  int size() const { return size_; }

  // Return maximum phrase span length.
  int maxlen() const { return maxlen_; }

  // Get document part for chart.
  const Document *document() const { return document_; }
  int begin() const { return begin_; }
  int end() const { return end_; }

  // Return phrase for chart item. The begin and end are relative to the chart.
  string phrase(int b, int e) const {
    return document_->PhraseText(b + begin_, e + begin_);
  }

  // Return token for chart item. The index is relative to the chart.
  const Token &token(int index) const {
    return document_->token(index + begin_);
  }

  // Return fingerprint for phrase. The index is relative to the chart.
  uint64 fingerprint(int b, int e) const {
    return document_->PhraseFingerprint(b + begin_, e + begin_);
  }

 private:
  // Document and token span for chart.
  const Document *document_;
  int begin_;
  int end_;

  // Maximum phrase length considered for matching.
  int maxlen_;

  // Chart items indexed by span start and length.
  std::vector<Item> items_;
  int size_;

  // Tracked frame handles.
  Handles tracking_;
};

}  // namespace nlp
}  // namespace sling

#endif  // SLING_NLP_NER_CHART_H_
