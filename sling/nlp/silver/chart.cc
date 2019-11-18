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

#include "sling/nlp/silver/chart.h"

#include <vector>
#include <utility>

namespace sling {
namespace nlp {

SpanChart::SpanChart(const Document *document, int begin, int end, int maxlen)
    : document_(document), begin_(begin), end_(end), maxlen_(maxlen),
      tracking_(document->store()) {
  // The chart height is equal to the number of tokens.
  DCHECK_GE(begin, 0);
  DCHECK_LT(begin, end);
  size_ = end_ - begin_;

  // Phrase matches cannot be longer than the number of document tokens.
  if (size_ < maxlen_) maxlen_ = size_;

  // Initialize chart.
  items_.resize(size_ * size_);
  for (int b = 0; b < size_; ++b) {
    for (int e = b + 1; e <= size_; ++e) {
      item(b, e).cost = e - b;
    }
  }
}

void SpanChart::Add(int begin, int end, Handle match, int flags) {
  Item &span = item(begin - begin_, end - begin_);
  span.aux = match;
  span.flags |= flags;
  span.cost = 1;
  if (match.IsRef()) tracking_.push_back(match);
  if (end - begin > maxlen_) maxlen_ = end - begin;
}

void SpanChart::Solve() {
  // Segment document into parts without crossing spans.
  int segment_begin = 0;
  while (segment_begin < size_) {
    // Find next segment.
    int segment_end = segment_begin + 1;
    for (int b = segment_begin; b < segment_end; ++b) {
      for (int l = 1; l <= maxlen_; l++) {
        int e = b + l;
        if (e > size_) break;
        Item &span = item(b, e);
        if (span.matches != nullptr || !span.aux.IsNil()) {
          if (e > segment_end) segment_end = e;
        }
      }
    }

    // Compute best span covering for the segment.
    int segment_size = segment_end - segment_begin;
    for (int l = 2; l <= segment_size; ++l) {
      // Find best covering for all spans of length l.
      for (int s = segment_begin; s <= segment_end - l; ++s) {
        // Find best split of span [s;s+l).
        Item &span = item(s, s + l);
        for (int n = 1; n < l; ++n) {
          // Consider the split [s;s+n) and [s+n;s+l).
          Item &left = item(s, s + n);
          Item &right = item(s + n, s + l);
          float cost = left.cost + right.cost;
          if (cost < span.cost) {
            span.cost = cost;
            span.split = n;
          }
        }
      }
    }

    if (segment_end != size_) {
      // Mark segment split.
      item(segment_begin, size_).split = segment_end - segment_begin;
    }

    // Move on to next segment.
    segment_begin = segment_end;
  }
}

void SpanChart::Extract(const Extractor &extractor) {
  std::vector<std::pair<int, int>> queue;
  queue.emplace_back(0, size_);
  while (!queue.empty()) {
    // Get next span from queue.
    int b = queue.back().first;
    int e = queue.back().second;
    queue.pop_back();

    Item &s = item(b, e);
    if (!s.aux.IsNil() || s.matches != nullptr) {
      // Output annotation.
      extractor(begin_ + b, begin_ + e, s);
    } else if (s.split != -1) {
      // Queue best split.
      queue.emplace_back(b + s.split, e);
      queue.emplace_back(b, b + s.split);
    }
  }
}

void SpanChart::Extract(Document *output) {
  Extract([output](int begin, int end, const Item &item) {
    if (!item.aux.IsNil()) {
      // Add span annotation for auxiliary item.
      Span *span = output->AddSpan(begin, end);
      if (span != nullptr) {
        span->Evoke(item.aux);
      }
    } else if (item.matches != nullptr) {
      // Add span annotation for match.
      output->AddSpan(begin, end);
    }
  });
}

}  // namespace nlp
}  // namespace sling

