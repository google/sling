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

#include "sling/base/logging.h"
#include "sling/base/types.h"
#include "sling/nlp/document/document.h"
#include "sling/nlp/document/document-tokenizer.h"
#include "sling/nlp/document/lex.h"
#include "sling/task/documents.h"
#include "sling/task/accumulator.h"
#include "sling/task/reducer.h"

namespace sling {
namespace nlp {

// Extract outgoing links from mentions and themes in documents.
class WikipediaLinkExtractor : public task::DocumentProcessor {
 public:
  void Startup(task::Task *task) override {
    // Get parameters.
    task->Fetch("extract_mention_links", &extract_mention_links_);
    task->Fetch("extract_theme_links", &extract_theme_links_);
    task->Fetch("extract_infobox_links", &extract_infobox_links_);

    // Initialize fan-in output.
    fanin_ = task->GetSink("fanin");
    if (fanin_ != nullptr) {
      fanin_counts_.Init(fanin_);
    }

    // Statistics.
    num_links_ = task->GetCounter("links");
    num_mention_links_ = task->GetCounter("mention_links");
    num_theme_links_ = task->GetCounter("theme_links");
  }

  void Process(Slice key, const Document &document) override {
    // Collect outbound links from document.
    HandleMap<int> links;
    Store *store = document.store();
    ExtractDocumentLinks(document, &links);

    if (!links.empty()) {
      // Output frame with link statistics.
      Builder b(store);
      for (auto it : links) {
        b.Add(it.first, it.second);
      }
      FrameProcessor::Output(key, b.Create());

      // Accumulate fan-in statistics.
      if (fanin_ != nullptr) {
        for (auto it : links) {
          Frame target(store, it.first);
          fanin_counts_.Increment(target.Id(), it.second);
        }
      }
    }
  }

  void Flush(task::Task *task) override {
    if (fanin_ != nullptr) {
      fanin_counts_.Flush();
    }
  }

  void ExtractDocumentLinks(const Document &document, HandleMap<int> *links) {
    // Collect all links in mentions.
    Store *store = document.store();
    if (extract_mention_links_) {
      Handles evoked(store);
      for (const Span *span : document.spans()) {
        span->AllEvoked(&evoked);
        for (Handle link : evoked) {
          link = store->Resolve(link);
          if (store->IsPublic(link)) {
            (*links)[link]++;
            num_mention_links_->Increment();
            num_links_->Increment();
          }
        }
      }
    }

    // Collect all thematic links.
    for (Handle link : document.themes()) {
      link = store->Resolve(link);
      if (store->IsFrame(link)) {
        if (store->IsPublic(link)) {
          if (extract_mention_links_) {
            (*links)[link]++;
            num_theme_links_->Increment();
            num_links_->Increment();
          }
        } else if (extract_infobox_links_) {
          // Extract links from info box theme.
          Frame f(store, link);
          if (f.IsA(n_infobox_)) {
            ExtractInfoboxLinks(f, links);
          }
        }
      }
    }
  }

  void ExtractInfoboxLinks(const Frame &frame, HandleMap<int> *links) {
    DocumentLexer lexer(&tokenizer_);
    Store *store = frame.store();
    for (const Slot &s : frame) {
      if (store->IsString(s.value)) {
        // Convert LEX-encoded field to document and extract links.
        Document field(store, docnames());
        if (lexer.Lex(&field,  store->GetString(s.value)->str())) {
          ExtractDocumentLinks(field, links);
        }
      } else if (store->IsFrame(s.value)) {
        Frame sub(store, s.value);
        if (sub.IsAnonymous()) {
          // Extract links from sub-frames.
          ExtractInfoboxLinks(sub, links);
        }
      }
    }
  }

 private:
  // Parameters.
  bool extract_mention_links_ = true;
  bool extract_theme_links_ = true;
  bool extract_infobox_links_ = true;

  // Output channel and accumulator for fan-in statistics.
  task::Channel *fanin_ = nullptr;
  task::Accumulator fanin_counts_;

  // Symbols.
  Name n_infobox_{names_, "/wp/infobox"};

  // Counters.
  task::Counter *num_links_;
  task::Counter *num_mention_links_;
  task::Counter *num_theme_links_;

  // Document tokenizer for LEX decoding.
  DocumentTokenizer tokenizer_;
};

REGISTER_TASK_PROCESSOR("wikipedia-link-extractor", WikipediaLinkExtractor);

// Collect fact targets from items and output aggregate target counts.
class FactTargetExtractor : public task::FrameProcessor {
 public:
  void Startup(task::Task *task) override {
    accumulator_.Init(output());
  }

  void Process(Slice key, const Frame &frame) override {
    // Accumulate fact targets for the item.
    Store *store = frame.store();
    for (const Slot &slot : frame) {
      if (slot.name == Handle::isa()) continue;
      if (slot.name == n_lang_) continue;

      Handle target = store->Resolve(slot.value);
      if (!store->IsFrame(target)) continue;

      Text id = store->FrameId(target);
      if (id.empty()) continue;

      accumulator_.Increment(id);
    }
  }

  void Flush(task::Task *task) override {
    accumulator_.Flush();
  }

 private:
  // Accumulator for fanin counts.
  task::Accumulator accumulator_;

  // Symbols.
  Name n_lang_{names_, "lang"};
};

REGISTER_TASK_PROCESSOR("fact-target-extractor", FactTargetExtractor);

// Merge links and output link frames for each item.
class WikipediaLinkMerger : public task::Reducer {
 public:
  void Start(task::Task *task) override {
    task::Reducer::Start(task);
    CHECK(names_.Bind(&commons_));
    commons_.Freeze();
  }

  void Reduce(const task::ReduceInput &input) override {
    // Merge links from all documents.
    HandleMap<int> links;
    Store store(&commons_);
    for (task::Message *message : input.messages()) {
      // Get next set of links.
      Frame batch = DecodeMessage(&store, message);

      // Aggregate links for item.
      for (const Slot &s : batch) {
        Handle link = s.name;
        int count = s.value.AsInt();
        links[link] += count;
      }
    }

    // Output frame with link map.
    Builder linkmap(&store);
    for (auto it : links) {
      linkmap.Add(it.first, it.second);
    }
    Builder b(&store);
    b.Add(n_links_, linkmap.Create());
    Output(input.shard(), task::CreateMessage(input.key(), b.Create()));
  }

 private:
  // Commons store.
  Store commons_;

  // Symbols.
  Names names_;
  Name n_links_{names_, "/w/item/links"};
};

REGISTER_TASK_PROCESSOR("wikipedia-link-merger", WikipediaLinkMerger);

// Sum item popularity and output popularity frame for each item.
class ItemPopularityReducer : public task::SumReducer {
 public:
  void Aggregate(int shard, const Slice &key, uint64 sum) override {
    // Output popularity frame for item.
    Store store;
    Builder b(&store);
    int popularity = sum;
    b.Add("/w/item/popularity", popularity);
    Output(shard, task::CreateMessage(key, b.Create()));
  }
};

REGISTER_TASK_PROCESSOR("item-popularity-reducer", ItemPopularityReducer);

}  // namespace nlp
}  // namespace sling

