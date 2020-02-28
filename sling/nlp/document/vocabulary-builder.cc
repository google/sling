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

#include <algorithm>
#include <string>
#include <vector>

#include "sling/file/textmap.h"
#include "sling/nlp/document/document.h"
#include "sling/task/accumulator.h"
#include "sling/task/documents.h"
#include "sling/util/unicode.h"

namespace sling {
namespace nlp {

using namespace task;

// Process documents and output counts for normalized words in documents.
class WordVocabularyMapper : public DocumentProcessor {
 public:
  void Startup(Task *task) override {
    // Initialize accumulator.
    accumulator_.Init(output(), 1 << 24);

    // Get parameters.
    normalization_ = ParseNormalization(task->Get("normalization", ""));
    task->Fetch("only_lowercase", &only_lowercase_);
    task->Fetch("skip_section_titles", &skip_section_titles_);
  }

  void Process(Slice key, const Document &document) override {
    // Output normalize token words.
    bool in_header = false;
    for (const Token &token : document.tokens()) {
      // Track section headings.
      if (token.style() & HEADING_BEGIN) in_header = true;
      if (token.style() & HEADING_END) in_header = false;
      if (in_header && skip_section_titles_) continue;

      // Check for lowercase words.
      if (only_lowercase_ && token.Form() != CASE_LOWER) continue;

      // Normalize token.
      string normalized;
      UTF8::Normalize(token.word(), normalization_, &normalized);

      // Discard empty tokens.
      if (normalized.empty()) continue;

      // Output normalized token word.
      accumulator_.Increment(normalized);
    }
  }

  void Flush(Task *task) override {
    accumulator_.Flush();
  }

 private:
  // Accumulator for word counts.
  Accumulator accumulator_;

  // Token normalization flags.
  Normalization normalization_;

  // Only extract lowercase words.
  bool only_lowercase_ = false;

  // Skip section titles.
  bool skip_section_titles_ = false;
};

REGISTER_TASK_PROCESSOR("word-vocabulary-mapper", WordVocabularyMapper);

// Collect vocabulary and output text map with words and counts.
class WordVocabularyReducer : public SumReducer {
 public:
  void Start(Task *task) override {
    // Initialize sum reducer.
    SumReducer::Start(task);

    // Get max vocabulary size and threshold for discarding words.
    min_freq_ = task->Get("min_freq", 30);
    max_words_ = task->Get("max_words", 1000000);

    // Add OOV item to vocabulary as the first entry.
    vocabulary_.emplace_back("<UNKNOWN>", 0);

    // Statistics.
    num_words_ = task->GetCounter("words");
    word_count_ = task->GetCounter("word_count");
    num_words_discarded_ = task->GetCounter("num_words_discarded");
  }

  void Aggregate(int shard, const Slice &key, uint64 sum) override {
    if (sum < min_freq_) {
      // Add counts for discarded words to OOV entry.
      vocabulary_[0].count += sum;
      num_words_discarded_->Increment();
    } else {
      // Add entry to vocabulary.
      vocabulary_.emplace_back(key.str(), sum);
    }
    num_words_->Increment();
    word_count_->Increment(sum);
  }

  void Done(Task *task) override {
    // Sort word entries in decreasing frequency. The OOV entry is kept as the
    // first entry.
    std::sort(vocabulary_.begin() + 1, vocabulary_.end(),
      [](const Entry &a, const Entry &b) {
        return a.count > b.count;
      });

    // Add counts for all discarded entries to OOV.
    for (int i = max_words_; i < vocabulary_.size(); ++i) {
      vocabulary_[0].count += vocabulary_[i].count;
    }

    // Write vocabulary to output.
    int words = 0;
    for (auto &entry : vocabulary_) {
      if (++words > max_words_) break;
      Output(0, new Message(entry.word, std::to_string(entry.count)));
    }
  }

 private:
  // Word entry with count.
  struct Entry {
    Entry(const string &word, int64 count) : word(word), count(count) {}
    string word;
    int64 count;
  };

  // Threshold for discarding words.
  int min_freq_;

  // Maximum number of words in vocabulary.
  int max_words_;

  // Vocabulary. The first item is the OOV item.
  std::vector<Entry> vocabulary_;

  // Statistics.
  Counter *num_words_ = nullptr;
  Counter *word_count_ = nullptr;
  Counter *num_words_discarded_ = nullptr;
};

REGISTER_TASK_PROCESSOR("word-vocabulary-reducer", WordVocabularyReducer);

}  // namespace nlp
}  // namespace sling

