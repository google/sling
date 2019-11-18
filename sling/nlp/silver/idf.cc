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

#include "sling/nlp/silver/idf.h"

#include <math.h>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "sling/base/logging.h"
#include "sling/base/types.h"
#include "sling/file/repository.h"
#include "sling/nlp/document/document.h"
#include "sling/nlp/document/fingerprinter.h"
#include "sling/string/numbers.h"
#include "sling/task/documents.h"
#include "sling/task/accumulator.h"
#include "sling/util/unicode.h"

namespace sling {
namespace nlp {

using namespace task;

// Collect vocabulary from document corpus and output word fingerprints with
// counts.
class VocabularyMapper : public DocumentProcessor {
 public:
  void Startup(Task *task) override {
    // Initialize accumulator.
    accumulator_.Init(output(), 1 << 24);

    // Get normalization flags.
    normalization_ = ParseNormalization(task->Get("normalization", "cln"));

    // Pruning threshold for skipping short documents.
    task->Fetch("min_document_length", &min_document_length_);
    if (min_document_length_ > 0) {
      num_short_documents_ = task->GetCounter("short_documents");
      num_short_tokens_ = task->GetCounter("short_tokens");
    }

    // Only extract lowercase words.
    task->Fetch("only_lowercase", &only_lowercase_);
    if (only_lowercase_) {
      num_discarded_tokens_ = task->GetCounter("discarded_tokens");
    }

    // Get document counter.
    num_idf_documents_ = task->GetCounter("idf_documents");
  }

  void Process(Slice key, const Document &document) override {
    // Skip short document.
    if (document.num_tokens() < min_document_length_) {
      num_short_documents_->Increment();
      num_short_tokens_->Increment(document.num_tokens());
      return;
    }

    // Collect fingerprints for all the words in the document.
    std::unordered_set<uint64> fingerprints;
    for (const Token &token : document.tokens()) {
      // Get fingerprint for token.
      uint64 fp = Fingerprinter:: Fingerprint(token.word(), normalization_);

      // Discard empty tokens.
      if (fp == 1) continue;

      // Check for lowercase words.
      if (only_lowercase_ && token.Form() != CASE_LOWER) {
        num_discarded_tokens_->Increment();
        continue;
      }

      // Add word to the document word set.
      fingerprints.insert(fp);
    }

    // Accumulate word fingerprints.
    num_idf_documents_->Increment();
    for (uint64 fp : fingerprints) {
      accumulator_.Increment(fp);
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

  // Minimum number of tokens in document that are used for extacting words.
  int min_document_length_ = 0;
  Counter *num_short_documents_ = nullptr;
  Counter *num_short_tokens_ = nullptr;

  // Only extract lowercase words.
  bool only_lowercase_ = false;
  Counter *num_discarded_tokens_ = nullptr;

  // Counter for aggregating the total number of documents. This is used in the
  // reducer for computing IDF.
  Counter *num_idf_documents_ = nullptr;
};

REGISTER_TASK_PROCESSOR("vocabulary-mapper", VocabularyMapper);

// Collect vocabulary and build repository with IDF table. The IDF maps
// word fingerprint to the inverse document frequency for the word.
class IDFTableBuilder : public SumReducer {
 public:
  void Start(Task *task) override {
    // Initialize sum reducer.
    SumReducer::Start(task);

    // Statistics.
    num_words_ = task->GetCounter("words");
    num_word_instances_ = task->GetCounter("word_instances");
  }

  void Aggregate(int shard, const Slice &key, uint64 sum) override {
    // Get word fingerprint.
    uint64 fp;
    CHECK(safe_strtou64_base(key.data(), key.size(), &fp, 10));

    // Add entry to vocabulary.
    vocabulary_.push_back(new WordEntry(fp, sum));
    num_words_->Increment();
    num_word_instances_->Increment(sum);
  }

  void Done(Task *task) override {
    // Get total number of documents.
    float num_docs = task->GetCounter("idf_documents")->value();

    // Set up IDF repository header.
    Repository repository;
    IDFTable::Header header;
    memset(&header, 0, sizeof(IDFTable::Header));
    header.version = IDFTable::VERSION;
    header.num_docs = num_docs;
    string normalization = task->Get("normalization", "cln");
    CHECK_LT(normalization.size(), sizeof(IDFTable::Header::normalization));
    strcpy(header.normalization, normalization.c_str());
    repository.AddBlock("IDFHeader", &header, sizeof(IDFTable::Header));

    // Compute IDF for all the words.
    for (auto *entry : vocabulary_) {
      WordEntry *word = static_cast<WordEntry *>(entry);
      word->idf = log(num_docs / word->count);
    }

    // Write word map.
    int num_words = vocabulary_.size();
    int num_buckets = (num_words + 8) / 8;
    repository.WriteMap("IDF", &vocabulary_, num_buckets);

    // Write repository to file.
    const string &filename = task->GetOutput("repository")->resource()->name();
    CHECK(!filename.empty());
    repository.Write(filename);

    // Clear collected data.
    for (auto *word : vocabulary_) delete word;
  }

 private:
  // Word entry with count and IDF score.
  struct WordEntry : public RepositoryMapItem {
    // Initialize new phrase.
    WordEntry(uint64 fingerprint, uint64 count)
      : fingerprint(fingerprint), count(count) {}

    // Write word to repository.
    int Write(File *file) const override {
      file->WriteOrDie(&fingerprint, sizeof(uint64));
      file->WriteOrDie(&idf, sizeof(float));
      return sizeof(uint64) + sizeof(float);
    }

    // Use word fingerprint as the hash code.
    uint64 Hash() const override { return fingerprint; }

    uint64 fingerprint;      // word fingerprint
    uint64 count;            // number of document containing word
    float idf = 0.0;         // inverse document frequency
  };

  // List of word entries in IDF table.
  std::vector<RepositoryMapItem *> vocabulary_;

  // Statistics.
  Counter *num_words_ = nullptr;
  Counter *num_word_instances_ = nullptr;
};

REGISTER_TASK_PROCESSOR("idf-table-builder", IDFTableBuilder);

void IDFTable::Load(const string &filename) {
  // Load IDF repository from file.
  repository_.Read(filename);

  // Check IDF header information.
  repository_.FetchBlock("IDFHeader", &header_);
  CHECK(header_ != nullptr) << "Invalid IDF table: " << filename;
  CHECK_EQ(header_->version, VERSION) << "Unsupported IDF table: " << filename;
  normalization_ = ParseNormalization(header_->normalization);

  // Initialize word index table.
  index_.Initialize(repository_);

  // Assume singleton for out-of-vocabulary words.
  oov_idf_ = log(header_->num_docs);
}

const IDFTable::Word *IDFTable::Find(uint64 fp) const {
  int bucket = fp % index_.num_buckets();
  const Word *word = index_.GetBucket(bucket);
  const Word *end = index_.GetBucket(bucket + 1);
  while (word < end) {
    if (word->fingerprint == fp) return word;
    word++;
  }

  return nullptr;
}

float IDFTable::GetIDF(uint64 fingerprint) const {
  const Word *word = Find(fingerprint);
  return word != nullptr ? word->idf : oov_idf_;
}

}  // namespace nlp
}  // namespace sling

