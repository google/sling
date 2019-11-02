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

#include <string>
#include <unordered_map>
#include <vector>

#include "sling/base/perf.h"
#include "sling/file/recordio.h"
#include "sling/file/textmap.h"
#include "sling/frame/serialization.h"
#include "sling/myelin/builder.h"
#include "sling/myelin/compiler.h"
#include "sling/myelin/profile.h"
#include "sling/nlp/document/document.h"
#include "sling/nlp/embedding/embedding-model.h"
#include "sling/task/process.h"
#include "sling/util/embeddings.h"
#include "sling/util/random.h"
#include "sling/util/thread.h"
#include "sling/util/unicode.h"

namespace sling {
namespace nlp {

using namespace task;

// Vocabulary sampling for word embeddings.
class WordVocabularySampler {
 public:
  // Load vocabulary table.
  void Load(const string &filename, float subsampling) {
    // Read words and frequencies.
    TextMapInput input(filename);
    double sum = 0.0;
    int index;
    string word;
    int64 count;
    while (input.Read(&index, &word, &count)) {
      if (word == "<UNKNOWN>") oov_ = index;
      dictionary_[word] = index;
      entry_.emplace_back(word, count);
      permutation_.emplace_back(index, count);
      sum += count;
    }
    threshold_ = subsampling * sum;

    // Shuffle words (Fisher-Yates shuffle).
    int n = permutation_.size();
    Random rnd;
    for (int i = 0; i < n - 1; ++i) {
      int j = rnd.UniformInt(n - i);
      std::swap(permutation_[i], permutation_[i + j]);
    }

    // Convert counts to accumulative distribution.
    double acc = 0.0;
    for (int i = 0; i < n; ++i) {
      acc += permutation_[i].probability / sum;
      permutation_[i].probability = acc;
    }
  }

  // Look up word in dictionary. Return OOV for unknown words.
  int Lookup(const string &word, Normalization flags) const {
    string normalized;
    UTF8::Normalize(word, flags, &normalized);
    auto f = dictionary_.find(normalized);
    return f != dictionary_.end() ? f->second : oov_;
  }

  // Sample word according to distribution. Used for sampling negative examples.
  int Sample(float p) const {
    int n = permutation_.size();
    int low = 0;
    int high = n - 1;
    while (low < high) {
      int center = (low + high) / 2;
      if (permutation_[center].probability < p) {
        low = center + 1;
      } else {
        high = center;
      }
    }
    return permutation_[low].index;
  }

  // Sub-sampling probability for word. Used for sub-sampling words in
  // sentences for skip-grams. This implements the sub-sampling strategy from
  // Mikolov 2013.
  float SubsamplingProbability(int index) const {
    float count = entry_[index].count;
    return (sqrt(count / threshold_) + 1.0) * threshold_ / count;
  }

  // Clear data.
  void Clear() {
    dictionary_.clear();
    entry_.clear();
    permutation_.clear();
  }

  // Return the number of words in the vocabulary.
  size_t size() const { return dictionary_.size(); }

  // Get word for index.
  const string &word(int index) const { return entry_[index].word; }

 private:
  struct Entry {
    Entry(const string &word, float count) : word(word), count(count) {}
    string word;
    float count;
  };

  struct Element {
    Element(int i, float p) : index(i), probability(p) {}
    int index;
    float probability;
  };

  // Mapping from word to vocabulary index.
  std::unordered_map<string, int> dictionary_;

  // Word list.
  std::vector<Entry> entry_;

  // Permutation of words for sampling.
  std::vector<Element> permutation_;

  // Threshold for sub-sampling words
  float threshold_;

  // Entry for unknown words.
  int oov_ = 0;
};

// Trainer for word embeddings model using Mikolov word2vec skipgram model. The
// trainer supports running training on multiple threads concurrently. While
// this can significantly speed up the processing time, this will also lead to
// non-determinism because of concurrent access to shared data structure, i.e.
// the weight matrices in the network. However, the updates are usually small,
// so in practice these unsafe updates are usually not harmful and adding
// mutexes to serialize access to the model slows down training considerably.
class WordEmbeddingsTrainer : public Process {
 public:
  // Run training of embedding net.
  void Run(Task *task) override {
    // Get training parameters.
    task->Fetch("iterations", &iterations_);
    task->Fetch("max_epochs", &max_epochs_);
    task->Fetch("negative", &negative_);
    task->Fetch("window", &window_);
    task->Fetch("learning_rate", &learning_rate_);
    task->Fetch("min_learning_rate", &min_learning_rate_);
    task->Fetch("embedding_dims", &embedding_dims_);
    task->Fetch("subsampling", &subsampling_);

    // Load vocabulary.
    normalization_ = ParseNormalization(task->Get("normalization", ""));
    vocabulary_.Load(task->GetInputFile("vocabulary"), subsampling_);
    int vocabulary_size = vocabulary_.size();

    // Build embedding model.
    flow_.inputs = flow_.outputs = vocabulary_size;
    flow_.dims = embedding_dims_;
    flow_.in_features = window_ * 2;
    flow_.Build();

    // Compile embedding model.
    myelin::Compiler compiler;
    myelin::Network model;
    compiler.Compile(&flow_, &model);

    // Initialize weights.
    Random rnd;
    myelin::TensorData W0 = model[flow_.W0];
    myelin::TensorData W1 = model[flow_.W1];
    for (int i = 0; i < vocabulary_size; ++i) {
      for (int j = 0; j < embedding_dims_; ++j) {
        W0.at<float>(i, j) = rnd.UniformFloat(1.0, -0.5);
        W1.at<float>(i, j) = 0.0;
      }
    }

    // Initialize commons store.
    commons_ = new Store();
    docnames_ = new DocumentNames(commons_);
    commons_->Freeze();

    // Statistics.
    num_documents_ = task->GetCounter("documents");
    total_documents_ = task->GetCounter("total_documents");
    num_tokens_ = task->GetCounter("tokens");
    num_instances_ = task->GetCounter("instances");
    epochs_completed_ = task->GetCounter("epochs_completed");

    // Start training threads. Use one worker thread per input file.
    std::vector<string> filenames = task->GetInputFiles("documents");
    int threads = filenames.size();
    task->Fetch("threads", &threads);
    WorkerPool pool;
    pool.Start(threads, [this, &filenames, &model](int index) {
      Worker(index, filenames[index % filenames.size()], &model);
    });

    // Wait until workers completes.
    pool.Join();

    // Output profile.
    myelin::LogProfile(model);

    // Write embeddings to output file.
    const string &output_filename = task->GetOutputFile("output");
    EmbeddingWriter writer(output_filename, vocabulary_size, embedding_dims_);
    std::vector<float> embedding(embedding_dims_);
    for (int i = 0; i < vocabulary_size; ++i) {
      const string &word = vocabulary_.word(i);
      for (int j = 0; j < embedding_dims_; ++j) {
        embedding[j] = W1.at<float>(i, j);
      }
      writer.Write(word, embedding);
    }
    CHECK(writer.Close());

    // Clean up.
    vocabulary_.Clear();
    docnames_->Release();
    delete commons_;
    docnames_ = nullptr;
    commons_ = nullptr;
  }

  // Worker thread for training embedding model.
  void Worker(int index, const string &filename, myelin::Network *model) {
    Random rnd;
    rnd.seed(index);
    int epoch = 0;
    std::vector<int> words;

    // Set up model compute instances.
    myelin::Instance l0(flow_.layer0);
    myelin::Instance l1(flow_.layer1);
    myelin::Instance l0b(flow_.layer0b);

    int *features = l0.Get<int>(flow_.fv);
    int *fend = features + flow_.in_features;
    int *target = l1.Get<int>(flow_.target);
    float *label = l1.Get<float>(flow_.label);
    float *alpha = l1.Get<float>(flow_.alpha);
    *alpha = learning_rate_;

    l1.Set(flow_.l1_l0, &l0);
    l0b.Set(flow_.l0b_l0, &l0);
    l0b.Set(flow_.l0b_l1, &l1);

    RecordFileOptions options;
    RecordReader input(filename, options);
    Record record;
    for (;;) {
      // Check for end of corpus.
      if (input.Done()) {
        epochs_completed_->Increment();
        if (++epoch < iterations_) {
          // Seek back to the beginning.
          input.Rewind();

          // Update learning rate.
          float progress = static_cast<float>(epoch) / iterations_;
          *alpha = learning_rate_ * (1.0 - progress);
          if (*alpha < min_learning_rate_) *alpha = min_learning_rate_;
          continue;
        } else {
          break;
        }
      }

      // Read next record from input.
      CHECK(input.Read(&record));
      num_documents_->Increment();
      if (epoch == 0) total_documents_->Increment();

      // Create document.
      Store store(commons_);
      StringDecoder decoder(&store, record.value.data(), record.value.size());
      Document document(decoder.Decode().AsFrame(), docnames_);
      num_tokens_->Increment(document.num_tokens());

      // Go over each sentence in the document.
      for (SentenceIterator s(&document); s.more(); s.next()) {
        // Get all the words in the sentence with sub-sampling.
        words.clear();
        for (int t = s.begin(); t < s.end(); ++t) {
          // Sub-sample words.
          const string &word = document.token(t).word();
          int index = vocabulary_.Lookup(word, normalization_);
          if (rnd.UniformProb() < vocabulary_.SubsamplingProbability(index)) {
            words.push_back(index);
          }
        }

        // Use each word in the sentence as a training example.
        for (int pos = 0; pos < words.size(); ++pos) {
          // Get features from window around word.
          int *f = features;
          for (int i = pos - window_; i <= pos + window_; ++i) {
            if (i == pos) continue;
            if (i < 0) continue;
            if (i >= words.size()) continue;
            *f++ = words[i];
          }
          if (f == features) continue;
          if (f < fend) *f = -1;
          num_instances_->Increment();

          // Propagate input to hidden layer.
          l0.Compute();

          // Propagate hidden to output and back. This also accumulates the
          // errors that should be propagated back to the input layer.
          l1.Clear(flow_.error);
          *label = 1.0;
          *target = words[pos];
          l1.Compute();

          // Randomly sample negative examples.
          *label = 0.0;
          for (int d = 0; d < negative_; ++d) {
            *target = vocabulary_.Sample(rnd.UniformProb());
            l1.Compute();
          }

          // Propagate hidden to input.
          l0b.Compute();
        }
      }

      // Check for early stopping.
      if (max_epochs_ != -1) {
        if (num_instances_->value() >= max_epochs_) break;
      }
    }
  }

 private:
  // Training parameters.
  int iterations_ = 5;                 // number of training iterations
  int64 max_epochs_ = -1;              // maximum number of training epochs
  int negative_ = 5;                   // negative examples per positive example
  int window_ = 5;                     // window size for skip-grams
  double learning_rate_ = 0.025;       // learning rate
  double min_learning_rate_ = 0.0001;  // minimum learning rate
  int embedding_dims_ = 256;           // size of embedding vectors
  double subsampling_ = 1e-3;          // sub-sampling rate

  // Flow model for word embedding trainer.
  MikolovFlow flow_;

  // Token normalization flags.
  Normalization normalization_;

  // Vocabulary for embeddings.
  WordVocabularySampler vocabulary_;

  // Commons store.
  Store *commons_ = nullptr;
  const DocumentNames *docnames_ = nullptr;

  // Statistics.
  Counter *num_documents_ = nullptr;
  Counter *total_documents_ = nullptr;
  Counter *num_tokens_ = nullptr;
  Counter *num_instances_ = nullptr;
  Counter *epochs_completed_ = nullptr;
};

REGISTER_TASK_PROCESSOR("word-embeddings-trainer", WordEmbeddingsTrainer);

}  // namespace nlp
}  // namespace sling

