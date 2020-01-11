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

#include <math.h>
#include <atomic>
#include <utility>

#include "sling/file/textmap.h"
#include "sling/frame/object.h"
#include "sling/frame/store.h"
#include "sling/frame/serialization.h"
#include "sling/myelin/builder.h"
#include "sling/myelin/compiler.h"
#include "sling/myelin/learning.h"
#include "sling/nlp/embedding/embedding-model.h"
#include "sling/task/frames.h"
#include "sling/task/learner.h"
#include "sling/util/embeddings.h"
#include "sling/util/random.h"

namespace sling {
namespace nlp {

using namespace task;

// Trainer for fact embeddings model.
class FactEmbeddingsTrainer : public LearnerTask {
 public:
  // Run training of embedding net.
  void Run(Task *task) override {
    // Get training parameters.
    task->Fetch("embedding_dims", &embedding_dims_);
    task->Fetch("batch_size", &batch_size_);
    task->Fetch("batches_per_update", &batches_per_update_);
    task->Fetch("max_features", &max_features_);
    task->Fetch("learning_rate", &learning_rate_);
    task->Fetch("min_learning_rate", &min_learning_rate_);

    // Set up counters.
    Counter *num_instances = task->GetCounter("instances");
    Counter *num_instances_skipped = task->GetCounter("instances_skipped");
    num_feature_overflows_ = task->GetCounter("feature_overflows");

    // Bind names.
    names_.Bind(&store_);

    // Read fact lexicon.
    std::vector<string> fact_lexicon;
    TextMapInput factmap(task->GetInputFile("factmap"));
    while (factmap.Next()) fact_lexicon.push_back(factmap.key());
    int fact_dims = fact_lexicon.size();
    task->GetCounter("facts")->Increment(fact_dims);

    // Read category lexicon.
    std::vector<string> category_lexicon;
    TextMapInput catmap(task->GetInputFile("catmap"));
    while (catmap.Next()) {
      category_lexicon.push_back(catmap.key());
    }
    int category_dims = category_lexicon.size();
    task->GetCounter("categories")->Increment(category_dims);

    // Build dual encoder model with facts on the left side and categories on
    // the right side.
    myelin::Compiler compiler;
    flow_.dims = embedding_dims_;
    flow_.batch_size = batch_size_;
    flow_.normalize = task->Get("normalize", false);
    flow_.left.dims = fact_dims;
    flow_.left.max_features = max_features_;
    flow_.right.dims = category_dims;
    flow_.right.max_features = max_features_;
    flow_.Build();
    loss_.Build(&flow_, flow_.sim_cosine, flow_.gsim_d_cosine);
    optimizer_ = GetOptimizer(task);
    optimizer_->Build(&flow_);

    // Compile embedding model.
    myelin::Network model;
    compiler.Compile(&flow_, &model);
    optimizer_->Initialize(model);
    loss_.Initialize(model);

    // Initialize weights.
    model.InitModelParameters(task->Get("seed", 0));

    // Read training instances from input.
    LOG(INFO) << "Reading training data";
    Queue input(this, task->GetSources("input"));
    Message *message;
    while (input.Read(&message)) {
      // Parse message into frame.
      Frame instance = DecodeMessage(&store_, message);
      Array facts = instance.Get(p_facts_).AsArray();
      Array categories = instance.Get(p_categories_).AsArray();
      if (facts.length() > 0 && categories.length() > 0) {
        instances_.push_back(instance.handle());
        num_instances->Increment();
      } else {
        num_instances_skipped->Increment();
      }

      delete message;
    }
    store_.Freeze();

    // Run training.
    Train(task, &model);

    // Write fact embeddings to output file.
    LOG(INFO) << "Writing embeddings";
    myelin::TensorData W0 = model[flow_.left.embeddings];
    std::vector<float> embedding(embedding_dims_);
    EmbeddingWriter fact_writer(task->GetOutputFile("factvecs"),
                                fact_lexicon.size(), embedding_dims_);
    for (int i = 0; i < fact_lexicon.size(); ++i) {
      for (int j = 0; j < embedding_dims_; ++j) {
        embedding[j] = W0.at<float>(i, j);
      }
      fact_writer.Write(fact_lexicon[i], embedding);
    }
    CHECK(fact_writer.Close());

    // Write category embeddings to output file.
    myelin::TensorData W1 =  model[flow_.right.embeddings];
    EmbeddingWriter category_writer(task->GetOutputFile("catvecs"),
                                    category_lexicon.size(), embedding_dims_);
    for (int i = 0; i < category_lexicon.size(); ++i) {
      for (int j = 0; j < embedding_dims_; ++j) {
        embedding[j] = W1.at<float>(i, j);
      }
      category_writer.Write(category_lexicon[i], embedding);
    }
    CHECK(category_writer.Close());

    delete optimizer_;
  }

  // Worker thread for training embedding model.
  void Worker(int index, myelin::Network *model) override {
    // Initialize batch.
    Random rnd;
    rnd.seed(index);
    DualEncoderBatch batch(flow_, *model, loss_);

    for (;;) {
      // Compute gradients for epoch.
      batch.Reset();
      float epoch_loss = 0.0;
      for (int b = 0; b < batches_per_update_; ++b) {
        // Random sample instances for batch.
        for (int i = 0; i < flow_.batch_size; ++i) {
          int sample = rnd.UniformInt(instances_.size());
          Frame instance(&store_, instances_[sample]);
          Array facts = instance.Get(p_facts_).AsArray();
          Array categories = instance.Get(p_categories_).AsArray();

          // Set fact features for instance.
          int *f = batch.left_features(i);
          int *fend = f + flow_.left.max_features;
          for (int i = 0; i < facts.length(); ++i) {
            if (f == fend) {
              num_feature_overflows_->Increment();
              break;
            }
            *f++ = facts.get(i).AsInt();
          }
          if (f < fend) *f = -1;

          // Set category features for instance.
          int *c = batch.right_features(i);
          int *cend = c + flow_.right.max_features;
          for (int i = 0; i < categories.length(); ++i) {
            if (c == cend) {
              num_feature_overflows_->Increment();
              break;
            }
            *c++ = categories.get(i).AsInt();
          }
          if (c < cend) *c = -1;
        }

        // Process batch.
        float loss = batch.Compute();
        epoch_loss += loss;
      }

      // Update parameters.
      optimizer_mu_.Lock();
      optimizer_->Apply(batch.gradients());
      loss_sum_ += epoch_loss;
      loss_count_ += batches_per_update_;
      optimizer_mu_.Unlock();

      // Check if we are done.
      if (EpochCompleted()) break;
    }
  }

  // Evaluate model.
  bool Evaluate(int64 epoch, myelin::Network *model) override {
    // Skip evaluation if there are no data.
    if (loss_count_ == 0) return true;

    // Compute average loss of epochs since last eval.
    float loss = loss_sum_ / loss_count_;
    float p = exp(-loss) * 100.0;
    loss_sum_ = 0.0;
    loss_count_ = 0;

    // Decay learning rate if loss increases.
    if (prev_loss_ != 0.0 &&
        prev_loss_ < loss &&
        learning_rate_ > min_learning_rate_) {
      learning_rate_ = optimizer_->DecayLearningRate();
    }
    prev_loss_ = loss;

    LOG(INFO) << "epoch=" << epoch
              << ", lr=" << learning_rate_
              << ", loss=" << loss
              << ", p=" << p;

    return true;
  }

 private:
  // Flow model for fact embedding trainer.
  DualEncoderFlow flow_;
  myelin::CrossEntropyLoss loss_;
  myelin::Optimizer *optimizer_ = nullptr;

  // Store for training instances.
  Store store_;

  // Training instances.
  Handles instances_{&store_};

  // Training parameters.
  int embedding_dims_ = 256;           // size of embedding vectors
  int max_features_ = 512;             // maximum features per item
  int batch_size_ = 1024;              // number of examples per batch
  int batches_per_update_ = 1;         // number of batches per epoch

  // Mutex for serializing access to optimizer.
  Mutex optimizer_mu_;

  // Evaluation statistics.
  float learning_rate_ = 1.0;
  float min_learning_rate_ = 0.01;
  float prev_loss_ = 0.0;
  float loss_sum_ = 0.0;
  int loss_count_ = 0.0;

  // Symbols.
  Names names_;
  Name p_item_{names_, "item"};
  Name p_facts_{names_, "facts"};
  Name p_categories_{names_, "categories"};

  // Statistics.
  Counter *num_feature_overflows_ = nullptr;
};

REGISTER_TASK_PROCESSOR("fact-embeddings-trainer", FactEmbeddingsTrainer);

}  // namespace nlp
}  // namespace sling
