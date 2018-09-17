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
#include "sling/nlp/kb/facts.h"
#include "sling/task/frames.h"
#include "sling/task/learner.h"
#include "sling/util/bloom.h"
#include "sling/util/embeddings.h"
#include "sling/util/random.h"
#include "sling/util/sortmap.h"

namespace sling {
namespace nlp {

using namespace task;

// Extract fact and category lexicons from items.
class FactLexiconExtractor : public Process {
 public:
  void Run(Task *task) override {
    // Get parameters.
    int64 bloom_size = task->Get("bloom_size", 4000000000L);
    int bloom_hashes = task->Get("bloom_hashes", 4);
    int fact_threshold = task->Get("fact_threshold", 10);
    int category_threshold = task->Get("category_threshold", 10);

    // Set up counters.
    Counter *num_items = task->GetCounter("items");
    Counter *num_facts = task->GetCounter("facts");
    Counter *num_fact_types = task->GetCounter("fact_types");
    Counter *num_filtered = task->GetCounter("filtered_facts");
    Counter *num_facts_selected = task->GetCounter("facts_selected");
    Counter *num_categories_selected = task->GetCounter("categories_selected");

    // Load knowledge base.
    Store commons;
    LoadStore(task->GetInputFile("kb"), &commons);

    // Resolve symbols.
    Names names;
    Name p_item_category(names, "/w/item/category");
    Name n_item(names, "/w/item");
    Name p_instance_of(names, "P31");
    Name n_wikimedia_category(names, "Q4167836");
    Name n_wikimedia_disambiguation(names, "Q4167836");

    names.Bind(&commons);

    // Initialize fact catalog.
    FactCatalog catalog;
    catalog.Init(&commons);
    commons.Freeze();

    // A Bloom filter is used for checking for singleton facts. It is used as
    // a fast and compact check for detecting if a fact is a new fact. The
    // probabilistic nature of the Bloom filter means that the fact instance
    // counts can be off by one.
    BloomFilter filter(bloom_size, bloom_hashes);

    // The categories are collected in a sortable hash map so the most frequent
    // categories can be selected.
    SortableMap<Handle, int64, HandleHash> category_lexicon;

    // The facts are collected in a sortable hash map where the key is the
    // fact fingerprint. The name of the fact is stored in a nul-terminated
    // dynamically allocated string.
    SortableMap<int64, std::pair<int64, char *>> fact_lexicon;

    // Extract facts from all items in the knowledge base.
    commons.ForAll([&](Handle handle) {
      Frame item(&commons, handle);
      if (!item.IsA(n_item)) return;

      // Skip categories and disambiguation page items.
      Handle cls = item.GetHandle(p_instance_of);
      if (cls == n_wikimedia_category) return;
      if (cls == n_wikimedia_disambiguation) return;

      // Extract facts from item.
      Store store(&commons);
      Facts facts(&catalog, &store);
      facts.Extract(handle);

      // Add facts to fact lexicon.
      for (Handle fact : facts.list()) {
        int64 fp = store.Fingerprint(fact);
        if (filter.add(fp)) {
          auto &entry = fact_lexicon[fp];
          if (entry.second == nullptr) {
            entry.second = strdup(ToText(&store, fact).c_str());
            num_fact_types->Increment();
          }
          entry.first++;
        } else {
          num_filtered->Increment();
        }
      }
      num_facts->Increment(facts.list().size());

      // Extract categories from item.
      for (const Slot &s : item) {
        if (s.name == p_item_category) {
          category_lexicon[s.value]++;
        }
      }

      num_items->Increment();
    });
    task->GetCounter("num_categories")->Increment(category_lexicon.map.size());

    // Write fact lexicon to text map.
    fact_lexicon.sort();
    TextMapOutput factout(task->GetOutputFile("factmap"));
    for (int i = fact_lexicon.array.size() - 1; i >= 0; --i) {
      Text fact(fact_lexicon.array[i]->second.second);
      int64 count = fact_lexicon.array[i]->second.first;
      if (count < fact_threshold) break;
      factout.Write(fact, count);
      num_facts_selected->Increment();
    }
    factout.Close();

    // Write category lexicon to text map.
    category_lexicon.sort();
    TextMapOutput catout(task->GetOutputFile("catmap"));
    for (int i = category_lexicon.array.size() - 1; i >= 0; --i) {
      Frame cat(&commons, category_lexicon.array[i]->first);
      int64 count = category_lexicon.array[i]->second;
      if (count < category_threshold) break;
      catout.Write(cat.Id(), count);
      num_categories_selected->Increment();
    }
    catout.Close();

    // Clean up.
    for (auto &it : fact_lexicon.map) free(it.second.second);
  }
};

REGISTER_TASK_PROCESSOR("fact-lexicon-extractor", FactLexiconExtractor);

// Extract facts items.
class FactExtractor : public Process {
 public:
  void Run(Task *task) override {
    // Set up counters.
    Counter *num_items = task->GetCounter("items");
    Counter *num_facts = task->GetCounter("facts");
    Counter *num_facts_extracted = task->GetCounter("facts_extracted");
    Counter *num_facts_skipped = task->GetCounter("facts_skipped");
    Counter *num_no_facts = task->GetCounter("items_without_facts");
    Counter *num_cats = task->GetCounter("categories");
    Counter *num_cats_extracted = task->GetCounter("categories_extracted");
    Counter *num_cats_skipped = task->GetCounter("categories_skipped");
    Counter *num_no_cats = task->GetCounter("items_without_categories");

    // Load knowledge base.
    LoadStore(task->GetInputFile("kb"), &commons_);

    // Resolve symbols.
    names_.Bind(&commons_);

    // Initialize fact catalog.
    FactCatalog catalog;
    catalog.Init(&commons_);
    commons_.Freeze();

    // Read fact and category lexicons.
    ReadFactLexicon(task->GetInputFile("factmap"));
    ReadCategoryLexicon(task->GetInputFile("catmap"));

    // Get output channel for resolved fact frames.
    Channel *output = task->GetSink("output");

    // Extract facts from all items in the knowledge base.
    commons_.ForAll([&](Handle handle) {
      Frame item(&commons_, handle);
      if (!item.IsA(n_item_)) return;

      // Skip categories and disambiguation page items.
      Handle cls = item.GetHandle(p_instance_of_);
      if (cls == n_wikimedia_category_) return;
      if (cls == n_wikimedia_disambiguation_) return;

      // Extract facts from item.
      Store store(&commons_);
      Facts facts(&catalog, &store);
      facts.Extract(handle);

      // Add facts to fact lexicon.
      Handles fact_indices(&store);
      for (Handle fact : facts.list()) {
        uint64 fp = store.Fingerprint(fact);
        auto f = fact_lexicon_.find(fp);
        if (f != fact_lexicon_.end()) {
          fact_indices.push_back(Handle::Integer(f->second));
        }
      }
      int total = facts.list().size();
      int extracted = fact_indices.size();
      int skipped = total - extracted;
      num_facts->Increment(total);
      num_facts_extracted->Increment(extracted);
      num_facts_skipped->Increment(skipped);
      if (extracted == 0) num_no_facts->Increment();

      // Extract categories from item.
      Handles category_indices(&store);
      for (const Slot &s : item) {
        if (s.name == p_item_category_) {
          auto f = category_lexicon_.find(s.value);
          if (f != category_lexicon_.end()) {
            category_indices.push_back(Handle::Integer(f->second));
            num_cats_extracted->Increment();
          } else {
            num_cats_skipped->Increment();
          }
          num_cats->Increment();
        }
      }
      if (category_indices.empty()) num_no_cats->Increment();

      // Build frame with resolved facts.
      Builder builder(&store);
      builder.Add(p_item_, item.id());
      builder.Add(p_facts_, Array(&store, fact_indices));
      builder.Add(p_categories_, Array(&store, category_indices));

      // Output frame with resolved facts on output channel.
      output->Send(CreateMessage(item.Id(), builder.Create()));
      num_items->Increment();
    });
  }

 private:
  // Read fact lexicon.
  void ReadFactLexicon(const string &filename) {
    Store store(&commons_);
    TextMapInput factmap(filename);
    string key;
    int index;
    while (factmap.Read(&index, &key, nullptr)) {
      uint64 fp = FromText(&store, key).Fingerprint();
      fact_lexicon_[fp] = index;
    }
  }

  // Read category lexicon.
  void ReadCategoryLexicon(const string &filename) {
    TextMapInput catmap(filename);
    string key;
    int index;
    while (catmap.Read(&index, &key, nullptr)) {
      Handle cat = commons_.Lookup(key);
      category_lexicon_[cat] = index;
    }
  }

  // Commons store with knowledge base.
  Store commons_;

  // Fact lexicon mapping from fact fingerprint to fact index.
  std::unordered_map<uint64, int> fact_lexicon_;

  // Category lexicon mapping from category handle to category index.
  HandleMap<int> category_lexicon_;

  // Symbols.
  Names names_;
  Name p_item_category_{names_, "/w/item/category"};
  Name n_item_{names_, "/w/item"};
  Name p_instance_of_{names_, "P31"};
  Name n_wikimedia_category_{names_, "Q4167836"};
  Name n_wikimedia_disambiguation_{names_, "Q4167410"};

  Name p_item_{names_, "item"};
  Name p_facts_{names_, "facts"};
  Name p_categories_{names_, "categories"};
};

REGISTER_TASK_PROCESSOR("fact-extractor", FactExtractor);

// Trainer for fact embeddings model.
class FactEmbeddingsTrainer : public LearnerTask {
 public:
  // Run training of embedding net.
  void Run(Task *task) override {
    // Get training parameters.
    task->Fetch("embedding_dims", &embedding_dims_);
    task->Fetch("batch_size", &batch_size_);
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
    flow_.Build(*compiler.library());
    loss_.Build(&flow_, flow_.sim_cosine, flow_.gsim_d_cosine);
    optimizer_ = GetOptimizer(task);
    optimizer_->Build(&flow_);

    // Compile embedding model.
    myelin::Network model;
    compiler.Compile(&flow_, &model);
    optimizer_->Initialize(model);
    loss_.Initialize(model);

    // Initialize weights.
    model.InitLearnableWeights(task->Get("seed", 0), 0.0, 0.01);

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
    LOG(INFO) << "Starting training";
    Train(task, &model);

    // Output profile.
    myelin::LogProfile(model);

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
      // Reset gradients.
      batch.Reset();

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

      // Update parameters.
      optimizer_mu_.Lock();
      optimizer_->Apply(batch.gradients());
      loss_sum_ += loss;
      loss_count_++;
      optimizer_mu_.Unlock();

      // Check if we are done.
      if (EpochCompleted()) break;
    }
  }

  // Evaluate model.
  bool Evaluate(int64 epoch, myelin::Network *model) override {
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
  int batch_size_ = 1024;              // number of examples per epoch

  // Mutex for serializing access to optimizer.
  Mutex optimizer_mu_;

  // Evaluation statistics.
  float learning_rate_ = 0.01;
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
