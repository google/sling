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

#include "sling/file/textmap.h"
#include "sling/frame/object.h"
#include "sling/frame/serialization.h"
#include "sling/frame/store.h"
#include "sling/myelin/compiler.h"
#include "sling/myelin/builder.h"
#include "sling/myelin/gradient.h"
#include "sling/myelin/learning.h"
#include "sling/task/frames.h"
#include "sling/task/learner.h"
#include "sling/util/random.h"

namespace sling {
namespace nlp {

using namespace task;
using namespace myelin;

// Fact plausibility flow.
struct FactPlausibilityFlow : public Flow {
  // Build fact plausibility model.
  void Build(bool learn) {
    scorer = AddFunction("scorer");
    FlowBuilder f(this, scorer);
    auto *embeddings =
        f.RandomNormal(f.Parameter("embeddings", DT_FLOAT, {facts, dims}));

    premise = f.Placeholder("premise", DT_INT32, {1, max_features});
    auto *pencoding = f.GatherSum(embeddings, premise);

    hypothesis = f.Placeholder("hypothesis", DT_INT32, {1, max_features});
    auto *hencoding = f.GatherSum(embeddings, hypothesis);

    auto *fv = f.Concat({pencoding, hencoding});
    logits = f.Name(f.FFLayers(fv, {dims * 2, 2}, -1, true, "Relu"), "logits");

    if (learn) {
      // Create gradient computations.
      gscorer = Gradient(this, scorer);
      d_logits = GradientVar(logits);
      primal = PrimalVar(scorer);
    } else {
      // Output probabilities. This outputs two probabilities. The first is the
      // probability that the fact is false (implausible) and the second is the
      // probability that the fact is true (plausible), such that
      // p(plausible) + p(implausible) = 1.
      probs = f.Name(f.Softmax(logits), "probs");
    }
  }

  int facts = 1;                   // number of fact types
  int dims = 64;                   // dimension of embedding vectors
  int max_features = 512;          // maximum number of features per example

  Function *scorer;                // plausibility scoring function
  Function *gscorer;               // plausibility scoring gradient function
  Variable *premise;               // premise facts
  Variable *hypothesis;            // hypothesis facts
  Variable *logits;                // plausibility prediction
  Variable *d_logits;              // plausibility gradient
  Variable *primal;                // primal reference for scorer
  Variable *probs;                 // output probabilities
};

// Trainer for fact plausibility model.
class FactPlausibilityTrainer : public LearnerTask {
 public:
  // Feature vector.
  typedef std::vector<int> Features;

  // Run training of embedding net.
  void Run(task::Task *task) override {
    // Get training parameters.
    task->Fetch("embedding_dims", &embedding_dims_);
    task->Fetch("batch_size", &batch_size_);
    task->Fetch("batches_per_update", &batches_per_update_);
    task->Fetch("min_facts", &min_facts_);
    task->Fetch("max_features", &max_features_);
    task->Fetch("learning_rate", &learning_rate_);
    task->Fetch("min_learning_rate", &min_learning_rate_);
    task->Fetch("eval_ratio", &eval_ratio_);
    task->Fetch("seed", &seed_);

    // Set up counters.
    Counter *num_train_instances = task->GetCounter("training_instances");
    Counter *num_eval_instances = task->GetCounter("evaluation_instances");
    Counter *num_facts = task->GetCounter("facts");
    Counter *num_fact_groups = task->GetCounter("fact_groups");
    Counter *num_skipped_instances = task->GetCounter("skipped_instances");
    num_training_examples_ = task->GetCounter("training_examples");
    num_feature_overflows_ = task->GetCounter("feature_overflows");
    num_contradictions_ = task->GetCounter("contradictions");

    // Bind names.
    names_.Bind(&store_);

    // Get model output filename.
    model_filename_ = task->GetOutputFile("model");

    // Read fact lexicon.
    TextMapInput factmap(task->GetInputFile("factmap"));
    Handles facts(&store_);
    while (factmap.Next()) {
      Object fact = FromText(&store_, factmap.key());
      facts.push_back(fact.handle());
    }
    fact_lexicon_ = Array(&store_, facts);
    task->GetCounter("facts")->Increment(fact_lexicon_.length());

    // Build plausibility model.
    Build(&flow_, true);
    loss_.Build(&flow_, flow_.logits, flow_.d_logits);
    optimizer_ = GetOptimizer(task);
    optimizer_->Build(&flow_);

    // Compile plausibility model.
    Network model;
    compiler_.Compile(&flow_, &model);
    optimizer_->Initialize(model);
    loss_.Initialize(model);

    // Initialize weights.
    model.InitModelParameters(seed_);

    // Read training instances from input.
    LOG(INFO) << "Reading training data";
    Queue input(this, task->GetSources("input"));
    Message *message;
    Random rnd;
    rnd.seed(seed_);
    while (input.Read(&message)) {
      // Parse message into frame.
      Frame instance = DecodeMessage(&store_, message);
      Array facts = instance.Get(p_facts_).AsArray();
      Array groups = instance.Get(p_groups_).AsArray();
      if (groups.length() >= min_facts_) {
        // Split into training and evaluation set.
        if (eval_ratio_ == 0 || rnd.UniformProb() > eval_ratio_) {
          training_instances_.push_back(instance.handle());
          num_train_instances->Increment();
        } else {
          evaluation_instances_.push_back(instance.handle());
          num_eval_instances->Increment();
        }
      } else {
        num_skipped_instances->Increment();
      }
      num_facts->Increment(facts.length());
      num_fact_groups->Increment(groups.length());

      delete message;
    }
    store_.Freeze();

    // Run training.
    Train(task, &model);

    // Save final model.
    if (!model_filename_.empty()) {
      LOG(INFO) << "Saving model to " << model_filename_;
      Save(model, model_filename_);
    }

    delete optimizer_;
  }

  // Worker thread for training embedding model.
  void Worker(int index, Network *model) override {
    // Initialize random number generator.
    Random rnd;
    rnd.seed(seed_ + index);

    // Premises and hypotheses for one batch.
    std::vector<Features> premises(batch_size_);
    std::vector<Features> hypotheses(batch_size_);

    // Set up plausibility scorer.
    Instance scorer(flow_.scorer);
    Instance gscorer(flow_.gscorer);
    int *premise = scorer.Get<int>(flow_.premise);
    int *hypothesis = scorer.Get<int>(flow_.hypothesis);
    float *logits = scorer.Get<float>(flow_.logits);
    float *dlogits = gscorer.Get<float>(flow_.d_logits);
    std::vector<Instance *> gradients{&gscorer};

    for (;;) {
      // Compute gradients for epoch.
      gscorer.Clear();
      gscorer.Set(flow_.primal, &scorer);
      float epoch_loss = 0.0;
      int epoch_count = 0;
      Benchmark batch_benchmark;

      for (int b = 0; b < batches_per_update_; ++b) {
        // Random sample instances for batch.
        for (int i = 0; i < batch_size_; ++i) {
          int sample = rnd.UniformInt(training_instances_.size());
          Frame instance(&store_, training_instances_[sample]);

          // Split instance into premise and hypothesis.
          int num_groups = instance.Get(p_groups_).AsArray().length();
          int heldout = rnd.UniformInt(num_groups);
          Split(instance, &premises[i], &hypotheses[i], heldout);
        }

        // Do forward and backward propagation for each premise/hypothesis pair.
        // Each sampled item is a positive example. Negative examples are
        // generated by using the premise from one item and the hypothesis from
        // another item.
        for (int i = 0; i < batch_size_; ++i) {
          for (int j = 0; j < batch_size_; ++j) {
            // Check that main hypothesis is not in premise for negative
            // examples.
            if (i != j && Has(premises[i], hypotheses[j][0])) {
              num_contradictions_->Increment();
              continue;
            }

            // Set premise and hypothesis for example.
            Copy(premises[i], premise);
            Copy(hypotheses[j], hypothesis);

            // Compute plausibility scores.
            scorer.Compute();

            // Compute accuracy.
            batch_benchmark.add(i == j, logits[1] > logits[0]);

            // Compute loss.
            int label = i == j ? 1 : 0;
            float loss = loss_.Compute(logits, label, dlogits);
            epoch_loss += loss;
            epoch_count++;

            // Backpropagate.
            gscorer.Compute();
            num_training_examples_->Increment();
          }
        }
      }

      // Update parameters.
      optimizer_mu_.Lock();
      optimizer_->Apply(gradients);
      loss_sum_ += epoch_loss;
      loss_count_ += epoch_count;
      train_benchmark_.add(batch_benchmark);
      optimizer_mu_.Unlock();

      // Check if we are done.
      if (EpochCompleted()) break;
    }
  }

  // Evaluate model.
  bool Evaluate(int64 epoch, Network *model) override {
    // Skip evaluation if there are no data.
    if (loss_count_ == 0) return true;

    // Compute average loss of epochs since last eval.
    float loss = loss_sum_ / loss_count_;
    float p = exp(-loss) * 100.0;
    loss_sum_ = 0.0;
    loss_count_ = 0;

    // Compute accuracy on training data for positive and negative examples.
    float pos_accuracy = train_benchmark_.positive_accuracy() * 100.0;
    float neg_accuracy = train_benchmark_.negative_accuracy() * 100.0;
    train_benchmark_.clear();

    // Compute accuracy in evaluation data.
    Benchmark eval;
    EvaluateModel(evaluation_instances_, &eval);

    // Decay learning rate if loss increases.
    if (prev_loss_ != 0.0 &&
        prev_loss_ < loss &&
        learning_rate_ > min_learning_rate_) {
      learning_rate_ = optimizer_->DecayLearningRate();
    }
    prev_loss_ = loss;

    LOG(INFO) << "epoch=" << epoch
              << " lr=" << learning_rate_
              << " loss=" << loss
              << " p=" << p
              << " train: +ve=" << pos_accuracy
              << " -ve=" << neg_accuracy
              << " eval: +ve=" << (eval.positive_accuracy() * 100.0)
              << " -ve=" << (eval.negative_accuracy() * 100.0);

    return true;
  }

  // Checkpoint model.
  void Checkpoint(int64 epoch, Network *model) override {
    if (!model_filename_.empty()) {
      LOG(INFO) << "Checkpoint model to " << model_filename_;
      Save(*model, model_filename_);
    }
  }

 private:
  // Benchmark statistics.
  struct Benchmark {
    // Clear statistics.
    void clear() {
      positive_correct = positive_wrong = negative_correct = negative_wrong = 0;
    }

    // Add result to benchmark.
    void add(bool golden, bool predicted) {
      if (golden) {
        if (predicted) {
          positive_correct++;
        } else {
          positive_wrong++;
        }
      } else {
        if (predicted) {
          negative_wrong++;
        } else {
          negative_correct++;
        }
      }
    }

    // Add statistics from another benchmark.
    void add(const Benchmark &other) {
      positive_correct += other.positive_correct;
      positive_wrong += other.positive_wrong;
      negative_correct += other.negative_correct;
      negative_wrong += other.negative_wrong;
    }

    // Total nummber of positive/negative examples.
    float positive() const { return positive_correct + positive_wrong; }
    float negative() const { return negative_correct + negative_wrong; }

    // Accuracy for positive/negative examples.
    float positive_accuracy() const { return positive_correct / positive(); }
    float negative_accuracy() const { return negative_correct / negative(); }

    // Number of positive/negative correct/wrong predictions.
    int positive_correct = 0;
    int positive_wrong = 0;
    int negative_correct = 0;
    int negative_wrong = 0;
  };

  // Evaluate model on positive and negative examples generated from instances.
  void EvaluateModel(const Handles &instances, Benchmark *benchmark) {
    static const int batch_size = 2;
    Random rnd;
    rnd.seed(seed_);
    std::vector<Features> premises(batch_size);
    std::vector<Features> hypotheses(batch_size);

    Instance scorer(flow_.scorer);
    int *premise = scorer.Get<int>(flow_.premise);
    int *hypothesis = scorer.Get<int>(flow_.hypothesis);
    float *logits = scorer.Get<float>(flow_.logits);

    for (int n = 0; n < instances.size(); ++n) {
      // Get instances for batch.
      for (int i = 0; i < batch_size; ++i) {
        int sample = rnd.UniformInt(instances.size());
        Frame instance(&store_, instances[sample]);

        // Split instance into premise and hypothesis.
        int num_groups = instance.Get(p_groups_).AsArray().length();
        int heldout = rnd.UniformInt(num_groups);
        Split(instance, &premises[i], &hypotheses[i], heldout);
      }

      // Compute scores for all pairs in batch.
      for (int i = 0; i < batch_size; ++i) {
        for (int j = 0; j < batch_size; ++j) {
          // Check that main hypothesis is not in premise for negative examples.
          if (i != j && Has(premises[i], hypotheses[j][0])) continue;

          // Set premise and hypothesis for example.
          Copy(premises[i], premise);
          Copy(hypotheses[j], hypothesis);

          // Compute plausibility scores.
          scorer.Compute();

          // Compute accuracy.
          benchmark->add(i == j, logits[1] > logits[0]);
        }
      }
    }
  }

  // Split instance into premise and hypothesis.
  void Split(const Frame &instance,
             Features *premise, Features *hypothesis,
             int heldout) {
    Array facts = instance.Get(p_facts_).AsArray();
    Array groups = instance.Get(p_groups_).AsArray();
    int num_groups = groups.length();

    // Add facts to premise, except for one heldout fact group which is
    // added to the hypothesis.
    premise->clear();
    hypothesis->clear();
    for (int g = 0; g < num_groups; ++g) {
      // Get range for fact group.
      int start = g == 0 ? 0 : groups.get(g - 1).AsInt();
      int end = groups.get(g).AsInt();

      if (g == heldout) {
        // Add fact group to hyothesis.
        for (int f = start; f < end; ++f) {
          hypothesis->push_back(facts.get(f).AsInt());
        }
      } else {
        // Add fact group to premise.
        for (int f = start; f < end; ++f) {
          premise->push_back(facts.get(f).AsInt());
        }
      }
    }

    if (premise->size() >= max_features_) {
      premise->resize(max_features_ - 1);
      num_feature_overflows_->Increment();
    }
    if (hypothesis->size() >= max_features_) {
      hypothesis->resize(max_features_ - 1);
      num_feature_overflows_->Increment();
    }
  }

  // Copy features into feature vector and terminate it with -1.
  static void Copy(const Features &src, int *dest) {
    for (int f : src) *dest++ = f;
    *dest = -1;
  }

  // Check if feature is in feature vector.
  static bool Has(const Features &fv, int feature) {
    for (int f : fv) {
      if (f == feature) return true;
    }
    return false;
  }

  // Add plausibility model to flow.
  void Build(FactPlausibilityFlow *flow, bool learn) {
    flow->facts = fact_lexicon_.length();
    flow->dims = embedding_dims_;
    flow->max_features = max_features_;
    flow->Build(learn);
  }

  // Save trained model to file.
  void Save(const Network &model, const string &filename) {
    // Build model.
    FactPlausibilityFlow flow;
    Build(&flow, false);

    // Copy weights from trained model.
    model.SaveParameters(&flow);

    // Add fact lexicon.
    string encoded = Encode(fact_lexicon_);
    Flow::Blob *facts = flow.AddBlob("facts", "store");
    facts->data = flow.AllocateMemory(encoded);
    facts->size = encoded.size();

    // Save model to file.
    flow.Save(filename);
  }

  // Training parameters.
  int embedding_dims_ = 128;           // size of embedding vectors
  int min_facts_ = 4;                  // minimum number of facts for example
  int max_features_ = 512;             // maximum features per item
  int batch_size_ = 4;                 // number of examples per batch
  int batches_per_update_ = 256;       // number of batches per epoch
  float eval_ratio_ = 0.001;           // train/eval split ratio
  float learning_rate_ = 1.0;
  float min_learning_rate_ = 0.001;

  // Store for training instances.
  Store store_;

  // Fact lexicon.
  Array fact_lexicon_;

  // Flow model for fact plausibility trainer.
  FactPlausibilityFlow flow_;
  Compiler compiler_;
  CrossEntropyLoss loss_;
  Optimizer *optimizer_ = nullptr;

  // Training and test instances.
  Handles training_instances_{&store_};
  Handles evaluation_instances_{&store_};

  // Mutex for serializing access to optimizer.
  Mutex optimizer_mu_;

  // Evaluation statistics.
  float prev_loss_ = 0.0;
  float loss_sum_ = 0.0;
  int loss_count_ = 0.0;
  Benchmark train_benchmark_;

  // Seed for random number generator.
  int seed_ = 0;

  // Output filename for trained model.
  string model_filename_;

  // Symbols.
  Names names_;
  Name p_item_{names_, "item"};
  Name p_facts_{names_, "facts"};
  Name p_groups_{names_, "groups"};

  // Statistics.
  Counter *num_training_examples_ = nullptr;
  Counter *num_feature_overflows_ = nullptr;
  Counter *num_contradictions_ = nullptr;
};

REGISTER_TASK_PROCESSOR("fact-plausibility-trainer", FactPlausibilityTrainer);

}  // namespace nlp
}  // namespace sling

