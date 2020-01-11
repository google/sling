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

#ifndef SLING_NLP_PARSER_PARSER_TRAINER_H_
#define SLING_NLP_PARSER_PARSER_TRAINER_H_

#include <string>
#include <unordered_map>
#include <vector>

#include "sling/frame/store.h"
#include "sling/myelin/flow.h"
#include "sling/myelin/builder.h"
#include "sling/myelin/compiler.h"
#include "sling/myelin/compute.h"
#include "sling/nlp/document/document.h"
#include "sling/nlp/document/document-corpus.h"
#include "sling/nlp/document/lexical-encoder.h"
#include "sling/nlp/parser/frame-evaluation.h"
#include "sling/nlp/parser/parser-action.h"
#include "sling/nlp/parser/parser-features.h"
#include "sling/nlp/parser/roles.h"
#include "sling/task/learner.h"
#include "sling/util/mutex.h"

namespace sling {
namespace nlp {

class DelegateLearnerInstance;

// Interface for delegate learner.
class DelegateLearner {
 public:
  virtual ~DelegateLearner() = default;

  // Build flow for delegate learner.
  virtual void Build(myelin::Flow *flow,
                     myelin::Flow::Variable *activation,
                     myelin::Flow::Variable *dactivation,
                     bool learn) = 0;

  // Initialize network for delegate.
  virtual void Initialize(const myelin::Network &network) = 0;

  // Create instance of delegate.
  virtual DelegateLearnerInstance *CreateInstance() = 0;

  // Save model data to flow.
  virtual void Save(myelin::Flow *flow, Builder *spec) = 0;
};

// Interface for delegate learner instance.
class DelegateLearnerInstance {
 public:
  virtual ~DelegateLearnerInstance() = default;

  // Collect gradients.
  virtual void CollectGradients(std::vector<myelin::Instance *> *gradients) = 0;

  // Clear gradients.
  virtual void ClearGradients() = 0;

  // Compute loss and gradient for delegate with respect to golden action.
  virtual float Compute(float *activation,
                        float *dactivation,
                        const ParserAction &action) = 0;

  // Predict action for delegate.
  virtual void Predict(float *activations, ParserAction *action) = 0;
};

// Basic task for training for transition-based frame-semantic parser.
class ParserTrainer : public task::LearnerTask {
 public:
  virtual ~ParserTrainer();

  // Learner task interface.
  void Run(task::Task *task) override;
  void Worker(int index, myelin::Network *model) override;
  bool Evaluate(int64 epoch, myelin::Network *model) override;
  void Checkpoint(int64 epoch, myelin::Network *model) override;

  // Abstract method for setting up parser trainer.
  virtual void Setup(task::Task *task) = 0;

  // Abstract method for converting document to transition sequence.
  virtual void GenerateTransitions(const Document &document,
                                   std::vector<ParserAction> *transitions) = 0;

 private:
  // Build flow graph for parser model.
  void Build(myelin::Flow *flow, bool learn);

  // Build linked feature.
  static myelin::Flow::Variable *LinkedFeature(
      myelin::FlowBuilder *f,
      const string &name,
      myelin::Flow::Variable *embeddings,
      int size, int dim);

  // Read next training document into store. The caller owns the returned
  // document.
  Document *GetNextTrainingDocument(Store *store);

  // Parse document using current model.
  void Parse(Document *document) const;

  // Save trained model to file.
  void Save(const string &filename);

 protected:
  // Parallel corpus for evaluating parser on golden corpus.
  class ParserEvaulationCorpus : public ParallelCorpus {
   public:
    ParserEvaulationCorpus(ParserTrainer *trainer);

    // Parse next evaluation document using parser model.
    bool Next(Store **store, Document **golden, Document **predicted) override;

   private:
    ParserTrainer *trainer_;   // parser trainer with current model
  };

  // Commons store for parser.
  Store commons_;

  // Training corpus.
  DocumentCorpus *training_corpus_ = nullptr;

  // Evaluation corpus.
  DocumentCorpus *evaluation_corpus_ = nullptr;

  // File name for trained model.
  string model_filename_;

  // Word vocabulary.
  std::unordered_map<string, int> words_;

  // Role set.
  RoleSet roles_;

  // Reset parser state between sentences in a document.
  bool sentence_reset_ = false;

  // Lexical feature specification for encoder.
  LexicalFeatures::Spec spec_;

  // Neural network.
  myelin::Flow flow_;
  myelin::Network model_;
  myelin::Compiler compiler_;
  myelin::Optimizer *optimizer_ = nullptr;

  // Document input encoder.
  LexicalEncoder encoder_;

  // Parser feature model.
  ParserFeatureModel feature_model_;

  // Decoder model.
  myelin::Cell *decoder_ = nullptr;
  myelin::Tensor *encodings_ = nullptr;
  myelin::Tensor *activations_ = nullptr;
  myelin::Tensor *activation_ = nullptr;

  myelin::Cell *gdecoder_ = nullptr;
  myelin::Tensor *primal_ = nullptr;
  myelin::Tensor *dencodings_ = nullptr;
  myelin::Tensor *dactivations_ = nullptr;
  myelin::Tensor *dactivation_ = nullptr;

  // Delegates.
  std::vector<DelegateLearner *> delegates_;

  // Mutexes for serializing access to global state.
  Mutex input_mu_;
  Mutex update_mu_;

  // Model hyperparameters.
  int rnn_type_ = myelin::RNN::LSTM;
  int rnn_dim_ = 256;
  int rnn_layers_ = 1;
  bool rnn_bidir_ = true;
  bool rnn_highways_ = false;
  int mark_depth_ = 1;
  int frame_limit_ = 5;
  int history_size_ = 5;
  int out_roles_size_ = 32;
  int in_roles_size_ = 32;
  int labeled_roles_size_ = 32;
  int unlabeled_roles_size_ = 32;
  int roles_dim_ = 16;
  int activations_dim_ = 128;
  int link_dim_token_ = 32;
  int link_dim_step_ = 64;
  int mark_dim_ = 32;
  int seed_ = 0;
  int batch_size_ = 32;
  int learning_rate_cliff_ = 0;
  float learning_rate_ = 1.0;
  float min_learning_rate_ = 0.001;
  float dropout_ = 0.0;
  float ff_l2reg_ = 0.0;

  // Evaluation statistics.
  float prev_loss_ = 0.0;
  float loss_sum_ = 0.0;
  int loss_count_ = 0.0;

  // Statistics.
  task::Counter *num_documents_;
  task::Counter *num_tokens_;
  task::Counter *num_transitions_;
};

}  // namespace nlp
}  // namespace sling

#endif  // SLING_NLP_PARSER_PARSER_TRAINER_H_

