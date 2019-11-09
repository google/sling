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

#include "sling/myelin/builder.h"
#include "sling/myelin/compute.h"
#include "sling/myelin/flow.h"
#include "sling/myelin/gradient.h"
#include "sling/myelin/learning.h"
#include "sling/nlp/parser/action-table.h"
#include "sling/nlp/parser/parser-trainer.h"
#include "sling/nlp/parser/trainer/transition-generator.h"

namespace sling {
namespace nlp {

using namespace task;
using namespace myelin;

// Deletegate for fixed action classification using a softmax cross-entropy
// loss.
class MultiClassDelegateLearner : public DelegateLearner {
 public:
  MultiClassDelegateLearner(const string &name)
      : name_(name), loss_(name + "_loss") {}

  void Build(Flow *flow,
             Flow::Variable *activations,
             Flow::Variable *dactivations,
             bool learn) override {
    FlowBuilder f(flow, name_);
    int dim = activations->elements();
    int size = actions_.size();
    auto *W = f.Random(f.Parameter("W", DT_FLOAT, {dim, size}));
    auto *b = f.Random(f.Parameter("b", DT_FLOAT, {1, size}));

    auto *input = f.Placeholder("input", DT_FLOAT, {1, dim}, true);
    auto *logits = f.Name(f.Add(f.MatMul(input, W), b), "logits");
    logits->set_out();
    f.Name(f.ArgMax(logits), "output");

    flow->Connect({activations, input});
    if (learn) {
      Gradient(flow, f.func());
      auto *dlogits = flow->GradientVar(logits);
      loss_.Build(flow, logits, dlogits);
    }
  }

  void Initialize(const Network &network) override {
    cell_ = network.GetCell(name_);
    input_ = cell_->GetParameter(name_ + "/input");
    logits_ = cell_->GetParameter(name_ + "/logits");
    output_ = cell_->GetParameter(name_ + "/output");

    dcell_ = cell_->Gradient();
    primal_ = cell_->Primal();
    dinput_ = input_->Gradient();
    dlogits_ = logits_->Gradient();
    loss_.Initialize(network);
  }

  DelegateLearnerInstance *CreateInstance() override {
    return new DelegateInstance(this);
  }

  void Save(Flow *flow, Builder *data) override {
    // Save delegate type.
    data->Add("name", name_);
    data->Add("runtime", "SoftmaxDelegate");
    data->Add("cell", cell_->name());

    // Save the action table.
    Store *store = data->store();
    Handle n_type = store->Lookup("/table/action/type");
    Handle n_length = store->Lookup("/table/action/length");
    Handle n_source = store->Lookup("/table/action/source");
    Handle n_target = store->Lookup("/table/action/target");
    Handle n_role = store->Lookup("/table/action/role");
    Handle n_label = store->Lookup("/table/action/label");
    Handle n_delegate = store->Lookup("/table/action/delegate");

    Array actions(store, actions_.size());
    int index = 0;
    for (const ParserAction &action : actions_.list()) {
      auto type = action.type;
      Builder b(store);
      b.Add(n_type, static_cast<int>(type));

      if (type == ParserAction::REFER || type == ParserAction::EVOKE) {
        if (action.length > 0) {
          b.Add(n_length, static_cast<int>(action.length));
        }
      }
      if (type == ParserAction::ASSIGN ||
          type == ParserAction::ELABORATE ||
          type == ParserAction::CONNECT) {
        if (action.source != 0) {
          b.Add(n_source, static_cast<int>(action.source));
        }
      }
      if (type == ParserAction::EMBED ||
          type == ParserAction::REFER ||
          type == ParserAction::CONNECT) {
        if (action.target != 0) {
          b.Add(n_target, static_cast<int>(action.target));
        }
      }
      if (type == ParserAction::CASCADE) {
        b.Add(n_delegate, static_cast<int>(action.delegate));
      }
      if (!action.role.IsNil()) b.Add(n_role, action.role);
      if (!action.label.IsNil()) b.Add(n_label, action.label);
      actions.set(index++, b.Create().handle());
    }
    data->Add("actions", actions);
  }

  // Multi-class delegate instance.
  class DelegateInstance : public DelegateLearnerInstance {
   public:
    DelegateInstance(MultiClassDelegateLearner *learner)
        : learner_(learner),
          forward_(learner->cell_),
          backward_(learner->dcell_) {
    }

    void CollectGradients(std::vector<myelin::Instance *> *gradients) override {
      gradients->push_back(&backward_);
    }

    void ClearGradients() override {
      backward_.Clear();
    }

    float Compute(float *activations,
                  float *dactivations,
                  const ParserAction &action) override {
      // Look up index for action. Skip backpropagation if action is unknown.
      int target = learner_->actions_.Index(action);
      if (target == -1) return 0.0;

      // Compute logits from activations.
      forward_.SetReference(learner_->input_, activations);
      forward_.Compute();

      // Compute loss.
      float *logits = forward_.Get<float>(learner_->logits_);
      float *dlogits = backward_.Get<float>(learner_->dlogits_);
      float loss = learner_->loss_.Compute(logits, target, dlogits);

      // Backpropagate loss.
      backward_.Set(learner_->primal_, &forward_);
      backward_.SetReference(learner_->dinput_, dactivations);
      backward_.Compute();

      return loss;
    }

    void Predict(float *activations, ParserAction *action) override {
      // Predict action from activations.
      forward_.SetReference(learner_->input_, activations);
      forward_.Compute();
      int argmax = *forward_.Get<int>(learner_->output_);
      *action = learner_->actions_.Action(argmax);
    }

   private:
    MultiClassDelegateLearner *learner_;
    Instance forward_;
    Instance backward_;
  };

 protected:
  string name_;                // delegate name
  ActionTable actions_;        // action table for multi-class classification
  CrossEntropyLoss loss_;      // loss function

  Cell *cell_ = nullptr;       // cell for forward computation
  Tensor *input_ = nullptr;    // input for activations
  Tensor *logits_ = nullptr;   // logits for actions
  Tensor *output_ = nullptr;   // output prediction

  Cell *dcell_ = nullptr;      // cell for backward computation
  Tensor *primal_ = nullptr;   // primal reference
  Tensor *dinput_ = nullptr;   // gradient for activations
  Tensor *dlogits_ = nullptr;  // gradient for logits
};

// Main delegate for coarse-grained shift/mark/other classification.
class ShiftMarkOtherDelegateLearner : public MultiClassDelegateLearner {
 public:
  ShiftMarkOtherDelegateLearner(int other)
      : MultiClassDelegateLearner("coarse") {
    // Set up coarse actions.
    actions_.Add(ParserAction(ParserAction::SHIFT));
    actions_.Add(ParserAction(ParserAction::MARK));
    actions_.Add(ParserAction(ParserAction::CASCADE, other));
  }
};

// Delegate for fine-grained parser action classification.
class ClassificationDelegateLearner : public MultiClassDelegateLearner {
 public:
  ClassificationDelegateLearner(const ActionTable &actions)
      : MultiClassDelegateLearner("fine") {
    for (const ParserAction &action : actions.list()) {
      actions_.Add(action);
    }
  }
};

// Parser trainer for simple cascaded parser with a coarse-grained main delegate
// for shift and mark and a fine-grained delegate for the rest of the actions.
class CasparTrainer : public ParserTrainer {
 public:
   // Set up caspar parser model.
   void Setup(task::Task *task) override {
    // Collect word and action vocabularies from training corpus.
    training_corpus_->Rewind();
    for (;;) {
      // Get next document.
      Document *document = training_corpus_->Next(&commons_);
      if (document == nullptr) break;

      // Update word vocabulary.
      for (const Token &t : document->tokens()) words_[t.word()]++;

      // Generate action table for fine-grained classifier.
      Generate(*document, [&](const ParserAction &action) {
        bool skip = false;
        switch (action.type) {
          case ParserAction::SHIFT:
          case ParserAction::MARK:
            skip = true;
            break;
          case ParserAction::CONNECT:
            if (action.source > max_source_) skip = true;
            if (action.target > max_target_) skip = true;
            break;
          case ParserAction::ASSIGN:
          case ParserAction::EMBED:
          case ParserAction::ELABORATE:
            if (action.source > max_source_) skip = true;
            break;
          default:
            break;
        }
        if (!skip) actions_.Add(action);
      });

      delete document;
    }
    roles_.Init(actions_.list());

    // Set up delegates.
    delegates_.push_back(new ShiftMarkOtherDelegateLearner(1));
    delegates_.push_back(new ClassificationDelegateLearner(actions_));
  }

  // Transition generator.
  void GenerateTransitions(const Document &document,
                           std::vector<ParserAction> *transitions) override {
    transitions->clear();
    Generate(document, [&](const ParserAction &action) {
      if (action.type != ParserAction::SHIFT &&
          action.type != ParserAction::MARK) {
        transitions->emplace_back(ParserAction::CASCADE, 1);
      }
      transitions->push_back(action);
    });
  }

  // Save action table in model.
  void SaveModel(Flow *flow, Store *store) override {
    // Save action table in store.
    Builder table(store);
    table.AddId("/table");
    actions_.Write(&table);
    table.Create();
  }

 private:
  // Parser actions.
  ActionTable actions_;
};

REGISTER_TASK_PROCESSOR("caspar-trainer", CasparTrainer);

}  // namespace nlp
}  // namespace sling

