// Copyright 2018 Google Inc.
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

#ifndef SLING_MYELIN_LEARNING_H_
#define SLING_MYELIN_LEARNING_H_

#include <string>
#include <map>

#include "sling/myelin/builder.h"
#include "sling/myelin/compute.h"
#include "sling/myelin/flow.h"

namespace sling {
namespace myelin {

// Cross entropy loss for multi-class classification.
class CrossEntropyLoss {
 public:
  CrossEntropyLoss(const string &name = "loss") : name_(name) {}

  // Build loss function together with gradient computation.
  void Build(Flow *flow, Flow::Variable *logits, Flow::Variable *dlogits);

  // Initialize loss for model.
  void Initialize(const Network &network);

  // Compute loss from logits and output loss gradient.
  float Compute(float *logits, int target, float *dlogits) const;

 private:
  // Name of loss function.
  string name_;

  // Cell for loss computation.
  Cell *cell_ = nullptr;

  // Tensors for loss computation.
  Tensor *logits_ = nullptr;
  Tensor *target_ = nullptr;
  Tensor *loss_ = nullptr;
  Tensor *dlogits_ = nullptr;
};

// A parameter optimizer applies updates to the learnable parameters of a model
// based on the (accumulated) gradients from backpropagation.
class Optimizer {
 public:
  // Mapping from learnable variables to their gradients.
  typedef std::map<Flow::Variable *, Flow::Variable *> GradientMap;

  Optimizer(const string &name = "optimizer") : name_(name) {}
  virtual ~Optimizer() { delete data_; }

  // Build update function for applying gradients.
  void Build(Flow *flow);

  // Initialize gradient update for model.
  void Initialize(const Network &network);

  // Apply gradients to update learnable parameters.
  virtual void Apply(const std::vector<Instance *> &gradients);

  // Decay learning rate. Returns new learning rate.
  virtual float DecayLearningRate() { return 0.0; }

  // Data instance for optimizer.
  Instance *data() const { return data_; }

  // Norm clipping threshold.
  float clipping_threshold() const { return clipping_threshold_; }
  void set_clipping_threshold(float t) { clipping_threshold_ = t; }

  // Local or global norm clipping.
  bool local_clipping() const { return local_clipping_; }
  void set_local_clipping(bool b) { local_clipping_ = b; }

 protected:
  // Let subclass build the parameter update using the gradient map.
  virtual void BuildOptimizer(const GradientMap &gradmap,
                              FlowBuilder *update) = 0;

  // Let subclass initialize update function for optimizer.
  virtual void InitializeOptimizer() = 0;

  // Get parameter from network.
  Tensor *GetParameter(const string &name) {
    return data_->cell()->GetParameter(name);
  }

  // Name of optimizer.
  string name_;

  // Mapping from gradient computation cell to instance variable in update.
  std::map<Flow::Function *, Flow::Variable *> instance_;
  std::map<const Cell *, Tensor *> refs_;

  // Data instance for updating the learnable parameters from the gradients.
  Instance *data_ = nullptr;

  float clipping_threshold_ = 0.0;  // norm clipping threshold (0=no clipping)
  bool local_clipping_ = false;     // compute norm per tensor
};

// Stochastic gradient descent optimizer.
class GradientDescentOptimizer : public Optimizer {
 public:
  // Decay learning rate.
  float DecayLearningRate() override;

  // Initial learning rate.
  float learning_rate() const { return lr_; }
  void set_learning_rate(float lr) { lr_ = lr; }

  // Learning rate decay.
  float decay() const { return decay_; }
  void set_decay(float decay) { decay_ = decay; }

  // Regularization parameter for L2 regularization.
  float lambda() const { return lambda_; }
  void set_lambda(float lambda) { lambda_ = lambda; }

 protected:
  void BuildOptimizer(const GradientMap &gradmap, FlowBuilder *update) override;
  void InitializeOptimizer() override;

  Tensor *alpha_ = nullptr;         // current learning rate
  float lr_ = 0.01;                 // initial learning rate
  float decay_  = 1.0;              // learning rate decay
  float lambda_ = 0.0;              // regularization parameter (0=none)
};

// Momentum optimizer.
class MomentumOptimizer : public Optimizer {
 public:
  MomentumOptimizer(const string &name = "optimizer") : Optimizer(name) {}

  // Apply gradients to update learnable parameters.
  void Apply(const std::vector<Instance *> &gradients) override;

  // Decay learning rate.
  float DecayLearningRate() override;

  // Learning rate.
  float learning_rate() const { return lr_; }
  void set_learning_rate(float lr) { lr_ = lr; }

  // Learning rate decay.
  float decay() const { return decay_; }
  void set_decay(float decay) { decay_ = decay; }

  // Momentum for blending in previous updates.
  float momentum() const { return momentum_; }
  void set_momentum(float momentum) { momentum_ = momentum; }

 protected:
  void BuildOptimizer(const GradientMap &gradmap, FlowBuilder *update) override;
  void InitializeOptimizer() override;

  Tensor *alpha_ = nullptr;         // current learning rate
  float lr_ = 0.01;                 // initial learning rate
  float decay_  = 1.0;              // learning rate decay
  float momentum_ = 0.9;            // blending ratio for previous update
  int num_linked_ = 0;              // number of linked updates
};

// Adam optimizer.
class AdamOptimizer : public Optimizer {
 public:
  AdamOptimizer(const string &name = "optimizer") : Optimizer(name) {}

  // Apply gradients to update learnable parameters.
  void Apply(const std::vector<Instance *> &gradients) override;

  // Decay learning rate.
  float DecayLearningRate() override;

  // Learning rate.
  float learning_rate() const { return lr_; }
  void set_learning_rate(float lr) { lr_ = lr; }

  // Learning rate decay.
  float decay() const { return decay_; }
  void set_decay(float decay) { decay_ = decay; }

  // The exponential decay rate for the first moment estimates.
  float beta1() const { return beta1_; }
  void set_beta1(float beta1) { beta1_ = beta1; }

  // The exponential decay rate for the second moment estimates.
  float beta2() const { return beta2_; }
  void set_beta2(float beta2) { beta2_ = beta2; }

  // Underflow correction.
  float epsilon() const { return epsilon_; }
  void set_epsilon(float epsilon) { epsilon_ = epsilon; }

 protected:
  void BuildOptimizer(const GradientMap &gradmap, FlowBuilder *update) override;
  void InitializeOptimizer() override;

  Tensor *alpha_ = nullptr;         // current learning rate
  float lr_ = 0.01;                 // initial learning rate
  float decay_  = 1.0;              // learning rate decay

  float beta1_ = 0.9;               // mean decay rate
  float beta2_ = 0.999;             // variance decay rate
  float epsilon_ = 1e-8;            // underflow correction
};

}  // namespace myelin
}  // namespace sling

#endif  // SLING_MYELIN_LEARNING_H_

