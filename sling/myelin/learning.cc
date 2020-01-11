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

#include "sling/myelin/learning.h"

#include "sling/base/logging.h"
#include "sling/myelin/builder.h"

namespace sling {
namespace myelin {

void CrossEntropyLoss::Build(Flow *flow,
                             Flow::Variable *logits,
                             Flow::Variable *dlogits) {
  // Assume logits batch dimension is one.
  CHECK_EQ(logits->rank(), 2);
  CHECK(logits->shape == dlogits->shape);
  int size = logits->dim(1);

  // Build loss and loss gradient computation.
  FlowBuilder tf(flow, name_);

  // Inputs are logits and target label.
  auto *input = tf.Placeholder("logits", DT_FLOAT, {1, size});
  input->set_ref();
  auto *target = tf.Placeholder("target", DT_INT32, {});

  // Compute softmax for logits.
  auto *softmax = tf.Softmax(tf.Reshape(input, {size}));

  // Compute loss (negative log-likelihood).
  auto *loss = tf.Name(tf.Neg(tf.Log(tf.Slice(softmax, target, {1}))), "loss");
  loss->set_out();

  // Compute gradient.
  auto *gradient = tf.Sub(softmax, tf.OneHot(target, size));
  auto *output = tf.Name(tf.Reshape(gradient, {1, size}), "d_logits");
  output->set_ref();

  // Connect input and output logits.
  flow->Connect({logits, input});
  flow->Connect({dlogits, output});

  // Loss is only needed at training-time.
  tf.func()->set_training();
}

void CrossEntropyLoss::Initialize(const Network &network) {
  // Get loss computation cell.
  cell_ = network.GetCell(name_);

  // Get tensors.
  logits_ = network.GetParameter(name_ + "/logits");
  target_ = network.GetParameter(name_ + "/target");
  loss_ = network.GetParameter(name_ + "/loss");
  dlogits_ = network.GetParameter(name_ + "/d_logits");
}

float CrossEntropyLoss::Compute(float *logits, int target,
                                float *dlogits) const {
  Instance data(cell_);
  data.SetReference(logits_, logits);
  data.SetReference(dlogits_, dlogits);
  *data.Get<int>(target_) = target;
  data.Compute();
  return *data.Get<float>(loss_);
}

void Optimizer::Build(Flow *flow) {
  // Build mapping from learnable variable to gradient for variable.
  FlowBuilder tf(flow, name_);
  GradientMap gradmap;
  for (Flow::Variable *var : flow->vars()) {
    if (!var->learnable()) continue;

    // Get gradient variable for learnable variable.
    Flow::Variable *dvar = flow->GradientVar(var);
    CHECK(dvar != nullptr) << "No gradient found for " << var->name;

    // Find function for gradient variable.
    Flow::Operation *producer = nullptr;
    if (dvar->producer != nullptr) {
      producer = dvar->producer;
    } else if (!dvar->consumers.empty()) {
      producer = dvar->consumers[0];
    }
    CHECK(producer != nullptr) << "No producer for gradient " << dvar->name;
    Flow::Function *func = producer->func;
    CHECK(func != nullptr) << "No producer function for gradient "
                           << dvar->name;

    // Add instance variables for producer functions.
    if (instance_[func] == nullptr) instance_[func] = tf.Instance(func);

    // Add reference to gradient in update function.
    gradmap[var] = tf.Ref(instance_[func], dvar);
  }

  // Optionally add gradient clipping.
  if (clipping_threshold_ != 0.0 && !gradmap.empty()) {
    auto *threshold = tf.Name(tf.Const(clipping_threshold_), "threshold");
    if (local_clipping_) {
      // Clip each weight tensor separately.
      for (auto &it : gradmap) {
        auto &dv = it.second;
        auto *norm = tf.Norm(dv);
        auto *clip = tf.Div(threshold, tf.Maximum(norm, threshold));
        dv = tf.Mul(dv, clip);
      }
    } else {
      // Compute global norm over all gradient tensors.
      Flow::Variable *sum = nullptr;
      for (auto &it : gradmap) {
        auto &dv = it.second;
        auto *squared_sum = tf.Sum(tf.Square(dv));
        if (sum == nullptr) {
          sum = squared_sum;
        } else {
          sum = tf.Add(sum, squared_sum);
        }
      }
      auto *norm = tf.Sqrt(sum);

      // Clip gradients by global norm.
      auto *clip = tf.Div(threshold, tf.Maximum(norm, threshold));
      for (auto &it : gradmap) {
        auto &dv = it.second;
        dv = tf.Mul(dv, clip);
      }
    }
  }

  // Build optimizer.
  BuildOptimizer(gradmap, &tf);

  // Optimizer is only needed at training-time.
  tf.func()->set_training();
}

void Optimizer::Initialize(const Network &network) {
  // Get cell for update.
  Cell *cell = network.GetCell(name_);

  // Create data instance for update.
  data_ = new Instance(cell);

  // Create mapping from gradient computation cell to instance variable in
  // update cell.
  for (auto it : instance_) {
    Cell *gradient_cell = network.GetCell(it.first->name);
    Tensor *gradient_instance = cell->GetParameter(it.second->name);
    refs_[gradient_cell] = gradient_instance;
  }

  // Initialize optimizer.
  InitializeOptimizer();
}

void Optimizer::Apply(const std::vector<Instance *> &gradients) {
  // Set instance references to gradients in update.
  for (Instance *g : gradients) {
    auto f = refs_.find(g->cell());
    CHECK(f != refs_.end()) << g->cell()->name();
    data_->Set(f->second, g);
  }

  // Apply gradient update to learnable parameters.
  data_->Compute();
}

void GradientDescentOptimizer::BuildOptimizer(const GradientMap &gradmap,
                                              FlowBuilder *update) {
  // Add learning rate to update function.
  FlowBuilder &tf = *update;
  auto *alpha = tf.Var("alpha", DT_FLOAT, {})->set_in()->set_out();
  auto *multiplier = tf.Neg(alpha);

  // Update learnable variables from gradients.
  for (auto it : gradmap) {
    auto *v = it.first;
    auto *dv = it.second;

    // Add scaled gradient to parameters.
    float lambda = v->GetAttr("l2reg", lambda_);
    if (lambda != 0.0) {
      tf.Assign(v, tf.Add(tf.Mul(tf.Sub(tf.One(), tf.Const(lambda)), v),
                          tf.Mul(dv, multiplier)));
    } else {
      tf.AssignAdd(v, tf.Mul(dv, multiplier));
    }
  }
}

void GradientDescentOptimizer::InitializeOptimizer() {
  // Set initial learning rate.
  alpha_ = GetParameter(name_ + "/alpha");
  *data_->Get<float>(alpha_) = lr_;
}

float GradientDescentOptimizer::DecayLearningRate() {
  float &lr = *data_->Get<float>(alpha_);
  lr *= decay_;
  return lr;
}

void MomentumOptimizer::BuildOptimizer(const GradientMap &gradmap,
                                       FlowBuilder *update) {
  // Momentum update rule:
  //   t <- t + 1
  //   v_t <- momentum * v_{t-1} + alpha * dv
  //   var <- var - v_t
  //
  // See also: http://ruder.io/optimizing-gradient-descent/index.html#momentum
  FlowBuilder &tf = *update;

  // Add hyperparameter inputs.
  auto *alpha = tf.Var("alpha", DT_FLOAT, {})->set_in()->set_out();
  auto *gamma = tf.Name(tf.Const(momentum_), "gamma");

  // Update learnable variables from gradients.
  int i = 0;
  for (auto it : gradmap) {
    auto *var = it.first;
    auto *dv = it.second;

    // Reference to previous update.
    auto *v_in = tf.Var("vin" + std::to_string(i), DT_FLOAT, var->shape);
    v_in->set_in()->set_ref();

    // Blend current and new update.
    auto *v_out = tf.Add(tf.Mul(gamma, v_in), tf.Mul(alpha, dv));
    v_out->set_out();
    tf.Name(v_out, "vout" + std::to_string(i));

    // Update parameters.
    tf.Assign(var, tf.Sub(var, v_out));
    i++;

    // Link input to output.
    tf.flow()->Connect({v_in, v_out});
  }
  num_linked_ = i;
}

void MomentumOptimizer::InitializeOptimizer() {
  // Set initial learning rate.
  alpha_ = GetParameter(name_ + "/alpha");
  *data_->Get<float>(alpha_) = lr_;

  // Link input and output.
  for (int i = 0; i < num_linked_; ++i) {
    Tensor *vin = GetParameter(name_ + "/vin" + std::to_string(i));
    Tensor *vout = GetParameter(name_ + "/vout" + std::to_string(i));
    data_->SetReference(vin, data_->GetAddress(vout));
  }
}

void MomentumOptimizer::Apply(const std::vector<Instance *> &gradients) {
  Optimizer::Apply(gradients);
}

float MomentumOptimizer::DecayLearningRate() {
  float &lr = *data_->Get<float>(alpha_);
  lr *= decay_;
  return lr;
}

void AdamOptimizer::BuildOptimizer(const GradientMap &gradmap,
                                   FlowBuilder *update) {
  // Adam update rule:
  //   t <- t + 1
  //   lr_t <- learning_rate * sqrt(1 - beta2^t) / (1 - beta1^t)
  //   m_t <- beta1 * m_{t-1} + (1 - beta1) * dv
  //   v_t <- beta2 * v_{t-1} + (1 - beta2) * dv^2
  //   var <- var - lr_t * m_t / (sqrt(v_t) + epsilon)
  //
  // See also: http://ruder.io/optimizing-gradient-descent/index.html#adam
  FlowBuilder &tf = *update;

  // Add hyperparameter inputs.
  auto *alpha = tf.Var("alpha", DT_FLOAT, {})->set_in()->set_out();
  auto *beta1 = tf.Name(tf.Const(beta1_), "beta1");
  auto *beta2 = tf.Name(tf.Const(beta2_), "beta2");
  auto *epsilon = tf.Name(tf.Const(epsilon_), "epsilon");

  auto *one_minus_beta1 = tf.Sub(tf.One(), beta1);
  auto *one_minus_beta2 = tf.Sub(tf.One(), beta2);

  // Decay beta1 and beta2.
  auto *beta1_t_var = tf.Var("beta1_t", DT_FLOAT, {});
  auto *beta2_t_var = tf.Var("beta2_t", DT_FLOAT, {});
  auto *beta1_t = tf.Accumulate(beta1_t_var, tf.Mul(beta1_t_var, beta1));
  auto *beta2_t = tf.Accumulate(beta2_t_var, tf.Mul(beta2_t_var, beta2));

  // Compute learning rate.
  auto *lr = tf.Mul(alpha, tf.Div(tf.Sqrt(tf.Sub(tf.One(), beta2_t)),
                                  tf.Sub(tf.One(), beta1_t)));

  // Update learnable variables from gradients.
  int i = 0;
  for (auto it : gradmap) {
    auto *var = it.first;
    auto *dv = it.second;

    // Aggregate mean and variance.
    auto *m_var = tf.Var("m" + std::to_string(i), dv->type, dv->shape);
    auto *mw = one_minus_beta1;
    auto *m = tf.Accumulate(m_var, tf.Add(tf.Mul(m_var, beta1),
                                          tf.Mul(dv, mw)));

    auto *v_var = tf.Var("v" + std::to_string(i), dv->type, dv->shape);
    auto *vw = one_minus_beta2;
    auto *v = tf.Accumulate(v_var, tf.Add(tf.Mul(v_var, beta2),
                                          tf.Square(tf.Mul(dv, vw))));

    // Update parameters.
    auto *delta = tf.Div(tf.Mul(m, lr), tf.Add(tf.Sqrt(v), epsilon));
    tf.Assign(var, tf.Sub(var, delta));
    i++;
  }
}

void AdamOptimizer::InitializeOptimizer() {
  // Set initial learning rate.
  alpha_ = GetParameter(name_ + "/alpha");
  *data_->Get<float>(alpha_) = lr_;

  // Initialize bias correction parameters.
  auto *beta1_t = GetParameter(name_ + "/beta1_t");
  auto *beta2_t = GetParameter(name_ + "/beta2_t");
  *data_->Get<float>(beta1_t) = 1.0;
  *data_->Get<float>(beta2_t) = 1.0;
}

void AdamOptimizer::Apply(const std::vector<Instance *> &gradients) {
  Optimizer::Apply(gradients);
}

float AdamOptimizer::DecayLearningRate() {
  float &lr = *data_->Get<float>(alpha_);
  lr *= decay_;
  return lr;
}

}  // namespace myelin
}  // namespace sling

