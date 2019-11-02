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

#include "sling/myelin/rnn.h"

#include "sling/myelin/builder.h"
#include "sling/myelin/gradient.h"

namespace sling {
namespace myelin {

void BiLSTM::LSTM::Initialize(const Network &net, const string &name) {
  // Initialize LSTM cell.
  cell = net.GetCell(name);
  input = net.GetParameter(name + "/input");
  h_in = net.GetParameter(name + "/h_in");
  h_out = net.GetParameter(name + "/h_out");
  c_in = net.GetParameter(name + "/c_in");
  c_out = net.GetParameter(name + "/c_out");

  // Initialize gradient cell for LSTM.
  gcell = cell->Gradient();
  if (gcell != nullptr) {
    primal = cell->Primal();
    dinput = input->Gradient();
    dh_in = h_in->Gradient();
    dh_out = h_out->Gradient();
    dc_in = c_in->Gradient();
    dc_out = c_out->Gradient();
  }
}

// Build flows for LSTMs.
BiLSTM::Outputs BiLSTM::Build(Flow *flow, int dim,
                              Flow::Variable *input,
                              Flow::Variable *dinput) {
  Outputs out;

  // Build left-to-right LSTM flow.
  FlowBuilder lr(flow, name_ + "/lr");
  auto *lr_input = lr.Placeholder("input", input->type, input->shape, true);
  out.lr = lr.LSTMLayer(lr_input, dim);

  // Build right-to-left LSTM flow.
  FlowBuilder rl(flow, name_ + "/rl");
  auto *rl_input = rl.Placeholder("input", input->type, input->shape, true);
  out.rl = rl.LSTMLayer(rl_input, dim);

  // Connect input to LSTMs.
  flow->Connect({input, lr_input, rl_input});

  // Build gradients for learning.
  if (dinput != nullptr) {
    Gradient(flow, lr.func());
    Gradient(flow, rl.func());
    out.dlr = flow->GradientVar(lr_input);
    out.drl = flow->GradientVar(rl_input);
    flow->Connect({dinput, out.dlr, out.drl});
  } else {
    out.dlr = nullptr;
    out.drl = nullptr;
  }

  return out;
}

void BiLSTM::Initialize(const Network &net) {
  lr_.Initialize(net, name_ + "/lr");
  rl_.Initialize(net, name_ + "/rl");
}

BiLSTMInstance::BiLSTMInstance(const BiLSTM &bilstm)
    : bilstm_(bilstm),
      lr_(bilstm.lr_.cell),
      rl_(bilstm.rl_.cell),
      lr_hidden_(bilstm.lr_.h_out),
      lr_control_(bilstm.lr_.c_out),
      rl_hidden_(bilstm.rl_.h_out),
      rl_control_(bilstm.rl_.c_out) {}

BiChannel BiLSTMInstance::Compute(Channel *input) {
  // Reset hidden and control channels.
  int length = input->size();
  lr_hidden_.reset(length + 1);
  rl_hidden_.reset(length + 1);
  lr_control_.resize(length + 1);
  rl_control_.resize(length + 1);
  lr_control_.zero(length);
  rl_control_.zero(length);

  // Compute left-to-right LSTM.
  for (int i = 0; i < length; ++i) {
    // Input.
    lr_.Set(bilstm_.lr_.input, input, i);
    lr_.Set(bilstm_.lr_.h_in, &lr_hidden_, i > 0 ? i - 1 : length);
    lr_.Set(bilstm_.lr_.c_in, &lr_control_, i > 0 ? i - 1 : length);

    // Output.
    lr_.Set(bilstm_.lr_.h_out, &lr_hidden_, i);
    lr_.Set(bilstm_.lr_.c_out, &lr_control_, i);

    // Compute LSTM cell.
    lr_.Compute();
  }

  // Compute right-to-left LSTM.
  for (int i = length - 1; i >= 0; --i) {
    // Input.
    rl_.Set(bilstm_.rl_.input, input, i);
    rl_.Set(bilstm_.rl_.h_in, &rl_hidden_, i + 1);
    rl_.Set(bilstm_.rl_.c_in, &rl_control_, i + 1);

    // Output.
    rl_.Set(bilstm_.rl_.h_out, &rl_hidden_, i);
    rl_.Set(bilstm_.rl_.c_out, &rl_control_, i);

    // Compute LSTM cell.
    rl_.Compute();
  }

  return BiChannel(&lr_hidden_, &rl_hidden_);
}

BiLSTMLearner::BiLSTMLearner(const BiLSTM &bilstm)
    : bilstm_(bilstm),
      lr_gradient_(bilstm.lr_.gcell),
      rl_gradient_(bilstm.rl_.gcell),
      lr_hidden_(bilstm.lr_.h_out),
      lr_control_(bilstm.lr_.c_out),
      rl_hidden_(bilstm.rl_.h_out),
      rl_control_(bilstm.rl_.c_out),
      dlr_hidden_(bilstm.lr_.dh_in),
      dlr_control_(bilstm.lr_.dc_in),
      drl_hidden_(bilstm.rl_.dh_in),
      drl_control_(bilstm.rl_.dc_in),
      dinput_(bilstm.lr_.dinput) {}

BiLSTMLearner::~BiLSTMLearner() {
  for (Instance *data : lr_) delete data;
  for (Instance *data : rl_) delete data;
}

BiChannel BiLSTMLearner::Compute(Channel *input) {
  // Allocate instances.
  int length = input->size();
  for (auto *data : lr_) delete data;
  for (auto *data : rl_) delete data;
  lr_.resize(length);
  rl_.resize(length);
  for (int i = 0; i < length; ++i) {
    lr_[i] = new Instance(bilstm_.lr_.cell);
    rl_[i] = new Instance(bilstm_.rl_.cell);
  }

  // Reset hidden and control channels.
  lr_hidden_.reset(length + 1);
  rl_hidden_.reset(length + 1);
  lr_control_.resize(length + 1);
  rl_control_.resize(length + 1);
  lr_control_.zero(length);
  rl_control_.zero(length);

  // Compute left-to-right LSTM.
  for (int i = 0; i < length; ++i) {
    Instance *lr = lr_[i];

    // Input.
    lr->Set(bilstm_.lr_.input, input, i);
    lr->Set(bilstm_.lr_.h_in, &lr_hidden_, i > 0 ? i - 1 : length);
    lr->Set(bilstm_.lr_.c_in, &lr_control_, i > 0 ? i - 1 : length);

    /// Output.
    lr->Set(bilstm_.lr_.h_out, &lr_hidden_, i);
    lr->Set(bilstm_.lr_.c_out, &lr_control_, i);

    // Compute LSTM cell.
    lr->Compute();
  }

  // Compute right-to-left LSTM.
  for (int i = length - 1; i >= 0; --i) {
    Instance *rl = rl_[i];

    // Input.
    rl->Set(bilstm_.rl_.input, input, i);
    rl->Set(bilstm_.rl_.h_in, &rl_hidden_, i + 1);
    rl->Set(bilstm_.rl_.c_in, &rl_control_, i + 1);

    // Output.
    rl->Set(bilstm_.rl_.h_out, &rl_hidden_, i);
    rl->Set(bilstm_.rl_.c_out, &rl_control_, i);

    // Compute LSTM cell.
    rl->Compute();
  }

  return BiChannel(&lr_hidden_, &rl_hidden_);
}

BiChannel BiLSTMLearner::PrepareGradientChannels(int length) {
  dlr_hidden_.reset(length + 1);
  drl_hidden_.reset(length + 1);
  dlr_control_.resize(length + 1);
  drl_control_.resize(length + 1);
  dlr_control_.zero(length);
  drl_control_.zero(length);

  return BiChannel(&dlr_hidden_, &drl_hidden_);
}

Channel *BiLSTMLearner::Backpropagate() {
  // Clear input gradient.
  int length = lr_.size();
  dinput_.reset(length);

  // Propagate gradients for left-to-right LSTM.
  for (int i = length - 1; i >= 0; --i) {
    // Set reference to primal cell.
    lr_gradient_.Set(bilstm_.lr_.primal, lr_[i]);

    // Gradient inputs.
    lr_gradient_.Set(bilstm_.lr_.dh_out, &dlr_hidden_, i);
    lr_gradient_.Set(bilstm_.lr_.dc_out, &dlr_control_, i);

    // Gradient outputs.
    lr_gradient_.Set(bilstm_.lr_.dh_in, &dlr_hidden_, i > 0 ? i - 1 : length);
    lr_gradient_.Set(bilstm_.lr_.dc_in, &dlr_control_, i > 0 ? i - 1 : length);
    lr_gradient_.Set(bilstm_.lr_.dinput, &dinput_, i);

    // Compute backward.
    lr_gradient_.Compute();
  }

  // Propagate gradients for right-to-left LSTM.
  for (int i = 0; i < length; ++i) {
    // Set reference to primal cell.
    rl_gradient_.Set(bilstm_.rl_.primal, rl_[i]);

    // Gradient inputs.
    rl_gradient_.Set(bilstm_.rl_.dh_out, &drl_hidden_, i);
    rl_gradient_.Set(bilstm_.rl_.dc_out, &drl_control_, i);

    // Gradient outputs.
    rl_gradient_.Set(bilstm_.rl_.dh_in, &drl_hidden_, i + 1);
    rl_gradient_.Set(bilstm_.rl_.dc_in, &drl_control_, i + 1);
    rl_gradient_.Set(bilstm_.rl_.dinput, &dinput_, i);

    // Compute backward.
    rl_gradient_.Compute();
  }

  // Return input gradient.
  return &dinput_;
}

}  // namespace myelin
}  // namespace sling

