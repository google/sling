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

#ifndef SLING_MYELIN_RNN_H_
#define SLING_MYELIN_RNN_H_

#include <vector>

#include "sling/myelin/compute.h"
#include "sling/myelin/flow.h"

namespace sling {
namespace myelin {

// Channel pair with left-to-right and right-to-left channels.
struct BiChannel {
  BiChannel(Channel *lr, Channel *rl) : lr(lr), rl(rl) {}
  Channel *lr;  // left-to-right channel
  Channel *rl;  // right-to-left channel
};

// Bi-directional long short-term memory (LSTM) module.
class BiLSTM {
 public:
  // Flow output variables.
  struct Outputs {
    Flow::Variable *lr;   // output from left-to-right LSTM (hidden)
    Flow::Variable *rl;   // output from right-to-left LSTM (hidden)
    Flow::Variable *dlr;  // gradient output from right-to-left LSTM (dinput)
    Flow::Variable *drl;  // gradient output from right-to-left LSTM (dinput)
  };

  // Initialize bi-directional LSTM.
  BiLSTM(const string &name = "lstm") : name_(name) {}

  // Build flows for LSTMs.
  Outputs Build(Flow *flow, const Library &library, int dim,
                Flow::Variable *input, Flow::Variable *dinput = nullptr);

  // Initialize LSTMs.
  void Initialize(const Network &net);

 private:
  // Network for LSTM cell.
  struct LSTM {
    // Initialize LSTM cell from network.
    void Initialize(const Network &net, const string &name);

    Cell *cell = nullptr;            // LSTM cell
    Tensor *input = nullptr;         // LSTM feature input
    Tensor *h_in = nullptr;          // link to LSTM hidden input
    Tensor *h_out = nullptr;         // link to LSTM hidden output
    Tensor *c_in = nullptr;          // link to LSTM control input
    Tensor *c_out = nullptr;         // link to LSTM control output

    Cell *gcell = nullptr;           // LSTM gradient cell
    Tensor *dinput = nullptr;        // input gradient
    Tensor *primal = nullptr;        // link to primal LSTM cell
    Tensor *dh_in = nullptr;         // gradient for LSTM hidden input
    Tensor *dh_out = nullptr;        // gradient for LSTM hidden output
    Tensor *dc_in = nullptr;         // gradient for LSTM control input
    Tensor *dc_out = nullptr;        // gradient for LSTM control output
  };

  string name_;   // LSTM cell name prefix
  LSTM lr_;       // left-to-right LSTM
  LSTM rl_;       // right-to-left LSTM

  friend class BiLSTMInstance;
  friend class BiLSTMLearner;
};

// Bi-directional LSTM instance.
class BiLSTMInstance {
 public:
  // Initialize bi-directional LSTM instance.
  BiLSTMInstance(const BiLSTM &bilstm);

  // Compute left-to-right and right-to-left LSTM sequences for input.
  BiChannel Compute(Channel *input);

 private:
  const BiLSTM &bilstm_;     // bi-directional LSTM

  Instance lr_;              // left-to-right LSTM instance
  Instance rl_;              // right-to-left LSTM instance

  Channel lr_hidden_;        // left-to-right LSTM hidden channel
  Channel lr_control_;       // left-to-right LSTM control channel
  Channel rl_hidden_;        // right-to-left LSTM hidden channel
  Channel rl_control_;       // right-to-left LSTM control channel
};

// Bi-directional LSTM learner.
class BiLSTMLearner {
 public:
  // Initialize bi-directional LSTM learner.
  BiLSTMLearner(const BiLSTM &bilstm);
  ~BiLSTMLearner();

  // Compute left-to-right and right-to-left LSTM sequences for input.
  BiChannel Compute(Channel *input);

  // Prepare gradient channels.
  BiChannel PrepareGradientChannels(int length);

  // Backpropagate hidden gradients to input gradient.
  Channel *Backpropagate();

  // Collect gradients.
  void CollectGradients(std::vector<Instance *> *gradients) {
    gradients->push_back(&lr_gradient_);
    gradients->push_back(&rl_gradient_);
  }

  // Clear gradients.
  void Clear() {
    lr_gradient_.Clear();
    rl_gradient_.Clear();
  }

 private:
  const BiLSTM &bilstm_;        // bi-directional LSTM

  std::vector<Instance *> lr_;  // left-to-right LSTM instances
  std::vector<Instance *> rl_;  // right-to-left LSTM instances
  Instance lr_gradient_;        // left-to-right LSTM gradients
  Instance rl_gradient_;        // right-to-left LSTM gradients

  Channel lr_hidden_;           // left-to-right LSTM hidden channel
  Channel lr_control_;          // left-to-right LSTM control channel
  Channel rl_hidden_;           // right-to-left LSTM hidden channel
  Channel rl_control_;          // right-to-left LSTM control channel

  Channel dlr_hidden_;          // left-to-right LSTM hidden gradient channel
  Channel dlr_control_;         // left-to-right LSTM control gradient channel
  Channel drl_hidden_;          // right-to-left LSTM hidden gradient channel
  Channel drl_control_;         // right-to-left LSTM control gradient channel

  Channel dinput_;              // input gradient channel
};

}  // namespace myelin
}  // namespace sling

#endif  // SLING_MYELIN_RNN_H_

