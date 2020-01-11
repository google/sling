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

#include <string>
#include <random>
#include <vector>

#include "sling/myelin/compute.h"
#include "sling/myelin/flow.h"

namespace sling {
namespace myelin {

class RNNInstance;
class RNNLearner;

// Recurrent Neural Network (RNN) cell.
struct RNN {
  // RNN types.
  enum Type {
    // Standard LSTM [Hochreiter & Schmidhuber 1997].
    LSTM = 0,

    // LSTM with peephole connections [Gers & Schmidhuber 2000] and coupled
    // forget and input gates [Greff et al. 2015].
    DRAGNN_LSTM = 1,

    // Standard LSTM with one matrix multiplication [Dozat & Manning 2017].
    DOZAT_LSTM = 2,

    // Standard LSTM with two matrix multiplications [Paszke et al. 2019].
    PYTORCH_LSTM = 3,

    // Gated Recurrent Unit (GRU) [Cho et al. 2014].
    GRU = 4,
  };

  // RNN specification.
  struct Spec {
    Type type = LSTM;        // RNN type
    int dim = 128;           // RNN dimension
    bool highways = false;   // use highway connections between layers
    float dropout = 0.0;     // dropout rate during training (0=no dropout)
  };

  // Flow input/output variables.
  struct Variables {
    Flow::Variable *input = nullptr;    // input to forward path
    Flow::Variable *output = nullptr;   // output from forward path
    Flow::Variable *doutput = nullptr;  // gradient input to backward path
    Flow::Variable *dinput = nullptr;   // gradient output from backward path
  };

  // Initialize RNN.
  RNN(const string &name, const Spec &spec) : name(name), spec(spec) {}

  // Build flow for RNN. If dinput is not null, the corresponding gradient
  // function is also built.
  Variables Build(Flow *flow,
                  Flow::Variable *input,
                  Flow::Variable *dinput = nullptr);

  // Initialize RNN.
  void Initialize(const Network &net);

  // Control channel is optional for RNN.
  bool has_control() const { return c_in != nullptr; }

  // Dropout is only needed during training.
  bool has_mask() const { return mask != nullptr; }

  string name;                     // RNN cell name
  Spec spec;                       // RNN specification

  Cell *cell = nullptr;            // RNN cell
  Tensor *input = nullptr;         // RNN feature input
  Tensor *h_in = nullptr;          // link to RNN hidden input
  Tensor *h_out = nullptr;         // link to RNN hidden output
  Tensor *c_in = nullptr;          // link to RNN control input
  Tensor *c_out = nullptr;         // link to RNN control output
  Tensor *zero = nullptr;          // zero element for channels
  Tensor *mask = nullptr;          // dropout mask input

  Tensor *nodropout = nullptr;     // dropout mask with no dropout

  Cell *gcell = nullptr;           // RNN gradient cell
  Tensor *dinput = nullptr;        // input gradient
  Tensor *primal = nullptr;        // link to primal RNN cell
  Tensor *dh_in = nullptr;         // gradient for RNN hidden input
  Tensor *dh_out = nullptr;        // gradient for RNN hidden output
  Tensor *dc_in = nullptr;         // gradient for RNN control input
  Tensor *dc_out = nullptr;        // gradient for RNN control output
  Tensor *sink = nullptr;          // scratch element for channels
};

// Channel merger cell for merging the outputs from two RNNs.
struct RNNMerger {
  // Flow input/output variables.
  struct Variables {
    Flow::Variable *left;     // left input to forward path
    Flow::Variable *right;    // right input to forward path
    Flow::Variable *merged;   // merged output from forward path

    Flow::Variable *dmerged;  // merged gradient from backward path
    Flow::Variable *dleft;    // left gradient output from backward path
    Flow::Variable *dright;   // right gradient output from backward path
  };

  // Initialize RNN merger.
  RNNMerger(const string &name) : name(name) {}

  // Build flow for channel merger. If dleft and dright are not null, the
  // corresponding gradient function is also built.
  Variables Build(Flow *flow,
                  Flow::Variable *left, Flow::Variable *right,
                  Flow::Variable *dleft, Flow::Variable *dright);

  // Initialize channel merger.
  void Initialize(const Network &net);

  string name;                     // cell name

  Cell *cell = nullptr;            // merger cell
  Tensor *left = nullptr;          // left channel input
  Tensor *right = nullptr;         // right channel input
  Tensor *merged = nullptr;        // merged output channel

  Cell *gcell = nullptr;           // merger gradient cell
  Tensor *dmerged = nullptr;       // gradient for merged channel
  Tensor *dleft = nullptr;         // gradient for left channel
  Tensor *dright = nullptr;        // gradient for right channel
};

// An RNN layer can be either unidirectional (left-to-right) or bidirectional
// (both left-to-right and right-to-left). The outputs from the the two RNNs
// in a bidirectional RNN are merged using an RNN channel merger.
class RNNLayer {
 public:
  // Set up RNN layer.
  RNNLayer(const string &name, const RNN::Spec &spec, bool bidir);

  // Build flow for RNN. If dinput is not null, the corresponding gradient
  // function is also built.
  RNN::Variables Build(Flow *flow,
                       Flow::Variable *input,
                       Flow::Variable *dinput = nullptr);

  // Initialize RNN.
  void Initialize(const Network &net);

 private:
  string name_;       // cell name prefix
  bool bidir_;        // bidirectional RNN
  float dropout_;     // dropout ratio during learning.

  RNN lr_;            // left-to-right RNN
  RNN rl_;            // right-to-left RNN (if bidirectional)
  RNNMerger merger_;  // channel merger for bidirectional RNN

  friend class RNNInstance;
  friend class RNNLearner;
};

// Instance of RNN layer for inference.
class RNNInstance {
 public:
  RNNInstance(const RNNLayer *rnn);

  // Compute RNN over input sequence and return output sequence.
  Channel *Compute(Channel *input);

 private:
  // Descriptor for RNN layer.
  const RNNLayer *rnn_;

  // Left-to-right RNN.
  Instance lr_;
  Channel lr_hidden_;
  Channel lr_control_;

  // Right-to-left RNN for bidirectional RNN.
  Instance rl_;
  Channel rl_hidden_;
  Channel rl_control_;

  // RNN channel merger for bidirectional RNN.
  Instance merger_;
  Channel merged_;
};

// Instance of RNN layer for learning.
class RNNLearner {
 public:
  RNNLearner(const RNNLayer *rnn);

  // Compute RNN over input sequence and return output sequence. Dropout is
  // only applied in learning mode.
  Channel *Compute(Channel *input);

  // Backpropagate gradients returning the output of backpropagation, i.e. the
  // gradient of the input sequence.
  Channel *Backpropagate(Channel *doutput);

  // Clear accumulated gradients.
  void Clear();

  // Collect instances with gradient updates.
  void CollectGradients(std::vector<Instance *> *gradients);

 private:
  // Generate uniform random number between 0 and 1.
  float Random() { return prob_(prng_); }

  // Descriptor for RNN layer.
  const RNNLayer *rnn_;

  // Left-to-right RNN.
  InstanceArray lr_fwd_;
  Channel lr_hidden_;
  Channel lr_control_;

  Instance lr_bkw_;
  Channel lr_dhidden_;
  Channel lr_dcontrol_;

  // Right-to-left RNN for bidirectional RNN.
  InstanceArray rl_fwd_;
  Channel rl_hidden_;
  Channel rl_control_;

  Instance rl_bkw_;
  Channel rl_dhidden_;
  Channel rl_dcontrol_;

  // Channel for gradient output.
  Channel dinput_;

  // RNN channel merger for bidirectional RNN.
  Instance merger_;
  Instance splitter_;
  Channel merged_;
  Channel dleft_;
  Channel dright_;

  // Channel for dropout mask.
  Channel mask_;

  // Random generator for dropout.
  std::mt19937_64 prng_;
  std::uniform_real_distribution<float> prob_{0.0, 1.0};
};

// Multi-layer RNN.
class RNNStack {
 public:
  RNNStack(const string &name) : name_(name) {}

  // Add RNN layer.
  void AddLayer(const RNN::Spec &spec, bool bidir);

  // Add multiple RNN layers of the same type.
  void AddLayers(int layers, const RNN::Spec &spec, bool bidir);

  // Build flow for RNNs.
  RNN::Variables Build(Flow *flow,
                       Flow::Variable *input,
                       Flow::Variable *dinput = nullptr);

  // Initialize RNN stack.
  void Initialize(const Network &net);

  // Layers in RNN stack.
  const std::vector<RNNLayer> &layers() const { return layers_; }

 private:
  // Name prefix for RNN cells.
  string name_;

  // RNN layers.
  std::vector<RNNLayer> layers_;
};

// Multi-layer RNN instance for prediction.
class RNNStackInstance {
 public:
  RNNStackInstance(const RNNStack &stack);

  // Compute RNN over input sequence and return output sequence.
  Channel *Compute(Channel *input);

 private:
  // RNN prediction instances for all layers.
  std::vector<RNNInstance> layers_;
};

// Multi-layer RNN layer for learning.
class RNNStackLearner {
 public:
  RNNStackLearner(const RNNStack &stack);

  // Compute RNN over input sequence and return output sequence.
  Channel *Compute(Channel *input);

  // Backpropagate gradients returning the output of backpropagation, i.e. the
  // gradient of the input sequence.
  Channel *Backpropagate(Channel *doutput);

  // Clear accumulated gradients.
  void Clear();

  // Collect instances with gradient updates.
  void CollectGradients(std::vector<Instance *> *gradients);

 private:
  // RNN learner instances for all layers.
  std::vector<RNNLearner> layers_;
};

}  // namespace myelin
}  // namespace sling

#endif  // SLING_MYELIN_RNN_H_

