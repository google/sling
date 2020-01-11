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

RNN::Variables RNN::Build(Flow *flow,
                          Flow::Variable *input,
                          Flow::Variable *dinput) {
  // Build RNN cell.
  bool learn = dinput != nullptr;
  Variables vars;
  FlowBuilder tf(flow, name);
  auto dt = input->type;
  int input_dim = input->dim(1);
  int rnn_dim = spec.dim;

  // Build inputs.
  auto *x = tf.Placeholder("input", dt, input->shape, true);
  auto *h_in = tf.Placeholder("h_in", dt, {1, rnn_dim}, true);
  Flow::Variable *c_in = nullptr;
  if (spec.type != GRU) {
    c_in = tf.Placeholder("c_in", dt, {1, rnn_dim}, true);
  }

  // Build recurrent unit.
  Flow::Variable *h_out = nullptr;     // hidden output
  Flow::Variable *c_out = nullptr;     // control output
  Flow::Variable *residual = nullptr;  // residial gate for highway connection
  switch (spec.type) {
    case LSTM: {
      // Standard LSTM.
      auto *x2i = tf.Parameter("x2i", dt, {input_dim, rnn_dim});
      auto *h2i = tf.Parameter("h2i", dt, {rnn_dim, rnn_dim});
      auto *bi = tf.Parameter("bi", dt, {1, rnn_dim});
      tf.RandomOrtho(x2i);
      tf.RandomOrtho(h2i);

      auto *x2f = tf.Parameter("x2f", dt, {input_dim, rnn_dim});
      auto *h2f = tf.Parameter("h2f", dt, {rnn_dim, rnn_dim});
      auto *bf = tf.Parameter("bf", dt, {1, rnn_dim});
      tf.RandomOrtho(x2f);
      tf.RandomOrtho(h2f);

      auto *x2g = tf.Parameter("x2g", dt, {input_dim, rnn_dim});
      auto *h2g = tf.Parameter("h2g", dt, {rnn_dim, rnn_dim});
      auto *bg = tf.Parameter("bg", dt, {1, rnn_dim});
      tf.RandomOrtho(x2g);
      tf.RandomOrtho(h2g);

      auto *x2o = tf.Parameter("x2o", dt, {input_dim, rnn_dim});
      auto *h2o = tf.Parameter("h2o", dt, {rnn_dim, rnn_dim});
      auto *bo = tf.Parameter("bo", dt, {1, rnn_dim});
      tf.RandomOrtho(x2o);
      tf.RandomOrtho(h2o);

      // i = sigmoid(x * x2i + h_in * h2i + bi)
      auto *ia = tf.Add(tf.MatMul(x, x2i), tf.Add(tf.MatMul(h_in, h2i), bi));
      auto *i = tf.Name(tf.Sigmoid(ia), "i");

      // f = sigmoid(x * x2f + h_in * h2f + bf)
      auto *fa = tf.Add(tf.MatMul(x, x2f), tf.Add(tf.MatMul(h_in, h2f), bf));
      auto *f = tf.Name(tf.Sigmoid(fa), "f");

      // g = tanh(x * x2g + h_in * h2g + bg)
      auto *ga = tf.Add(tf.MatMul(x, x2g), tf.Add(tf.MatMul(h_in, h2g), bg));
      auto *g = tf.Name(tf.Tanh(ga), "g");

      // o = sigmoid(x * x2o + h_in * h2o + bo)
      auto *oa = tf.Add(tf.MatMul(x, x2o), tf.Add(tf.MatMul(h_in, h2o), bo));
      auto *o = tf.Name(tf.Sigmoid(oa), "o");

      // residual = sigmoid(x * x2r + h_in * h2r + br)
      if (spec.highways) {
        auto *x2r = tf.Parameter("x2r", dt, {input_dim, rnn_dim});
        auto *h2r = tf.Parameter("h2r", dt, {rnn_dim, rnn_dim});
        auto *br = tf.Parameter("br", dt, {1, rnn_dim});
        tf.RandomOrtho(x2r);
        tf.RandomOrtho(h2r);

        auto *ra = tf.Add(tf.Add(tf.MatMul(x, x2r),tf.MatMul(h_in, h2r)), br);
        residual = tf.Name(tf.Sigmoid(ra), "r");
      }

      // c_out = f * c_in + i * g
      c_out = tf.Add(tf.Mul(f, c_in), tf.Mul(i, g));

      // h_out = o * tanh(c_out)
      h_out = tf.Mul(o, tf.Tanh(c_out));
      break;
    }

    case DRAGNN_LSTM: {
      // DRAGNN LSTM with peephole and coupled gates.
      auto *x2i = tf.Parameter("x2i", dt, {input_dim, rnn_dim});
      auto *h2i = tf.Parameter("h2i", dt, {rnn_dim, rnn_dim});
      auto *c2i = tf.Parameter("c2i", dt, {rnn_dim, rnn_dim});
      auto *bi = tf.Parameter("bi", dt, {1, rnn_dim});
      tf.RandomOrtho(x2i);
      tf.RandomOrtho(h2i);
      tf.RandomOrtho(c2i);

      auto *x2o = tf.Parameter("x2o", dt, {input_dim, rnn_dim});
      auto *h2o = tf.Parameter("h2o", dt, {rnn_dim, rnn_dim});
      auto *c2o = tf.Parameter("c2o", dt, {rnn_dim, rnn_dim});
      auto *bo = tf.Parameter("bo", dt, {1, rnn_dim});
      tf.RandomOrtho(x2o);
      tf.RandomOrtho(h2o);
      tf.RandomOrtho(c2o);

      auto *x2c = tf.Parameter("x2c", dt, {input_dim, rnn_dim});
      auto *h2c = tf.Parameter("h2c", dt, {rnn_dim, rnn_dim});
      auto *bc = tf.Parameter("bc", dt, {1, rnn_dim});
      tf.RandomOrtho(x2c);
      tf.RandomOrtho(h2c);

      // i = sigmoid(x * x2i + h_in * h2i + c_in * c2i + bi)
      auto *ia = tf.Add(tf.MatMul(x, x2i),
                 tf.Add(tf.MatMul(h_in, h2i),
                 tf.Add(tf.MatMul(c_in, c2i), bi)));
      auto *i = tf.Name(tf.Sigmoid(ia), "i");

      // f = 1 - i
      auto *f = tf.Name(tf.Sub(tf.One(), i), "f");

      // w = tanh(x * x2c + h_in * h2c + bc)
      auto *wa = tf.Add(tf.MatMul(x, x2c),
                 tf.Add(tf.MatMul(h_in, h2c), bc));
      auto *w = tf.Name(tf.Tanh(wa), "w");

      // c_out = i * w + f * c_in
      c_out = tf.Add(tf.Mul(i, w), tf.Mul(f, c_in));

      // o = sigmoid(x * x2o + c_out * c2o + h_in * h2o + bo)
      auto *oa = tf.Add(tf.MatMul(x, x2o),
                 tf.Add(tf.MatMul(c_out, c2o),
                 tf.Add(tf.MatMul(h_in, h2o), bo)));
      auto *o = tf.Name(tf.Sigmoid(oa), "o");

      // r = sigmoid(x * x2r + h_in * h2r + br)
      if (spec.highways) {
        auto *x2r = tf.Parameter("x2r", dt, {input_dim, rnn_dim});
        auto *h2r = tf.Parameter("h2r", dt, {rnn_dim, rnn_dim});
        auto *br = tf.Parameter("br", dt, {1, rnn_dim});
        tf.RandomOrtho(x2r);
        tf.RandomOrtho(h2r);
        auto *ra = tf.Add(tf.Add(tf.MatMul(x, x2r),tf.MatMul(h_in, h2r)), br);
        residual = tf.Name(tf.Sigmoid(ra), "r");
      }

      // h_out = o * tanh(c_out)
      h_out = tf.Mul(o, tf.Tanh(c_out));
      break;
    }

    case DOZAT_LSTM: {
      // Standard LSTM with one matrix multiplication.
      int gates = spec.highways ? 5 : 4;
      auto *w = tf.Parameter("W", dt, {input_dim + rnn_dim, gates * rnn_dim});
      auto *b = tf.Parameter("b", dt, {1, gates * rnn_dim});
      tf.RandomOrtho(w);

      // Preactivations.
      auto *xh = tf.Concat({x, h_in});
      auto p = tf.Split(tf.Add(tf.MatMul(xh, w), b), gates, 1);

      // Gates.
      auto *f = tf.Name(tf.Sigmoid(p[0]), "f");
      auto *i = tf.Name(tf.Sigmoid(p[1]), "i");
      auto *o = tf.Name(tf.Sigmoid(p[2]), "o");
      auto *g = tf.Name(tf.Tanh(p[3]), "g");
      if (spec.highways) {
        residual = tf.Name(tf.Sigmoid(p[4]), "r");
      }

      // Outputs.
      c_out = tf.Add(tf.Mul(f, c_in), tf.Mul(i, g));
      h_out = tf.Mul(o, tf.Tanh(c_out));
      break;
    }

    case PYTORCH_LSTM: {
      // Standard LSTM with two matrix multiplications.
      int gates = spec.highways ? 5 : 4;
      auto *w_ih = tf.Parameter("w_ih", dt, {input_dim, gates * rnn_dim});
      auto *w_hh = tf.Parameter("w_hh", dt, {rnn_dim, gates * rnn_dim});
      auto *b_ih = tf.Parameter("b_ih", dt, {1, gates * rnn_dim});
      auto *b_hh = tf.Parameter("b_hh", dt, {1, gates * rnn_dim});
      tf.RandomOrtho(w_ih);
      tf.RandomOrtho(w_hh);

      // Preactivations.
      auto *ih = tf.Add(tf.MatMul(x, w_ih), b_ih);
      auto *hh = tf.Add(tf.MatMul(h_in, w_hh), b_hh);
      auto p = tf.Split(tf.Add(ih, hh), gates, 1);

      // Gates.
      auto *f = tf.Name(tf.Sigmoid(p[0]), "f");
      auto *i = tf.Name(tf.Sigmoid(p[1]), "i");
      auto *o = tf.Name(tf.Sigmoid(p[2]), "o");
      auto *g = tf.Name(tf.Tanh(p[3]), "g");
      if (spec.highways) {
        residual = tf.Name(tf.Sigmoid(p[4]), "r");
      }

      // Outputs.
      c_out = tf.Add(tf.Mul(f, c_in), tf.Mul(i, g));
      h_out = tf.Mul(o, tf.Tanh(c_out));
      break;
    }

    case GRU: {
      // Gated Recurrent Unit.
      auto *x2z = tf.Parameter("x2z", dt, {input_dim, rnn_dim});
      auto *h2z = tf.Parameter("h2z", dt, {rnn_dim, rnn_dim});
      tf.RandomOrtho(x2z);
      tf.RandomOrtho(h2z);

      auto *x2r = tf.Parameter("x2r", dt, {input_dim, rnn_dim});
      auto *h2r = tf.Parameter("h2r", dt, {rnn_dim, rnn_dim});
      tf.RandomOrtho(x2r);
      tf.RandomOrtho(h2r);

      auto *x2h = tf.Parameter("x2h", dt, {input_dim, rnn_dim});
      auto *h2h = tf.Parameter("h2h", dt, {rnn_dim, rnn_dim});
      tf.RandomOrtho(x2h);
      tf.RandomOrtho(h2h);

      // z = sigmoid(x * x2z + h_in * h2z)
      auto *za = tf.Add(tf.MatMul(x, x2z), tf.MatMul(h_in, h2z));
      auto *z = tf.Name(tf.Sigmoid(za), "z");

      // r = sigmoid(x * x2r + h_in * h2r)
      auto *ra = tf.Add(tf.MatMul(x, x2r), tf.MatMul(h_in, h2r));
      auto *r = tf.Name(tf.Sigmoid(ra), "r");

      // h = tanh(x * x2h + (r * h_in) * h2h)
      auto *ha = tf.Add(tf.MatMul(x, x2h), tf.MatMul(tf.Mul(r, h_in), h2h));
      auto *h = tf.Name(tf.Tanh(ha), "h");

      // residual = sigmoid(x * x2b + h_in * h2b)
      if (spec.highways) {
        auto *x2b = tf.Parameter("x2b", dt, {input_dim, rnn_dim});
        auto *h2b = tf.Parameter("h2b", dt, {rnn_dim, rnn_dim});
        tf.RandomOrtho(x2b);
        tf.RandomOrtho(h2b);
        auto *ra = tf.Add(tf.MatMul(x, x2b),tf.MatMul(h_in, h2b));
        residual = tf.Name(tf.Sigmoid(ra), "r");
      }

      // h_out = (1 - z) * h_in + z * h
      h_out = tf.Add(tf.Mul(tf.Sub(tf.One(), z), h_in), tf.Mul(z, h));
      break;
    }

    default:
      LOG(FATAL) << "RNN type not supported: " << spec.type;
  }

  // Highway connection.
  if (residual != nullptr) {
    // Highway connection.
    auto *bypass = x;
    if (input_dim != rnn_dim) {
      // Linear transform from input to output dimension.
      auto *wx = tf.RandomOrtho(tf.Parameter("Wr", dt, {input_dim, rnn_dim}));
      bypass = tf.MatMul(x, wx);
    }
    h_out = tf.Add(tf.Mul(residual, h_out),
                   tf.Mul(tf.Sub(tf.One(), residual), bypass));
  }

  // Apply dropout to output.
  if (learn && spec.dropout != 0.0) {
    auto *mask = tf.Placeholder("mask", DT_FLOAT, {1, rnn_dim}, true);
    mask->set(Flow::Variable::NOGRADIENT);
    h_out = tf.Mul(h_out, mask);

    // The no-dropout mask is used for testing during training when no dropout
    // should be applied.
    std::vector<float> ones(rnn_dim, 1.0);
    auto *nodropout = tf.Name(tf.Const(ones), "nodropout");
    nodropout->set_out();
    flow->Connect({nodropout, mask});
  }

  // Name RNN outputs.
  if (h_out != nullptr) tf.Name(h_out, "h_out");
  if (c_out != nullptr) tf.Name(c_out, "c_out");

  // Make zero element.
  auto *zero = tf.Name(tf.Const(nullptr, dt, {1, rnn_dim}), "zero");
  zero->set_out();

  // Connect RNN units.
  vars.input = x;
  vars.output = h_out;
  flow->Connect({x, input});
  h_out->set_out()->set_ref();
  flow->Connect({h_in, h_out, zero});
  if (c_in != nullptr) {
    c_out->set_out()->set_ref();
    flow->Connect({c_in, c_out, zero});

    // The control channel has a single-source gradient.
    c_in->set_unique();
  }

  // Build gradients for learning.
  if (learn) {
    auto *gf = Gradient(flow, tf.func());
    vars.dinput = flow->GradientVar(vars.input);
    vars.doutput = flow->GradientVar(vars.output);
    flow->Connect({vars.dinput, dinput});

    // Make sink variable for final channel gradients.
    auto *sink = tf.Var("sink", dt, {1, rnn_dim})->set_out();
    gf->unused.push_back(sink);
    auto *dh_in = flow->GradientVar(h_in);
    auto *dh_out = flow->GradientVar(h_out);
    flow->Connect({dh_in, dh_out, sink});
    if (c_out != nullptr) {
      auto *dc_in = flow->GradientVar(c_in);
      auto *dc_out = flow->GradientVar(c_out);
      flow->Connect({dc_in, dc_out, sink});
    }
  }

  return vars;
}

void RNN::Initialize(const Network &net) {
  // Initialize RNN cell. Control channel is optional.
  cell = net.GetCell(name);
  input = net.GetParameter(name + "/input");
  h_in = net.GetParameter(name + "/h_in");
  h_out = net.GetParameter(name + "/h_out");
  c_in = net.LookupParameter(name + "/c_in");
  c_out = net.LookupParameter(name + "/c_out");
  zero = net.GetParameter(name + "/zero");

  // Initialize gradient cell for RNN.
  gcell = cell->Gradient();
  if (gcell != nullptr) {
    primal = cell->Primal();
    dinput = input->Gradient();
    dh_in = h_in->Gradient();
    dh_out = h_out->Gradient();
    dc_in = c_in == nullptr ? nullptr : c_in->Gradient();
    dc_out = c_out == nullptr ? nullptr : c_out->Gradient();
    sink = net.GetParameter(name + "/sink");
  }

  // Initialize dropout mask.
  if (spec.dropout != 0.0) {
    mask = net.GetParameter(name + "/mask");
    nodropout = net.GetParameter(name + "/nodropout");
  }
}

RNNMerger::Variables RNNMerger::Build(Flow *flow,
                                      Flow::Variable *left,
                                      Flow::Variable *right,
                                      Flow::Variable *dleft,
                                      Flow::Variable *dright) {
  Variables vars;

  // Build merger cell.
  FlowBuilder f(flow, name);
  vars.left = f.Placeholder("left", left->type, left->shape);
  vars.left->set_dynamic()->set_unique();

  vars.right = f.Placeholder("right", right->type, right->shape);
  vars.right->set_dynamic()->set_unique();

  vars.merged = f.Name(f.Concat({vars.left, vars.right}, 1), "merged");
  vars.merged->set_dynamic();
  flow->Connect({vars.left, left});
  flow->Connect({vars.right, right});

  // Build gradients for learning.
  if (dleft != nullptr && dright != nullptr) {
    Gradient(flow, f.func());
    vars.dmerged = flow->GradientVar(vars.merged);
    vars.dleft = flow->GradientVar(vars.left);
    vars.dright = flow->GradientVar(vars.right);
    flow->Connect({vars.dleft, dleft});
    flow->Connect({vars.dright, dright});
  } else {
    vars.dmerged = vars.dleft = vars.dright = nullptr;
  }

  return vars;
}

void RNNMerger::Initialize(const Network &net) {
  cell = net.GetCell(name);
  left = net.GetParameter(name + "/left");
  right = net.GetParameter(name + "/right");
  merged = net.GetParameter(name + "/merged");

  gcell = cell->Gradient();
  if (gcell != nullptr) {
    dmerged = merged->Gradient();
    dleft = left->Gradient();
    dright = right->Gradient();
  }
}

RNNLayer::RNNLayer(const string &name, const RNN::Spec &spec, bool bidir)
    : name_(name),
      bidir_(bidir),
      dropout_(spec.dropout),
      lr_(bidir ? name + "/lr" : name, spec),
      rl_(name + "/rl", spec),
      merger_(name) {}

RNN::Variables RNNLayer::Build(Flow *flow,
                               Flow::Variable *input,
                               Flow::Variable *dinput) {
  if (bidir_) {
    // Build left-to-right and right-to-left RNNs.
    auto l = lr_.Build(flow, input, dinput);
    auto r = rl_.Build(flow, input, dinput);

    // Build channel merger.
    auto m = merger_.Build(flow, l.output, r.output, l.doutput, r.doutput);

    // Return outputs.
    RNN::Variables vars;
    vars.input = l.input;
    vars.output = m.merged;
    vars.dinput = l.dinput;
    vars.doutput = m.dmerged;

    return vars;
  } else {
    return lr_.Build(flow, input, dinput);
  }
}

void RNNLayer::Initialize(const Network &net) {
  lr_.Initialize(net);
  if (bidir_) {
    rl_.Initialize(net);
    merger_.Initialize(net);
  }
}

RNNInstance::RNNInstance(const RNNLayer *rnn)
    : rnn_(rnn),
      lr_(rnn->lr_.cell),
      lr_hidden_(rnn->lr_.h_out),
      lr_control_(rnn->lr_.c_out),
      rl_(rnn->rl_.cell),
      rl_hidden_(rnn->rl_.h_out),
      rl_control_(rnn->rl_.c_out),
      merger_(rnn->merger_.cell),
      merged_(rnn->merger_.merged) {}

Channel *RNNInstance::Compute(Channel *input) {
  // Get sequence length.
  int length = input->size();
  bool ctrl = rnn_->lr_.has_control();

  // Set pass-through dropout mask.
  if (rnn_->lr_.has_mask()) {
    lr_.SetReference(rnn_->lr_.mask, rnn_->lr_.nodropout->data());
  }

  // Compute left-to-right RNN.
  lr_hidden_.resize(length);
  if (ctrl) lr_control_.resize(length);

  if (length > 0) {
    lr_.Set(rnn_->lr_.input, input, 0);
    lr_.SetReference(rnn_->lr_.h_in, rnn_->lr_.zero->data());
    lr_.Set(rnn_->lr_.h_out, &lr_hidden_, 0);
    if (ctrl) {
      lr_.SetReference(rnn_->lr_.c_in, rnn_->lr_.zero->data());
      lr_.Set(rnn_->lr_.c_out, &lr_control_, 0);
    }
    lr_.Compute();
  }

  for (int i = 1; i < length; ++i) {
    lr_.Set(rnn_->lr_.input, input, i);
    lr_.Set(rnn_->lr_.h_in, &lr_hidden_, i - 1);
    lr_.Set(rnn_->lr_.h_out, &lr_hidden_, i);
    if (ctrl) {
      lr_.Set(rnn_->lr_.c_in, &lr_control_, i - 1);
      lr_.Set(rnn_->lr_.c_out, &lr_control_, i);
    }
    lr_.Compute();
  }

  // Return left-to-right hidden channel for unidirectional RNN.
  if (!rnn_->bidir_) return &lr_hidden_;

  // Set pass-through dropout mask.
  if (rnn_->rl_.has_mask()) {
    rl_.SetReference(rnn_->rl_.mask, rnn_->rl_.nodropout->data());
  }

  // Compute right-to-left RNN.
  rl_hidden_.resize(length);
  if (ctrl) rl_control_.resize(length);

  if (length > 0) {
    rl_.Set(rnn_->rl_.input, input, length - 1);
    rl_.SetReference(rnn_->rl_.h_in, rnn_->rl_.zero->data());
    rl_.Set(rnn_->rl_.h_out, &rl_hidden_, length - 1);
    if (ctrl) {
      rl_.SetReference(rnn_->rl_.c_in, rnn_->rl_.zero->data());
      rl_.Set(rnn_->rl_.c_out, &rl_control_, length - 1);
    }
    rl_.Compute();
  }

  for (int i = length - 2; i >= 0; --i) {
    rl_.Set(rnn_->rl_.input, input, i);
    rl_.Set(rnn_->rl_.h_in, &rl_hidden_, i + 1);
    rl_.Set(rnn_->rl_.h_out, &rl_hidden_, i);
    if (ctrl) {
      rl_.Set(rnn_->rl_.c_in, &rl_control_, i + 1);
      rl_.Set(rnn_->rl_.c_out, &rl_control_, i);
    }
    rl_.Compute();
  }

  // Merge outputs.
  merged_.resize(length);
  merger_.SetChannel(rnn_->merger_.left, &lr_hidden_);
  merger_.SetChannel(rnn_->merger_.right, &rl_hidden_);
  merger_.SetChannel(rnn_->merger_.merged, &merged_);
  merger_.Compute();

  return &merged_;
}

RNNLearner::RNNLearner(const RNNLayer *rnn)
    : rnn_(rnn),
      lr_fwd_(rnn->lr_.cell),
      lr_hidden_(rnn->lr_.h_out),
      lr_control_(rnn->lr_.c_out),
      lr_bkw_(rnn->lr_.gcell),
      lr_dhidden_(rnn->lr_.dh_in),
      lr_dcontrol_(rnn->lr_.dc_in),
      rl_fwd_(rnn->rl_.cell),
      rl_hidden_(rnn->rl_.h_out),
      rl_control_(rnn->rl_.c_out),
      rl_bkw_(rnn->rl_.gcell),
      rl_dhidden_(rnn->rl_.dh_in),
      rl_dcontrol_(rnn->rl_.dc_in),
      dinput_(rnn_->lr_.dinput),
      merger_(rnn->merger_.cell),
      splitter_(rnn->merger_.gcell),
      merged_(rnn->merger_.merged),
      dleft_(rnn->merger_.dleft),
      dright_(rnn->merger_.dright),
      mask_(rnn_->lr_.mask) {
  if (rnn->dropout_ != 0.0) {
    mask_.resize(1);
  }
}

Channel *RNNLearner::Compute(Channel *input) {
  // Get sequence length.
  int length = input->size();
  bool ctrl = rnn_->lr_.has_control();

  // Set up dropout mask.
  bool dropout = rnn_->dropout_ != 0.0;
  if (dropout) {
    float *mask = reinterpret_cast<float *>(mask_.at(0));
    float rate = rnn_->dropout_;
    float scaler = 1.0 / (1.0 - rate);
    int size = rnn_->lr_.spec.dim;
    for (int i = 0; i < size; ++i) {
      mask[i] = Random() < rate ? 0.0 : scaler;
    }
  }

  // Compute left-to-right RNN.
  lr_fwd_.Resize(length);
  lr_hidden_.resize(length);
  if (ctrl) lr_control_.resize(length);

  if (length > 0) {
    Instance &data = lr_fwd_[0];
    data.Set(rnn_->lr_.input, input, 0);
    data.SetReference(rnn_->lr_.h_in, rnn_->lr_.zero->data());
    data.Set(rnn_->lr_.h_out, &lr_hidden_, 0);
    if (ctrl) {
      data.SetReference(rnn_->lr_.c_in, rnn_->lr_.zero->data());
      data.Set(rnn_->lr_.c_out, &lr_control_, 0);
    }
    if (dropout) {
      data.Set(rnn_->lr_.mask, &mask_, 0);
    }
    data.Compute();
  }

  for (int i = 1; i < length; ++i) {
    Instance &data = lr_fwd_[i];
    data.Set(rnn_->lr_.input, input, i);
    data.Set(rnn_->lr_.h_in, &lr_hidden_, i - 1);
    data.Set(rnn_->lr_.h_out, &lr_hidden_, i);
    if (ctrl) {
      data.Set(rnn_->lr_.c_in, &lr_control_, i - 1);
      data.Set(rnn_->lr_.c_out, &lr_control_, i);
    }
    if (dropout) {
      data.Set(rnn_->lr_.mask, &mask_, 0);
    }
    data.Compute();
  }

  // Return left-to-right hidden channel for unidirectional RNN.
  if (!rnn_->bidir_) return &lr_hidden_;

  // Compute right-to-left RNN.
  rl_fwd_.Resize(length);
  rl_hidden_.resize(length);
  if (ctrl) rl_control_.resize(length);

  if (length > 0) {
    Instance &data = rl_fwd_[length - 1];
    data.Set(rnn_->rl_.input, input, length - 1);
    data.SetReference(rnn_->rl_.h_in, rnn_->rl_.zero->data());
    data.Set(rnn_->rl_.h_out, &rl_hidden_, length - 1);
    if (ctrl) {
      data.SetReference(rnn_->rl_.c_in, rnn_->rl_.zero->data());
      data.Set(rnn_->rl_.c_out, &rl_control_, length - 1);
    }
    if (dropout) {
      data.Set(rnn_->rl_.mask, &mask_, 0);
    }
    data.Compute();
  }

  for (int i = length - 2; i >= 0; --i) {
    Instance &data = rl_fwd_[i];
    data.Set(rnn_->rl_.input, input, i);
    data.Set(rnn_->rl_.h_in, &rl_hidden_, i + 1);
    data.Set(rnn_->rl_.h_out, &rl_hidden_, i);
    if (ctrl) {
      data.Set(rnn_->rl_.c_in, &rl_control_, i + 1);
      data.Set(rnn_->rl_.c_out, &rl_control_, i);
    }
    if (dropout) {
      data.Set(rnn_->rl_.mask, &mask_, 0);
    }
    data.Compute();
  }

  // Merge outputs.
  merged_.resize(length);
  merger_.SetChannel(rnn_->merger_.left, &lr_hidden_);
  merger_.SetChannel(rnn_->merger_.right, &rl_hidden_);
  merger_.SetChannel(rnn_->merger_.merged, &merged_);
  merger_.Compute();

  return &merged_;
}

Channel *RNNLearner::Backpropagate(Channel *doutput) {
  // Clear input gradient.
  int length = doutput->size();
  dinput_.reset(length);
  bool ctrl = rnn_->lr_.has_control();

  // Split gradient for bidirectional RNN.
  Channel *dleft;
  Channel *dright;
  if (rnn_->bidir_) {
    // Split gradients.
    dleft_.resize(length);
    dright_.resize(length);
    splitter_.SetChannel(rnn_->merger_.dmerged, doutput);
    splitter_.SetChannel(rnn_->merger_.dleft, &dleft_);
    splitter_.SetChannel(rnn_->merger_.dright, &dright_);
    splitter_.Compute();
    dleft = &dleft_;
    dright = &dright_;
  } else {
    dleft = doutput;
    dright = nullptr;
  }

  // Propagate gradients for left-to-right RNN.
  if (dleft != nullptr) {
    if (ctrl) lr_dcontrol_.reset(length);

    for (int i = length - 1; i > 0; --i) {
      lr_bkw_.Set(rnn_->lr_.primal, &lr_fwd_[i]);
      lr_bkw_.Set(rnn_->lr_.dh_out, dleft, i);
      lr_bkw_.Set(rnn_->lr_.dh_in, dleft, i - 1);
      lr_bkw_.Set(rnn_->lr_.dinput, &dinput_, i);
      if (ctrl) {
        lr_bkw_.Set(rnn_->lr_.dc_out, &lr_dcontrol_, i);
        lr_bkw_.Set(rnn_->lr_.dc_in, &lr_dcontrol_, i - 1);
      }
      lr_bkw_.Compute();
    }

    if (length > 0) {
      void *sink = lr_bkw_.GetAddress(rnn_->lr_.sink);
      lr_bkw_.Set(rnn_->lr_.primal, &lr_fwd_[0]);
      lr_bkw_.Set(rnn_->lr_.dh_out, dleft, 0);
      lr_bkw_.SetReference(rnn_->lr_.dh_in, sink);
      lr_bkw_.Set(rnn_->lr_.dinput, &dinput_, 0);
      if (ctrl) {
        lr_bkw_.Set(rnn_->lr_.dc_out, &lr_dcontrol_, 0);
        lr_bkw_.SetReference(rnn_->lr_.dc_in, sink);
      }
      lr_bkw_.Compute();
    }
  }

  // Propagate gradients for right-to-left RNN.
  if (dright != nullptr) {
    if (ctrl) rl_dcontrol_.reset(length);

    for (int i = 0; i < length - 1; ++i) {
      rl_bkw_.Set(rnn_->rl_.primal, &rl_fwd_[i]);
      rl_bkw_.Set(rnn_->rl_.dh_out, dright, i);
      rl_bkw_.Set(rnn_->rl_.dh_in, dright, i + 1);
      rl_bkw_.Set(rnn_->rl_.dinput, &dinput_, i);
      if (ctrl) {
        rl_bkw_.Set(rnn_->rl_.dc_out, &rl_dcontrol_, i);
        rl_bkw_.Set(rnn_->rl_.dc_in, &rl_dcontrol_, i + 1);
      }
      rl_bkw_.Compute();
    }

    if (length > 0) {
      void *sink = rl_bkw_.GetAddress(rnn_->rl_.sink);
      rl_bkw_.Set(rnn_->rl_.primal, &rl_fwd_[length - 1]);
      rl_bkw_.Set(rnn_->rl_.dh_out, dright, length - 1);
      rl_bkw_.SetReference(rnn_->rl_.dh_in, sink);
      rl_bkw_.Set(rnn_->rl_.dinput, &dinput_, length - 1);
      if (ctrl) {
        rl_bkw_.Set(rnn_->rl_.dc_out, &rl_dcontrol_, length - 1);
        rl_bkw_.SetReference(rnn_->rl_.dc_in, sink);
      }
      rl_bkw_.Compute();
    }
  }

  // Return input gradient.
  return &dinput_;
}

void RNNLearner::Clear() {
  lr_bkw_.Clear();
  if (rnn_->bidir_) rl_bkw_.Clear();
}

void RNNLearner::CollectGradients(std::vector<Instance *> *gradients) {
  gradients->push_back(&lr_bkw_);
  if (rnn_->bidir_) gradients->push_back(&rl_bkw_);
}

void RNNStack::AddLayer(const RNN::Spec &spec, bool bidir) {
  string name = name_ + "/rnn" + std::to_string(layers_.size());
  layers_.emplace_back(name, spec, bidir);
}

void RNNStack::AddLayers(int layers, const RNN::Spec &spec, bool bidir) {
  for (int l = 0; l < layers; ++l) {
    AddLayer(spec, bidir);
  }
}

RNN::Variables RNNStack::Build(Flow *flow,
                               Flow::Variable *input,
                               Flow::Variable *dinput) {
  RNN::Variables vars;
  vars.input = vars.output = input;
  vars.dinput = vars.doutput = dinput;
  for (RNNLayer &l : layers_) {
    RNN::Variables v = l.Build(flow, vars.output, vars.doutput);
    vars.output = v.output;
    vars.doutput = v.doutput;
  }
  return vars;
}

void RNNStack::Initialize(const Network &net) {
  for (RNNLayer &l : layers_) {
    l.Initialize(net);
  }
}

RNNStackInstance::RNNStackInstance(const RNNStack &stack) {
  layers_.reserve(stack.layers().size());
  for (const RNNLayer &l : stack.layers()) {
    layers_.emplace_back(&l);
  }
}

Channel *RNNStackInstance::Compute(Channel *input) {
  Channel *channel = input;
  for (RNNInstance &l : layers_) {
    channel = l.Compute(channel);
  }
  return channel;
}

RNNStackLearner::RNNStackLearner(const RNNStack &stack) {
  layers_.reserve(stack.layers().size());
  for (const RNNLayer &l : stack.layers()) {
    layers_.emplace_back(&l);
  }
}

Channel *RNNStackLearner::Compute(Channel *input) {
  Channel *channel = input;
  for (RNNLearner &l : layers_) {
    channel = l.Compute(channel);
  }
  return channel;
}

Channel *RNNStackLearner::Backpropagate(Channel *doutput) {
  Channel *channel = doutput;
  for (int i = layers_.size() - 1; i >= 0; --i) {
    channel = layers_[i].Backpropagate(channel);
  }
  return channel;
}

void RNNStackLearner::Clear() {
  for (RNNLearner &l : layers_) {
    l.Clear();
  }
}

void RNNStackLearner::CollectGradients(std::vector<Instance *> *gradients) {
  for (RNNLearner &l : layers_) {
    l.CollectGradients(gradients);
  }
}

}  // namespace myelin
}  // namespace sling

