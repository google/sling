# Copyright 2017 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http:#www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# PyTorch module implementations for Caspar.

import math
import numpy as np
import torch
import torch.nn as nn

from cascade import Delegate
from cascade import SoftmaxDelegate
from lexical_encoder import LexicalEncoder
from parser_state import ParserState
from trace import Trace

import sling
import sling.myelin.nn as flownn
import sling.myelin.flow as flow
import sling.myelin.builder as builder

Param = nn.Parameter
Var = torch.autograd.Variable

# Utility for dumping a (list of) PyTorch tensor/variable.
def fstr(var, fmt="%.9f"):
  if type(var) is tuple and len(var) == 1: var = var[0]

  dim = var.dim()
  if dim == 1 or (dim == 2 and (var.size(0) == 1 or var.size(1) == 1)):
    var = var.view(1, -1)
  ls = var.data.numpy().tolist()
  if type(ls[0]) is list: ls = ls[0]
  ls = [fmt % x for x in ls]
  return "[" + ",".join(ls) + "]"


# Reads the parameter 'param' from a flow variable 'var'.
def parameter_from_flow(var, param):
  shape = list(param.data.shape)   # from torch.Size to python list
  numpy_array = np.frombuffer(bytearray(var.data), dtype=np.float32)
  numpy_array = numpy_array.reshape(shape)
  param.data = torch.from_numpy(numpy_array)

# Projects input x to xA + b. This is in contrast to nn.Linear which does Ax + b
# and consequently has different parameter dimensionalities.
class Projection(nn.Module):
  def __init__(self, num_in, num_out, bias=True):
    super(Projection, self).__init__()
    self.weight = Param(torch.randn(num_in, num_out))
    self.bias = None
    if bias:
      self.bias = Param(torch.randn(1, num_out))


  # Initializes the weights with gaussian(mean=0, stddev='weight_stddev'),
  # and all biases with 'bias_const'.
  def init(self, weight_stddev, bias_const=None):
    self.weight.data.normal_()
    self.weight.data.mul_(weight_stddev)
    if bias_const is not None and self.bias is not None:
      self.bias.data.fill_(bias_const)


  # Computes and returns xA + b.
  def forward(self, x):
    if x.size()[0] != 1: x = x.view(1, -1)
    out = torch.mm(x, self.weight)
    if self.bias is not None:
      out = out + self.bias
    return out


  # Returns a string specification of the module.
  def __repr__(self):
    s = self.weight.size()
    return self.__class__.__name__ + "(in=" + str(s[0]) + \
        ", out=" + str(s[1]) + ", bias=" + str(self.bias is not None) + ")"


# Transforms input x to x * A. If x is None, returns OOV * A.
class LinkTransform(nn.Module):
  def __init__(self, activation_size, dim):
    super(LinkTransform, self).__init__()
    self.projection = Projection(activation_size, dim, bias=False)
    self.oov = Param(torch.randn(1, activation_size))


  # Forward pass.
  def forward(self, activation=None):
    if activation is None:
      activation = self.oov
    return self.projection.forward(activation)


  # Initializes all weights with (mean=0, stddev=stddev).
  def init(self, stddev):
    self.projection.init(stddev)
    self.oov.data.normal_()
    self.oov.data.mul_(stddev)


  # Returns a string specification of the module.
  def __repr__(self):
    s = self.projection.weight.size()
    return self.__class__.__name__ + "(input_activation=" + str(s[0]) + \
        ", dim=" + str(s[1]) + ", oov_vector=" + str(s[0])+ ")"


# PyTorch replication of the DRAGNN LSTM Cell. This is slightly different from
# PyTorch's built-in LSTM implementation.
class DragnnLSTM(nn.Module):
  def __init__(self, input_dim, hidden_dim):
    super(DragnnLSTM, self).__init__()
    self.input_dim = input_dim
    self.hidden_dim = hidden_dim

    self._x2i = Param(torch.randn(input_dim, hidden_dim))
    self._h2i = Param(torch.randn(hidden_dim, hidden_dim))
    self._c2i = Param(torch.randn(hidden_dim, hidden_dim))
    self._bi = Param(torch.randn(1, hidden_dim))

    self._x2o = Param(torch.randn(input_dim, hidden_dim))
    self._h2o = Param(torch.randn(hidden_dim, hidden_dim))
    self._c2o = Param(torch.randn(hidden_dim, hidden_dim))
    self._bo = Param(torch.randn(1, hidden_dim))

    self._x2c = Param(torch.randn(input_dim, hidden_dim))
    self._h2c = Param(torch.randn(hidden_dim, hidden_dim))
    self._bc = Param(torch.randn(1, hidden_dim))


  # Runs the LSTM cell one step using the previous step's hidden and control
  # outputs in 'prev_h' and 'prev_c', and the input at the current step in
  # 'input_tensor'.
  # Currently limited to a batch size of 1.
  def forward_one_step(self, input_tensor, prev_h, prev_c):
    i_ait = torch.mm(input_tensor, self._x2i) + \
        torch.mm(prev_h, self._h2i) + \
        torch.mm(prev_c, self._c2i) + \
        self._bi
    i_it = torch.sigmoid(i_ait)
    i_ft = 1.0 - i_it
    i_awt = torch.mm(input_tensor, self._x2c) + \
        torch.mm(prev_h, self._h2c) + self._bc
    i_wt = torch.tanh(i_awt)
    ct = torch.mul(i_it, i_wt) + torch.mul(i_ft, prev_c)
    i_aot = torch.mm(input_tensor, self._x2o) + \
        torch.mm(ct, self._c2o) + torch.mm(prev_h, self._h2o) + self._bo
    i_ot = torch.sigmoid(i_aot)
    ph_t = torch.tanh(ct)
    ht = torch.mul(i_ot, ph_t)

    return (ht, ct)


  # Runs the LSTM Cell as many as the major size of 'input_tensors', using
  # all-zeros initial hidden and control states.
  # 'input_tensors' should be (Length x LSTM Input Dim).
  # Outputs the hidden and control states for all the steps.
  def forward(self, input_tensors):
    h = Var(torch.zeros(1, self.hidden_dim))
    c = Var(torch.zeros(1, self.hidden_dim))
    hidden = []
    cell = []

    length = input_tensors.size(0)
    for i in range(length):
      (h, c) = self.forward_one_step(input_tensors[i].view(1, -1), h, c)
      hidden.append(h)
      cell.append(c)

    return (hidden, cell)


  # Copies the parameters to the Myelin flow LSTM instance in 'flow_lstm'.
  def copy_to_flow_lstm(self, flow_lstm):
    for p in ["x2i", "h2i", "c2i", "bi", \
              "x2o", "h2o", "c2o", "bo", "x2c", "h2c", "bc"]:
      assert hasattr(self, "_" + p), p
      assert hasattr(flow_lstm, p), p
      getattr(flow_lstm, p).data = getattr(self, "_" + p).data.numpy()


  # Copies parameters from Myelin flow.
  def copy_from_flow(self, fl, func_name):
    for p in ["x2i", "h2i", "c2i", "bi", \
              "x2o", "h2o", "c2o", "bo", "x2c", "h2c", "bc"]:
      assert hasattr(self, "_" + p), p
      var = fl.var(func_name + "/" + p)
      param = getattr(self, "_" + p)
      parameter_from_flow(var, param)


  # Returns a string specification of the module.
  def __repr__(self):
    return self.__class__.__name__ + "(in=" + str(self.input_dim) + \
        ", hidden=" + str(self.hidden_dim) + ")"


# Head of the FF unit for Softmax delegates.
class SoftmaxHead:
  def __init__(self, input_size, output_size):
    self.softmax = Projection(input_size, output_size)

  def __call__(self, ff_activation, train=False):
    logits = self.softmax(ff_activation)
    if not train:
      best_score, best_index = torch.max(logits, 1)
      return best_index.data[0]
    return logits


# Default loss function for Softmax delegates.
class SoftmaxLoss:
  def __init__(self):
    self.fn = nn.CrossEntropyLoss()

  def __call__(self, logits, gold_index):
    gold_var = Var(torch.LongTensor([gold_index]))
    return self.fn(logits, gold_var)


# Keeps track of per-delegate per-transition losses.
class Losses:
  def __init__(self):
    self.losses = {}  # delegate -> (loss, num transitions)

  # Adds specified delegate loss.
  def add(self, delegate_index, step_loss, count=1):
    if delegate_index not in self.losses:
      self.losses[delegate_index] = [Var(torch.Tensor([0.0])), 0]
    self.losses[delegate_index][0] += step_loss
    self.losses[delegate_index][1] += count

  # Adds losses in 'other' to itself.
  def aggregate(self, other):
    for delegate, value in other.losses.items():
      self.add(delegate, value[0], value[1])

  # Returns (total loss, total number of transitions).
  def total(self):
    loss = 0
    count = 0
    for _,v in self.losses.items():
      loss += v[0]
      count += v[1]
    return (loss, count)

  # Returns average per-transition loss across all delegates.
  def average(self):
    (loss, count) = self.total()
    return loss / count

  # Prints all losses.
  def tostring(self, after=None):
    s = ""
    for k in sorted(self.losses.keys()):
      if s != "": s += "\n"
      s += "AvgDelegateLoss for " + str(k)
      if after is not None:
        s += " after " + str(after) + " examples "
      l = self.losses[k]
      s += "= " + str(l[0].data[0]) + "/" + str(l[1]) + " = "
      s += str(l[0].data[0] / l[1])
    return s

  # Returns a dict where all values are numbers instead of tensors/variables.
  def tonum(self):
    output = {}
    total = 0
    total_count = 0
    for k,v in self.losses.items():
      output[k] = (v[0].data[0], v[1])
      total += v[0].data[0]
      total_count += v[1]
    output["total"] = (total.item(), total_count)
    return output


# Asserts 'delegate' to be a softmax delegate.
def assert_softmax_delegate(delegate):
  assert isinstance(delegate, SoftmaxDelegate), delegate.__class__.__name__


# Top-level module for CASPAR.
class Caspar(nn.Module):
  def __init__(self, spec):
    super(Caspar, self).__init__()
    self.spec = spec

    # LSTM Embeddings.
    self.lstm_embeddings = []
    for f in spec.lstm_features:
      embedding = nn.EmbeddingBag(f.vocab_size, f.dim, mode='sum')
      self.add_module('lstm_embedding_' + f.name, embedding)
      self.lstm_embeddings.append(embedding)
      f.embedding = embedding

    # LR and RL LSTM cells.
    self.lr_lstm = DragnnLSTM(spec.lstm_input_dim, spec.lstm_hidden_dim)
    self.rl_lstm = DragnnLSTM(spec.lstm_input_dim, spec.lstm_hidden_dim)

    # FF Embeddings and network.
    for f in spec.ff_fixed_features:
      embedding = nn.EmbeddingBag(f.vocab_size, f.dim, mode='sum')
      self.add_module('ff_fixed_embedding_' + f.name, embedding)
      f.bag = embedding

    for f in spec.ff_link_features:
      transform = LinkTransform(f.activation_size, f.dim)
      self.add_module('ff_link_transform_' + f.name, transform)
      f.transform = transform

    # Feedforward unit trunk.
    h = spec.ff_hidden_dim
    self.ff_layer = Projection(spec.ff_input_dim, h)   # hidden layer
    self.ff_relu = nn.ReLU()                           # non-linearity

    # Feedforward unit heads.
    cascade = self.spec.cascade
    self.ff_heads = []
    for index, delegate in enumerate(cascade.delegates):
      assert_softmax_delegate(delegate)
      head = SoftmaxHead(h, delegate.size())
      self.ff_heads.append(head)
      self.add_module("ff_softmax_" + str(index), head.softmax)
      delegate.set_model(head)
      delegate.set_loss(SoftmaxLoss())

    # Only regularize the FF hidden layer weights.
    self.regularized_params = [self.ff_layer.weight]
    print("Caspar:", self)


  # Prints the first and last few elements of each parameter.
  def print_parameters(self):
    for name, p in self.named_parameters():
      tensor = p.data.view([-1]).numpy()
      s1 = ["%.3f" % f for f in tensor[0:5]]
      s2 = ["%.3f" % f for f in tensor[-5:]]
      print(name, ' '.join(s1), "...", ' '.join(s2))


  # Initializes various module parameters.
  def initialize(self, word_embeddings_file):
    word_embeddings = None
    word_embedding_indices = None
    if word_embeddings_file is not None:
      word_embeddings, word_embedding_indices = \
        self.spec.load_word_embeddings(word_embeddings_file)

    # Initialize the embeddings to gaussian(mean=0, stddev=1/sqrt(dim)),
    # where 'dim' is the dimensionality of the embedding.
    for f in self.spec.lstm_features:
      coeff = 1.0 / math.sqrt(f.dim)
      matrix = f.embedding.weight.data
      matrix.normal_()
      matrix.mul_(coeff)

      # Override with pre-trained word embeddings, if provided.
      if f.name == "word" and word_embeddings is not None:
        indices = torch.LongTensor(word_embedding_indices)
        data = torch.Tensor(word_embeddings)

        # Separately normalize each embedding row
        data = torch.nn.functional.normalize(data)

        # Copy the normalized embeddings at appropriate indices.
        matrix.index_copy_(0, indices, data)
        print("Overwrote", len(word_embeddings), f.name, \
            "embedding vectors with normalized pre-trained vectors.")

    # Initialize the FF's fixed and link embeddings like those in the LSTMs.
    for f in self.spec.ff_fixed_features:
      f.bag.weight.data.normal_()
      f.bag.weight.data.mul_(1.0 / math.sqrt(f.dim))

    for f in self.spec.ff_link_features:
      f.transform.init(1.0 / math.sqrt(f.dim))

    # Initialize the LSTM and FF parameters with gaussian(mean=0, stddev=1e-4).
    params = [self.ff_layer.weight]
    params += [p for p in self.lr_lstm.parameters()]
    params += [p for p in self.rl_lstm.parameters()]
    for p in params:
      if len(p.shape) >= 2:
        nn.init.orthogonal(p.data)
      else:
        nn.init.normal(p.data, mean=0, stddev=1e-4)

    # Initialize output layer weights.
    for head in self.ff_heads:
      nn.init.xavier_normal(head.softmax.weight.data)

    # Positive bias for the hidden layer, and zero bias for the output layers.
    self.ff_layer.bias.data.fill_(0.2)
    for head in self.ff_heads:
      head.softmax.bias.data.fill_(0.0)


  # Looks up the embedding bags for respective features indices, and returns
  # their concatenated results.
  def _embedding_lookup(self, embedding_bags, features):
    assert len(embedding_bags) == len(features)
    values = []
    for feature, bag in zip(features, embedding_bags):
      if not feature.has_multi and not feature.has_empty:
        # This case covers features that return exactly one value per call.
        # So this covers word features and all fallback features.
        indices = Var(torch.LongTensor(feature.indices))
        values.append(bag(indices.view(len(feature.indices), 1)))
      else:
        # Other features, e.g. suffixes, may return 0 or >1 ids.
        subvalues = []
        dim = bag.weight.size(1)
        for index, i in enumerate(feature.indices):
          if type(i) is int:  # one feature id
            subvalues.append(bag(Var(torch.LongTensor([i])).view(1, 1)))
          elif len(i) > 0:    # multiple feature ids
            subvalues.append(bag(Var(torch.LongTensor(i)).view(1, len(i))))
          else:               # no feature id => zero vector
            subvalues.append(Var(torch.zeros(1, dim)))
        values.append(torch.cat(subvalues, 0))

    return torch.cat(values, 1)


  # Returns the outputs from both the LSTMs for all tokens in 'document'.
  def _lstm_outputs(self, document):
    # Compute all raw token features just once for both the LSTMs.
    raw_features = self.spec.raw_lstm_features(document)
    length = document.size()

    # 'lstm_inputs' should have shape (length, lstm_input_dim).
    lstm_inputs = self._embedding_lookup(self.lstm_embeddings, raw_features)
    assert length == lstm_inputs.size(0)

    lr_out, _ = self.lr_lstm.forward(lstm_inputs)

    # Note: Negative strides are not supported, otherwise we would just do:
    #   rl_input = lstm_inputs[::-1]
    inverse_indices = torch.arange(length - 1, -1, -1).long()
    rl_inputs = lstm_inputs[inverse_indices]
    rl_out, _ = self.rl_lstm.forward(rl_inputs)
    rl_out.reverse()

    # lr_out[i], rl_out[i] are the LSTM outputs for the ith token.
    return (lr_out, rl_out, raw_features)


  # Returns the FF activation, given the LSTM outputs and previous activations.
  def _ff_activation(
      self, lr_lstm_output, rl_lstm_output, ff_activations, state, debug=False):
    assert len(ff_activations) == state.steps
    ff_input_parts = []
    ff_input_parts_debug = []

    # Fixed features.
    for f in self.spec.ff_fixed_features:
      raw_features = self.spec.raw_ff_fixed_features(f, state)

      embedded_features = None
      if len(raw_features) == 0:
        embedded_features = Var(torch.zeros(1, f.dim))
      else:
        embedded_features = f.bag(
          Var(torch.LongTensor(raw_features)), Var(torch.LongTensor([0])))
      ff_input_parts.append(embedded_features)
      if debug: ff_input_parts_debug.append((f, raw_features))

    # Link features.
    for f in self.spec.ff_link_features:
      link_debug = (f, [])

      # Figure out where we need to pick the activations from.
      activations = ff_activations
      if f.name in ["lr", "frame-end-lr", "mark-lr"]:
        activations = lr_lstm_output
      elif f.name in ["rl" , "frame-end-rl", "mark-rl"]:
        activations = rl_lstm_output

      # Get indices into the activations. Recall that missing indices are
      # indicated via None and they will map to an OOV vector.
      indices = self.spec.translated_ff_link_features(f, state)
      assert len(indices) == f.num, (f.num, len(indices))

      for index in indices:
        activation = None
        if index is not None:
          l = len(activations)
          assert index >= 0 and index < l, (f.name, index, l)
          activation = activations[index]
        vec = f.transform.forward(activation)
        ff_input_parts.append(vec)

      if debug:
        link_debug[1].extend(indices)
        ff_input_parts_debug.append(link_debug)

    ff_input = torch.cat(ff_input_parts, 1).view(-1, 1)
    ff_activation = self.ff_layer(ff_input)
    ff_activation = self.ff_relu(ff_activation)

    # Store the FF activation for future steps.
    ff_activations.append(ff_activation)

    return ff_activation, ff_input_parts_debug


  # Makes a forward pass over 'document'.
  def forward(self, document, train=False, debug=False):
    # Compute LSTM outputs for all tokens.
    lr_out, rl_out, lstm_features = self._lstm_outputs(document)

    # Run FF unit.
    state = ParserState(document, self.spec)
    actions = self.spec.actions
    cascade = self.spec.cascade
    ff_activations = []

    if train:
      losses = Losses()

      # Translate the gold actions into their cascade equivalents.
      cascade_gold = cascade.translate(document.gold)
      gold_index = 0
      while not state.done:
        # Compute the hidden layer once for all cascade delegates.
        ff_activation, _ = self._ff_activation(
            lr_out, rl_out, ff_activations, state)
        cascading = True
        delegate_index = 0   # assume we start the cascade at delegate 0
        while cascading:
          # Get the gold action for the delegate and compute loss w.r.t. it.
          gold = cascade_gold[gold_index]
          step_loss = cascade.loss(delegate_index, state, ff_activation, gold)
          losses.add(delegate_index, step_loss)

          # If the gold action was a CASCADE, move to the next delegate.
          if gold.is_cascade():
            delegate_index = gold.delegate
          else:
            state.advance(gold)
            cascading = False
          gold_index += 1

      return losses
    else:
      if document.size() == 0: return state

      shift = actions.action(actions.shift())
      stop = actions.action(actions.stop())
      disallowed_counts = [0] * cascade.size()
      total_counts = [0] * cascade.size()
      trace = Trace(self.spec, state, lstm_features) if debug else None
      while not state.done:
        # Compute the FF activation once for all cascade delegates.
        ff_activation, ff_features = self._ff_activation(
            lr_out, rl_out, ff_activations, state, debug=debug)
        if trace:
          trace.start_step(state, ff_features)

        # Store the last CASCADE action in a cascade.
        delegate_index = 0
        last = None
        while True:
          # Get the highest scoring action from the cascade delegate.
          # Note: We don't have to do any filtering or checking here, we
          # can just return the top-scoring action.
          best = cascade.predict(delegate_index, state, last, ff_activation)
          final = best
          if best.is_cascade():
            delegate_index = best.delegate
            last = best
          else:
            # If the action isn't allowed or can't be applied to the state,
            # then default to SHIFT or STOP.
            index = actions.index(best)
            total_counts[delegate_index] += 1
            if actions.disallowed[index] or not state.is_allowed(index):
              disallowed_counts[delegate_index] += 1
              final = shift
              if state.current == state.end:
                final = stop

          if trace:
            trace.action(best, final)
          if not final.is_cascade():
            # Apply the action and stop the cascade.
            state.advance(final)
            break

      return state, disallowed_counts, total_counts, trace


  # Reads the model from the specified flow.
  # Assumes that the spec has already been read from the flow.
  def from_flow(self, fl):
    # Read LSTM embeddings.
    for index, f in enumerate(self.spec.lstm_features):
      param = self.lstm_embeddings[index].weight
      var = fl.var("features/" + f.name + "_embeddings")
      parameter_from_flow(var, param)
      assert [f.vocab_size, f.dim] == list(param.shape)

    # Read LSTM weights.
    self.lr_lstm.copy_from_flow(fl, "lstm/lr")
    self.rl_lstm.copy_from_flow(fl, "lstm/rl")

    # Read FF fixed feature embeddings.
    for f in self.spec.ff_fixed_features:
      param = f.bag.weight
      var = fl.var("ff_trunk/" + f.name + "_embedding")
      parameter_from_flow(var, param)

    # Read FF link transforms.
    for f in self.spec.ff_link_features:
      param = f.transform.projection.weight
      var = fl.var("ff_trunk/" + f.name + "/transform")
      parameter_from_flow(var, param)

      param = f.transform.oov
      var = fl.var("ff_trunk/" + f.name + "/Gather/oov")
      parameter_from_flow(var, param)

    # Read FF hidden layer.
    parameter_from_flow(fl.var("ff_trunk/W0"), self.ff_layer.weight)
    parameter_from_flow(fl.var("ff_trunk/b0"), self.ff_layer.bias)

    # Read softmax head(s).
    prefix = "delegate"
    for i, head in enumerate(self.ff_heads):
      parameter_from_flow(fl.var(prefix + str(i) + "/W"), head.softmax.weight)
      parameter_from_flow(fl.var(prefix + str(i) + "/b"), head.softmax.bias)


  # Writes model as a Myelin flow to 'fl'.
  def to_flow(self, fl):
    spec = self.spec

    # Specify the encoder.
    lstm_embeddings = [e.weight.data.numpy() for e in self.lstm_embeddings]
    lex = LexicalEncoder(fl, spec, lstm_embeddings, self.lr_lstm, self.rl_lstm)

    # Write the specification to the flow.
    spec.to_flow(fl)

    # Adds a flow variable that will store raw indices for the given feature.
    def index_vars(bldr, feature_spec):
      return bldr.var(name=feature.name, dtype="int32", shape=[1, feature.num])

    # Adds flow variables and ops for the given fixed feature.
    # The output variable of the feature is added as an input to 'concat_op'.
    def write_fixed_feature(feature, bag, bldr, concat_op):
      indices = index_vars(bldr, feature)
      s = bag.weight.size()
      embedding = bldr.var(name=feature.name + "_embedding", shape=[s[0], s[1]])
      embedding.data = bag.weight.data.numpy()

      name = feature.name + '/GatherSum'
      embedded = bldr.gather_sum(embedding, indices, name=name)
      concat_op.add_input(embedded)

    # Finishes the concatenation op assuming all inputs have been added.
    def finish_concat_op(bldr, op):
      op.add_attr("N", len(op.inputs))
      axis = bldr.const(1, "int32")
      op.add_input(axis)

    # Specify the FF trunk = FF feature vector + hidden layer computation.
    ff = builder.Builder(fl, "ff_trunk")
    ff_input = ff.var(name="input", shape=[1, spec.ff_input_dim])
    flow_ff = flownn.FF(
        ff, \
        input=ff_input, \
        layers=[spec.ff_hidden_dim],
        hidden=0)
    flow_ff.set_layer_data(0, self.ff_layer.weight.data.numpy(), \
                           self.ff_layer.bias.data.numpy())

    ff_concat_op = ff.rawop(optype="Concat", name="concat")
    ff_concat_op.add_output(ff_input)

    # Add link variable to the given connector.
    def link(bldr, name, dim, cnx, prefix=True):
      if prefix: name = "link/" + name
      l = bldr.var(name, shape=[-1, dim])
      l.ref = True
      cnx.add(l)
      return l

    # Add links to the two LSTMs.
    ff_lr = link(ff, "lr_lstm", spec.lstm_hidden_dim, lex.lr_lstm.cnx_hidden)
    ff_rl = link(ff, "rl_lstm", spec.lstm_hidden_dim, lex.rl_lstm.cnx_hidden)

    # Add link and connector for previous FF steps.
    ff_cnx = ff.cnx("step", args=[])
    ff_steps = link(ff, "steps", spec.ff_hidden_dim, ff_cnx, False)
    ff_cnx.add(flow_ff.hidden_out)

    # Add FF's input variables.
    for feature in spec.ff_fixed_features:
      write_fixed_feature(feature, feature.bag, ff, ff_concat_op)

    for feature in spec.ff_link_features:
      indices = index_vars(ff, feature)

      activations = None
      n = feature.name
      if n == "frame-end-lr" or n == "lr" or n == "mark-lr":
        activations = ff_lr
      elif n == "frame-end-rl" or n == "rl" or n == "mark-rl":
        activations = ff_rl
      elif n in ["frame-creation-steps", "frame-focus-steps", \
        "history", "mark-step"]:
        activations = ff_steps
      else:
        raise ValueError("Unknown feature %r" % n)

      activation_dim = activations.shape[1]
      name = feature.name + "/Gather"
      oov_var = ff.var(name=name + "/oov", shape=[1, activation_dim])
      oov_var.data = feature.transform.oov.data.numpy()
      gathered = ff.gather(activations, indices, oov_var, name=name)

      projection = feature.transform.projection.weight
      sz = projection.size()
      transform = ff.var(name=feature.name + "/transform", shape=[sz[0], sz[1]])
      transform.data = projection.data.numpy()

      name = feature.name + "/MatMul"
      output = ff.matmul(gathered, transform, name=name)
      ff_concat_op.add_input(output)

    finish_concat_op(ff, ff_concat_op)

    # Specify one cell per FF head (= delegate).
    ff_trunk_width = flow_ff.hidden_out.shape[1]
    delegate_cell_prefix = "delegate"
    for i, head in enumerate(self.ff_heads):
      delegate = spec.cascade.delegates[i]
      assert_softmax_delegate(delegate)
      d = builder.Builder(fl, delegate_cell_prefix + str(i))
      head_input = link(d, "input", ff_trunk_width, ff_cnx, False)

      W = d.var("W", shape=[ff_trunk_width, delegate.size()])
      W.data = head.softmax.weight.data.numpy()
      output = d.matmul(head_input, W)
      output.type = W.type
      output.shape = [1, delegate.size()]

      b = d.var("b", shape=[1, delegate.size()])
      b.data = head.softmax.bias.data.numpy()
      logits = d.add(output, b)
      logits.type = b.type
      logits.shape = [1, delegate.size()]

      best_op = d.rawop(optype="ArgMax")
      best_op.add_input(logits)
      best = d.var("output")
      best.type = builder.DT_INT
      best.shape = [1]
      best_op.add_output(best)
      best.producer.add_attr("output", 1)
