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


# PyTorch module implementations for Sempar.

import math
import sys
import torch
import torch.nn as nn

sys.path.insert(0, "sling/nlp/parser/trainer")
from parser_state import ParserState

import sling.myelin.nn as flownn
import sling.myelin.flow as flow
import sling.myelin.builder as builder

Param = nn.Parameter
Var = torch.autograd.Variable

# Utility for dumping a (list of) PyTorch tensor/variable.
def fstr(var):
  if type(var) is tuple and len(var) == 1: var = var[0]

  dim = var.dim()
  if dim == 1 or (dim == 2 and (var.size(0) == 1 or var.size(1) == 1)):
    var = var.view(1, -1)
  ls = var.data.numpy().tolist()
  if type(ls[0]) is list: ls = ls[0]
  ls = ["%.9f" % x for x in ls]
  return "[" + ",".join(ls) + "]"


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


# Transforms input x to x * A. If x is None, returns a special vector.
class LinkTransform(Projection):
  def __init__(self, activation_size, dim):
    super(LinkTransform, self).__init__(activation_size + 1, dim, bias=False)


  # Forward pass.
  def forward(self, activation=None):
    if activation is None:
      return self.weight[-1].view(1, -1)  # last row
    else:
      return torch.mm(activation.view(1, -1), self.weight[0:-1])


  # Returns a string specification of the module.
  def __repr__(self):
    s = self.weight.size()
    return self.__class__.__name__ + "(input_activation=" + str(s[0] - 1) + \
        ", dim=" + str(s[1]) + ", oov_vector=" + str(s[1])+ ")"


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
    for i in xrange(length):
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


  # Returns a string specification of the module.
  def __repr__(self):
    return self.__class__.__name__ + "(in=" + str(self.input_dim) + \
        ", hidden=" + str(self.hidden_dim) + ")"


# Top-level module for SEMPAR.
class Sempar(nn.Module):
  def __init__(self, spec):
    super(Sempar, self).__init__()
    self.spec = spec

    # LSTM Embeddings.
    self.lr_embeddings = []
    self.rl_embeddings = []
    for f in spec.lstm_features:
      lr_embedding = nn.EmbeddingBag(f.vocab_size, f.dim, mode='sum')
      self.add_module('lr_lstm_embedding_' + f.name, lr_embedding)
      self.lr_embeddings.append(lr_embedding)
      f.lr_embedding = lr_embedding

      rl_embedding = nn.EmbeddingBag(f.vocab_size, f.dim, mode='sum')
      self.add_module('rl_lstm_embedding_' + f.name, rl_embedding)
      self.rl_embeddings.append(rl_embedding)
      f.rl_embedding = rl_embedding

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

    # Feedforward unit.
    h = spec.ff_hidden_dim
    self.ff_layer = Projection(spec.ff_input_dim, h)   # hidden layer
    self.ff_relu = nn.ReLU()                           # non-linearity
    self.ff_softmax = Projection(h, spec.num_actions)  # output layer
    self.loss_fn = nn.CrossEntropyLoss()               # loss function

    # Only regularize the FF hidden layer weights.
    self.regularized_params = [self.ff_layer.weight]
    print "Sempar:", self


  # Initializes various module parameters.
  def initialize(self):
    # Initialize the embeddings to gaussian(mean=0, stddev=1/sqrt(dim)),
    # where 'dim' is the dimensionality of the embedding.
    for f in self.spec.lstm_features:
      coeff = 1.0 / math.sqrt(f.dim)
      lr = f.lr_embedding.weight.data
      rl = f.rl_embedding.weight.data
      lr.normal_()
      lr.mul_(coeff)
      rl.normal_()
      rl.mul_(coeff)

      # Override with pre-trained word embeddings, if provided.
      if f.name == "words" and self.spec.word_embeddings is not None:
        indices = torch.LongTensor(self.spec.word_embedding_indices)
        data = torch.Tensor(self.spec.word_embeddings)

        # Separately normalize each embedding row
        data = torch.nn.functional.normalize(data)

        # Copy the normalized embeddings at appropriate indices.
        lr.index_copy_(0, indices, data)
        rl.index_copy_(0, indices, data)
        print "Overwrote", len(self.spec.word_embeddings), f.name, \
            "embedding vectors with normalized pre-trained vectors."

    # Initialize the FF's fixed and link embeddings like those in the LSTMs.
    for f in self.spec.ff_fixed_features:
      f.bag.weight.data.normal_()
      f.bag.weight.data.mul_(1.0 / math.sqrt(f.dim))

    for f in self.spec.ff_link_features:
      f.transform.init(1.0 / math.sqrt(f.dim))

    # Initialize the LSTM and FF parameters with gaussan(mean=0, stddev=1e-4).
    params = [self.ff_layer.weight, self.ff_softmax.weight]
    params += [p for p in self.lr_lstm.parameters()]
    params += [p for p in self.rl_lstm.parameters()]
    for p in params:
      p.data.normal_()
      p.data.mul_(1e-4)

    # Positive bias for the hidden layer, and zero bias for the output layer.
    self.ff_layer.bias.data.fill_(0.2)
    self.ff_softmax.bias.data.fill_(0.0)


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

    # Each of {lr,rl}_inputs should have shape (length, lstm_input_dim).
    lr_inputs = self._embedding_lookup(self.lr_embeddings, raw_features)
    rl_inputs = self._embedding_lookup(self.rl_embeddings, raw_features)
    assert length == lr_inputs.size(0)
    assert length == rl_inputs.size(0)

    lr_out, _ = self.lr_lstm.forward(lr_inputs)

    # Note: Negative strides are not supported, otherwise we would just do:
    #   rl_input = rl_inputs[::-1]
    inverse_indices = torch.arange(length - 1, -1, -1).long()
    rl_inputs = rl_inputs[inverse_indices]
    rl_out, _ = self.rl_lstm.forward(rl_inputs)
    rl_out.reverse()

    # lr_out[i], rl_out[i] are the LSTM outputs for the ith token.
    return (lr_out, rl_out, raw_features)


  # Returns the FF output, given the LSTM outputs and previous FF activations.
  def _ff_output(
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
      if f.name == "lr" or f.name == "frame-end-lr":
        activations = lr_lstm_output
      elif f.name == "rl" or f.name == "frame-end-rl":
        activations = rl_lstm_output

      # Get indices into the activations. Recall that missing indices are
      # indicated via None, and they map to the last row in 'transform'.
      indices = self.spec.translated_ff_link_features(f, state)
      assert len(indices) == f.num

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
    ff_hidden = self.ff_layer(ff_input)
    ff_hidden = self.ff_relu(ff_hidden)
    softmax_output = self.ff_softmax(ff_hidden)

    # Store the FF activation for future steps.
    ff_activations.append(ff_hidden)

    return softmax_output.view(self.spec.num_actions), ff_input_parts_debug


  # Makes a forward pass over 'document'.
  def forward(self, document, train=False, debug=False):
    # Compute LSTM outputs for all tokens.
    lr_out, rl_out, _ = self._lstm_outputs(document)

    # Run FF unit.
    state = ParserState(document, self.spec)
    actions = self.spec.actions
    ff_activations = []

    if train:
      loss = Var(torch.FloatTensor([1]).zero_())
      for index, gold in enumerate(document.gold):
        gold_index = actions.indices.get(gold, None)
        assert gold_index is not None, "Unknown gold action %r" % gold

        ff_output, _ = self._ff_output(lr_out, rl_out, ff_activations, state)
        gold_var = Var(torch.LongTensor([gold_index]))
        step_loss = self.loss_fn(ff_output.view(1, -1), gold_var)
        loss += step_loss

        assert state.is_allowed(gold_index), "Disallowed gold action: %r" % gold
        state.advance(gold)
      return loss, len(document.gold)
    else:
      if document.size() == 0: return state

      # Number of top-k actions to consider. If all top-k actions are
      # infeasible, then we default to SHIFT or STOP.
      topk = self.spec.num_actions

      shift = actions.shift()
      stop = actions.stop()
      predicted = shift
      while predicted != stop:
        ff_output, _ = self._ff_output(lr_out, rl_out, ff_activations, state)

        # Find the highest scoring allowed action among the top-k.
        # If all top-k actions are disallowed, then use a fallback action.
        _, topk_indices = torch.topk(ff_output, topk)
        found = False
        rank = "(fallback)"
        for candidate in topk_indices.view(-1).data:
          if not actions.disallowed[candidate] and state.is_allowed(candidate):
            rank = str(candidate)
            found = True
            predicted = candidate
            break
        if not found:
          # Fallback.
          predicted = shift if state.current < state.end else stop

        action = actions.table[predicted]
        state.advance(action)
        if debug:
          print "Predicted", action, "at rank ", rank

      return state


  # Traces the model as it runs through 'document'.
  def model_trace(self, document):
    length = document.size()
    lr_out, rl_out, lstm_features = self._lstm_outputs(document)

    assert len(self.spec.lstm_features) == len(lstm_features)
    for f in lstm_features:
      assert len(f.offsets) == length

    print length, "tokens in document"
    for index, t in enumerate(document.tokens()):
      print "Token", index, "=", t.text
    print

    state = ParserState(document, self.spec)
    actions = self.spec.actions
    ff_activations = []
    steps = 0
    for gold in document.gold:
      print "State:", state
      gold_index = actions.indices.get(gold, None)
      assert gold_index is not None, "Unknown gold action %r" % gold

      if state.current < state.end:
        print "Token", state.current, "=", document.tokens()[state.current].text
        for feature_spec, values in zip(self.spec.lstm_features, lstm_features):
          # Recall that 'values' has indices at all sequence positions.
          # We need to get the slice of feature indices at the current token.
          start = values.offsets[state.current - state.begin]
          end = None
          if state.current < state.end - 1:
            end = values.offsets[state.current - state.begin + 1]

          current = values.indices[start:end]
          print "  LSTM feature:", feature_spec.name, ", indices=", current,\
              "=", self.spec.lstm_feature_strings(current)

      ff_output, ff_debug = self._ff_output(
          lr_out, rl_out, ff_activations, state, debug=True)
      for f, indices in ff_debug:
        debug = self.spec.ff_fixed_features_debug(f, indices)
        print "  FF Feature", f.name, "=", str(indices), debug
      assert ff_output.view(1, -1).size(1) == self.spec.num_actions

      assert state.is_allowed(gold_index), "Disallowed gold action: %r" % gold
      state.advance(gold)
      print "Step", steps, ": advancing using gold action", gold
      print
      steps += 1


  # Writes model as a Myelin flow to 'flow_file'.
  def write_flow(self, flow_file):
    fl = flow.Flow()
    spec = self.spec
    spec.write_flow(fl)

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

      lookup = bldr.rawop(optype="Lookup", name=feature.name + "/Lookup")
      lookup.add_input(indices)
      lookup.add_input(embedding)
      embedded = bldr.var(name=feature.name + "_embedded", shape=[1, s[1]])
      lookup.add_output(embedded)
      concat_op.add_input(embedded)

    # Finishes the concatenation op assuming all inputs have been added.
    def finish_concat_op(bldr, op):
      op.add_attr("N", len(op.inputs))
      axis = bldr.const(1, "int32")
      op.add_input(axis)

    # Specify LSTMs.
    lr = builder.Builder(fl, "lr_lstm")
    lr_input = lr.var(name="input", shape=[1, spec.lstm_input_dim])
    flow_lr_lstm = flownn.LSTM(lr, input=lr_input, size=spec.lstm_hidden_dim)
    self.lr_lstm.copy_to_flow_lstm(flow_lr_lstm)
    lr_concat_op = lr.rawop(optype="ConcatV2", name="concat")
    lr_concat_op.add_output(lr_input)

    rl = builder.Builder(fl, "rl_lstm")
    rl_input = rl.var(name="input", shape=[1, spec.lstm_input_dim])
    flow_rl_lstm = flownn.LSTM(rl, input=rl_input, size=spec.lstm_hidden_dim)
    self.rl_lstm.copy_to_flow_lstm(flow_rl_lstm)
    rl_concat_op = rl.rawop(optype="ConcatV2", name="concat")
    rl_concat_op.add_output(rl_input)

    # Add LSTM inputs.
    for feature in spec.lstm_features:
      write_fixed_feature(feature, feature.lr_embedding, lr, lr_concat_op)
      write_fixed_feature(feature, feature.rl_embedding, rl, rl_concat_op)

    finish_concat_op(lr, lr_concat_op)
    finish_concat_op(rl, rl_concat_op)

    # Specify the FF unit.
    ff = builder.Builder(fl, "ff")
    ff_input = ff.var(name="input", shape=[1, spec.ff_input_dim])
    flow_ff = flownn.FF(
        ff, \
        input=ff_input, \
        layers=[spec.ff_hidden_dim, spec.num_actions], \
        hidden=0)
    flow_ff.set_layer_data(0, self.ff_layer.weight.data.numpy(), \
                           self.ff_layer.bias.data.numpy())
    flow_ff.set_layer_data(1, self.ff_softmax.weight.data.numpy(), \
                           self.ff_softmax.bias.data.numpy())

    ff_concat_op = ff.rawop(optype="ConcatV2", name="concat")
    ff_concat_op.add_output(ff_input)

    # Add link variable to the given connector.
    def link(bldr, name, dim, cnx, prefix=True):
      if prefix: name = "link/" + name
      l = bldr.var(name, shape=[-1, dim])
      l.ref = True
      cnx.add(l)
      return l

    # Add links to the two LSTMs.
    ff_lr = link(ff, "lr_lstm", spec.lstm_hidden_dim, flow_lr_lstm.cnx_hidden)
    ff_rl = link(ff, "rl_lstm", spec.lstm_hidden_dim, flow_rl_lstm.cnx_hidden)

    # Add link and connector for previous FF steps.
    ff_cnx = ff.cnx("step", args=[])
    ff_steps = link(ff, "steps", spec.ff_hidden_dim, ff_cnx, False)

    # Add FF's input variables.
    for feature in spec.ff_fixed_features:
      write_fixed_feature(feature, feature.bag, ff, ff_concat_op)

    for feature in spec.ff_link_features:
      indices = index_vars(ff, feature)

      activations = None
      n = feature.name
      if n == "frame-end-lr" or n == "lr":
        activations = ff_lr
      elif n == "frame-end-rl" or n == "rl":
        activations = ff_rl
      elif n in ["frame-creation-steps", "frame-focus-steps", "history"]:
        activations = ff_steps
      else:
        raise ValueError("Unknown feature %r" % n)

      name = feature.name + "/Collect"
      collect = ff.rawop(optype="Collect", name=name)
      collect.add_input(indices)
      collect.add_input(activations)
      collected = ff.var(
          name=name + ":0", shape=[feature.num, activations.shape[1] + 1])
      collect.add_output(collected)

      sz = feature.transform.weight.size()
      transform = ff.var(name=feature.name + "/transform", shape=[sz[0], sz[1]])
      transform.data = feature.transform.weight.data.numpy()

      name = feature.name + "/MatMul"
      matmul = ff.rawop(optype="MatMul", name=name)
      matmul.add_input(collected)
      matmul.add_input(transform)
      output = ff.var(name + ":0", shape=[feature.num, sz[1]])
      matmul.add_output(output)
      ff_concat_op.add_input(output)

    finish_concat_op(ff, ff_concat_op)
    fl.save(flow_file)

