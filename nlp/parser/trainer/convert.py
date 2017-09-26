# Copyright 2017 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License")

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

"""Converts sempar model to myelin format
"""

import sys
import numpy as np
import tensorflow as tf

sys.path.insert(0, "third_party/syntaxnet")
sys.path.insert(0, "python")

from flow import Flow
from flow import FlowBuilder
from dragnn.protos import spec_pb2

tf.load_op_library("bazel-bin/nlp/parser/trainer/sempar.so")

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string("input", "", "Tensorflow input model directory.")
flags.DEFINE_string("output", "", "Myelin output model.")

# DRAGNN op names.
LSTM_H_IN = "lstm_h_in"
LSTM_H_OUT = "lstm_h"
LSTM_C_IN = "lstm_c_in"
LSTM_C_OUT = "lstm_c"
LSTM_FV = "feature_vector"

FF_HIDDEN = "Relu"
FF_OUTPUT = "logits"
FF_FV = "feature_vector"

FIXED_EMBEDDING = "/embedding_lookup/Enter"
LINKED_EMBEDDING = "/MatMul/Enter"

GET_SESSION = "annotation/ComputeSession/GetSession"

def read_file(filename):
  fin = open(filename, "r")
  data = fin.read()
  fin.close()
  return data

class Component:
  def __init__(self, spec, builder, connectors):
    self.spec = spec
    self.name = spec.name
    self.builder = builder
    self.flow = builder.flow
    self.sess = builder.sess
    self.func = self.flow.func(self.name)
    self.features = None
    self.connectors = connectors
    self.links = {}

  def path(self):
    return "annotation/inference_" + self.name + "/" + self.name

  def tfvar(self, name):
    return self.sess.graph.get_tensor_by_name(self.path() + "/" + name + ":0")

  def flowop(self, name):
    return self.flow.op(self.path() + "/" + name)

  def flowvar(self, name):
    return self.flow.var(self.path() + "/" + name + ":0")

  def newvar(self, name, type="float", shape=[0], data=None):
    var = self.flowvar(name)
    var.type = type
    var.shape = shape
    var.data = data
    return var

  def newop(self, name, optype, type="float", shape=[0]):
    var = self.newvar(name, type, shape)
    op = self.flowop(name)
    self.func.add(op)
    op.type = optype
    op.add_output(var)
    return op, var

  def extract(self):
    # Extract cell ops for component.
    component_type = self.spec.network_unit.registered_name
    if component_type == 'LSTMNetwork':
      self.extract_lstm()
    elif component_type == 'FeedForwardNetwork':
      self.extract_feed_forward()
    else:
      print "Warning: Unknown component type:", component_type

    # Extract ops for fixed features.
    for feature in self.spec.fixed_feature:
      self.extract_fixed_feature(feature)

    # Extract ops for linked features.
    for feature in self.spec.linked_feature:
      self.extract_linked_feature(feature)

    # Set the number of feature inputs.
    if self.features != None:
      one = np.array(1, dtype=np.int32)
      axis = self.newvar("axis", "int32", [], one)
      self.features.add_attr("N", len(self.features.inputs))
      self.features.add_input(axis)

  def extract_lstm(self):
    # The LSTM cell has inputs and outputs for the hidden and control channels
    # and the input features are collected into a concatenated dense feature
    # vector. First, extract LSTM cell ops excluding feature inputs.
    tf_h_in = self.tfvar(LSTM_H_IN)
    tf_c_in = self.tfvar(LSTM_C_IN)
    tf_h_out = self.tfvar(LSTM_H_OUT)
    tf_c_out = self.tfvar(LSTM_C_OUT)
    tf_fv = self.tfvar(LSTM_FV)
    self.builder.add(self.func,
                     [tf_h_in, tf_c_in, tf_fv],
                     [tf_h_out, tf_c_out])
    self.add_feature_concatenation(self.flowvar(LSTM_FV))

    # The LSTM cells are connected through the hidden and control channels. The
    # hidden output from the previous step is connected to the hidden input
    # for the current step. Likewise, the control output from the previous step
    # is linked to the control input for the current step.
    dims = int(self.spec.network_unit.parameters["hidden_layer_sizes"])

    h_in = self.flowvar(LSTM_H_IN)
    h_out = self.flowvar(LSTM_H_OUT)
    c_in = self.flowvar(LSTM_C_IN)
    c_out = self.flowvar(LSTM_C_OUT)

    h_cnx = self.flow.cnx(self.path() + "/hidden")
    h_cnx.add(h_in)
    h_cnx.add(h_out)

    c_cnx = self.flow.cnx(self.path() + "/control")
    c_cnx.add(c_in)
    c_cnx.add(c_out)

    self.connectors[self.name] = h_cnx
    for v in [h_in, h_out, c_in, c_out]:
      v.type = "&" + v.type
      v.shape = [1, dims]

  def extract_feed_forward(self):
    # The FF cell produces output logits as well as step activations from the
    # hidden layer. The input features are collected into a concatenated dense
    # feature vector. First, extract FF cell ops excluding feature inputs.
    tf_hidden = self.tfvar(FF_HIDDEN)
    tf_output = self.tfvar(FF_OUTPUT)
    tf_fv = self.tfvar(FF_FV)
    self.builder.add(self.func, [tf_fv], [tf_hidden, tf_output])
    self.add_feature_concatenation(self.flowvar(FF_FV))

    # The activations from the hidden layer is output to a connector channel in
    # each step and fed back into the cell through the feature functions. A
    # reference variable that points to the channel with all the previous step
    # activations is added to the cell so these can be used by the recurrent
    # feature functions to look up activations from previous steps.
    dims = int(self.spec.network_unit.parameters["hidden_layer_sizes"])

    activation = self.flowvar(FF_HIDDEN)
    activation.type = "&" + activation.type
    activation.shape = [1, dims]

    self.steps = self.flow.var(self.path() + "/steps")
    self.steps.type = "&float"
    self.steps.shape = [-1, dims]

    step_cnx = self.flow.cnx(self.path() + "/step")
    step_cnx.add(self.flowvar(FF_HIDDEN))
    step_cnx.add(self.steps)

    self.connectors[self.name] = step_cnx
    self.links[self.name] = self.steps

  def add_feature_concatenation(self, output):
    """Add concat op that the features can be fed into."""
    self.features = self.flowop("concat")
    self.features.type = "ConcatV2"
    self.features.add_output(output)
    self.func.add(self.features)

  def extract_fixed_feature(self, feature):
    # Create feature input variable.
    input = self.flow.var(self.path() + "/" + feature.name)
    input.type = "int32"
    input.shape = [1, feature.size]

    # Extract embedding matrix.
    prefix = "fixed_embedding_" + feature.name
    embedding_name = prefix + FIXED_EMBEDDING
    tf_embedding = self.tfvar(embedding_name)
    self.builder.add(self.func, [], [tf_embedding])
    embedding = self.flowvar(embedding_name)

    # Look up feature(s) in embedding.
    lookup, embedded = self.newop(feature.name + "/Lookup", "Lookup")
    lookup.add_input(input)
    lookup.add_input(embedding)

    # Add features to feature vector.
    self.features.add_input(embedded)

  def extract_linked_feature(self, feature):
    # Create feature input variable.
    input = self.flow.var(self.path() + "/" + feature.name)
    input.type = "int32"
    input.shape = [1, feature.size]

    # A recurrent feature takes activations from its own cell as input.
    source = feature.source_component
    recurrent = source == self.spec.name
    if recurrent:
      prefix = "activation_lookup_recurrent_" + feature.name
    else:
      prefix = "activation_lookup_other_" + feature.name

    # Extract embedding matrix.
    embedding_name = prefix + LINKED_EMBEDDING
    tf_embedding = self.tfvar(embedding_name)
    self.builder.add(self.func, [], [tf_embedding])
    embedding = self.flowvar(embedding_name)

    # Get or create link variable for activation lookup.
    link = self.links.get(source, None)
    if link is None:
      cnx = self.connectors[source]
      link = self.flow.var(self.path() + "/link/" + source)
      link.type = "&float32"
      link.shape = [-1] + cnx.links[0].shape[1:]
      self.links[source] = link
      cnx.add(link)

    # Collect activation vectors for features.
    collect, activations = self.newop(feature.name + "/Collect", "Collect")
    collect.add_input(input)
    collect.add_input(link)

    # Multiply activation vectors with embedding matrix.
    matmul, embedded = self.newop(feature.name + "/MatMul", "MatMul")
    matmul.add_input(activations)
    matmul.add_input(embedding)

    # Reshape embedded feature output.
    shape = np.array([1, feature.size * feature.embedding_dim], dtype=np.int32)
    reshape, reshaped = self.newop(feature.name + "/Reshape", "Reshape")
    reshape.add_input(embedded)
    reshape.add_input(self.newvar(feature.name + "/shape", "int32", [2], shape))

    # Add features to feature vector.
    self.features.add_input(reshaped)

def convert_model(master_spec, sess):
  # Create flow.
  flow = Flow()
  builder = FlowBuilder(sess, flow)

  # Get components.
  components = []
  connectors = {}
  for c in master_spec.component:
    component = Component(c, builder, connectors)
    components.append(component)

  # Extract components.
  for c in components: c.extract()

  # Sanitize names.
  for c in components: flow.rename_prefix(c.path() + "/", c.name + "/")
  flow.rename_suffix("/ExponentialMovingAverage:0", "")
  flow.rename_suffix(LSTM_H_IN + ":0", "h_in")
  flow.rename_suffix(LSTM_H_OUT + ":0", "h_out")
  flow.rename_suffix(LSTM_C_IN + ":0", "c_in")
  flow.rename_suffix(LSTM_C_OUT + ":0", "c_out")
  flow.rename_suffix(FF_HIDDEN + ":0", "hidden")
  flow.rename_suffix(FF_OUTPUT + ":0", "output")

  # Get external resources.
  lexicon_file = None
  prefix_file = None
  suffix_file = None
  commons_file = None
  actions_file = None
  for c in master_spec.component:
    for r in c.resource:
      if r.name == "word-vocab":
        lexicon_file = r.part[0].file_pattern
      elif r.name == "prefix-table":
        prefix_file = r.part[0].file_pattern
      elif r.name == "suffix-table":
        suffix_file = r.part[0].file_pattern
      elif r.name == "commons":
        commons_file = r.part[0].file_pattern
      elif r.name == "action-table":
        actions_file = r.part[0].file_pattern

  # Add lexicon to flow.
  if lexicon_file != None:
    lexicon = flow.blob("lexicon")
    lexicon.type = "dict"
    lexicon.add_attr("delimiter", 10)
    lexicon.add_attr("oov", 0)
    lexicon.add_attr("normalize_digits", 1)
    lexicon.data = read_file(lexicon_file)

  # Add prefix table to flow.
  if prefix_file != None:
    prefixes = flow.blob("prefixes")
    prefixes.type = "affix"
    prefixes.data = read_file(prefix_file)

  # Add suffix table to flow.
  if suffix_file != None:
    suffixes = flow.blob("suffixes")
    suffixes.type = "affix"
    suffixes.data = read_file(suffix_file)

  # Add commons to flow.
  if commons_file != None:
    commons = flow.blob("commons")
    commons.type = "frames"
    commons.data = read_file(commons_file)

  # Add action table to flow.
  if actions_file != None:
    actions = flow.blob("actions")
    actions.type = "frames"
    actions.data = read_file(actions_file)

  return flow

def main(argv):
  # Load Tensorflow checkpoint for sempar model.
  sess = tf.Session()
  saver = tf.train.import_meta_graph(FLAGS.input + "/checkpoints/best.meta")
  saver.restore(sess, FLAGS.input + "/checkpoints/best")

  # Read master spec.
  master = sess.graph.get_operation_by_name(GET_SESSION)
  master_spec = spec_pb2.MasterSpec()
  master_spec.ParseFromString(master.get_attr("master_spec"))

  # Convert model to flow.
  flow = convert_model(master_spec, sess)

  # Write flow.
  flow.save(FLAGS.output)

if __name__ == '__main__':
  FLAGS.alsologtostderr = True
  tf.app.run()

