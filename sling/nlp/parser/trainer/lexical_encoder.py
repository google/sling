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

import os
import tempfile

import sling.myelin.builder as builder
import sling.myelin.nn as nn

# Adds a lexical encoder to 'flow', as per 'spec' (which is a Spec object).
# 'lstm_feature_embeddings' is a list of embeddings data (e.g. numpy arrays),
# and 'lr_lstm'/'rl_lstm' are LSTM implementations to be copied to the flow.
class LexicalEncoder:
  def __init__(self, flow,  spec, lstm_feature_embeddings, lr_lstm, rl_lstm):
    # Add blobs for the lexical resources.
    lexicon = flow.blob("lexicon")
    lexicon.type = "dict"
    lexicon.add_attr("delimiter", 10)
    lexicon.add_attr("oov", spec.words.oov_index)
    normalization = ""
    if spec.words.normalize_digits: normalization = "d"
    lexicon.add_attr("normalization", normalization)
    lexicon.data = str(spec.words) + "\n"
    self.lexicon_blob = lexicon

    def read_file(filename):
      fin = open(filename, "rb")
      data = fin.read()
      fin.close()
      return data

    f = tempfile.NamedTemporaryFile(delete=False)
    fname = f.name
    spec.commons.save(fname, binary=True)
    f.close()

    commons = flow.blob("commons")
    commons.type = "frames"
    commons.data = read_file(fname)
    os.unlink(fname)
    self.commons_blob = commons

    suffix = flow.blob("suffixes")
    suffix.type = "affix"
    suffix.data = spec.write_suffix_table()
    self.suffix_blob = suffix

    # Add feature extraction related ops.
    bldr = builder.Builder(flow, "features")
    self.feature_ids = []
    concat_args = []
    for f, e in zip(spec.lstm_features, lstm_feature_embeddings):
      shape=[f.vocab_size, f.dim]
      embedding = bldr.var(name=f.name + "_embeddings", shape=shape)
      embedding.data = e
      ids_input = bldr.var(name=f.name, dtype="int32", shape=[1, f.num])
      self.feature_ids.append(ids_input)

      gather_op_type = "Gather"
      if f.num > 1: gather_op_type = "GatherSum"
      gather_op = bldr.rawop(gather_op_type)
      gather_op.dtype = "float32"
      gather_op.add_input(embedding)
      gather_op.add_input(ids_input)
      gather_output = bldr.var(gather_op.name + ":0", "float32", [1, f.dim])
      gather_op.add_output(gather_output)
      concat_args.append(gather_output)

    self.feature_vector = bldr.concat(concat_args)
    bldr.rename(self.feature_vector, "feature_vector")
    self.feature_vector.ref = True
    self.feature_vector.input = True
    self.feature_vector.output = True

    # Add BiLSTM.
    lr = builder.Builder(flow, "lstm/lr")
    lr_input = lr.var(name="input", shape=[1, spec.lstm_input_dim])
    lr_input.ref = True
    flow_lr_lstm = nn.LSTM(lr, input=lr_input, size=spec.lstm_hidden_dim)
    lr_lstm.copy_to_flow_lstm(flow_lr_lstm)
    self.lr_lstm = flow_lr_lstm

    rl = builder.Builder(flow, "lstm/rl")
    rl_input = rl.var(name="input", shape=[1, spec.lstm_input_dim])
    rl_input.ref = True
    flow_rl_lstm = nn.LSTM(rl, input=rl_input, size=spec.lstm_hidden_dim)
    rl_lstm.copy_to_flow_lstm(flow_rl_lstm)
    self.rl_lstm = flow_rl_lstm

    cnxin = flow.cnx("features")
    cnxin.add(self.feature_vector)
    cnxin.add(lr_input)
    cnxin.add(rl_input)

