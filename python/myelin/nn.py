# Copyright 2017 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Neural network cells."""

class FF:
  def __init__(
      self, tf, input, layers, hidden=None, bias=True, activation="Relu"):
    self.weights = []
    self.biases = []
    self.hidden_out = None
    self.output = None

    v = input
    for l in range(len(layers)):
      type = v.type
      height = v.shape[1]
      width = layers[l]

      W = tf.var("W" + str(l), type, [height, width])
      self.weights.append(W)
      v = tf.matmul(v, W)
      v.type = type
      v.shape = [1, width]

      if bias:
        b = tf.var("b" + str(l), type, [1, width])
        self.biases.append(b)
        v = tf.add(v, b)
        v.type = type
        v.shape = [1, width]

      if l == hidden or l != len(layers) - 1:
        v = tf.op(activation, [v])
        v.type = type
        v.shape = [1, width]

      if l == hidden:
        v = tf.identity(v, name="hidden")
        v.type = type
        v.shape = [1, width]
        v.ref = True
        v.input = True
        v.output = True
        self.hidden_out = v

    self.output = tf.identity(v, name='output')
    self.output.type = v.type
    self.output.shape = v.shape


  def set_layer_data(self, index, weight, bias=None):
    assert len(self.weights) > index, index
    self.weights[index].data = weight
    if bias is not None:
      self.biases[index].data = bias


class LSTM:
  def __init__(self, tf, input, size, output_dim=None):
    # Get LSTM dimensions.
    type = input.type
    input_dim = input.shape[1]

    # Define parameters.
    self.x2i = tf.var("x2i", type, [input_dim, size])
    self.h2i = tf.var("h2i", type, [size, size])
    self.c2i = tf.var("c2i", type, [size, size])
    self.bi = tf.var("bi", type, [1, size])

    self.x2o = tf.var("x2o", type, [input_dim, size])
    self.h2o = tf.var("h2o", type, [size, size])
    self.c2o = tf.var("c2o", type, [size, size])
    self.bo = tf.var("bo", type, [1, size])

    self.x2c = tf.var("x2c", type, [input_dim, size])
    self.h2c = tf.var("h2c", type, [size, size])
    self.bc = tf.var("bc", type, [1, size])

    # h_in, c_in = h_{t-1}, c_{t-1}
    h_in = tf.var("h_in", type, [1, size])
    h_in.ref = True
    c_in = tf.var("c_in", type, [1, size])
    c_in.ref = True

    # input --  i_t = sigmoid(affine(x_t, h_{t-1}, c_{t-1}))
    i_ait = tf.add(
        tf.matmul(input, self.x2i),
        tf.add(tf.matmul(h_in, self.h2i),
               tf.add(tf.matmul(c_in, self.c2i), self.bi)), name='i_ait')
    i_it = tf.sigmoid(i_ait, name='i_it')

    # forget -- f_t = 1 - i_t
    i_ft = tf.sub(1.0, i_it, name='i_ft')

    # write memory cell -- tanh(affine(x_t, h_{t-1}))
    i_awt = tf.add(
        tf.matmul(input, self.x2c),
        tf.add(tf.matmul(h_in, self.h2c), self.bc), name='i_awt')
    i_wt = tf.tanh(i_awt, name='i_wt')

    # c_t = f_t \odot c_{t-1} + i_t \odot tanh(affine(x_t, h_{t-1}))
    ct = tf.add(tf.mul(i_it, i_wt), tf.mul(i_ft, c_in), name='c_out')
    ct.ref = True
    ct.input = True
    ct.output = True
    self.ct = ct

    # Connector for control channel.
    self.cnx_control = tf.cnx('control', [c_in, ct])

    # output -- o_t = sigmoid(affine(x_t, h_{t-1}, c_t))
    i_aot = tf.add(tf.matmul(input, self.x2o),
            tf.add(tf.matmul(ct, self.c2o),
            tf.add(tf.matmul(h_in, self.h2o), self.bo)), name='i_aot')
    i_ot = tf.sigmoid(i_aot, name='i_ot')

    # ht = o_t \odot tanh(ct)
    ph_t = tf.tanh(ct)
    ht = tf.mul(i_ot, ph_t, name='h_out')
    ht.ref = True
    ht.input = True
    ht.output = True
    self.ht = ht

    # Connector for hidden channel.
    self.cnx_hidden = tf.cnx('hidden', [h_in, ht])

    if output_dim != None:
      self.W = tf.var("W", type, [size, output_dim])
      self.b = tf.var("b", type, [1, output_dim])


