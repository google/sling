# Copyright 2018 Google Inc. All Rights Reserved.
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

def attr_str(value):
  """ Convert attribute to string value."""
  if isinstance(value, bool):
    return "true" if value else "false"
  elif isinstance(value, int):
    return str(value)
  elif isinstance(value, long):
    return str(value)
  elif isinstance(value, str):
    return value
  elif isinstance(value, list):
    l = []
    for v in value: l.append(attr_str(v))
    return ",".join(l)
  elif value.__class__.__name__ == "TensorShapeProto":
    dims = []
    for d in value.dim: dims.append(str(d.size))
    return "x".join(dims)
  elif value.__class__.__name__ == "TensorProto":
    return str(value)
  elif value.__class__.__name__ == "DType":
    return value.name
  else:
    return str(type(value)) + ":" + str(value).replace('\n', ' ')

class Extractor:
  """Extract myelin flow from tensorflow graph."""

  def __init__(self, sess, flow):
    """Initialize empty flow builder."""
    self.sess = sess
    self.feed = None
    self.flow = flow
    self.vars = []
    self.ops = []

  def add(self, func, inputs, outputs):
    """Add ops to flow."""
    for var in outputs:
      self.expand(func, var, inputs)

  def expand(self, func, var, inputs):
    """Traverse graphs and add ops to flow."""
    if var not in self.vars:
      # Add new variable to flow.
      self.vars.append(var)
      v = self.flow.var(var.name, var.dtype.base_dtype.name, [])

      # Get data for constants and variables.
      if var.op.type in ["Const", "ConstV2"]:
        v.data = tf.contrib.util.constant_value(var)
      elif var.op.type in ["Variable", "VariableV2"]:
        if self.feed is None:
          v.data = var.eval(session=self.sess)
        else:
          v.data = self.sess.run(var, feed_dict=self.feed)

      # Get shape.
      if v.data is None:
        shape = var.get_shape()
        for d in shape.as_list():
          if d != None:
            v.shape.append(d)
          else:
            v.shape.append(-1)
      else:
        for d in v.data.shape:
          v.shape.append(d)

      if not var in inputs:
        op = var.op
        if op not in self.ops:
          # Add new operation to flow function.
          self.ops.append(op)
          o = self.flow.op(op.name)
          func.add(o)
          o.type = op.type
          for input in op.inputs:
            o.add_input(self.flow.var(input.name))
          for output in op.outputs:
            o.add_output(self.flow.var(output.name))
          for a in op.node_def.attr:
            o.add_attr(a, attr_str(op.get_attr(a)))

          # Traverse dependencies.
          for dep in op.inputs:
            self.expand(func, dep, inputs)


  def compute_shapes(self):
    """Compute shapes for variables with missing shape information."""
    # Find all variables with missing shape information.
    missing = {}
    for var in self.vars:
      v = self.flow.var(var.name)
      if not v.shape_defined():
        missing[v] = var

    if len(missing) > 0:
      # Compute variables from feed.
      results = self.sess.run(missing, feed_dict=self.feed)

      # Use the shape of the computed variables for the flow.
      for v in results:
        v.shape = results[v].shape