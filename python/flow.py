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

""" Myelin flow builder

Convert tensorflow graphs to myelin flow files.
"""

import tensorflow as tf
from struct import pack
from tensorflow.python.platform import gfile

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


class File:
  """Flow file writer."""

  def __init__(self, filename):
    """Open flow file for writing."""
    self.f = gfile.GFile(filename, 'wb')

  def close(self):
    """Close flow file."""
    self.f.close()

  def write(self, d):
    """Write data to flow file."""
    self.f.write(d)

  def write_int(self, n):
    """Write 32-bit integer to flow file."""
    self.f.write(pack('i', n))

  def write_long(self, n):
    """Write 64-bit integer to flow file."""
    self.f.write(pack('Q', n))

  def write_string(self, s):
    """Write string to flow file."""
    if s is None:
      self.write_int(0)
    else:
      self.write_int(len(s))
      self.f.write(s)

  def write_array(self, a):
    """Write array to flow file."""
    if a is None:
      self.write_long(0)
    elif isinstance(a, str):
      self.write_long(len(a))
      self.f.write(a)
    else:
      self.write_long(a.nbytes)
      self.write(a.tostring())


class Variable:
  """Flow variable."""

  def __init__(self, name):
    """Initialize new variable."""
    self.name = name
    self.type = None
    self.shape = []
    self.data = None

  def shape_defined(self):
    for d in self.shape:
      if d == -1: return False
    return True


class Operation:
  """Flow operation with inputs and outputs."""

  def __init__(self, name):
    """Initialize new operation."""
    self.name = name
    self.type = None
    self.inputs = []
    self.outputs = []
    self.attrs = {}

  def add_input(self, input):
    """Add input to operation."""
    self.inputs.append(input)

  def add_output(self, output):
    """Add output from operation."""
    self.outputs.append(output)

  def add_attr(self, name, value):
    """Add operation attribute."""
    self.attrs[name] = str(value)


class Function:
  """Flow function with operations."""

  def __init__(self, name):
    """Initialize new function."""
    self.name = name
    self.ops = []

  def add(self, op):
    """Add operation to function."""
    self.ops.append(op)


class Connector:
  """Flow connector with linked variables."""

  def __init__(self, name):
    """Initialize new connector."""
    self.name = name
    self.links = []

  def add(self, var):
    """Add linked variable to connector."""
    self.links.append(var)


class Blob:
  """Blob for storing extra data like lexicons and feature maps."""

  def __init__(self, name):
    """Initialize new blob."""
    self.name = name
    self.type = ""
    self.data = None
    self.attrs = {}

  def add_attr(self, name, value):
    """Add blob attribute."""
    self.attrs[name] = str(value)


class Flow:
  """Flow with variables, operations, and functions."""

  def __init__(self):
    """Initialize empty flow."""
    self.vars = {}
    self.ops = {}
    self.funcs = {}
    self.cnxs = {}
    self.blobs = {}

  def func(self, name):
    """Add function to flow."""
    f = self.funcs.get(name, None)
    if f == None:
      f = Function(name)
      self.funcs[name] = f
    return f

  def var(self, name):
    """Add variable to flow."""
    v = self.vars.get(name, None)
    if v == None:
      v = Variable(name)
      self.vars[name] = v
    return v

  def op(self, name):
    """Add operation to flow."""
    o = self.ops.get(name, None)
    if o == None:
      o = Operation(name)
      self.ops[name] = o
    return o

  def cnx(self, name):
    """Add connector to flow."""
    c = self.cnxs.get(name, None)
    if c == None:
      c = Connector(name)
      self.cnxs[name] = c
    return c

  def blob(self, name):
    """Add blob to flow."""
    b = self.blobs.get(name, None)
    if b == None:
      b = Blob(name)
      self.blobs[name] = b
    return b

  def rename_prefix(self, prefix, replacement):
    """Replace prefix in all names."""
    for mapping in [self.vars, self.ops, self.funcs, self.cnxs]:
      for name in mapping.keys():
        if name.startswith(prefix):
          element = mapping.pop(name)
          newname = replacement + name[len(prefix):]
          element.name = newname
          mapping[newname] = element

  def rename_suffix(self, suffix, replacement):
    """Replace suffix in all names."""
    for mapping in [self.vars, self.ops, self.funcs, self.cnxs]:
      for name in mapping.keys():
        if name.endswith(suffix):
          element = mapping.pop(name)
          newname = name[:-len(suffix)] + replacement
          element.name = newname
          mapping[newname] = element


  def save(self, filename):
    """Write flow to file."""

    # Write flow file header
    f = File(filename)
    f.write('flow')
    f.write_int(4)

    # Write variables.
    f.write_int(len(self.vars))
    for name in self.vars:
      var = self.vars[name]
      f.write_string(var.name)
      f.write_int(0)  # no aliases
      f.write_string(var.type)
      f.write_int(len(var.shape))
      for d in var.shape: f.write_int(d)
      f.write_array(var.data)

    # Write operations.
    f.write_int(len(self.ops))
    for name in self.ops:
      op = self.ops[name]
      f.write_string(op.name)
      f.write_string(op.type)
      f.write_int(len(op.inputs))
      for i in op.inputs:
        f.write_string(i.name)
      f.write_int(len(op.outputs))
      for o in op.outputs:
        f.write_string(o.name)
      f.write_int(len(op.attrs))
      for a in op.attrs:
        f.write_string(a)
        f.write_string(op.attrs[a])

    # Write functions.
    f.write_int(len(self.funcs))
    for name in self.funcs:
      func = self.funcs[name]
      f.write_string(func.name)
      f.write_int(len(func.ops))
      for op in func.ops:
        f.write_string(op.name)

    # Write connectors.
    f.write_int(len(self.cnxs))
    for name in self.cnxs:
      cnx = self.cnxs[name]
      f.write_string(cnx.name)
      f.write_int(len(cnx.links))
      for link in cnx.links:
        f.write_string(link.name)

    # Write blobs.
    f.write_int(len(self.blobs))
    for name in self.blobs:
      blob = self.blobs[name]
      f.write_string(blob.name)
      f.write_string(blob.type)
      f.write_int(len(blob.attrs))
      for a in blob.attrs:
        f.write_string(a)
        f.write_string(blob.attrs[a])
      f.write_array(blob.data)

    f.close()


class FlowBuilder:
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
      v = self.flow.var(var.name)
      v.type = var.dtype.base_dtype.name

      # Get data for constants and variables.
      if var.op.type in ["Const", "ConstV2"]:
        v.data = tf.contrib.util.constant_value(var)
      elif var.op.type in ["Variable", "VariableV2"]:
        if self.feed is None:
          v.data = var.eval(session=self.sess)
        else:
          v.data = self.sess.run(var, feed_dict=self.feed)

      # Get shape.
      if v.data is not None:
        for d in v.data.shape:
          v.shape.append(d)
      else:
        shape = var.get_shape()
        if shape.dims != None:
          undef = True
          for d in shape.as_list():
            if d is None:
              v.shape.append(-1)
            else:
              v.shape.append(d)
              undef = False
          if undef: v.shape = [0] * len(shape.dims)
        else:
          v.shape = [0]

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
