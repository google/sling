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

"""Myelin computation flows."""

from struct import pack

class File:
  """Flow file writer."""

  def __init__(self, file):
    """Open flow file for writing."""
    if type(file) is str:
      self.f = open(file, 'wb')
    else:
      self.f = file

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

  def write_object(self, a):
    """Write array to flow file."""
    if a is None:
      self.write_long(0)
    elif isinstance(a, str):
      self.write_long(len(a))
      self.f.write(a)
    elif isinstance(a, float):
      data = pack('f', a);
      self.write_long(len(data))
      self.f.write(data)
    elif isinstance(a, int):
      data = pack('i', a);
      self.write_long(len(data))
      self.f.write(data)
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
    self.ref = False
    self.data = None
    self.producer = None
    self.consumers = []

  def shape_defined(self):
    for d in self.shape:
      if d == -1: return False
    return True

  def __repr__(self):
    return self.name

  def __str__(self):
    s = "var " + self.name + " : " + self.typestr()
    if self.data is not None:
      #s += " " + str(self.data.nbytes) + "bytes"
      s += " = " + str(self.data)
    s += " {\n"
    if self.producer != None:
      s += "  from " + self.producer.name + "\n"
    for op in self.consumers:
      s += "  to " + op.name + "\n"
    s += "}\n"
    return s

  def typestr(self):
    t = ""
    if self.ref: t += "&"
    t += self.type if self.type != None else "???"
    if len(self.shape) != 0: t += "[" + 'x'.join(map(str, self.shape)) + "]"
    return t


class Operation:
  """Flow operation with inputs and outputs."""

  def __init__(self, name):
    """Initialize new operation."""
    self.name = name
    self.type = None
    self.inputs = []
    self.outputs = []
    self.attrs = {}
    self.func = None

  def add_input(self, input):
    """Add input to operation."""
    self.inputs.append(input)
    input.consumers.append(self)

  def add_output(self, output):
    """Add output from operation."""
    self.outputs.append(output)
    output.producer = self

  def add_attr(self, name, value):
    """Add operation attribute."""
    if type(value) is bool:
      if value == True: value = 1
      elif value == False: value = 0
    self.attrs[name] = str(value)

  def attr(self, name):
    """Look up attribute for operation."""
    return self.attrs.get(name, None)

  def __repr__(self):
    return self.name

  def __str__(self):
    s = "op " + self.name + " : " + self.type + " {\n"
    for v in self.inputs:
      s += "  input " + v.name + " : " + v.typestr() + "\n"
    for v in self.outputs:
      s += "  output " + v.name + " : " + v.typestr() + "\n"
    for key in self.attrs:
      s += "  " + key + " = " + str(self.attrs[key]) + "\n"
    s += "}\n"
    return s


class Function:
  """Flow function with operations."""

  def __init__(self, name):
    """Initialize new function."""
    self.name = name
    self.ops = []

  def add(self, op):
    """Add operation to function."""
    self.ops.append(op)
    op.func = self

  def __str__(self):
    s = "func " + self.name + " {\n"
    for op in self.ops:
      s += "  " + op.name + " : " + op.type + "\n"
    s += "}\n"
    return s


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
    if type(value) is bool:
      if value == True: value = 1
      elif value == False: value = 0
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
    if isinstance(name, Function): return name
    f = self.funcs.get(name, None)
    if f == None:
      f = Function(name)
      self.funcs[name] = f
    return f

  def var(self, name, type="float32", shape=[]):
    """Add variable to flow."""
    if isinstance(name, Variable): return name
    v = self.vars.get(name, None)
    if v == None:
      v = Variable(name)
      self.vars[name] = v
      v.type = type
      v.shape = shape
    return v

  def op(self, name):
    """Add operation to flow."""
    if isinstance(name, Operation): return name
    o = self.ops.get(name, None)
    if o == None:
      o = Operation(name)
      self.ops[name] = o
    return o

  def cnx(self, name):
    """Add connector to flow."""
    if isinstance(name, Connector): return name
    c = self.cnxs.get(name, None)
    if c == None:
      c = Connector(name)
      self.cnxs[name] = c
    return c

  def blob(self, name):
    """Add blob to flow."""
    if isinstance(name, Blob): return name
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


  def inputs(self, func=None, variable=True, const=True):
    """ Return list of inputs, i.e. variables with no producer."""
    ivars = []
    for v in self.vars.values():
      if v.producer != None: continue
      if func != None:
        used = False
        for op in v.consumers:
          if op.func == func:
            used = True
            break
        if not used: continue
      if v.data is None:
        if not variable: continue
      else:
        if not const: continue
      ivars.append(v)
    return ivars

  def outputs(self, func=None, variable=True, const=True):
    """ Return list of outputs, i.e. variables with no consumers."""
    ovars = []
    for v in self.vars.values():
      if len(v.consumers) > 0: continue
      if func != None:
        if v.producer == None: continue
        if v.producer.func != func: continue
      if v.data is None:
        if not variable: continue
      else:
        if not const: continue
      ovars.append(v)
    return ovars

  def order(self, func=None):
    """Return variables and operations in dependency order."""
    ordered_vars = []
    ordered_ops = []

    # Add all variables with no producer.
    for v in self.vars.values():
      if v.producer != None: continue
      if func != None:
        used = False
        for op in v.consumers:
          if op.func == func:
            used = True
            break
        if not used: continue
      ordered_vars.append(v)

    # Compute the number of missing inputs for each operation and add operations
    # that do not depend on other operations to the ready queue.
    ready = []
    remaining = {}
    for op in self.ops.values():
      if func != None and op.func != func: continue
      num_missing = 0
      for v in op.inputs:
        if v.producer != None: num_missing += 1
      if num_missing == 0:
        ready.append(op)
      else:
        remaining[op] = num_missing

    # Keep adding ops that are ready to be computed.
    while len(ready) > 0:
      # Get the next op that is ready.
      op = ready.pop()

      # Add it to the ordered set of ops.
      ordered_ops.append(op);

      # Propagate readiness to consumers.
      for v in op.outputs:
        ordered_vars.append(v)
        for consumer in v.consumers:
          if func != None and consumer.func != func: continue
          remaining[consumer] -= 1
          if remaining[consumer] == 0:
            ready.append(consumer)

    return ordered_vars, ordered_ops

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
      f.write_string("&" + var.type if var.ref else var.type)
      f.write_int(len(var.shape))
      for d in var.shape: f.write_int(d)
      f.write_object(var.data)

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
      f.write_object(blob.data)

    f.close()

  def __str__(self):
    s = ""
    for var in self.vars.values(): s += str(var)
    for op in self.ops.values(): s += str(op)
    for func in self.funcs.values(): s += str(func)
    for cnx in self.cnxs.values(): s += str(cnx)
    for blob in self.blobs.values(): s += str(blob)
    return s

