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

import array
import os
from struct import calcsize, pack, unpack, unpack_from

def dummy_factory_builder(flow, name):
  raise Exception("No flow builder defined")

builder_factory_method = dummy_factory_builder

def set_builder_factory(factory):
  global builder_factory_method
  builder_factory_method = factory

class FileWriter:
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
    if type(d) is str:
      d = d.encode()
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
      s = s.encode()
      self.write_int(len(s))
      self.f.write(s)

  def write_object(self, a):
    """Write array to flow file."""
    if a is None:
      self.write_long(0)
    elif isinstance(a, bytes):
      self.write_long(len(a))
      self.f.write(a)
    elif isinstance(a, bytearray):
      b = bytes(a)
      self.write_long(len(b))
      self.f.write(b)
    elif isinstance(a, str):
      a = a.encode()
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
    elif isinstance(a, array.array):
      m = memoryview(a)
      self.write_long(m.nbytes)
      self.f.write(m)
    elif isinstance(a, memoryview):
      self.write_long(a.nbytes)
      self.f.write(a)
    else:
      self.write_long(a.nbytes)
      self.write(a.tostring())


class FileReader:
  """Flow file reader."""
  def __init__(self, filename):
    size = os.path.getsize(filename)
    with open(filename, 'rb') as f:
      b = bytearray(size)
      f.readinto(b)
    self.view = memoryview(b)
    self.next = 0  # index of next byte to read

  def slice(self, size):
    """Returns a slice of given size from the current byte and skips ahead."""
    current = self.next
    self.next += size
    return self.view[current:self.next]

  def read(self, n):
    """Returns the next n bytes."""
    s = self.view[self.next:self.next + n]
    self.next += n
    return s

  def _read_fmt(self, fmt):
    """Reads an element of format 'fmt'."""
    size = calcsize(fmt)
    val = unpack_from(fmt, self.view, self.next)[0]
    self.next += size
    return val

  def read_int(self):
    """Reads an integer."""
    return self._read_fmt('i')

  def read_long(self):
    """Reads a long integer."""
    return self._read_fmt('Q')

  def read_string(self):
    """Reads a string."""
    size = self.read_int()
    if size > 0:
      return self.read(size).tobytes().decode()
    return ''


class Variable(object):
  """Flow variable."""

  def __init__(self, name):
    """Initialize new variable."""
    self.name = name
    self.flags = 0
    self.aliases = []
    self.type = None
    self.shape = []
    self.attrs = {}
    self.data = None
    self.producer = None
    self.consumers = []

  @property
  def input(self):
    return (self.flags & 1) != 0

  @input.setter
  def input(self, value):
    if value:
      self.flags |= 1
    else:
      self.flags &= ~1

  @property
  def output(self):
    return (self.flags & 2) != 0

  @output.setter
  def output(self, value):
    if value:
      self.flags |= 2
    else:
      self.flags &= ~2

  @property
  def ref(self):
    return (self.flags & 4) != 0

  @ref.setter
  def ref(self, value):
    if value:
      self.flags |= 4
    else:
      self.flags &= ~4

  @property
  def learnable(self):
    return (self.flags & 8) != 0

  @learnable.setter
  def learnable(self, value):
    if value:
      self.flags |= 8
    else:
      self.flags &= ~8

  @property
  def unique(self):
    return (self.flags & 16) != 0

  @unique.setter
  def unique(self, value):
    if value:
      self.flags |= 16
    else:
      self.flags &= ~16

  def add_attr(self, name, value):
    if type(value) is bool: value = int(value)
    self.attrs[name] = str(value)

  def attr(self, name):
    return self.attrs.get(name, None)

  def shape_defined(self):
    for d in self.shape:
      if d == -1: return False
    return True

  def elements(self):
    n = 1
    for d in self.shape: n *= d
    return n

  def rank(self):
    return len(self.shape)

  def __repr__(self):
    return self.name

  def __str__(self):
    s = "var " + self.name + " : " + self.typestr()
    if self.data is not None:
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


class Operation(object):
  """Flow operation with inputs and outputs."""

  def __init__(self, name):
    """Initialize new operation."""
    self.name = name
    self.flags = 0
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


class Function(object):
  """Flow function with operations."""

  def __init__(self, name):
    """Initialize new function."""
    self.name = name
    self.flags = 0
    self.ops = []

  def add(self, op):
    """Add operation to function."""
    self.ops.append(op)
    op.func = self

  @property
  def backprop(self):
    return (self.flags & 2) != 0

  @backprop.setter
  def backprop(self, value):
    if value:
      self.flags |= 2
    else:
      self.flags &= ~2

  def __str__(self):
    s = "func " + self.name + " {\n"
    for op in self.ops:
      s += "  " + op.name + " : " + op.type + "\n"
    s += "}\n"
    return s


class Connector(object):
  """Flow connector with linked variables."""

  def __init__(self, name):
    """Initialize new connector."""
    self.name = name
    self.flags = 0
    self.links = []

  def add(self, var):
    """Add linked variable to connector."""
    self.links.append(var)

  def __str__(self):
    s = "connector " + self.name + " {\n"
    for l in self.links:
      s += "  " + l.name + "\n"
    s += "}\n"
    return s


class Blob(object):
  """Blob for storing extra data like lexicons and feature maps."""

  def __init__(self, name):
    """Initialize new blob."""
    self.name = name
    self.flags = 0
    self.type = ""
    self.data = None
    self.attrs = {}

  def add_attr(self, name, value):
    """Add blob attribute."""
    if type(value) is bool:
      if value == True: value = 1
      elif value == False: value = 0
    self.attrs[name] = str(value)

  def get_attr(self, name):
    """Get blob attribute as a string or None."""
    return self.attrs.get(name, None)

  def __str__(self):
    s = "blob " + self.name + " : " + self.type
    if self.data is not None:
      s += " = " + str(self.data)
    s += " {\n"
    for a in self.attrs:
      s += "  " + a + " = " + self.attrs[a] + "\n"
    s += "}\n"
    return s


class Flow:
  """Flow with variables, operations, and functions."""

  def __init__(self):
    """Initialize empty flow."""
    self.flags = 0
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

  def var(self, name, type="float32", shape=None):
    """Add variable to flow."""
    if isinstance(name, Variable): return name
    v = self.vars.get(name, None)
    if v == None:
      v = Variable(name)
      self.vars[name] = v
      v.type = type
      if shape is not None: v.shape = shape
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

  def define(self, name):
    """Create a builder for a new funtion."""
    return builder_factory_method(self, name)

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
    f = FileWriter(filename)
    f.write('flow')
    f.write_int(6)
    f.write_int(self.flags)

    # Write variables.
    f.write_int(len(self.vars))
    for name in self.vars:
      var = self.vars[name]
      f.write_int(var.flags)
      f.write_string(var.name)
      f.write_int(len(var.aliases))
      for alias in var.aliases: f.write_string(alias)
      f.write_string(var.type)
      f.write_int(len(var.shape))
      for d in var.shape: f.write_int(d)
      f.write_int(len(var.attrs))
      for a in var.attrs:
        f.write_string(a)
        f.write_string(op.attrs[a])
      f.write_object(var.data)

    # Write operations.
    f.write_int(len(self.ops))
    for name in self.ops:
      op = self.ops[name]
      f.write_int(op.flags)
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
      f.write_int(func.flags)
      f.write_string(func.name)
      f.write_int(len(func.ops))
      for op in func.ops:
        f.write_string(op.name)

    # Write connectors.
    f.write_int(len(self.cnxs))
    for name in self.cnxs:
      cnx = self.cnxs[name]
      f.write_int(cnx.flags)
      f.write_string(cnx.name)
      f.write_int(len(cnx.links))
      for link in cnx.links:
        f.write_string(link.name)

    # Write blobs.
    f.write_int(len(self.blobs))
    for name in self.blobs:
      blob = self.blobs[name]
      f.write_int(blob.flags)
      f.write_string(blob.name)
      f.write_string(blob.type)
      f.write_int(len(blob.attrs))
      for a in blob.attrs:
        f.write_string(a)
        f.write_string(blob.attrs[a])
      f.write_object(blob.data)

    f.close()

  # Loads flow from 'filename'.
  def load(self, filename):
    f = FileReader(filename)
    magic = f.read(4)
    assert magic == memoryview(b'flow'), magic.tobytes()

    version = f.read_int()
    assert version == 4 or version == 5 or version == 6, version
    if version >= 5: self.flags = f.read_int()

    num_vars = f.read_int()
    for _ in range(num_vars):
      flags = 0
      if version >= 5: flags = f.read_int()
      name = f.read_string()
      num_aliases = f.read_int()
      aliases = []
      for i in range(num_aliases):
        aliases.append(f.read_string())
      t = f.read_string()
      if t[0] == '&':
        flags |= 4
        t = t[1:]
      shape_size = f.read_int()
      shape = []
      for _ in range(shape_size):
        shape.append(f.read_int())
      var = self.var(name, type=t, shape=shape)
      var.flags = flags
      if version >= 6:
        num_attr = f.read_int()
        for _ in range(num_attr):
          attr_name = f.read_string()
          attr_val = f.read_string()
          var.add_attr(attr_name, attr_val)
      size = f.read_long()
      if size > 0:
        var.data = f.slice(size)  # avoid creating a copy

    num_ops = f.read_int()
    for _ in range(num_ops):
      flags = 0
      if version >= 5: flags = f.read_int()
      name = f.read_string()
      op = self.op(name)
      op.flags = flags
      op.type = f.read_string()

      num_in = f.read_int()
      for _ in range(num_in):
        op.add_input(self.var(name=f.read_string()))

      num_out = f.read_int()
      for _ in range(num_out):
        op.add_output(self.var(name=f.read_string()))

      num_attr = f.read_int()
      for _ in range(num_attr):
        attr_name = f.read_string()
        attr_val = f.read_string()
        op.add_attr(attr_name, attr_val)

    num_funcs = f.read_int()
    for _ in range(num_funcs):
      flags = 0
      if version >= 5: flags = f.read_int()
      name = f.read_string()
      func = self.func(name)
      func.flags = flags
      n = f.read_int()
      for _ in range(n):
        func.add(self.op(f.read_string()))

    num_cnxs = f.read_int()
    for _ in range(num_cnxs):
      flags = 0
      if version >= 5: flags = f.read_int()
      name = f.read_string()
      cnx = self.cnx(name)
      cnx.flags = flags
      n = f.read_int()
      for _ in range(n):
        cnx.add(self.var(f.read_string()))

    num_blobs = f.read_int()
    for _ in range(num_blobs):
      flags = 0
      if version >= 5: flags = f.read_int()
      name = f.read_string()
      blob = self.blob(name)
      blob.flags = flags
      blob.type = f.read_string()
      n = f.read_int()
      for _ in range(n):
        name = f.read_string()
        val = f.read_string()
        blob.add_attr(name, val)
      size = f.read_long()
      if size > 0:
        blob.data = f.slice(size)  # avoid creating a copy

  def __str__(self):
    s = ""
    for var in self.vars.values(): s += str(var)
    for op in self.ops.values(): s += str(op)
    for func in self.funcs.values(): s += str(func)
    for cnx in self.cnxs.values(): s += str(cnx)
    for blob in self.blobs.values(): s += str(blob)
    return s

