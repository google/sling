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

"""Myelin function builder and expression evaluator."""

import array
from .flow import set_builder_factory, Variable

DT_FLOAT32 = "float32"
DT_FLOAT64 = "float64"

DT_INT8 = "int8"
DT_INT16 = "int16"
DT_INT32 = "int32"
DT_INT64 = "int64"
DT_BOOL = "bool"

DT_INT = DT_INT32
DT_FLOAT = DT_FLOAT32
DT_DOUBLE = DT_FLOAT64

typemap = {
  "f": DT_FLOAT32,
  "d": DT_FLOAT64,
  "i": DT_INT32,
  "l": DT_INT64,
  "B": DT_INT8,
  "h": DT_INT16,
  "b": DT_INT8,
  "q": DT_INT64,
  "?": DT_BOOL,
}

typecodes = {
  DT_FLOAT32: "f",
  DT_FLOAT64: "d",
  DT_INT8: "b",
  DT_INT16: "h",
  DT_INT32: "i",
  DT_INT64: "q",
  DT_BOOL: "?",
}

dtypes = {
  int: DT_INT32,
  float: DT_FLOAT32,
}

# Compute product of elements in list.
def list_product(l):
  n = 1
  for e in l: n *= e
  return n

# Compute the shape of a nested list.
def list_shape(l):
  shape = None
  for e in l:
    if type(e) is list:
      s = list_shape(e)
      if shape is None:
        shape = s
      elif shape != s:
        return None
  if shape is None: return [len(l)]
  return [len(l)] + shape

# Flatten nested list.
def flatten_list(flat, l):
  for e in l:
    if type(e) is list:
      flatten_list(flat, e)
    else:
      flat.append(e)

# Convert nested list to array.
def list_to_array(l, typecode=None):
  # Get list shape.
  shape = list_shape(l)
  if shape is None: raise TypeError("Unsupported list shape")

  # Flatten list.
  f = []
  flatten_list(f, l)

  # Determine list type.
  if typecode is None:
    for e in f:
      et = type(e)
      if et == float or typecode == 'f':
        typecode = 'f'
      elif et == int or typecode == 'i':
        typecode = 'i'
  if typecode is None: typecode = 'f'

  # Convert list to array.
  a = array.array(typecode, f)

  # Return array, type, and shape.
  return a, typecode, shape

class Builder:
  def __init__(self, flow, func):
    self.flow = flow
    self.func = flow.func(func)

  def var(self, name, dtype=DT_FLOAT, shape=[]):
    n = self.func.name + "/" + name
    if n in self.flow.vars: raise IndexError("variable already defined: " + n);
    v = self.flow.var(n)
    v.type = dtype
    v.shape = shape
    return v

  def rename(self, var, new_suffix):
    n = self.func.name + "/" + new_suffix
    if n in self.flow.vars: raise IndexError("variable already defined: " + n);
    var.name = n
    return var

  def cnx(self, name, args):
    c = self.flow.cnx(self.func.name + "/" + name)
    for a in args:
      c.add(a)
    return c

  def op(self, optype, args, name=None):
    if name is None:
      name = self.opname(optype)
    else:
      name = self.opname(name)
    op = self.flow.op(name)
    op.type = optype
    self.func.add(op)
    shape = []
    for a in args:
      if not isinstance(a, Variable): a = self.const(a)
      op.add_input(a)
      if len(a.shape) > len(shape): shape = a.shape
    dtype = op.inputs[0].type if len(op.inputs) > 0 else DT_FLOAT
    result = self.flow.var(name + ":0", dtype, shape)
    op.add_output(result)
    return result

  def rawop(self, optype, name=None):
    if name is None:
      name = self.opname(optype)
    else:
      name = self.opname(name)
    op = self.flow.op(name)
    op.type = optype
    self.func.add(op)
    return op

  def const(self, value, dtype=None, shape=None):
    # Scalar type conversion.
    if type(value) is int and dtype == DT_FLOAT: value = float(value)
    if type(value) is float and dtype == DT_INT: value = int(value)

    # Convert scalars and lists.
    if type(value) is float:
      if dtype is None: dtype = DT_FLOAT
      if shape is None: shape = []
    elif type(value) is int:
      if dtype is None: dtype = DT_INT
      if shape is None: shape = []
    elif type(value) is list:
      value, typecode, shape = list_to_array(value, typecodes.get(dtype))
      dtype = typemap[typecode]

    # Convert arrays.
    if type(value) is array.array:
      if dtype is None: dtype = typemap[value.typecode]
      if shape is None: shape = [len(value)]
      value = memoryview(value)

    # Get type and shape if missing.
    if dtype is None: dtype = str(value.dtype)
    if shape is None: shape = list(value.shape)

    var = self.flow.var(self.varname("const"), dtype, shape)
    var.data = value
    return var

  def array(self, name, value):
    # Convert lists to arrays that support the buffer protocol.
    if type(value) is list:
      value, _, _ = list_to_array(value)

    # Make constant from object with buffer support.
    view = memoryview(value)
    dtype = typemap[view.format]
    shape = list(view.shape)
    var = self.flow.var(self.varname(name), dtype, shape)
    var.data = value
    return var

  def opname(self, optype):
    name = self.func.name + '/' + optype
    if name not in self.flow.ops: return name
    index = 1
    while True:
      n = name + "_" + str(index)
      if n not in self.flow.ops: return n
      index += 1

  def varname(self, var):
    name = self.func.name + '/' + var
    if name not in self.flow.vars: return name
    index = 1
    while True:
      n = name + "_" + str(index)
      if n not in self.flow.vars: return n
      index += 1

  def concat(self, args, name=None):
    op = self.rawop("Concat", name)
    shape = [args[0].shape[0], 0]
    for arg in args:
      op.add_input(arg)
      shape[1] += arg.shape[1]
    op.add_attr("N", len(args))
    axis = self.const(1, DT_INT)
    op.add_input(axis)
    result = self.var(op.name + ":0", args[0].type, shape)
    op.add_output(result)

    return op.outputs[0]

  def split(self, x, splits, axis=0, name=None):
    op = self.rawop("Split", name)
    op.add_input(x)
    op.add_input(self.const(splits, DT_INT))
    op.add_input(self.const(axis, DT_INT))
    shape = x.shape[:]
    shape[axis] = x.shape[axis] / splits
    results = []
    for n in range(splits):
      o = self.var(op.name + ":" + str(n), x.type, shape)
      op.add_output(o)
      results.append(o)
    return tuple(results)

  def reshape(self, x, shape, name=None):
    result = self.op("Reshape", [x, self.const(shape)], name)
    result.shape = shape
    return result

  def add(self, x, y, name=None):
    return self.op("Add", [x, y], name)

  def sub(self, x, y, name=None):
    return self.op("Sub", [x, y], name)

  def mul(self, x, y, name=None):
    return self.op("Mul", [x, y], name)

  def div(self, x, y, name=None):
    return self.op("Div", [x, y], name)

  def minimum(self, x, y, name=None):
    return self.op("Minimum", [x, y], name)

  def maximum(self, x, y, name=None):
    return self.op("Maximum", [x, y], name)

  def argmin(self, x, name=None):
    result = self.op("ArgMin", [x], name)
    result.shape = []
    result.type = DT_INT
    return result

  def argmax(self, x, name=None):
    result = self.op("ArgMax", [x], name)
    result.shape = []
    result.type = DT_INT
    return result

  def gather(self, embedding, indices, oov=None, name=None):
    inputs = [embedding, indices]
    if oov is not None:
      inputs.append(oov)
    result = self.op("Gather", inputs, name)
    result.type = embedding.type
    if len(embedding.shape) == 2 and len(indices.shape) == 2:
      result.shape = [indices.shape[1], embedding.shape[1]]
    else:
      result.shape = [0]
    return result

  def pooling_gather(self, optype, embedding, indices, name=None):
    result = self.op(optype, [embedding, indices], name)
    result.type = embedding.type
    if len(embedding.shape) == 2:
      result.shape = [1, embedding.shape[1]]
    else:
      result.shape = [0]
    return result

  def gather_sum(self, embedding, indices, name=None):
    return self.pooling_gather("GatherSum", embedding, indices, name)

  def gather_max(self, embedding, indices, name=None):
    return self.pooling_gather("GatherMax", embedding, indices, name)

  def gather_avg(self, embedding, indices, name=None):
    return self.pooling_gather("GatherAvg", embedding, indices, name)

  def matmul(self, x, y, name=None):
    result = self.op("MatMul", [x, y], name)
    result.shape = x.shape[:-2] + [x.shape[-2], y.shape[-1]]
    return result

  def transpose(self, x, perm=None, name=None):
    rank = len(x.shape)
    result = self.op("Transpose", [x], name)
    if perm is None and rank == 2:
      # Matrix transpose.
      result.shape = [x.shape[1], x.shape[0]]
    else:
      # Tensor transpose.
      if perm is None: perm = list(reversed(range(rank)))
      if perm == list(range(rank)):
        result.producer.type = "Identity"
        result.shape = x.shape
      else:
        result.producer.add_attr("perm", perm)
        result.shape = [0] * rank
        for d in range(rank): result.shape[d] = x.shape[perm[d]]
    return result

  def t(self, x, perm=None, name=None):
    return self.transpose(x, perm, name)

  def log(self, x, name=None):
    return self.op("Log", [x], name)

  def exp(self, x, name=None):
    return self.op("Exp", [x], name)

  def pow(self, x, y, name=None):
    return self.op("Pow", [x, y], name)

  def erf(self, x, name=None):
    return self.op("Erf", [x], name)

  def sigmoid(self, x, name=None):
    return self.op("Sigmoid", [x], name)

  def relu(self, x, name=None):
    return self.op("Relu", [x], name)

  def sin(self, x, name=None):
    return self.op("Sin", [x], name)

  def cos(self, x, name=None):
    return self.op("Cos", [x], name)

  def tan(self, x, name=None):
    return self.op("Tan", [x], name)

  def cot(self, x, name=None):
    return self.op("Cot", [x], name)

  def sec(self, x, name=None):
    return self.op("Sec", [x], name)

  def csc(self, x, name=None):
    return self.op("Csc", [x], name)

  def asin(self, x, name=None):
    return self.op("Asin", [x], name)

  def acos(self, x, name=None):
    return self.op("Acos", [x], name)

  def atan(self, x, name=None):
    return self.op("Atan", [x], name)

  def acot(self, x, name=None):
    return self.op("Acot", [x], name)

  def asec(self, x, name=None):
    return self.op("Asec", [x], name)

  def acsc(self, x, name=None):
    return self.op("Acsc", [x], name)

  def sinh(self, x, name=None):
    return self.op("Sinh", [x], name)

  def cosh(self, x, name=None):
    return self.op("Cosh", [x], name)

  def tanh(self, x, name=None):
    return self.op("Tanh", [x], name)

  def coth(self, x, name=None):
    return self.op("Coth", [x], name)

  def sech(self, x, name=None):
    return self.op("Sech", [x], name)

  def csch(self, x, name=None):
    return self.op("Csch", [x], name)

  def asinh(self, x, name=None):
    return self.op("Asinh", [x], name)

  def acosh(self, x, name=None):
    return self.op("Acosh", [x], name)

  def atanh(self, x, name=None):
    return self.op("Atanh", [x], name)

  def acoth(self, x, name=None):
    return self.op("Acoth", [x], name)

  def asech(self, x, name=None):
    return self.op("Asech", [x], name)

  def acsch(self, x, name=None):
    return self.op("Acsch", [x], name)

  def square(self, x, name=None):
    return self.op("Square", [x], name)

  def sqrt(self, x, name=None):
    return self.op("Sqrt", [x], name)

  def rsqrt(self, x, name=None):
    return self.op("Rsqrt", [x], name)

  def neg(self, x, name=None):
    return self.op("Neg", [x], name)

  def abs(self, x, name=None):
    return self.op("Abs", [x], name)

  def sign(self, x, name=None):
    return self.op("Sign", [x], name)

  def rcp(self, x, name=None):
    return self.op("Reciprocal", [x], name)

  def floor(self, x, name=None):
    return self.op("Floor", [x], name)

  def ceil(self, x, name=None):
    return self.op("Ceil", [x], name)

  def round(self, x, name=None):
    return self.op("Round", [x], name)

  def trunc(self, x, name=None):
    return self.op("Trunc", [x], name)

  def equal(self, x, y, name=None):
    return self.op("Equal", [x, y], name)

  def not_equal(self, x, y, name=None):
    return self.op("NotEqual", [x, y], name)

  def less(self, x, y, name=None):
    return self.op("Less", [x, y], name)

  def less_equal(self, x, y, name=None):
    return self.op("LessEqual", [x, y], name)

  def greater(self, x, y, name=None):
    return self.op("Greater", [x, y], name)

  def greater_equal(self, x, y, name=None):
    return self.op("GreaterEqual", [x, y], name)

  def logical_and(self, x, y, name=None):
    return self.op("And", [x, y], name)

  def logical_or(self, x, y, name=None):
    return self.op("Or", [x, y], name)

  def logical_xor(self, x, y, name=None):
    return self.op("Xor", [x, y], name)

  def logical_not(self, x, name=None):
    return self.op("Not", [x], name)

  def cond(self, c, x, y, name=None):
    return self.op("Cond", [c, x, y], name)

  def select(self, c, x, name=None):
    return self.op("Select", [c, x], name)

  def identity(self, x, name=None):
    return self.op("Identity", [x], name)

  def reduce(self, optype, x, axis=None, keepdims=None, name=None):
    v = self.op(optype, [x], name)
    if axis is None:
      v.shape = []
    else:
      if axis < 0: axis = len(x.shape) + axis
      v.shape = x.shape.copy()
      v.producer.add_attr("axis", axis)
      if keepdims:
        v.shape[axis] = 1
        v.producer.add_attr("keepdims", True)
      else:
        del v.shape[axis]
    return v

  def sum(self, x, axis=None, keepdims=None, name=None):
    return self.reduce("Sum", x, axis, keepdims, name)

  def product(self, x, axis=None, keepdims=None, name=None):
    return self.reduce("Product", x, axis, keepdims, name)

  def min(self, x, axis=None, keepdims=None, name=None):
    return self.reduce("Min", x, axis, keepdims, name)

  def max(self, x, axis=None, keepdims=None, name=None):
    return self.reduce("Max", x, axis, keepdims, name)

  def all(self, x, axis=None, keepdims=None, name=None):
    return self.reduce("All", x, axis, keepdims, name)

  def any(self, x, axis=None, keepdims=None, name=None):
    return self.reduce("Any", x, axis, keepdims, name)

  def count(self, p, axis=None, dtype=DT_FLOAT32, name=None):
    r = self.reduce("Count", p, axis, name)
    r.type = dtype
    return r

  def mean(self, x, axis=None, keepdims=None, name=None):
    sum = self.sum(x, axis, keepdims)
    if axis is None:
      size = list_product(x.shape)
    else:
      if axis < 0: axis = len(x.shape) + axis
      size = x.shape[axis]
    return self.div(sum, self.const(size, x.type), name=name)

  def norm(self, x, name=None):
    return self.sqrt(self.sum(self.square(x)), name)

  def normalize(self, x, name=None):
    return self.mul(x, self.rcp(self.norm(x)), name)

  def softmax(self, x, name=None):
    return self.normalize(self.exp(self.sub(x, self.max(x))), name)

  def ref(self, instance, var, name=None):
    r = self.op("Reference", [instance], name)
    r.producer.add_attr("var", var.name)
    r.type = var.type
    r.shape = var.shape
    return r

  def shape(self, x, name=None):
    result = self.op("Shape", [x], name)
    result.shape = [x.rank()]
    result.type = DT_INT
    return result

  def size(self, x, name=None):
    result = self.op("Size", [x], name)
    result.shape = []
    result.type = DT_INT
    return result

  def rank(self, x, name=None):
    result = self.op("Rank", [x], name)
    result.shape = []
    result.type = DT_INT
    return result

  def assign(self, x, y, name=None):
    op = self.rawop("Assign", name)
    op.add_input(x)
    op.add_input(y)

  def assign_add_scatter(self, m, f, v, ref=False, name=None):
    op = self.rawop("AssignAddScatter", name)
    op.add_input(m)
    op.add_input(f)
    op.add_input(v)
    if ref:
      r = self.var(op.name + "/ref", m.type, m.shape)
      r.ref = True
      op.add_output(r)
      return r


# Set builder factory for flows.
def builder_factory(flow, name):
  return Builder(flow, name)

set_builder_factory(builder_factory)

