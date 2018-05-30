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

import math
from struct import pack

from flow import Variable
from flow import Function
from flow import Flow

DT_INT = "int32"
DT_FLOAT = "float32"

class Builder:
  def __init__(self, flow, func):
    self.flow = flow
    self.func = flow.func(func)

  def var(self, name, dtype=DT_FLOAT, shape=[]):
    v = self.flow.var(self.func.name + "/" + name)
    v.type = dtype
    v.shape = shape
    return v

  def rename(self, var, new_suffix):
    var.name = self.func.name + "/" + new_suffix
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
      name = self.func.name + "/" + name
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
      name = self.func.name + "/" + name
    op = self.flow.op(name)
    op.type = optype
    self.func.add(op)
    return op

  def const(self, value, dtype=None, shape=None):
    # Convert scalars.
    if type(value) is float:
      dtype = DT_FLOAT
      shape = []
      value = value
    elif type(value) is int:
      dtype = DT_INT
      shape = []
      value = value

    # Get type and shape if missing.
    if dtype is None: dtype = str(value.dtype)
    if shape is None: shape = list(value.shape)

    var = self.flow.var(self.varname("const"), dtype, shape)
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
    op = self.rawop("ConcatV2", name)
    op.dtype = args[0].type
    shape = [args[0].shape[0], 0]
    for arg in args:
      op.add_input(arg)
      shape[1] += arg.shape[1]
    op.add_attr("N", len(args))
    axis = self.const(1, DT_INT)
    op.add_input(axis)
    result = self.var(op.name + ":0", shape=shape)
    op.add_output(result)

    return op.outputs[0]

  def add(self, x, y, name=None):
    return self.op("Add", [x, y], name)

  def sub(self, x, y, name=None):
    return self.op("Sub", [x, y], name)

  def mul(self, x, y, name=None):
    return self.op("Mul", [x, y], name)

  def div(self, x, y, name=None):
    return self.op("Div", [x, y], name)

  def min(self, x, y, name=None):
    return self.op("Minimum", [x, y], name)

  def max(self, x, y, name=None):
    return self.op("Maximum", [x, y], name)

  def matmul(self, x, y, name=None):
    result = self.op("MatMul", [x, y], name)
    result.type = x.type
    if len(x.shape) == 2 and len(y.shape) == 2:
      result.shape = [x.shape[0], y.shape[1]]
    else:
      result.shape = [0]
    return result

  def t(self, x, name=None):
    result = self.op("Transpose", [x], name)
    if len(x.shape) == 2:
      result.shape = [x.shape[1], x.shape[0]]
    else:
      result.shape = [0]
    return result

  def log(self, x, name=None):
    return self.op("Log", [x], name)

  def exp(self, x, name=None):
    return self.op("Exp", [x], name)

  def tanh(self, x, name=None):
    return self.op("Tanh", [x], name)

  def sigmoid(self, x, name=None):
    return self.op("Sigmoid", [x], name)

  def relu(self, x, name=None):
    return self.op("Relu", [x], name)

  def sin(self, x, name=None):
    return self.op("Sin", [x], name)

  def cos(self, x, name=None):
    return self.op("Cos", [x], name)

  def square(self, x, name=None):
    return self.op("Square", [x], name)

  def neg(self, x, name=None):
    return self.op("Neg", [x], name)

  def abs(self, x, name=None):
    return self.op("Abs", [x], name)

  def sign(self, x, name=None):
    return self.op("Sign", [x], name)

  def rcp(self, x, name=None):
    return self.op("Reciprocal", [x], name)

  def identity(self, x, name=None):
    return self.op("Identity", [x], name)

  def ref(self, instance, var, name=None):
    r = self.op("Reference", [instance], name)
    r.producer.add_attr("var", var.name)
    r.type = var.type
    r.shape = var.shape
    return r

