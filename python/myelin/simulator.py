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

"""Myelin flow simulator using NumPy."""

import math
import numpy as np
import sling.myelin as myelin

# Myelin to NumPy type conversion.
nptypes = {
  myelin.DT_FLOAT32: np.float32,
  myelin.DT_FLOAT64: np.float64,
  myelin.DT_INT8: np.int8,
  myelin.DT_INT16: np.int16,
  myelin.DT_INT32: np.int32,
  myelin.DT_INT64: np.int64,
}

# NumPy functions.
def sigmoid(x):
  return 1 / (1 + np.exp(-x))

def relu(x):
  return np.maximum(x, 0)

def erf(x):
  return np.array([math.erf(v) for v in x])

def gather(d, i):
  return np.take(d, i, axis=0)

# Compute flow function using numpy.
def compute(flow, f, data):
  # Copy input tensors.
  v = {}
  for i in flow.inputs(f):
    if i.data is None:
      v[i] = np.asarray(data.tensor(i))
    else:
      v[i] = np.array(i.data, dtype=nptypes[i.type])

  # Get ops in computation order.
  _, ops = flow.order(f)

  # Compute ops using numpy.
  for op in ops:
    i = op.inputs
    o = op.outputs

    if op.type == "MatMul":
      v[o[0]] = np.matmul(v[i[0]], v[i[1]])
    elif op.type == "Exp":
      v[o[0]] = np.exp(v[i[0]])
    elif op.type == "Sigmoid":
      v[o[0]] = sigmoid(v[i[0]])
    elif op.type == "Log":
      v[o[0]] = np.log(v[i[0]])
    elif op.type == "Pow":
      v[o[0]] = np.power(v[i[0]], v[i[1]])
    elif op.type == "Erf":
      v[o[0]] = erf(v[i[0]])
    elif op.type == "Sin":
      v[o[0]] = np.sin(v[i[0]])
    elif op.type == "Cos":
      v[o[0]] = np.cos(v[i[0]])
    elif op.type == "Tan":
      v[o[0]] = np.tan(v[i[0]])
    elif op.type == "Asin":
      v[o[0]] = np.arcsin(v[i[0]])
    elif op.type == "Acos":
      v[o[0]] = np.arccos(v[i[0]])
    elif op.type == "Atan":
      v[o[0]] = np.arctan(v[i[0]])
    elif op.type == "Sinh":
      v[o[0]] = np.sinh(v[i[0]])
    elif op.type == "Cosh":
      v[o[0]] = np.cosh(v[i[0]])
    elif op.type == "Tanh":
      v[o[0]] = np.tanh(v[i[0]])
    elif op.type == "Asinh":
      v[o[0]] = np.arcsinh(v[i[0]])
    elif op.type == "Acosh":
      v[o[0]] = np.arccosh(v[i[0]])
    elif op.type == "Atanh":
      v[o[0]] = np.arctanh(v[i[0]])
    elif op.type == "Relu":
      v[o[0]] = relu(v[i[0]])
    elif op.type == "Sqrt":
      v[o[0]] = np.sqrt(v[i[0]])
    elif op.type == "Rsqrt":
      v[o[0]] = 1 / np.sqrt(v[i[0]])
    elif op.type == "Square":
      v[o[0]] = np.square(v[i[0]])
    elif op.type == "Neg":
      v[o[0]] = -v[i[0]]
    elif op.type == "Abs":
      v[o[0]] = np.abs(v[i[0]])
    elif op.type == "Sign":
      v[o[0]] = np.sign(v[i[0]])
    elif op.type == "Add":
      v[o[0]] = v[i[0]] + v[i[1]]
    elif op.type == "Sub":
      v[o[0]] = v[i[0]] - v[i[1]]
    elif op.type == "Mul":
      v[o[0]] = v[i[0]] * v[i[1]]
    elif op.type == "Div":
      v[o[0]] = np.divide(v[i[0]], v[i[1]]).astype(nptypes[o[0].type])
    elif op.type == "Minimum":
      v[o[0]] = np.minimum(v[i[0]], v[i[1]])
    elif op.type == "Maximum":
      v[o[0]] = np.maximum(v[i[0]], v[i[1]])
    elif op.type == "Reciprocal":
      v[o[0]] = np.divide(1, v[i[0]])
    elif op.type == "Floor":
      v[o[0]] = np.floor(v[i[0]])
    elif op.type == "Ceil":
      v[o[0]] = np.ceil(v[i[0]])
    elif op.type == "Round":
      v[o[0]] = np.round(v[i[0]])
    elif op.type == "Trunc":
      v[o[0]] = np.trunc(v[i[0]])
    elif op.type == "Sum":
      axis = op.attrs.get("axis")
      if axis is None:
        v[o[0]] = np.sum(v[i[0]])
      else:
        keepdims = bool(op.attrs.get("keepdims"))
        v[o[0]] = np.sum(v[i[0]], int(axis), keepdims=keepdims)
    elif op.type == "Max":
      axis = op.attrs.get("axis")
      if axis is None:
        v[o[0]] = np.max(v[i[0]])
      else:
        keepdims = bool(op.attrs.get("keepdims"))
        v[o[0]] = np.max(v[i[0]], int(axis), keepdims=keepdims)
    elif op.type == "Min":
      axis = op.attrs.get("axis")
      if axis is None:
        v[o[0]] = np.min(v[i[0]])
      else:
        keepdims = bool(op.attrs.get("keepdims"))
        v[o[0]] = np.min(v[i[0]], int(axis), keepdims=keepdims)
    elif op.type == "Product":
      axis = op.attrs.get("axis")
      if axis is None:
        v[o[0]] = np.prod(v[i[0]])
      else:
        keepdims = bool(op.attrs.get("keepdims"))
        v[o[0]] = np.prod(v[i[0]], int(axis), keepdims=keepdims)
    elif op.type == "All":
      axis = op.attrs.get("axis")
      if axis is None:
        v[o[0]] = np.all(v[i[0]])
      else:
        keepdims = bool(op.attrs.get("keepdims"))
        v[o[0]] = np.all(v[i[0]], int(axis), keepdims=keepdims)
    elif op.type == "Any":
      axis = op.attrs.get("axis")
      if axis is None:
        v[o[0]] = np.any(v[i[0]])
      else:
        keepdims = bool(op.attrs.get("keepdims"))
        v[o[0]] = np.any(v[i[0]], axis, keepdims=keepdims)
    elif op.type == "Count":
      v[o[0]] = np.array(np.count_nonzero(v[i[0]]), nptypes[o[0].type])
    elif op.type == "ArgMin":
      v[o[0]] = np.argmin(v[i[0]])
    elif op.type == "ArgMax":
      v[o[0]] = np.argmax(v[i[0]])
    elif op.type == "Equal":
      v[o[0]] = np.equal(v[i[0]], v[i[1]])
    elif op.type == "NotEqual":
      v[o[0]] = np.not_equal(v[i[0]], v[i[1]])
    elif op.type == "Less":
      v[o[0]] = np.less(v[i[0]], v[i[1]])
    elif op.type == "LessEqual":
      v[o[0]] = np.less_equal(v[i[0]], v[i[1]])
    elif op.type == "Greater":
      v[o[0]] = np.greater(v[i[0]], v[i[1]])
    elif op.type == "GreaterEqual":
      v[o[0]] = np.greater_equal(v[i[0]], v[i[1]])
    elif op.type == "And":
      v[o[0]] = np.logical_and(v[i[0]], v[i[1]])
    elif op.type == "Or":
      v[o[0]] = np.logical_or(v[i[0]], v[i[1]])
    elif op.type == "Xor":
      v[o[0]] = np.logical_xor(v[i[0]], v[i[1]])
    elif op.type == "Not":
      v[o[0]] = np.logical_not(v[i[0]])
    elif op.type == "Cond":
      v[o[0]] = np.where((v[i[0]] != 0), v[i[1]], v[i[2]])
    elif op.type == "Select":
      v[o[0]] = np.where((v[i[0]] != 0), v[i[1]], 0)
    elif op.type == "Transpose":
      if "perm" in op.attrs:
        perm = eval(op.attrs["perm"])
        v[o[0]] = np.transpose(v[i[0]], axes=perm)
      else:
        v[o[0]] = np.transpose(v[i[0]])
    elif op.type == "Shape":
      v[o[0]] = np.array(v[i[0]].shape)
    elif op.type == "Size":
      v[o[0]] = np.array(v[i[0]].size)
    elif op.type == "Rank":
      v[o[0]] = np.array(len(v[i[0]].shape))
    elif op.type == "Identity":
      v[o[0]] = v[i[0]]
    elif op.type == "Concat":
      n = int(op.attr("N"))
      axis = v[i[n]]
      seq = []
      for k in range(n): seq.append(v[i[k]])
      v[o[0]] = np.concatenate(tuple(seq), axis)
    elif op.type == "Split":
      splits = np.split(v[i[0]], v[i[1]], v[i[2]])
      for k in range(len(splits)): v[o[k]] = splits[k]
    elif op.type == "Gather":
      v[o[0]] = gather(v[i[0]], v[i[1]])
    elif op.type == "GatherSum":
      v[o[0]] = np.sum(gather(v[i[0]], v[i[1]]), axis=1)
    elif op.type == "GatherMax":
      v[o[0]] = np.max(gather(v[i[0]], v[i[1]]), axis=1)
    elif op.type == "GatherAvg":
      v[o[0]] = np.sum(gather(v[i[0]], v[i[1]]), axis=1) / v[i[1]].shape[1]
    elif op.type == "AssignAddScatter":
      m = v[i[0]]
      f = v[i[1]]
      x = v[i[2]]
      m[f] += x
    elif op.type == "Reshape":
      v[o[0]] = np.reshape(v[i[0]], v[i[1]])
    elif op.type == "Assign":
      v[i[0]] = v[i[1]]
    else:
      raise Exception("No NumPy support for " + op.type)

  # Return results.
  return v

