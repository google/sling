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

"""Compare Myelin flow computations with NumPy."""

import sling
import sling.flags as flags
import sling.myelin as myelin
import numpy as np
import sys
import struct
import math

flags.define("--dt", default=myelin.DT_FLOAT)
flags.define("--test")
flags.define("--thorough", default=False, action='store_true')
flags.define("--repeat", default=1, type=int)

flags.parse()
dt = flags.arg.dt

print("Myelin test suite for", dt, flags.arg.cpu)
print()

# Statistics for test runs.
class Test:
  def __init__(self, f):
    self.name = f.name
    self.runs = 0
    self.errors = 0

  def passed(self):
    return self.runs - self.errors

  def failed(self):
    return self.errors

tests = {}

# Initialize myelin compiler.
compiler = myelin.Compiler()

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

# Compute flow function using numpy.
def simulate(flow, f, data):
  # Copy input tensors.
  v = {}
  for i in flow.inputs(f):
    if i.data != None:
      v[i] = np.array(i.data, dtype=nptypes[i.type])
    else:
      v[i] = np.asarray(data.tensor(i))

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
      v[o[0]] = np.divide(v[i[0]], v[i[1]]).astype(nptypes[dt])
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
      v[o[0]] = np.sum(v[i[0]])
    elif op.type == "Max":
      v[o[0]] = np.max(v[i[0]])
    elif op.type == "Min":
      v[o[0]] = np.min(v[i[0]])
    elif op.type == "Product":
      v[o[0]] = np.prod(v[i[0]])
    elif op.type == "All":
      v[o[0]] = np.all(v[i[0]])
    elif op.type == "Any":
      v[o[0]] = np.any(v[i[0]])
    elif op.type == "Count":
      v[o[0]] = np.array(np.count_nonzero(v[i[0]]), nptypes[dt])
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
      v[o[0]] = np.transpose(v[i[0]])
    elif op.type == "Shape":
      v[o[0]] = np.array(v[i[0]].shape)
    elif op.type == "Size":
      v[o[0]] = np.array(v[i[0]].size)
    elif op.type == "Rank":
      v[o[0]] = np.array(len(v[i[0]].shape))
    elif op.type == "ConcatV2":
      n = int(op.attr("N"))
      axis = v[i[n]]
      seq = []
      for k in range(n): seq.append(v[i[k]])
      v[o[0]] = np.concatenate(tuple(seq), axis)
    elif op.type == "Split":
      splits = np.split(v[i[0]], v[i[1]], v[i[2]])
      for k in range(len(splits)): v[o[k]] = splits[k]
    else:
      raise Exception("No NumPy support for " + op.type)

  # Return results.
  return v

# Compare flow functions against numpy.
def check(flow, variant, lo=-10.0, hi=10.0, rtol=1e-5, atol=1e-8):
  # Ensure that inputs are not overwritten.
  for i in flow.inputs(): i.output = True

  if flags.arg.v >= 2:
    for f in flow.funcs.values():
      print("Compiling %s %s" % (f.name, str(variant)))

  # Compile flow.
  net = compiler.compile(flow)

  # Run all functions and compare results.
  for f in flow.funcs.values():
    # Output progress.
    if flags.arg.v >= 1:
      print("Running %s %s" % (f.name, str(variant)))

    # Create data instance for cell.
    cell = net.cell(f.name)
    data = cell.instance()

    # Fill inputs.
    for i in flow.inputs(f):
      if i.data != None: continue
      a = np.asarray(data.tensor(i))
      if type(lo) == int and type(hi) == int:
        r = np.random.randint(lo, hi, a.shape)
      else:
        r = np.random.ranf(a.shape) * (hi - lo) + lo
      np.copyto(a, r, casting="unsafe")

    # Compute cell.
    for n in range(flags.arg.repeat):
      data.compute()

    # Compute function using numpy.
    baseline = simulate(flow, f, data)

    # Check outputs.
    test = tests.get(f.name)
    if test == None:
      test = Test(f)
      tests[f.name] = test
    for o in flow.outputs(f):
      test.runs += 1
      t = data.tensor(o)
      b = baseline[o]

      if b.dtype == bool: t = np.array(t, dtype=bool)
      if not np.allclose(t, b, rtol=rtol, atol=atol):
        test.errors += 1
        print()
        print("mismatch in", f.name, variant, "for", o.name)
        print("inputs:")
        for i in flow.inputs(f):
          if i.data == None:
            print(i.name)
            print(np.asarray(data.tensor(i)))
        print("myelin:")
        print(np.asarray(t))
        print("numpy:")
        print(b)
        if b.dtype != bool:
          print("abs error:")
          print(b - np.asarray(t))
          print("rel error:")
          print((b - np.asarray(t)) / np.asarray(t))

  if flags.arg.profile:
    print(net.profile())

# Tests

def matmul_test(m, k, n):
  flow = myelin.Flow()
  f = flow.define("matmul")
  A = f.var("A", dt, [m, k])
  B = f.var("B", dt, [k, n])
  C = f.matmul(A, B, name="C")
  check(flow, (m, k, n), -10, 10)

def matmul_add_test(m, k, n):
  flow = myelin.Flow()
  f = flow.define("matmul_add")
  x = f.var("x", dt, [m, k])
  W = f.var("W", dt, [k, n])
  b = f.var("b", dt, [1, n])
  y = f.add(f.matmul(x, W), b)
  check(flow, (m, k, n), -10, 10)

def matmul_add_relu_test(m, k, n):
  flow = myelin.Flow()
  f = flow.define("matmul_add_relu")
  x = f.var("x", dt, [m, k])
  W = f.var("W", dt, [k, n])
  b = f.var("b", dt, [1, n])
  y = f.relu(f.add(f.matmul(x, W), b))
  check(flow, (m, k, n), -10, 10)

def matmul_transpose_test(m, n, k=1):
  flow = myelin.Flow()
  f = flow.define("matmul_transpose")
  x = f.var("x", dt, [k, m])
  y = f.var("y", dt, [n, k])
  z = f.var("z", dt, [m, k])
  W = f.var("W", dt, [n, m])
  f.matmul(x, f.t(W))
  f.matmul(W, f.t(x))
  f.matmul(f.t(y), W)
  f.matmul(f.t(z), f.t(W))
  check(flow, (m, n, k), -10, 10)

def matmul_order_test(m, k, n,
                      ta=False, tb=False, tc=False,
                      ra=None, rb=None, rc=None):
  flow = myelin.Flow()
  f = flow.define("matmul_order")

  a = f.var("A", dt, [k, m] if ta else [m, k])
  b = f.var("B", dt, [n, k] if tb else [k, n])

  if ra is True: a.flags |= 128
  if ra is False: a.flags |= 64

  if rb is True: b.flags |= 128
  if rb is False: b.flags |= 64

  if ta: a = f.t(a)
  if tb: b = f.t(b)

  if tc:
    c = f.t(f.matmul(a, b, name="C"))
  else:
    c = f.matmul(a, b, name="C")

  if rc is True: c.flags |= 128
  if rc is False: c.flags |= 64

  check(flow, ([m, k, n], [ta, tb, tc], [ra, rb, rc]), -10, 10)

def matmul_all_orders_test(m, k, n):
  for ra in [False, True]:
    for rb in [False, True]:
      for ta in [False, True]:
        for tb in [False, True]:
          for tc in [False, True]:
            matmul_order_test(m, k, n, ta, tb, tc, ra, rb)

def matmul_batch_test(m, k, n, b=8):
  flow = myelin.Flow()
  f = flow.define("matmul_batch")
  a = f.var("A", dt, [b, m, k])
  b = f.var("B", dt, [b, k, n])
  c = f.matmul(a, b, name="C")
  check(flow, (m, k, n, b), -10, 10)

def add_test(n):
  flow = myelin.Flow()
  f = flow.define("add")
  x = f.var("x", dt, [n])
  y = f.var("y", dt, [n])
  f.add(x, y)
  check(flow, n)

def sub_test(n):
  flow = myelin.Flow()
  f = flow.define("sub")
  x = f.var("x", dt, [n])
  y = f.var("y", dt, [n])
  f.sub(x, y)
  check(flow, n)

def mul_test(n):
  flow = myelin.Flow()
  f = flow.define("mul")
  x = f.var("x", dt, [n])
  y = f.var("y", dt, [n])
  f.mul(x, y)
  check(flow, n)

def div_test(n):
  flow = myelin.Flow()
  f = flow.define("div")
  x = f.var("x", dt, [n])
  y = f.var("y", dt, [n])
  f.div(x, y)
  check(flow, n, 1.0, 100.0)

def neg_test(n):
  flow = myelin.Flow()
  f = flow.define("neg")
  x = f.var("x", dt, [n])
  y = f.neg(x)
  check(flow, n)

def rcp_test(n):
  flow = myelin.Flow()
  f = flow.define("rcp")
  x = f.var("x", dt, [n])
  y = f.rcp(x)
  check(flow, n)

def abs_test(n):
  flow = myelin.Flow()
  f = flow.define("abs")
  x = f.var("x", dt, [n])
  y = f.abs(x)
  check(flow, n, -10.0, 10.0)

def sign_test(n):
  flow = myelin.Flow()
  f = flow.define("sign")
  x = f.var("x", dt, [n])
  y = f.sign(x)
  check(flow, n, -10.0, 10.0)

def floor_test(n):
  flow = myelin.Flow()
  f = flow.define("floor")
  x = f.var("x", dt, [n])
  y = f.floor(x)
  check(flow, n)

def ceil_test(n):
  flow = myelin.Flow()
  f = flow.define("ceil")
  x = f.var("x", dt, [n])
  y = f.ceil(x)
  check(flow, n)

def round_test(n):
  flow = myelin.Flow()
  f = flow.define("round")
  x = f.var("x", dt, [n])
  y = f.round(x)
  check(flow, n)

def trunc_test(n):
  flow = myelin.Flow()
  f = flow.define("trunc")
  x = f.var("x", dt, [n])
  y = f.trunc(x)
  check(flow, n)

def exp_test(n):
  flow = myelin.Flow()
  f = flow.define("exp")
  x = f.var("x", dt, [n])
  y = f.exp(x)
  check(flow, n)

def log_test(n):
  flow = myelin.Flow()
  f = flow.define("log")
  x = f.var("x", dt, [n])
  y = f.log(x)
  if flags.arg.gpu:
    check(flow, n, 0.1, 10.0, atol=1e-6)
  else:
    check(flow, n, 0.1, 10.0)

def tanh_test(n):
  flow = myelin.Flow()
  f = flow.define("tanh")
  x = f.var("x", dt, [n])
  y = f.tanh(x)
  if flags.arg.gpu:
    check(flow, n, -1.0, 1.0, atol=1e-7)
  else:
    check(flow, n, -1.0, 1.0)

def erf_test(n):
  flow = myelin.Flow()
  f = flow.define("erf")
  x = f.var("x", dt, [n])
  y = f.erf(x)
  check(flow, n, -2.0, 2.0, 1e-6, 1e-4)

def sin_test(n):
  flow = myelin.Flow()
  f = flow.define("sin")
  x = f.var("x", dt, [n])
  y = f.sin(x)
  if flags.arg.gpu:
    check(flow, n, atol=1e-6)
  else:
    check(flow, n)

def cos_test(n):
  flow = myelin.Flow()
  f = flow.define("cos")
  x = f.var("x", dt, [n])
  y = f.cos(x)
  if flags.arg.gpu:
    check(flow, n, atol=1e-6)
  else:
    check(flow, n)

def tan_test(n):
  flow = myelin.Flow()
  f = flow.define("tan")
  x = f.var("x", dt, [n])
  y = f.tan(x)
  if flags.arg.gpu:
    check(flow, n, atol=1e-6, rtol=1e-4)
  else:
    check(flow, n)

def trig_test(n):
  flow = myelin.Flow()
  f = flow.define("trig")
  x = f.var("x", dt, [n])
  ys = f.sin(x)
  yc = f.cos(x)
  yt = f.tan(x)
  if flags.arg.gpu:
    check(flow, n, atol=1e-6, rtol=1e-4)
  else:
    check(flow, n)

def asin_test(n):
  flow = myelin.Flow()
  f = flow.define("asin")
  x = f.var("x", dt, [n])
  y = f.asin(x)
  check(flow, n, -1.0, 1.0)

def acos_test(n):
  flow = myelin.Flow()
  f = flow.define("acos")
  x = f.var("x", dt, [n])
  y = f.acos(x)
  check(flow, n, -1.0, 1.0)

def atan_test(n):
  flow = myelin.Flow()
  f = flow.define("atan")
  x = f.var("x", dt, [n])
  y = f.atan(x)
  check(flow, n)

def sinh_test(n):
  flow = myelin.Flow()
  f = flow.define("sinh")
  x = f.var("x", dt, [n])
  y = f.sinh(x)
  check(flow, n)

def cosh_test(n):
  flow = myelin.Flow()
  f = flow.define("cosh")
  x = f.var("x", dt, [n])
  y = f.cosh(x)
  check(flow, n)

def tanh_test(n):
  flow = myelin.Flow()
  f = flow.define("tanh")
  x = f.var("x", dt, [n])
  y = f.tanh(x)
  check(flow, n)

def asinh_test(n):
  flow = myelin.Flow()
  f = flow.define("asinh")
  x = f.var("x", dt, [n])
  y = f.asinh(x)
  check(flow, n, -1.0, 1.0, rtol=1e-3, atol=1e-6)

def acosh_test(n):
  flow = myelin.Flow()
  f = flow.define("acosh")
  x = f.var("x", dt, [n])
  y = f.acosh(x)
  check(flow, n, 1.0, 10.0, atol=1e-6)

def atanh_test(n):
  flow = myelin.Flow()
  f = flow.define("atanh")
  x = f.var("x", dt, [n])
  y = f.atanh(x)
  check(flow, n, -1.0, -0.1)
  check(flow, n, 0.1, 1.0)

def sqrt_test(n):
  flow = myelin.Flow()
  f = flow.define("sqrt")
  x = f.var("x", dt, [n])
  y = f.sqrt(x)
  check(flow, n, 0.1, 10.0)

def rsqrt_test(n):
  flow = myelin.Flow()
  f = flow.define("rsqrt")
  x = f.var("x", dt, [n])
  y = f.rsqrt(x)
  check(flow, n, 1.0, 10.0, rtol=1e-3, atol=1e-4)

def rcpsqrt_test(n):
  flow = myelin.Flow()
  f = flow.define("rcpsqrt")
  x = f.var("x", dt, [n])
  y = f.div(f.const(1.0, dtype=dt), f.sqrt(x))
  check(flow, n, 1.0, 10.0, rtol=1e-3, atol=1e-4)

def onediv_test(n):
  flow = myelin.Flow()
  f = flow.define("onediv")
  x = f.var("x", dt, [n])
  y = f.div(f.const(1.0, dtype=dt), x)
  check(flow, n, 1.0, 10.0, rtol=1e-3, atol=1e-4)

def square_test(n):
  flow = myelin.Flow()
  f = flow.define("square")
  x = f.var("x", dt, [n])
  y = f.square(x)
  check(flow, n)

def sigmoid_test(n):
  flow = myelin.Flow()
  f = flow.define("sigmoid")
  x = f.var("x", dt, [n])
  y = f.sigmoid(x)
  check(flow, n)

def softmax_test(n):
  flow = myelin.Flow()
  f = flow.define("softmax")
  x = f.var("x", dt, [n])
  y = f.softmax(x)
  check(flow, n)

def relu_test(n):
  flow = myelin.Flow()
  f = flow.define("relu")
  x = f.var("x", dt, [n])
  y = f.relu(x)
  check(flow, n)

def min_test(n):
  flow = myelin.Flow()
  f = flow.define("min")
  x = f.var("x", dt, [n])
  y = f.min(x)
  check(flow, n)

def max_test(n):
  flow = myelin.Flow()
  f = flow.define("max")
  x = f.var("x", dt, [n])
  y = f.max(x)
  check(flow, n)

def sum_test(n):
  flow = myelin.Flow()
  f = flow.define("sum")
  x = f.var("x", dt, [n])
  y = f.sum(x)
  check(flow, n, 0.0, 10.0)

def product_test(n):
  flow = myelin.Flow()
  f = flow.define("product")
  x = f.var("x", dt, [n])
  y = f.product(x)
  check(flow, n, 0.0, 1.0)

def all_test(n):
  flow = myelin.Flow()
  f = flow.define("all")
  x = f.var("x", dt, [n])
  y = f.all(f.greater(x, f.const(0, dtype=dt)))
  check(flow, -10.0, 1.0)

def any_test(n):
  flow = myelin.Flow()
  f = flow.define("any")
  x = f.var("x", dt, [n])
  y = f.any(f.greater(x, f.const(0, dtype=dt)))
  check(flow, n, -1.0, 10.0)

def count_test(n):
  flow = myelin.Flow()
  f = flow.define("count")
  x = f.var("x", dt, [n])
  y = f.count(f.greater(x, f.const(0, dtype=dt)), dtype=dt)
  check(flow, n)

def norm_test(n):
  flow = myelin.Flow()
  f = flow.define("norm")
  x = f.var("x", dt, [n])
  y = f.norm(x)
  check(flow, n, 0.0, 1.0)

def argmin_test(n):
  flow = myelin.Flow()
  f = flow.define("argmin")
  x = f.var("x", dt, [n])
  y = f.argmin(x)
  check(flow, n)

def argmax_test(n):
  flow = myelin.Flow()
  f = flow.define("argmax")
  x = f.var("x", dt, [n])
  y = f.argmax(x)
  check(flow, n)

def minimum_test(n):
  flow = myelin.Flow()
  f = flow.define("minimum")
  x = f.var("x", dt, [n])
  y = f.var("y", dt, [n])
  z = f.minimum(x, y)
  check(flow, n)

def maximum_test(n):
  flow = myelin.Flow()
  f = flow.define("maximum")
  x = f.var("x", dt, [n])
  y = f.var("y", dt, [n])
  z = f.maximum(x, y)
  check(flow, n)

def bcast_test(n):
  flow = myelin.Flow()
  f = flow.define("bcast")
  x = f.var("x", dt, [n])
  y = f.mul(x, f.const(7, dt))
  check(flow, n)

def bcast_repeat_test(n):
  flow = myelin.Flow()
  f = flow.define("bcast_repeat")
  x = f.var("x", dt, [n, 256])
  y = f.var("y", dt, [n, 1])
  z = f.square(f.sub(x, y))
  check(flow, n)

def shape_test(n):
  flow = myelin.Flow()
  f = flow.define("shape")
  x = f.var("x", dt, [n])
  y = f.shape(x)
  check(flow, n)

def size_test(n):
  flow = myelin.Flow()
  f = flow.define("size")
  x = f.var("x", dt, [n])
  y = f.size(x)
  check(flow, n)

def rank_test(n):
  flow = myelin.Flow()
  f = flow.define("size")
  x = f.var("x", dt, [1] * n)
  y = f.rank(x)
  check(flow, n)

def concat_test(n, m):
  flow = myelin.Flow()
  f = flow.define("concat")
  a = f.var("a", dt, [1, n])
  b = f.var("b", dt, [1, m])
  c = f.concat([a, b])
  check(flow, (n, m))

def split_test(n, m):
  flow = myelin.Flow()
  f = flow.define("split")
  x = f.var("x", dt, [1, n])
  y = f.split(x, m, 1)
  check(flow, (n, m))

def equal_test(n):
  flow = myelin.Flow()
  f = flow.define("equal")
  f.equal(f.var("x", dt, [n]), f.var("y", dt, [n]))
  check(flow, n, 0, 10)

def not_equal_test(n):
  flow = myelin.Flow()
  f = flow.define("not_equal")
  f.not_equal(f.var("x", dt, [n]), f.var("y", dt, [n]))
  check(flow, n, 0, 10)

def less_test(n):
  flow = myelin.Flow()
  f = flow.define("less")
  f.less(f.var("x", dt, [n]), f.var("y", dt, [n]))
  check(flow, n, 0, 10)

def less_equal_test(n):
  flow = myelin.Flow()
  f = flow.define("less_equal")
  f.less_equal(f.var("x", dt, [n]), f.var("y", dt, [n]))
  check(flow, n, 0, 10)

def greater_test(n):
  flow = myelin.Flow()
  f = flow.define("greater")
  f.greater(f.var("x", dt, [n]), f.var("y", dt, [n]))
  check(flow, n, 0, 10)

def greater_equal_test(n):
  flow = myelin.Flow()
  f = flow.define("greater_equal")
  f.greater_equal(f.var("x", dt, [n]), f.var("y", dt, [n]))
  check(flow, n, 0, 10)

def logic_test(n):
  flow = myelin.Flow()
  f = flow.define("logic")
  x = f.var("x", dt, [n])
  y = f.var("y", dt, [n])
  eq = f.equal(x, y)
  gt = f.greater(x, y)
  f.logical_and(eq, gt, name="and")
  f.logical_or(eq, gt, name="or")
  f.logical_not(gt, name="not")
  f.logical_xor(eq, gt, name="xor")
  f.logical_and(f.logical_not(eq), gt, name="andn")
  check(flow, n, 0, 10)

def cond_test(n):
  flow = myelin.Flow()
  f = flow.define("cond")
  c = f.var("c", dt, [n])
  x = f.var("x", dt, [n])
  y = f.var("y", dt, [n])
  f.cond(f.equal(c, f.const(0, dtype=dt)), x, y)
  check(flow, n, -5, 5)

def select_test(n):
  flow = myelin.Flow()
  f = flow.define("select")
  c = f.var("c", dt, [n])
  x = f.var("x", dt, [n])
  f.select(f.equal(c, f.const(0, dtype=dt)), x)
  check(flow, n, -5, 5)

def add_const_test(n, c):
  flow = myelin.Flow()
  f = flow.define("add_const")
  x = f.var("x", dt, [n])
  y = f.const(c, dt)
  x = f.add(x, y)
  check(flow, (n, c))

def sub_const_test(n, c):
  flow = myelin.Flow()
  f = flow.define("sub_const")
  x = f.var("x", dt, [n])
  y = f.const(c, dt)
  x = f.sub(x, y)
  check(flow, (n, c))

def mul_const_test(n, c):
  flow = myelin.Flow()
  f = flow.define("mul_const")
  x = f.var("x", dt, [n])
  y = f.const(c, dt)
  x = f.mul(x, y)
  check(flow, (n, c))

def pow_test(n, p):
  flow = myelin.Flow()
  f = flow.define("pow")
  x = f.var("x", dt, [n])
  y = f.const(p, dt)
  x = f.pow(x, y)
  check(flow, (n, p), 0.0, 10.0)

def negfold_test(n):
  flow = myelin.Flow()
  f = flow.define("negfold")
  x = f.var("x", dt, [n])
  y = f.var("y", dt, [n])
  z = f.sub(x, f.neg(y))
  check(flow, n)

# Check for specific test to run.
if flags.arg.test:
  print("Running test", flags.arg.test)
  exec(flags.arg.test)
  print()
  quit()

# Run tests for different size ranges.
if flags.arg.thorough:
  sizes = list(range(1, 48)) + [64, 128, 256]
else:
  sizes = list(range(1, 8)) + [9, 14, 15, 16, 31, 32, 33, 64]

for i in sizes:
  for j in sizes:
    concat_test(i, j)
  for j in range(1, i + 1):
    if i % j == 0:
      split_test(i, j)

for i in sizes:
  add_test(i)
  sub_test(i)
  mul_test(i)
  div_test(i)
  minimum_test(i)
  maximum_test(i)
  neg_test(i)
  abs_test(i)
  square_test(i)
  relu_test(i)
  bcast_test(i)
  bcast_repeat_test(i)
  shape_test(i)
  size_test(i)
  if i < 32: rank_test(i)

  for c in [-1, 0, 1, 2, 3, 4, 5, 6, 7, 8]:
    add_const_test(i, c)
    sub_const_test(i, c)
    mul_const_test(i, c)

  if dt == myelin.DT_FLOAT or dt == myelin.DT_DOUBLE:
    rcp_test(i)
    sqrt_test(i)
    rsqrt_test(i)
    exp_test(i)
    log_test(i)
    for p in [0.0, 1.0, 2.0, 2.5, 3.0, -1.0, -2.0, 0.5, -0.5]:
      pow_test(i, p)
    sin_test(i)
    cos_test(i)
    tan_test(i)
    asin_test(i)
    acos_test(i)
    atan_test(i)
    sinh_test(i)
    cosh_test(i)
    tanh_test(i)
    asinh_test(i)
    acosh_test(i)
    atanh_test(i)
    trig_test(i)
    erf_test(i)
    sigmoid_test(i)
    softmax_test(i)
    sum_test(i)
    product_test(i)
    min_test(i)
    max_test(i)
    all_test(i)
    any_test(i)
    count_test(i)
    norm_test(i)
    sign_test(i)
    floor_test(i)
    ceil_test(i)
    round_test(i)
    trunc_test(i)

    equal_test(i)
    not_equal_test(i)
    less_test(i)
    less_equal_test(i)
    greater_test(i)
    greater_equal_test(i)
    logic_test(i)
    cond_test(i)
    select_test(i)

    if dt != myelin.DT_DOUBLE:
      # No support yet for argmax and argmin for doubles.
      argmax_test(i)
      argmin_test(i)

for i in sizes:
  for j in sizes:
    matmul_transpose_test(i, j)
    if not flags.arg.mkl: matmul_all_orders_test(i, j, 32)
    for k in sizes:
      matmul_test(i, j, k)
      matmul_add_test(i, j, k)
      matmul_batch_test(i, j, k, 8)
      if flags.arg.thorough and not flags.arg.mkl:
        matmul_all_orders_test(i, j, k)
      if dt != myelin.DT_INT8:
        # Rounding with MatMulAddRelu not compatible with NymPy for INT8.
        matmul_add_relu_test(i, j, k)
if flags.arg.thorough:
  matmul_test(1024, 1024, 1024)

# Output test results.
print("Test results")
print("============")
print()

errors = 0
for name in sorted(tests):
  t = tests[name]
  errors += t.failed()
  if t.failed() == 0:
    print("%-20s %7d passed" % (t.name, t.passed()))
  else:
    print("%-20s %7d passed %7d failed" % (t.name, t.passed(), t.failed()))
print

if errors > 0:
  print("*****", errors, "tests failed for ", " ".join(sys.argv[1:]), " *****")
  exit(1)

