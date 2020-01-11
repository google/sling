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
import sling.myelin.simulator as simulator
import numpy as np
import sys
import struct

flags.define("--dt", default=myelin.DT_FLOAT)
flags.define("--test")
flags.define("--thorough", default=False, action='store_true')
flags.define("--repeat", default=1, type=int)
flags.define("--skipdiff", default=False, action='store_true')

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

# Compare flow functions against numpy.
def check(flow, variant, lo=-10.0, hi=10.0, rtol=1e-5, atol=1e-8, check=None):
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
      if i.data is not None: continue
      a = np.asarray(data.tensor(i))
      if type(lo) == int and type(hi) == int:
        r = np.random.randint(lo, hi, a.shape)
      else:
        r = np.random.ranf(a.shape) * (hi - lo) + lo
      np.copyto(a, r, casting="unsafe")

    # Compute function using numpy.
    baseline = simulator.compute(flow, f, data)

    # Compute cell.
    for n in range(flags.arg.repeat):
      data.compute()

    # Create new test if does not already exist.
    test = tests.get(f.name)
    if test is None:
      test = Test(f)
      tests[f.name] = test

    # Check outputs
    if check is None: check = flow.outputs(f)
    for o in check:
      test.runs += 1
      t = data.tensor(o)
      b = baseline[o]

      if b.dtype == bool: t = np.array(t, dtype=bool)
      if not np.allclose(t, b, rtol=rtol, atol=atol):
        test.errors += 1
        print()
        print("ERROR: mismatch in", f.name, variant, "for", o.name)
        if not flags.arg.skipdiff:
          print("inputs:")
          for i in flow.inputs(f):
            if i.data is None:
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

def matmul_batch_test(m, k, n, batch=8):
  flow = myelin.Flow()
  f = flow.define("matmul_batch")
  a = f.var("A", dt, [batch, m, k])
  b = f.var("B", dt, [batch, k, n])
  c = f.matmul(a, b, name="C")
  check(flow, (m, k, n, batch), -10, 10)

def transpose_test(m, k, n, perm):
  flow = myelin.Flow()
  f = flow.define("transpose")
  x = f.var("x", dt, [m, k, n])
  y = f.transpose(x, perm=perm)
  check(flow, (m, k, n, perm))

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
  check(flow, n, rtol=1e-3, atol=1e-6)

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

def sum_axis_test(n, m, k, axis):
  flow = myelin.Flow()
  f = flow.define("sum_axis")
  x = f.var("x", dt, [n, m, k])
  y = f.sum(x, axis=axis, keepdims=True)
  check(flow, (n, m, k, axis), 0.0, 10.0)

def max_axis_test(n, m, k, axis):
  flow = myelin.Flow()
  f = flow.define("max_axis")
  x = f.var("x", dt, [n, m, k])
  y = f.max(x, axis=axis, keepdims=True)
  check(flow, (n, m, k, axis), 0.0, 10.0)

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

def bcast_repeat_test(n, m, k):
  flow = myelin.Flow()
  f = flow.define("bcast_repeat")
  x = f.var("x", dt, [n, m])
  y = f.var("y", dt, [n, k])
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

def gather_test(n, d, s):
  flow = myelin.Flow()
  f = flow.define("gather")
  emb = f.array("emb", np.random.ranf((n, d)).astype(simulator.nptypes[dt]))
  ind = f.var("ind", myelin.DT_INT32, [1, s])
  v = f.gather(emb, ind)
  check(flow, (n, d, s), 0, n)

def gather_sum_test(n, d, s):
  flow = myelin.Flow()
  f = flow.define("gather_sum")
  emb = f.array("emb", np.random.ranf((n, d)).astype(simulator.nptypes[dt]))
  ind = f.var("ind", myelin.DT_INT32, [1, s])
  v = f.gather_sum(emb, ind)
  check(flow, (n, d, s), 0, n)

def gather_max_test(n, d, s):
  flow = myelin.Flow()
  f = flow.define("gather_max")
  emb = f.array("emb", np.random.ranf((n, d)).astype(simulator.nptypes[dt]))
  ind = f.var("ind", myelin.DT_INT32, [1, s])
  v = f.gather_max(emb, ind)
  check(flow, (n, d, s), 0, n)

def gather_avg_test(n, d, s):
  flow = myelin.Flow()
  f = flow.define("gather_avg")
  emb = f.array("emb", np.random.ranf((n, d)).astype(simulator.nptypes[dt]))
  ind = f.var("ind", myelin.DT_INT32, [1, s])
  v = f.gather_avg(emb, ind)
  check(flow, (n, d, s), 0, n, rtol=1e-3)

def scatter_add_test(n, d, s):
  flow = myelin.Flow()
  f = flow.define("scatter_add")
  m = f.var("m", dt, [n, d])
  ind = f.var("ind", myelin.DT_INT32, [1, s])
  v = f.var("v", dt, [1, d])
  f.assign_add_scatter(m, ind, v)
  check(flow, (n, d, s), 0, n, check=[m])

def negfold_test(n):
  flow = myelin.Flow()
  f = flow.define("negfold")
  x = f.var("x", dt, [n])
  y = f.var("y", dt, [n])
  z = f.sub(x, f.neg(y))
  check(flow, n)

def acc_matmul_transpose_test():
  flow = myelin.Flow()
  f = flow.define("acc_matmul_transpose")
  w1 = f.var("w1", dt, [32, 2])
  l = f.var("l", dt, [1, 2])
  r = f.var("r", dt, [1, 32])
  f.assign(w1, f.add(w1, f.matmul(f.transpose(r), l)))
  check(flow, [], 0, 10, check=[w1])

def acc_matmul_test(m, k, n):
  flow = myelin.Flow()
  f = flow.define("acc_matmul")
  c = f.var("c", dt, [m, n])
  a = f.var("a", dt, [m, k])
  b = f.var("b", dt, [k, n])
  f.assign(c, f.add(c, f.matmul(a, b)))
  check(flow, (m, k, n), 0, 10, check=[c])

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
  argmax_test(i)
  argmin_test(i)
  neg_test(i)
  abs_test(i)
  square_test(i)
  relu_test(i)
  bcast_test(i)
  bcast_repeat_test(i, 256, 1)
  shape_test(i)
  size_test(i)
  if i < 32: rank_test(i)

  embsize = 32
  for f in [1, 2, 5]:
    gather_test(embsize, i, f)
    gather_sum_test(embsize, i, f)
    gather_max_test(embsize, i, f)
    if dt == myelin.DT_FLOAT or dt == myelin.DT_DOUBLE:
      gather_avg_test(embsize, i, f)
    scatter_add_test(embsize, i, f)

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

for i in sizes:
  for j in sizes:
    matmul_transpose_test(i, j)
    if not flags.arg.mkl: matmul_all_orders_test(i, j, 32)
    for k in sizes:
      matmul_test(i, j, k)
      matmul_add_test(i, j, k)
      matmul_batch_test(i, j, k, 8)
      acc_matmul_test(i, j, k)
      transpose_test(i, j, k, [1, 0, 2])
      transpose_test(i, j, k, [1, 2, 0])
      if flags.arg.thorough:
        transpose_test(i, j, k, [0, 1, 2])
        transpose_test(i, j, k, [0, 2, 1])
        transpose_test(i, j, k, [2, 0, 1])
        transpose_test(i, j, k, [2, 1, 0])
      if flags.arg.thorough and not flags.arg.mkl:
        matmul_all_orders_test(i, j, k)
      if dt != myelin.DT_INT8:
        # Rounding not compatible with NymPy for INT8.
        matmul_add_relu_test(i, j, k)
        for axis in [0, 1, 2]:
          sum_axis_test(i, j, k, axis)
          max_axis_test(i, j, k, axis)

if flags.arg.thorough:
  matmul_test(1024, 1024, 1024)

# Output test results.
print("Test results")
print("============")
print()

passed = 0
errors = 0
for name in sorted(tests):
  t = tests[name]
  passed += t.passed()
  errors += t.failed()
  if t.failed() == 0:
    print("%-20s %7d passed" % (t.name, t.passed()))
  else:
    print("%-20s %7d passed %7d failed" % (t.name, t.passed(), t.failed()))

if errors == 0:
  print("%-20s %7d passed" % ("TOTAL", passed))
else:
  print("%-20s %7d passed %7d failed" % ("TOTAL", passed, errors))
print

if errors > 0:
  print("*****", errors, "tests failed for ", " ".join(sys.argv[1:]), " *****")
  exit(1)

