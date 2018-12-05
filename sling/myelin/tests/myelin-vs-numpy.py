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

flags.define("--dt", default=myelin.DT_FLOAT)
flags.parse()
dt = flags.arg.dt

print "Myelin test suite for", dt, flags.arg.cpu
print

# Statistics for test runs.
class Test:
  def __init__(self, f):
    self.name = f.name
    self.runs = 0
    self.errors = 0

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

# Compute flow function using numpy.
def simulate(flow, f, data):
  # Copy input tensors.
  v = {}
  for i in flow.inputs(f):
    if i.data != None:
      v[i] = np.array(i.data, dtype=nptypes[i.type])
    else:
      v[i] = np.asarray(data[i])

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
    elif op.type == "Tanh":
      v[o[0]] = np.tanh(v[i[0]])
    elif op.type == "Relu":
      v[o[0]] = relu(v[i[0]])
    elif op.type == "Sqrt":
      v[o[0]] = np.sqrt(v[i[0]])
    elif op.type == "Square":
      v[o[0]] = np.square(v[i[0]])
    elif op.type == "Neg":
      v[o[0]] = -v[i[0]]
    elif op.type == "Abs":
      v[o[0]] = np.abs(v[i[0]])
    elif op.type == "Add":
      v[o[0]] = v[i[0]] + v[i[1]]
    elif op.type == "Sub":
      v[o[0]] = v[i[0]] - v[i[1]]
    elif op.type == "Mul":
      v[o[0]] = v[i[0]] * v[i[1]]
    elif op.type == "Div":
      v[o[0]] = np.divide(v[i[0]], v[i[1]])
    elif op.type == "Minimum":
      v[o[0]] = np.minimum(v[i[0]], v[i[1]])
    elif op.type == "Maximum":
      v[o[0]] = np.maximum(v[i[0]], v[i[1]])
    elif op.type == "Reciprocal":
      v[o[0]] = np.divide(1, v[i[0]])
    elif op.type == "Sum":
      v[o[0]] = np.sum(v[i[0]])
    elif op.type == "Max":
      v[o[0]] = np.max(v[i[0]])
    elif op.type == "Min":
      v[o[0]] = np.min(v[i[0]])
    elif op.type == "Product":
      v[o[0]] = np.prod(v[i[0]])
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
    elif op.type == "Transpose":
      v[o[0]] = np.transpose(v[i[0]])
    elif op.type == "ConcatV2":
      n = int(op.attr("N"))
      axis = v[i[n]]
      seq = []
      for k in range(n): seq.append(v[i[k]])
      v[o[0]] = np.concatenate(tuple(seq), axis)
    else:
      raise Exception("No NumPy support for " + op.type)

  # Return results.
  return v

# Compare flow functions against numpy.
def check(flow, variant, lo=-10.0, hi=10.0):
  # Ensure that inputs are not overwritten.
  for i in flow.inputs(): i.output = True

  # Compile flow.
  net = compiler.compile(flow)
  for f in flow.funcs.itervalues():
    # Output progress.
    print "\r" + "Running " + f.name + " " + str(variant) + ": \033[K",

    # Create data instance for cell.
    cell = net.cell(f.name)
    data = cell.instance()

    # Fill inputs.
    for i in flow.inputs(f):
      if i.data != None: continue
      a = np.asarray(data[i])
      if type(lo) == int and type(hi) == int:
        r = np.random.randint(lo, hi, a.shape)
      else:
        r = np.random.ranf(a.shape) * (hi - lo) + lo
      np.copyto(a, r, casting="unsafe")

    # Compute cell.
    data.compute()

    # Compute function using numpy.
    baseline = simulate(flow, f, data)

    # Check outputs.
    test = tests.get(f.name)
    if test == None:
      test = Test(f)
      tests[f.name] = test
    test.runs += 1
    for o in flow.outputs(f):
      t = data[o]
      b = baseline[o]
      if b.dtype == bool: t = np.array(t, dtype=bool)
      if not np.allclose(t, b):
        test.errors += 1
        print "mismatch in", f.name, "for", o.name
        print "inputs:"
        for i in flow.inputs(f):
          if i.data == None: print i.name, np.asarray(data[i])
        print "myelin:"
        print np.asarray(t)
        print "numpy:"
        print b
        if b.dtype != bool:
          print "diff:"
          print b - np.asarray(t)

# Tests

def matmul_test(m, k, n):
  flow = myelin.Flow()
  f = flow.define("matmul")
  x = f.var("x", dt, [m, k])
  W = f.var("W", dt, [k, n])
  y = f.matmul(x, W)
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

def matmul_transpose_test(m, n):
  flow = myelin.Flow()
  f = flow.define("matmul_transpose")
  x = f.var("x", dt, [1, m])
  W = f.var("W", dt, [n, m])
  f.matmul(x, f.t(W))
  f.matmul(W, f.t(x))
  check(flow, (m, n), -10, 10)

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
  check(flow, n, 0.1, 10.0)

def tanh_test(n):
  flow = myelin.Flow()
  f = flow.define("tanh")
  x = f.var("x", dt, [n])
  y = f.tanh(x)
  check(flow, n, -1.0, 1.0)

def sqrt_test(n):
  flow = myelin.Flow()
  f = flow.define("sqrt")
  x = f.var("x", dt, [n])
  y = f.sqrt(x)
  check(flow, n, 0.1, 10.0)

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

def norm_test(n):
  flow = myelin.Flow()
  f = flow.define("norm")
  x = f.var("x", dt, [n])
  y = f.norm(x)
  check(flow, n, 0.0, 1.0)

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

def concat_test(n, m):
  flow = myelin.Flow()
  f = flow.define("concat")
  a = f.var("a", dt, [1, n])
  b = f.var("b", dt, [1, m])
  c = f.concat([a, b])
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

# Run tests for different size ranges.
sizes = range(1, 48) + [64, 128, 256]

for i in sizes:
  for j in sizes:
    concat_test(i, j)

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

  for c in [-1, 0, 1, 2, 3, 4, 5, 6, 7, 8]:
    add_const_test(i, c)
    sub_const_test(i, c)
    mul_const_test(i, c)

  if dt == myelin.DT_FLOAT or dt == myelin.DT_DOUBLE:
    rcp_test(i)
    sqrt_test(i)
    exp_test(i)
    tanh_test(i)
    log_test(i)
    sigmoid_test(i)
    softmax_test(i)
    sum_test(i)
    product_test(i)
    min_test(i)
    max_test(i)
    norm_test(i)
    equal_test(i)
    not_equal_test(i)
    less_test(i)
    less_equal_test(i)
    greater_test(i)
    greater_equal_test(i)
    logic_test(i)

    if dt != myelin.DT_DOUBLE:
      # No support yet for argmax over doubles.
      argmax_test(i)

if dt == myelin.DT_FLOAT or dt == myelin.DT_DOUBLE:
  for i in sizes:
    for j in sizes:
      matmul_transpose_test(i, j)
      for k in sizes:
        matmul_test(i, j, k)
        matmul_add_test(i, j, k)
        matmul_add_relu_test(i, j, k)
  matmul_test(2048, 2048, 2048)
else:
  # Only vector-matrix matmul supported for integers.
  for i in sizes:
    for j in sizes:
      matmul_test(1, i, j)
      matmul_add_test(1, i, j)
      if dt != myelin.DT_INT8:
        # Rounding with MatMulAddRelu not compatible with NymPy for INT8.
        matmul_add_relu_test(1, i, j)


# Output test results.
print "\r\033[K"
print "Test results"
print "============"
print

errors = 0
for name in sorted(tests):
  t = tests[name]
  errors += t.errors
  if t.errors == 0:
    print "%-20s %7d runs" % (t.name, t.runs)
  else:
    print "%-20s %7d runs %7d errors" % (t.name, t.runs, t.errors)
print

if errors > 0:
  print "******", errors, "tests failed for ", " ".join(sys.argv[1:]), " ******"
  exit(1)

