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

"""Numerical gradient checking."""

import sling
import sling.myelin as myelin
import sling.flags as flags
import numpy as np
import math

flags.define("--fp64", default=False, action='store_true')

flags.parse()
compiler = myelin.Compiler()

shape = [16]
dtype = myelin.DT_FLOAT
nptype = np.float32
if flags.arg.fp64:
  dtype = "float64"
  nptype = np.float64

# Compute number of elements in shape.
def elements(shape):
  n = 1
  for d in shape: n *= d
  return n

# Make one-hot tensor with linear indexing.
def onehot(shape, index, value):
  v = np.zeros(elements(shape)).astype(nptype)
  v[index] = value
  return v.reshape(shape)

# Get variable name for adjoint.
def adjointvar(v):
  name = v.name
  slash = name.find('/')
  if slash == -1: return "gradients/d_" + name
  return "gradients/" + name[:slash] + "/d_" + name[slash + 1:];

# Get variable name for primal.
def primalvar(f):
  return "gradients/" + f.name + "/primal"

# Check gradient by comparing analytical and numerical derivatives.
def gradcheck(f, inputs, outputs, lo=-10.0, hi=10.0, eps=1e-3, tol=1e-4):
  # Get function from flow builder.
  flow = f.flow
  func = f.func

  # Mark inputs and outputs.
  for v in inputs:
    v.input = True
  for v in outputs:
    v.output = True

  # Enable backprop for function to compute gradient.
  func.backprop = True

  # Compile flow.
  net = compiler.compile(flow)
  cell = net.cell(func.name)
  gcell = net.cell("gradients/" + func.name)
  if primalvar(func) in gcell:
    primal = gcell.index(primalvar(func))
  else:
    primal = -1

  # Choose random input point for evaluating gradient.
  x = {}
  for v in inputs:
    x[v] = np.random.ranf(v.shape).astype(nptype) * (hi - lo) + lo

  # Compute f(x).
  data = cell.instance()
  for v in inputs: data[v] = x[v]
  data.compute()

  # Check gradient for each output variable element.
  minulp = None
  for output in outputs:
    # Get adjoint for output variable.
    if adjointvar(output) in gcell:
      adjoint = gcell.index(adjointvar(output))
    else:
      adjoint = -1

    for j in xrange(output.elements()):
      # Compute analytical gradient.
      gdata = gcell.instance()
      if primal != -1: gdata[primal] = data
      if adjoint != -1: gdata.tensor(adjoint)[j] = 1.0
      gdata.compute()

      # Check gradient for each input variable element.
      for input in inputs:
        gradient = gdata.tensor(adjointvar(input))
        for i in xrange(input.elements()):
          # Construct one-hot tensor with x_i set to epsilon.
          delta = onehot(input.shape, i, eps)

          # Compute f(x-delta) and f(x+delta).
          plus = cell.instance()
          minus = cell.instance()
          for v in inputs:
            if v == input:
              plus[v] = x[v] + delta
              minus[v] = x[v] - delta
            else:
              plus[v] = x[v]
              minus[v] = x[v]
          plus.compute()
          minus.compute()

          # Compute numerical estimate of gradient using small finite
          # difference:
          #
          # df_j       f_j(x+eps) - f_j(x-eps)
          # ---- (x) ~ -----------------------
          #  d_i                2*eps
          #
          fplus = plus.tensor(output)[j]
          fminus = minus.tensor(output)[j]
          numerical = (fplus - fminus) / (2 * eps)

          # Compare numerical gradient with analytical gradient.
          analytical = gradient[i]
          error = abs(analytical - numerical)
          deviation = error / (1 + abs(analytical))
          if deviation != 0.0:
            ulp = -int(math.log10(deviation))
            if minulp == None or ulp < minulp: minulp = ulp
          if not np.isclose(numerical, analytical, rtol=tol, atol=tol):
            print "%s: d%s_%d / d%s_%d: %g vs %g dev=%g ulp=%d" % (
              func.name, output.name, j, input.name, i,
              analytical, numerical, deviation, ulp)

  # Return the minimum unit of least precision.
  print func.name, ":", minulp, "ulp"
  return minulp

def check_add():
  flow = myelin.Flow()
  f = flow.define("add")
  x1 = f.var("x1", dtype, shape)
  x2 = f.var("x2", dtype, shape)
  y = f.add(x1, x2, "y")
  gradcheck(f, [x1, x2], [y], tol=1e-3)

def check_sub():
  flow = myelin.Flow()
  f = flow.define("sub")
  x1 = f.var("x1", dtype, shape)
  x2 = f.var("x2", dtype, shape)
  y = f.sub(x1, x2, "y")
  gradcheck(f, [x1, x2], [y], tol=1e-3)

def check_mul():
  flow = myelin.Flow()
  f = flow.define("mul")
  x1 = f.var("x1", dtype, shape)
  x2 = f.var("x2", dtype, shape)
  y = f.mul(x1, x2, "y")
  gradcheck(f, [x1, x2], [y], tol=1e-3)

def check_div():
  flow = myelin.Flow()
  f = flow.define("div")
  x1 = f.var("x1", dtype, shape)
  x2 = f.var("x2", dtype, shape)
  y = f.div(x1, x2, "y")
  gradcheck(f, [x1, x2], [y], tol=1e-3)

def check_minimum():
  flow = myelin.Flow()
  f = flow.define("minimum")
  x1 = f.var("x1", dtype, shape)
  x2 = f.var("x2", dtype, shape)
  y = f.minimum(x1, x2, "y")
  gradcheck(f, [x1, x2], [y], tol=1e-3)

def check_maximum():
  flow = myelin.Flow()
  f = flow.define("maximum")
  x1 = f.var("x1", dtype, shape)
  x2 = f.var("x2", dtype, shape)
  y = f.maximum(x1, x2, "y")
  gradcheck(f, [x1, x2], [y], tol=1e-3)

def check_square():
  flow = myelin.Flow()
  f = flow.define("square")
  x = f.var("x", dtype, shape)
  y = f.square(x, "y")
  gradcheck(f, [x], [y], tol=1e-3)

def check_sqrt():
  flow = myelin.Flow()
  f = flow.define("sqrt")
  x = f.var("x", dtype, shape)
  y = f.sqrt(x, "y")
  gradcheck(f, [x], [y], lo=0.0)

def check_rcp():
  flow = myelin.Flow()
  f = flow.define("rcp")
  x = f.var("x", dtype, shape)
  y = f.rcp(x, "y")
  gradcheck(f, [x], [y], tol=1e-3)

def check_neg():
  flow = myelin.Flow()
  f = flow.define("neg")
  x = f.var("x", dtype, shape)
  y = f.rcp(x, "y")
  gradcheck(f, [x], [y])

def check_abs():
  flow = myelin.Flow()
  f = flow.define("abs")
  x = f.var("x", dtype, shape)
  y = f.abs(x, "y")
  gradcheck(f, [x], [y], tol=1e-3)

def check_sign():
  flow = myelin.Flow()
  f = flow.define("sign")
  x = f.var("x", dtype, shape)
  y = f.sign(x, "y")
  gradcheck(f, [x], [y])

def check_exp():
  flow = myelin.Flow()
  f = flow.define("exp")
  x = f.var("x", dtype, shape)
  y = f.exp(x, "y")
  gradcheck(f, [x], [y], -3.0, 3.0)

def check_log():
  flow = myelin.Flow()
  f = flow.define("log")
  x = f.var("x", dtype, shape)
  y = f.log(x, "y")
  gradcheck(f, [x], [y], 0.0, 10.0)

def check_tanh():
  flow = myelin.Flow()
  f = flow.define("tanh")
  x = f.var("x", dtype, shape)
  y = f.tanh(x, "y")
  gradcheck(f, [x], [y])

def check_sigmoid():
  flow = myelin.Flow()
  f = flow.define("sigmoid")
  x = f.var("x", dtype, shape)
  y = f.sigmoid(x, "y")
  gradcheck(f, [x], [y])

def check_erf():
  flow = myelin.Flow()
  f = flow.define("erf")
  x = f.var("x", dtype, shape)
  y = f.erf(x, "y")
  gradcheck(f, [x], [y])

def check_relu():
  flow = myelin.Flow()
  f = flow.define("relu")
  x = f.var("x", dtype, shape)
  y = f.relu(x, "y")
  gradcheck(f, [x], [y], lo=-1.0, hi=1.0)

def check_norm():
  flow = myelin.Flow()
  f = flow.define("norm")
  x = f.var("x", dtype, shape)
  y = f.norm(x, "y")
  gradcheck(f, [x], [y], eps=1e-2)

def check_normalize():
  flow = myelin.Flow()
  f = flow.define("normalize")
  x = f.var("x", dtype, shape)
  y = f.normalize(x, "y")
  gradcheck(f, [x], [y], tol=1e-2)

def check_softmax():
  flow = myelin.Flow()
  f = flow.define("softmax")
  x = f.var("x", dtype, shape)
  y = f.softmax(x, "y")
  gradcheck(f, [x], [y])

def check_sum():
  flow = myelin.Flow()
  f = flow.define("sum")
  x = f.var("x", dtype, shape)
  y = f.sum(x, "y")
  gradcheck(f, [x], [y], tol=1e-3)

def check_max():
  flow = myelin.Flow()
  f = flow.define("max")
  x = f.var("x", dtype, shape)
  y = f.max(x, "y")
  gradcheck(f, [x], [y])

def check_min():
  flow = myelin.Flow()
  f = flow.define("min")
  x = f.var("x", dtype, shape)
  y = f.min(x, "y")
  gradcheck(f, [x], [y])

def check_matmul():
  flow = myelin.Flow()
  f = flow.define("matmul")
  x1 = f.var("x1", dtype, [8, 16])
  x2 = f.var("x2", dtype, [16, 8])
  y = f.matmul(x1, x2, "y")
  gradcheck(f, [x1, x2], [y], eps=1e-2, tol=1e-2)

def check_bcast_add():
  flow = myelin.Flow()
  f = flow.define("bcast_add")
  x1 = f.var("x1", dtype, shape)
  x2 = f.var("x2", dtype)
  y = f.add(x1, x2, "y")
  gradcheck(f, [x1, x2], [y], tol=1e-3)

def check_bcast_mul():
  flow = myelin.Flow()
  f = flow.define("bcast_mul")
  x1 = f.var("x1", dtype, shape)
  x2 = f.var("x2", dtype)
  y = f.mul(x1, x2, "y")
  gradcheck(f, [x1, x2], [y], tol=1e-3)

# Test gradients for all functions.
check_add()
check_sub()
check_mul()
check_minimum()
check_maximum()
check_div()
check_square()
check_sqrt()
check_rcp()
check_neg()
check_abs()
check_sign()
check_exp()
check_log()
check_tanh()
check_sigmoid()
check_erf()
check_relu()
check_norm()
check_normalize()
check_sum()
if not flags.arg.fp64:
  check_max()
  check_min()
  check_softmax()
check_matmul()
check_bcast_add()
check_bcast_mul()

