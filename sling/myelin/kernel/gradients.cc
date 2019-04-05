// Copyright 2018 Google Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "sling/myelin/kernel/gradients.h"

#include <math.h>

namespace sling {
namespace myelin {

namespace {

// z = x + y
// dx = dz
// dy = dz
void add_grad(Flow::Operation *op, Gradients *g) {
  auto x = op->inputs[0];
  auto y = op->inputs[1];
  auto z = op->outputs[0];
  g->add(x, g->d(z));
  g->add(y, g->d(z));
}

// z = x - y
// dx = dz
// dy = -dz
void sub_grad(Flow::Operation *op, Gradients *g) {
  auto x = op->inputs[0];
  auto y = op->inputs[1];
  auto z = op->outputs[0];
  g->add(x, g->d(z));
  g->add(y, g->Neg(g->d(z)));
}

// z = x * y
// dx = dz * y
// dy = x * dz
void mul_grad(Flow::Operation *op, Gradients *g) {
  auto x = op->inputs[0];
  auto y = op->inputs[1];
  auto z = op->outputs[0];
  g->add(x, g->Mul(g->d(z), g->v(y)));
  g->add(y, g->Mul(g->v(x), g->d(z)));
}

// z = x * y
// dx = dz * y^T
// dy = x^T * dz
void matmul_grad(Flow::Operation *op, Gradients *g) {
  auto x = op->inputs[0];
  auto y = op->inputs[1];
  auto z = op->outputs[0];
  bool ta = op->GetAttr("transpose_a", false);
  bool tb = op->GetAttr("transpose_b", false);
  bool tc = op->GetAttr("transpose_c", false);

  if (tc) {
    // c^T = b^T a^T
    std::swap(x, y);
    std::swap(ta, tb);
    ta = !ta;
    tb = !tb;
  }

  if (tb) {
    g->add(x, g->MatMul(g->d(z), g->v(y)));
  } else {
    g->add(x, g->MatMul(g->d(z), g->Transpose(g->v(y))));
  }

  if (ta) {
    g->add(y, g->MatMul(g->v(x), g->d(z)));
  } else {
    g->add(y, g->MatMul(g->Transpose(g->v(x)), g->d(z)));
  }
}

// z = x / y
// dx = z / x * dz = dz / y
// dy = (-x / y^2) * dz = -z / y * dz
void div_grad(Flow::Operation *op, Gradients *g) {
  auto x = op->inputs[0];
  auto y = op->inputs[1];
  auto z = op->outputs[0];
  g->add(x, g->Div(g->d(z), g->v(y)));
  g->add(y, g->Mul(g->d(z), g->Div(g->Neg(g->v(z)), g->v(y))));
}

// y = x^2
// dx = 2 * x * dy
void square_grad(Flow::Operation *op, Gradients *g) {
  auto x = op->inputs[0];
  auto y = op->outputs[0];
  auto two = g->Two(g->v(x)->type);
  g->add(x, g->Mul(g->d(y), g->Mul(two, g->v(x))));
}

// y = sqrt(x)
// dx = dy / (2 sqrt(x)) = dy / 2y
void sqrt_grad(Flow::Operation *op, Gradients *g) {
  auto x = op->inputs[0];
  auto y = op->outputs[0];
  auto two = g->Two(g->v(y)->type);
  g->add(x, g->Div(g->d(y), g->Mul(two, g->v(y))));
}

// y = 1 / x
// dx = -dy / x^2 = -dy * y^2
void reciprocal_grad(Flow::Operation *op, Gradients *g) {
  auto x = op->inputs[0];
  auto y = op->outputs[0];
  g->add(x, g->Neg(g->Mul(g->d(y), g->Square(g->v(y)))));
}

// y = -x
// dx = -dy
void neg_grad(Flow::Operation *op, Gradients *g) {
  auto x = op->inputs[0];
  auto y = op->outputs[0];
  g->add(x, g->Neg(g->d(y)));
}

// y = |x|
// dx = sign(x) * dy
void abs_grad(Flow::Operation *op, Gradients *g) {
  auto x = op->inputs[0];
  auto y = op->outputs[0];
  auto zero = g->Zero(x->type);
  g->add(x, g->Cond(g->Less(g->v(x), zero), g->Neg(g->d(y)), g->d(y)));
}

// y = sign(x)
// dx = dy
void sign_grad(Flow::Operation *op, Gradients *g) {
  auto x = op->inputs[0];
  auto zero = g->Zero(x->type);
  g->add(x, zero);
}

// z = min(x, y)
// dx = (x<y)?x:0 * dz
// dy = (x<y)?0:y * dz
void minimum_grad(Flow::Operation *op, Gradients *g) {
  auto x = op->inputs[0];
  auto y = op->inputs[1];
  auto z = op->outputs[0];
  auto zero = g->Zero(z->type);
  auto test = g->Less(g->v(x),g->v(y));
  g->add(x, g->Cond(test, g->d(z), zero));
  g->add(y, g->Cond(test, zero, g->d(z)));
}

// z = max(x, y)
// dx = (x>y)?x:0 * dz
// dy = (x>y)?0:y * dz
void maximum_grad(Flow::Operation *op, Gradients *g) {
  auto x = op->inputs[0];
  auto y = op->inputs[1];
  auto z = op->outputs[0];
  auto zero = g->Zero(z->type);
  auto test = g->Greater(g->v(x),g->v(y));
  g->add(x, g->Cond(test, g->d(z), zero));
  g->add(y, g->Cond(test, zero, g->d(z)));
}

// y = sin(x)
// dx = cos(x) * dy
void sin_grad(Flow::Operation *op, Gradients *g) {
  auto x = op->inputs[0];
  auto y = op->outputs[0];
  g->add(x, g->Mul(g->d(y), g->Cos(g->v(x))));
}

// y = cos(x)
// dx = -sin(x) * dy
void cos_grad(Flow::Operation *op, Gradients *g) {
  auto x = op->inputs[0];
  auto y = op->outputs[0];
  g->add(x, g->Neg(g->Mul(g->d(y), g->Sin(g->v(x)))));
}

// y = exp(x)
// dx = exp(x) * dy = y * dy
void exp_grad(Flow::Operation *op, Gradients *g) {
  auto x = op->inputs[0];
  auto y = op->outputs[0];
  g->add(x, g->Mul(g->d(y), g->v(y)));
}

// y = log(x)
// dx = dy / x
void log_grad(Flow::Operation *op, Gradients *g) {
  auto x = op->inputs[0];
  auto y = op->outputs[0];
  g->add(x, g->Div(g->d(y), g->v(x)));
}

// y = sigmoid(x)
// dx = sigmoid(x) * (1 - sigmoid(x)) * dy = y * (1 - y) * dy
void sigmoid_grad(Flow::Operation *op, Gradients *g) {
  auto x = op->inputs[0];
  auto y = op->outputs[0];
  auto one = g->One(g->v(y)->type);
  g->add(x, g->Mul(g->d(y), g->Mul(g->v(y), g->Sub(one, g->v(y)))));
}

// y = tanh(x)
// dx = (1 - tanh(x)^2) * dy = (1 - y^2) * dy
void tanh_grad(Flow::Operation *op, Gradients *g) {
  auto x = op->inputs[0];
  auto y = op->outputs[0];
  auto one = g->One(g->v(y)->type);
  g->add(x, g->Mul(g->d(y), g->Sub(one, g->Square(g->v(y)))));
}

// y = erf(x)
// dx = 2/sqrt(pi) exp(-x^2) * dy
void erf_grad(Flow::Operation *op, Gradients *g) {
  auto x = op->inputs[0];
  auto y = op->outputs[0];
  Flow::Variable *c;
  if (x->type == DT_FLOAT) {
    float v = 2.0f / sqrtf(M_PI);
    c = g->Const(v);
  } else {
    double v = 2.0 / sqrt(M_PI);
    c = g->Const(v);
  }
  g->add(x, g->Mul(g->d(y), g->Mul(c, g->Exp(g->Neg(g->Square(g->v(x)))))));
}

// y = relu(x) = max(0, x)
// dx = (x > 0) * dy
void relu_grad(Flow::Operation *op, Gradients *g) {
  auto x = op->inputs[0];
  auto y = op->outputs[0];
  auto zero = g->Zero(g->v(x)->type);
  g->add(x, g->Select(g->Greater(g->v(x), zero), g->d(y)));
}

// y = norm(x) = sqrt(sum(square(x))) = |x|
// dx = x/|x| * dy = x/y * dy
void norm_grad(Flow::Operation *op, Gradients *g) {
  auto x = op->inputs[0];
  auto y = op->outputs[0];
  g->add(x, g->Mul(g->d(y), g->Div(g->v(x), g->v(y))));
}

// y = x
// dx = dy
void identity_grad(Flow::Operation *op, Gradients *g) {
  auto x = op->inputs[0];
  auto y = op->outputs[0];
  g->add(x, g->d(y));
}

// v = gather(M, f)
// dM = scatter(dv, f)
void gather_grad(Flow::Operation *op, Gradients *g) {
  auto M = op->inputs[0];
  auto f = op->inputs[1];
  auto v = op->outputs[0];
  g->add(M, g->Scatter(g->v(f), g->d(v), M->dim(0)));
}

// v = gather_sum(M, f)
// dM = scatter(dv, f)
void gathersum_grad(Flow::Operation *op, Gradients *g) {
  auto M = op->inputs[0];
  auto f = op->inputs[1];
  auto v = op->outputs[0];
  g->add(M, g->Scatter(g->v(f), g->d(v), M->dim(0)));
}

// v = concat(v_1, ..., v_n, axis)
// dv_i = slice(dv, begin_i, size_i)
void concat_grad(Flow::Operation *op, Gradients *g) {
  auto v = op->outputs[0];
  int N = op->GetAttr("N", 0);
  int axis;
  CHECK(op->inputs.back()->GetData(&axis));
  Shape begin;
  begin.redim(op->outputs[0]->rank());
  for (int i = 0; i < N; ++i) {
    auto vi = op->inputs[i];
    g->add(vi, g->Slice(g->d(v), g->Const(begin), vi->shape));
    begin.set(axis, begin.dim(axis) + vi->shape.dim(axis));
  }
}

// y = sum(x)
// dx = dy
void sum_grad(Flow::Operation *op, Gradients *g) {
  auto x = op->inputs[0];
  auto y = op->outputs[0];
  g->add(x, g->Broadcast(g->d(y), x->shape));
}

// y = min(x)
// dx = onehot(argmin(x), dy)
void min_grad(Flow::Operation *op, Gradients *g) {
  auto x = op->inputs[0];
  auto y = op->outputs[0];
  g->add(x, g->OneHot(g->ArgMin(g->v(x)), g->d(y), x->elements()));
}

// y = max(x)
// dx = onehot(argmax(x), dy)
void max_grad(Flow::Operation *op, Gradients *g) {
  auto x = op->inputs[0];
  auto y = op->outputs[0];
  g->add(x, g->OneHot(g->ArgMax(g->v(x)), g->d(y), x->elements()));
}

// y = x^T
// dx = dy^T
void transpose_grad(Flow::Operation *op, Gradients *g) {
  auto x = op->inputs[0];
  auto y = op->outputs[0];
  g->add(x, g->Transpose(g->d(y)));
}

// y = select(p, x)
// dx = select(p, dy)
void select_grad(Flow::Operation *op, Gradients *g) {
  auto p = op->inputs[0];
  auto x = op->inputs[1];
  auto y = op->outputs[0];
  g->add(x, g->Select(g->v(p), g->d(y)));
}

// z = cond(p, x, y)
// dx = select(p, dz)
// dy = select(not(p), dz)
void cond_grad(Flow::Operation *op, Gradients *g) {
  auto p = op->inputs[0];
  auto x = op->inputs[1];
  auto y = op->inputs[2];
  auto z = op->outputs[0];
  g->add(x, g->Select(g->v(p), g->d(z)));
  g->add(y, g->Select(g->Not(g->v(p)), g->d(z)));
}

} // namespace

void RegisterStandardGradients(Transformations *library) {
  library->RegisterGradient("Add", add_grad);
  library->RegisterGradient("Sub", sub_grad);
  library->RegisterGradient("Mul", mul_grad);
  library->RegisterGradient("MatMul", matmul_grad);
  library->RegisterGradient("Div", div_grad);
  library->RegisterGradient("Square", square_grad);
  library->RegisterGradient("Sqrt", sqrt_grad);
  library->RegisterGradient("Reciprocal", reciprocal_grad);
  library->RegisterGradient("Neg", neg_grad);
  library->RegisterGradient("Abs", abs_grad);
  library->RegisterGradient("Sign", sign_grad);
  library->RegisterGradient("Minimum", minimum_grad);
  library->RegisterGradient("Maximum", maximum_grad);
  library->RegisterGradient("Sin", sin_grad);
  library->RegisterGradient("Cos", cos_grad);
  library->RegisterGradient("Exp", exp_grad);
  library->RegisterGradient("Log", log_grad);
  library->RegisterGradient("Sigmoid", sigmoid_grad);
  library->RegisterGradient("Tanh", tanh_grad);
  library->RegisterGradient("Erf", erf_grad);
  library->RegisterGradient("Relu", relu_grad);
  library->RegisterGradient("Norm", norm_grad);
  library->RegisterGradient("Identity", identity_grad);
  library->RegisterGradient("Gather", gather_grad);
  library->RegisterGradient("GatherSum", gathersum_grad);
  library->RegisterGradient("ConcatV2", concat_grad);
  library->RegisterGradient("Sum", sum_grad);
  library->RegisterGradient("Min", min_grad);
  library->RegisterGradient("Max", max_grad);
  library->RegisterGradient("Transpose", transpose_grad);
  library->RegisterGradient("Select", select_grad);
  library->RegisterGradient("Cond", cond_grad);
}

}  // namespace myelin
}  // namespace sling

