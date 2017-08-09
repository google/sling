// Copyright 2017 Google Inc.
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

#ifndef MYELIN_BUILDER_H_
#define MYELIN_BUILDER_H_

#include <map>
#include <vector>

#include "base/types.h"
#include "myelin/flow.h"

namespace sling {
namespace myelin {

// Flow builder utility for building flows from expressions, e.g.:
//   Flow flow;
//   Builder tf(&flow, "mnist");
//   auto *w = tf.Constant(weights, DT_FLOAT, {784, 10});
//   auto *b = tf.Constant(bias, DT_FLOAT, {10});
//   auto *x = tf.Var("x", DT_FLOAT, {1, 784});
//   auto *y = tf.Add(tf.MatMul(x, w), b);
class Builder {
 public:
  // Flow typedefs.
  typedef Flow::Variable Variable;
  typedef Flow::Operation Operation;
  typedef Flow::Function Function;

  // Initialize builder for existing function.
  Builder(Flow *flow, Function *func) : flow_(flow), func_(func) {}

  // Initialize builder for new function.
  Builder(Flow *flow, const string &name) : flow_(flow) {
    func_ = flow->AddFunction(name);
  }

  // Add variable to flow.
  Variable *Var(const string &name, Type type, const Shape &shape);

  // Add operation to function and return output variable.
  Variable *Op(const string &op, const std::vector<Variable *> &args);

  // Add constant to flow.
  Variable *Constant(const void *data, Type type, const Shape &shape);
  Variable *Constant(float value) { return Constant(&value, DT_FLOAT, {}); }
  Variable *Constant(int value) { return Constant(&value, DT_INT32, {}); }
  Variable *Constant(std::vector<float> &value) {
    int size = value.size();
    return Constant(value.data(), DT_FLOAT, {size});
  }
  Variable *Constant(std::vector<int> &value) {
    int size = value.size();
    return Constant(value.data(), DT_INT32, {size});
  }

  // Builder methods for common operations.
  Variable *Add(Variable *x, Variable *y) { return Op("Add", {x, y}); }
  Variable *Sub(Variable *x, Variable *y) { return Op("Sub", {x, y}); }
  Variable *Mul(Variable *x, Variable *y) { return Op("Mul", {x, y}); }
  Variable *Div(Variable *x, Variable *y) { return Op("Div", {x, y}); }
  Variable *Min(Variable *x, Variable *y) { return Op("Min", {x, y}); }
  Variable *Max(Variable *x, Variable *y) { return Op("Min", {x, y}); }
  Variable *MatMul(Variable *x, Variable *y) { return Op("MatMul", {x, y}); }
  Variable *Log(Variable *x) { return Op("Log", {x}); }
  Variable *Exp(Variable *x) { return Op("Exp", {x}); }
  Variable *Tanh(Variable *x) { return Op("Tanh", {x}); }
  Variable *Sigmoid(Variable *x) { return Op("Sigmoid", {x}); }
  Variable *Relu(Variable *x) { return Op("Relu", {x}); }

  Variable *Reshape(Variable *x, Variable *shape) {
    return Op("Reshape", {x, shape});
  }

  // Return unique name for operation.
  string OpName(const string &op);

  // Return function for builder.
  Flow::Function *func() const { return func_; }

 private:
  // Flow for builder.
  Flow *flow_;

  // Function for builder.
  Function *func_;

  // Next unused operation number for each operation type.
  std::map<string, int> opnum_;
};

}  // namespace myelin
}  // namespace sling

#endif  // MYELIN_BUILDER_H_

