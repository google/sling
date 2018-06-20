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

#include "sling/myelin/builder.h"

namespace sling {
namespace myelin {

Scope::Scope(Scope *parent, const string &name) : parent_(parent) {
  if (parent == nullptr) {
    // Top-level scope.
    root_ = current_ = this;
    name_ = name;
  } else {
    // Inner scope.
    root_ = parent_->root_;
    CHECK(root_->current_ == parent_);
    root_->current_ = current_ = this;
    name_ = parent_->name_ + "/" + name;
  }
}

Scope::~Scope() {
  CHECK(root_->current_ == this);
  root_->current_ = parent_;
}

string Scope::OpName(const string &op) {
  string name = current_->name_;
  name.push_back('/');
  name.append(op);
  int num = opnum_[op]++;
  if (num > 0) {
    name.push_back('_');
    name.append(std::to_string(num));
  }
  return name;
}

Flow::Variable *FlowBuilder::Var(const string &name, Type type,
                                 const Shape &shape) {
  return flow_->AddVariable(prefix() + "/" + name, type, shape);
}

Flow::Variable *FlowBuilder::Parameter(const string &name,
                                       Type type,
                                       const Shape &shape) {
  Variable *var = Var(name, type, shape)->set_learnable();
  return var;
}

Flow::Variable *FlowBuilder::Placeholder(const string &name,
                                         Type type,
                                         const Shape &shape,
                                         bool ref) {
  Variable *input = Var(name, type, shape)->set_in();
  if (ref) input->set_ref();
  return input;
}

Flow::Variable *FlowBuilder::Name(Variable *var, const string &name) {
  var->name = prefix() + "/" + name;
  return var;
}

Flow::Variable *FlowBuilder::Op(const string &op,
                                const std::vector<Flow::Variable *> &args,
                                Type type,
                                const Shape &shape) {
  string name = OpName(op);
  Variable *result = flow_->AddVariable(name + ":0", type, shape);
  flow_->AddOperation(func_, name, op, args, {result});
  return result;
}

Flow::Variable *FlowBuilder::Op(const string &op,
                                const std::vector<Flow::Variable *> &args) {
  // Use first argument for return type.
  Type type = args.empty() ? DT_INVALID : args[0]->type;

  // Determine output rank.
  Shape shape;
  int rank = 0;
  for (Flow::Variable *arg : args) {
    if (arg->rank() > rank) rank = arg->rank();
  }
  shape.fill(rank, 1);

  // Determine output shape based on broadcast semantics.
  for (Flow::Variable *arg : args) {
    int depth = rank - arg->rank();
    for (int d = 0; d < arg->rank(); ++d) {
      if (shape.dim(d + depth) < arg->dim(d)) {
        shape.set(d + depth, arg->dim(d));
      }
    }
  }

  return Op(op, args, type, shape);
}

Flow::Operation *FlowBuilder::Op0(const string &op,
                                  const std::vector<Flow::Variable *> &args) {
  string name = OpName(op);
  return flow_->AddOperation(func_, name, op, args, {});
}

Flow::Variable *FlowBuilder::Const(const void *data, Type type,
                                   const Shape &shape) {
  Variable *var = flow_->AddVariable(OpName("const"), type, shape);
  var->size = TypeTraits::of(type).size() * shape.elements();
  char *buffer = flow_->AllocateMemory(var->size);
  var->data = buffer;
  if (data != nullptr) {
    memcpy(buffer, data, var->size);
  } else {
    memset(buffer, 0, var->size);
  }
  return var;
}

Flow::Variable *FlowBuilder::Instance(Function *func) {
  Variable *instance = Var(func->name, DT_RESOURCE, {});
  instance->set_ref();
  return instance;
}

Flow::Variable *FlowBuilder::MatMul(Variable *x, Variable *y) {
  Variable *result = Op("MatMul", {x, y});
  if (x->rank() == 2 && y->rank() == 2) {
    result->shape = Shape({x->dim(0), y->dim(1)});
  }
  return result;
}

Flow::Variable *FlowBuilder::Ref(Variable *instance, Variable *external) {
  Variable *ref = Op("Reference", {instance});
  ref->type = external->type;
  ref->shape = external->shape;
  ref->set_ref();
  ref->producer->SetAttr("var", external->name);
  return ref;
}

Flow::Variable *FlowBuilder::Concat(const std::vector<Variable *> &parts,
                                    int axis) {
  CHECK(!parts.empty());
  Shape shape = parts[0]->shape;
  int n = parts.size();
  int width = 0;
  for (Variable *v : parts) width += v->shape.dim(axis);
  shape.set(axis, width);
  std::vector<Variable *> args = parts;
  args.push_back(Const(axis));
  auto *concat = Op("ConcatV2", args, parts[0]->type, shape);
  concat->producer->SetAttr("N", n);
  return concat;
}

Flow::Variable *FlowBuilder::FFLayers(Variable *input,
                                      std::vector<int> layers,
                                      int hidden,
                                      bool bias,
                                      const string &activation) {
  Variable *v = input;
  for (int l = 0; l < layers.size(); ++l) {
    // Get dimensions for next layer.
    Type type = v->type;
    int height = v->dim(1);
    int width = layers[l];

    // Add weight matrix.
    auto *W = Parameter("W" + std::to_string(l), type, {height, width});
    v = MatMul(v, W);

    // Optionally add bias.
    if (bias) {
      auto *b = Parameter("b" + std::to_string(l), type, {1, width});
      v = Add(v, b);
    }

    // Add activation function between layers.
    if (l != layers.size() - 1) {
      v = Op(activation, {v});
    }

    // Make hidden layer a reference.
    if (l == hidden) {
      v = Name(Identity(v), "hidden");
      v->set_in()->set_out()->set_ref();
    }
  }

  auto *logits = Name(Identity(v), "logits");
  return logits;
}

Flow::Variable *FlowBuilder::LSTMLayer(Variable *input, int size) {
  // Get LSTM dimensions.
  Type type = input->type;
  int input_dim = input->dim(1);

  // Define parameters.
  auto *x2i = Parameter("x2i", type, {input_dim, size});
  auto *h2i = Parameter("h2i", type, {size, size});
  auto *c2i = Parameter("c2i", type, {size, size});
  auto *bi = Parameter("bi", type, {1, size});

  auto *x2o = Parameter("x2o", type, {input_dim, size});
  auto *h2o = Parameter("h2o", type, {size, size});
  auto *c2o = Parameter("c2o", type, {size, size});
  auto *bo = Parameter("bo", type, {1, size});

  auto *x2c = Parameter("x2c", type, {input_dim, size});
  auto *h2c = Parameter("h2c", type, {size, size});
  auto *bc = Parameter("bc", type, {1, size});

  // Channels -- h_in, c_in = h_{t-1}, c_{t-1}
  auto *h_in = Placeholder("h_in", type, {1, size}, true);
  auto *c_in = Placeholder("c_in", type, {1, size}, true);

  // Input -- i_t = sigmoid(x_t * x2i + h_in * h2i + c_in * c2i + bi)
  auto *i_ait = Name(Add(MatMul(input, x2i),
                     Add(MatMul(h_in, h2i),
                     Add(MatMul(c_in, c2i), bi))),
                     "i_ait");
  auto *i_it = Name(Sigmoid(i_ait), "i_it");

  // Forget -- f_t = 1 - i_t
  auto *i_ft = Name(Sub(One(), i_it), "i_ft");

  // Memory -- tanh(x_t * x2c + h_in * h2c + h_in * h2c + bc)
  auto *i_awt = Name(Add(MatMul(input, x2c),
                     Add(MatMul(h_in, h2c), bc)),
                     "i_awt");
  auto *i_wt = Name(Tanh(i_awt), "i_wt");

  // Control -- c_out = c_t = i_t * w_t + f_t * c_in
  auto *c_out = Name(Add(Mul(i_it, i_wt), Mul(i_ft, c_in)), "c_out");
  c_out->set_out()->set_ref();

  // Output -- o_t = sigmoid(x_t * x2o + c_t * c2o + h_in * h2o + bo)
  auto *i_aot = Name(Add(MatMul(input, x2o),
                     Add(MatMul(c_out, c2o),
                     Add(MatMul(h_in, h2o), bo))),
                     "i_aot");
  auto *i_ot = Name(Sigmoid(i_aot), "i_ot");

  // Hidden -- h_out = h_t = o_t * tanh(c_out)
  auto *h_out = Name(Mul(i_ot, Tanh(c_out)), "h_out");
  h_out->set_out()->set_ref();

  // Connectors for hidden and control channels.
  flow_->Connect({h_in, h_out});
  flow_->Connect({c_in, c_out});

  // The control channel has a single-source gradient.
  c_in->set_unique();

  return h_out;
}

}  // namespace myelin
}  // namespace sling

