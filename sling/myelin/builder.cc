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

Flow::Variable *FlowBuilder::RandomUniform(Variable *var) {
  var->init = Flow::Variable::INIT_UNIFORM;
  return var;
}

Flow::Variable *FlowBuilder::RandomNormal(Variable *var) {
  var->init = Flow::Variable::INIT_NORMAL;
  return var;
}

Flow::Variable *FlowBuilder::RandomOrtho(Variable *var) {
  var->init = Flow::Variable::INIT_ORTHO;
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

Flow::Operation *FlowBuilder::RawOp(const string &op,
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

Flow::Variable *FlowBuilder::Const(double value, Type type) {
  switch (type) {
    case DT_FLOAT: {
      float v = value;
      return Const(v);
    }
    case DT_DOUBLE: {
      double v = value;
      return Const(v);
    }
    case DT_INT64: {
      int64 v = value;
      return Const(&v, DT_INT64, {});
    }
    case DT_INT32: {
      int32 v = value;
      return Const(&v, DT_INT32, {});
    }
    case DT_INT16: {
      int16 v = value;
      return Const(&v, DT_INT16, {});
    }
    case DT_INT8: {
      int8 v = value;
      return Const(&v, DT_INT16, {});
    }
    default: LOG(FATAL) << "Constant type not supported";
  }
}

Flow::Variable *FlowBuilder::Zero(Type type) {
  switch (type) {
    case DT_FLOAT: return Const(0.0f);
    case DT_DOUBLE: return Const(0.0);
    default: return Const(nullptr, type, {});
  }
}

Flow::Variable *FlowBuilder::One(Type type) {
  switch (type) {
    case DT_FLOAT: return Const(1.0f);
    case DT_DOUBLE: return Const(1.0);
    case DT_INT32: return Const(1);
    default: LOG(FATAL) << "Constant type not supported";
  }
}

Flow::Variable *FlowBuilder::Two(Type type) {
  switch (type) {
    case DT_FLOAT: return Const(2.0f);
    case DT_DOUBLE: return Const(2.0);
    case DT_INT32: return Const(2);
    default: LOG(FATAL) << "Constant type not supported";
  }
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
  auto *concat = Op("Concat", args, parts[0]->type, shape);
  concat->producer->SetAttr("N", n);
  return concat;
}

std::vector<Flow::Variable *> FlowBuilder::Split(Variable *v, int splits,
                                                 int axis) {
  CHECK(v->dim(axis) % splits == 0)
    << "Cannot split " << v->shape.ToString() << " into " << splits
    << " parts along dimension " << axis;
  std::vector<Variable *> parts;
  Operation *op = RawOp("Split", {v, Const(splits), Const(axis)});
  Shape shape = v->shape;
  shape.set(axis, shape.dim(axis) / splits);
  for (int i = 0; i < splits; ++i) {
    string name = op->name + ":" + std::to_string(i);
    Variable *out = flow_->AddVariable(name, v->type, shape);
    op->AddOutput(out);
    parts.push_back(out);
  }
  return parts;
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
    RandomNormal(W);
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

}  // namespace myelin
}  // namespace sling

