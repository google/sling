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

#ifndef MYELIN_FLOW_H_
#define MYELIN_FLOW_H_

#include <string>
#include <unordered_map>
#include <vector>

#include "base/status.h"
#include "base/types.h"

namespace sling {
namespace myelin {

class Typer;

// Data types.
enum Type {
  DT_INVALID        = 0,      // invalid data type
  DT_FLOAT          = 1,      // 32-bit IEEE floating point number
  DT_DOUBLE         = 2,      // 64-bit IEEE floating point number
  DT_INT32          = 3,      // 32-bit signed integer
  DT_UINT8          = 4,      // 8-bit unsigned integer
  DT_INT16          = 5,      // 16-bit signed integer
  DT_INT8           = 6,      // 8-bit signed integer
  DT_STRING         = 7,      // string
  DT_COMPLEX64      = 8,      // single-precision complex
  DT_INT64          = 9,      // 64-bit signed integer
  DT_BOOL           = 10,     // boolean
  DT_QINT8          = 11,     // quantized 8-bit signed integer
  DT_QUINT8         = 12,     // quantized 8-bit unsigned integer
  DT_QINT32         = 13,     // quantized 32-bit signed integer
  DT_BFLOAT16       = 14,     // float32 truncated to 16 bits
  DT_QINT16         = 15,     // quantized 16-bit signed integer
  DT_QUINT16        = 16,     // quantized 16-bit unsigned integer
  DT_UINT16         = 17,     // 16-bit unsigned integer
  DT_COMPLEX128     = 18,     // double-precision complex
  DT_HALF           = 19,     // 16-bit IEEE floating point number
  DT_RESOURCE       = 20,     // resource
};

// Type properties.
class TypeTraits {
 public:
  TypeTraits(Type type, const char *name, int size)
      : type_(type), name_(name), size_(size) {}

  Type type() const { return type_; }
  const string &name() const { return name_; }
  int size() const { return size_; }
  bool valid() const { return type_ != DT_INVALID; }
  string str(void *data) const;

  // Look up traits from type code.
  static const TypeTraits &of(Type type);

  // Look up traits from type name.
  static const TypeTraits &of(string &name);

 private:
  Type type_;
  string name_;
  int size_;
};

// Look up traits from type.
template<typename T> inline const TypeTraits &Traits();
template<> inline const TypeTraits &Traits<float>() {
  return TypeTraits::of(DT_FLOAT);
}
template<> inline const TypeTraits &Traits<double>() {
  return TypeTraits::of(DT_DOUBLE);
}
template<> inline const TypeTraits &Traits<int32_t>() {
  return TypeTraits::of(DT_INT32);
}
template<> inline const TypeTraits &Traits<uint8_t>() {
  return TypeTraits::of(DT_UINT8);
}
template<> inline const TypeTraits &Traits<int16_t>() {
  return TypeTraits::of(DT_INT16);
}
template<> inline const TypeTraits &Traits<int8_t>() {
  return TypeTraits::of(DT_INT8);
}
template<> inline const TypeTraits &Traits<int64_t>() {
  return TypeTraits::of(DT_INT64);
}
template<> inline const TypeTraits &Traits<int64>() {
  return TypeTraits::of(DT_INT64);
}
template<> inline const TypeTraits &Traits<bool>() {
  return TypeTraits::of(DT_BOOL);
}
template<> inline const TypeTraits &Traits<uint16_t>() {
  return TypeTraits::of(DT_UINT16);
}

// Flow graph transformations.
class Transformations {
 public:
  ~Transformations();

  // Combination of operations that can be replaced with a combined operation.
  struct Combination {
    Combination(const string &first,
                const string &second,
                const string &replacement)
        : first(first), second(second), replacement(replacement) {}
    string first;
    string second;
    string replacement;
  };

  // Register identity operation.
  void RegisterIdentityOp(const string &noop) {
    noops_.push_back(noop);
  }

  // Register operation combination.
  void RegisterCombinedOp(const string &first,
                          const string &second,
                          const string &replacement) {
    combinations_.emplace_back(first, second, replacement);
  }

  // Register type inference component. Transfers ownership from caller.
  void RegisterTyper(Typer *typer) {
    typers_.emplace_back(typer);
  }

  // Identity operations.
  const std::vector<string> &noops() const { return noops_; }

  // Pairs of operations that can be combined.
  const std::vector<Combination> &combinations() const { return combinations_; }

  // Type inference components.
  const std::vector<Typer *> typers() const { return typers_; }

 private:
  // Identity operations.
  std::vector<string> noops_;

  // Pairs of operations that can be combined.
  std::vector<Combination> combinations_;

  // Type inference components.
  std::vector<Typer *> typers_;
};

// Tensor shape.
class Shape {
 public:
  Shape() {}
  Shape(const Shape &shape) : dims_(shape.dims_) {}
  Shape(const std::vector<int> &dims) : dims_(dims) {}
  Shape(std::initializer_list<int> dims) : dims_(dims) {}

  // Assignment.
  void operator=(const Shape &other) { dims_ = other.dims_; }

  // Clear all dimensions.
  void clear() { dims_.clear(); }

  // Change the number of dimensions.
  void redim(int rank) { dims_.resize(rank); }

  // Set size of dimension.
  void set(int d, int size) { dims_[d] = size; }

  // Assign shape for vector or matrix.
  void assign(int d1) { clear(); add(d1); }
  void assign(int d1, int d2) { clear(); add(d1); add(d2); }

  // Set all dimensions to a certain size.
  void fill(int size) { dims_.assign(rank(), size); }
  void fill(int rank, int size) { dims_.assign(rank, size); }

  // Add dimension to shape.
  void add(int size) { dims_.push_back(size); }

  // Return the rank of the shape, i.e. the number of dimensions.
  int rank() const  { return dims_.size(); }

  // Check for scalar.
  bool scalar() const { return rank() == 0; }

  // Return size of dimension.
  int dim(int d) const { return dims_[d]; }

  // Return the total number of elements.
  int elements() const {
    int n = 1;
    for (int d : dims_) {
      if (d == -1) return -1;
      n *= d;
    }
    return n;
  }

  // Check for undefined shape, i.e. some dimensions have zero size.
  bool undefined() const { return elements() == 0; }

  // Check for partial shape, i.e. some dimensions have unspecifed (-1) size.
  bool partial() const { return elements() == 0; }

  // Check if shape is the same as another shape.
  bool IsSameSize(const Shape &other) const;
  bool operator==(const Shape &other) const { return IsSameSize(other); }
  bool operator!=(const Shape &other) const { return !IsSameSize(other); }

  // Return shape as string.
  string ToString() const;

 private:
  // Size of each dimension.
  std::vector<int> dims_;
};

// Flow graph for computation.
class Flow {
 public:
  struct Operation;
  struct Function;

  // Flow variable.
  struct Variable {
    // Add alias for variable.
    void AddAlias(const string &alias);

    // Return the rank of the variable tensor.
    int rank() const { return shape.rank(); }

    // Return size of dimension.
    int dim(int d) const { return shape.dim(d); }

    // Return the number of elements in the variable tensor.
    int elements() const { return shape.elements(); }

    // Return type as string.
    string TypeString() const;

    // Set data for variable. The storage is not owned by the variable.
    void SetData(void *buffer, int len) {
      data = static_cast<char *>(buffer);
      size = len;
    }

    string name;
    std::vector<string> aliases;

    Type type = DT_INVALID;
    bool ref = false;
    Shape shape;
    char *data = nullptr;
    uint64_t size = 0;
    bool function_input = false;
    bool function_output = false;

    Operation *producer = nullptr;
    std::vector<Operation *> consumers;
  };

  // Operation attribute.
  struct Attribute {
    Attribute(const string &n, const string &v) : name(n), value(v) {}
    string name;
    string value;
  };

  // Flow operation.
  struct Operation {
    // Add input to operation.
    void AddInput(Variable *var);

    // Add output to operation.
    void AddOutput(Variable *var);

    // Get attribute value.
    const string &GetAttr(const string &name);
    int GetAttr(const string &name, int defval);

    // Return in and out degree.
    int indegree() const { return inputs.size(); }
    int outdegree() const { return outputs.size(); }

    string name;
    string type;
    std::vector<Variable *> inputs;
    std::vector<Variable *> outputs;
    std::vector<Attribute> attrs;
    Function *func = nullptr;

    int task = 0;
    int priority = 3;
    int order = -1;
    int missing = 0;
  };

  // Flow function.
  struct Function {
    // Add operation to function.
    void AddOperation(Operation *op);

    string name;
    std::vector<Operation *> ops;
  };

  // Flow connector.
  struct Connector {
    // Add operation to function.
    void AddLink(Variable *var);

    string name;
    std::vector<Variable *> links;
  };

  Flow();
  ~Flow();

  // Load flow from file.
  Status Load(const string &filename);

  // Analyze flow.
  void Analyze(const Transformations &transformations);

  // Infer which variables are inputs and outputs to functions.
  void InferInputsAndOutputs();

  // Apply transformations to flow graph.
  void Transform(const Transformations &transformations);

  // Combine two op types to a single combined op type.
  void Combine(const string &first,
               const string &second,
               const string &combined);

  // Remove operation from flow.
  void Eliminate(Operation *op);

  // Merge two operations into a combined op.
  Operation *Merge(Operation *first,
                   Operation *second,
                   const string &combined);

  // Sort operations in topological order of computation.
  void Sort();

  // Infer types of variables. Return false if some variables are unresolved.
  bool InferTypes(const Transformations &transformations);

  // Add variable.
  Variable *AddVariable(const string &name,
                        Type type,
                        const Shape &shape);

  // Add operation.
  Operation *AddOperation(const string &name, const string &type);

  // Add operation to flow function.
  Operation *AddOperation(Function *func,
                          const string &name,
                          const string &type);

  // Add function.
  Function *AddFunction(const string &name);

  // Add connector.
  Connector *AddConnector(const string &name);

  // Delete variable.
  void DeleteVariable(Variable *var);

  // Delete operation.
  void DeleteOperation(Operation *op);

  // Look up variable by name.
  Variable *Var(const string &name);

  // Look up operation by name.
  Operation *Op(const string &name);

  // Look up function by name.
  Function *Func(const string &name);

  // Return flow in text format.
  string ToString() const;

  // Return all variables.
  const std::vector<Variable *> &vars() const { return vars_; }

  // Return all operations.
  const std::vector<Operation *> &ops() const { return ops_; }

  // Return all functions.
  const std::vector<Function *> &funcs() const { return funcs_; }

  // Return all connectors.
  const std::vector<Connector *> &cnxs() const { return cnxs_; }

  // Batch size.
  int batch_size() const { return batch_size_; }
  void set_batch_size(int batch_size) { batch_size_ = batch_size; }

 private:
  // Variables.
  std::vector<Variable *> vars_;

  // Operations.
  std::vector<Operation *> ops_;

  // Functions.
  std::vector<Function *> funcs_;

  // Connectors.
  std::vector<Connector *> cnxs_;

  // Data areas owned by flow.
  std::vector<char *> memory_;

  // Batch size.
  int batch_size_ = -1;
};

// Component type for inferring types and shapes of operation outputs.
class Typer {
 public:
  virtual ~Typer() = default;

  // Return true if the type of the outputs of the operations has been
  // inferred.
  virtual bool InferTypes(Flow::Operation *op) = 0;
};

}  // namespace myelin
}  // namespace sling

#endif  // MYELIN_FLOW_H_

