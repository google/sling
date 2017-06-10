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
class Transformer;

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
  TypeTraits(Type type, const char *name, int size, const char *ptx)
      : type_(type), name_(name), size_(size), ptx_(ptx) {}

  Type type() const { return type_; }
  const string &name() const { return name_; }
  size_t size() const { return size_; }
  bool valid() const { return type_ != DT_INVALID; }
  const char *ptx() const { return ptx_; }
  string str(void *data) const;

  // Look up traits from type code.
  static const TypeTraits &of(Type type);

  // Look up traits from type name.
  static const TypeTraits &of(string &name);

 private:
  Type type_;        // basic type
  string name_;      // type name
  size_t size_;      // size in bytes
  const char *ptx_;  // ptx type
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
    string first;        // first operation
    string second;       // second operation
    string replacement;  // replacement operation
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

  // Register flow transformation component. Transfers ownership from caller.
  void RegisterTransformer(Transformer *transformer) {
    transformers_.emplace_back(transformer);
  }

  // Register type inference component. Transfers ownership from caller.
  void RegisterTyper(Typer *typer) {
    typers_.emplace_back(typer);
  }

  // Identity operations.
  const std::vector<string> &noops() const { return noops_; }

  // Pairs of operations that can be combined.
  const std::vector<Combination> &combinations() const { return combinations_; }

  // Flow transformation components.
  const std::vector<Transformer *> transformers() const {
    return transformers_;
  }

  // Type inference components.
  const std::vector<Typer *> typers() const { return typers_; }

 private:
  // Identity operations.
  std::vector<string> noops_;

  // Pairs of operations that can be combined.
  std::vector<Combination> combinations_;

  // Flow transformation components.
  std::vector<Transformer *> transformers_;

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
  bool partial() const { return elements() == -1; }

  // Check if shape is the same as another shape. Undefined dimensions are
  // not compared.
  bool IsSameSize(const Shape &other) const;
  bool operator==(const Shape &other) const { return IsSameSize(other); }
  bool operator!=(const Shape &other) const { return !IsSameSize(other); }

  // Return shape as string.
  string ToString() const;

 private:
  // Size of each dimension.
  std::vector<int> dims_;
};

// Attribute with name and value.
struct Attribute {
  Attribute(const Attribute &other) : name(other.name), value(other.value) {}
  Attribute(const string &n, const string &v) : name(n), value(v) {}
  string name;   // attribute name
  string value;  // attribute value
};

// Attribute list with key value pairs.
class Attributes : public std::vector<Attribute> {
 public:
  // Get attribute value.
  const string &Get(const string &name) const;
  int Get(const string &name, int defval) const;

  // Check if attribute exists.
  bool Has(const string &name) const;

  // Set attribute.
  void Set(const string &name, const string &value);
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

    // Return data in text format.
    string DataString() const;

    // Set data for variable. The storage is not owned by the variable.
    void SetData(void *buffer, int len) {
      data = static_cast<char *>(buffer);
      size = len;
    }

    // Check if variable has a dependency on some operation.
    bool DependsOn(const Operation *op) const;

    string name;                         // variable name
    std::vector<string> aliases;         // additional aliases for variable

    Type type = DT_INVALID;              // element type for variable
    bool ref = false;                    // is variable a reference?
    Shape shape;                         // variable shape
    char *data = nullptr;                // data for constants (owned by flow)
    uint64_t size = 0;                   // size of data in bytes
    bool in = false;                     // is variable a function input?
    bool out = false;                    // is variable a function output?

    Operation *producer = nullptr;       // operation producing variable
    std::vector<Operation *> consumers;  // list of consumers of variable
  };

  // Flow operation.
  struct Operation {
    // Add input to operation.
    void AddInput(Variable *var);

    // Add output to operation.
    void AddOutput(Variable *var);

    // Get attribute value.
    const string &GetAttr(const string &name) const {
      return attrs.Get(name);
    };
    int GetAttr(const string &name, int defval) const {
      return attrs.Get(name, defval);
    }

    // Check if operation has attribute.
    bool HasAttr(const string &name) const {
      return attrs.Has(name);
    }

    // Set attribute.
    void SetAttr(const string &name, const string &value) {
      attrs.Set(name, value);
    }

    // Check if variable is an input to the operation.
    bool IsInput(const Variable *var) const;

    // Check if variable is an output from the operation.
    bool IsOutput(const Variable *var) const;

    // Remove input variable from operation.
    void RemoveInput(Variable *var);

    // Remove output variable from operation.
    void RemoveOutput(Variable *var);

    // Move input variable to another operation.
    void MoveInput(Variable *var, Operation *op);

    // Move output variable to another operation.
    void MoveOutput(Variable *var, Operation *op);

    // Return in and out degree.
    int indegree() const { return inputs.size(); }
    int outdegree() const { return outputs.size(); }

    string name;                      // operation name
    string type;                      // operation type
    std::vector<Variable *> inputs;   // input variables
    std::vector<Variable *> outputs;  // output variables
    Attributes attrs;                 // operation attributes
    Function *func = nullptr;         // function that operation belongs to

    int task = 0;                     // task id for operation for parallel op
    int priority = 3;                 // task priority for op compute ordering
    int order = -1;                   // placement in computation order
    int missing = 0;                  // number of inputs that are not yet ready
  };

  // Flow function.
  struct Function {
    // Add operation to function.
    void AddOperation(Operation *op);

    string name;                      // function name
    std::vector<Operation *> ops;     // ops for function in compute order
  };

  // Flow connector.
  struct Connector {
    // Add linked variable to connector.
    void AddLink(Variable *var);

    // Remove linked variable from connector. Return false if link was not
    // found.
    bool RemoveLink(Variable *var);

    // Replace linked variable with another variable. Return false if link was
    // not found.
    bool ReplaceLink(Variable *old, Variable *var);

    string name;                      // connector name
    std::vector<Variable *> links;    // variables linked to connector
  };

  Flow();
  ~Flow();

  // Allocate memory that is owned by the flow.
  char *AllocateMemory(size_t size);

  // Load flow from file.
  Status Load(const string &filename);

  // Analyze flow.
  void Analyze(const Transformations &transformations);

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
  Operation *AddOperation(Function *func,
                          const string &name,
                          const string &type,
                          const std::vector<Variable *> &inputs,
                          const std::vector<Variable *> &outputs);

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

  // Fuse two operations into a combined op.
  Operation *Fuse(Operation *first,
                  Operation *second,
                  const string &combined,
                  bool merge_inputs = false);

  // Find sequences of ops in flow graph. This only matches the first output
  // for each op in the sequence.
  std::vector<Operation *> Find(const std::vector<string> &ops);

  // Extract sub-flow from flow. A new function will be added to the subflow and
  // will contain all the dependencies of the outputs excluding the dependencies
  // of the inputs. The extracted flow may contain pointers to data blocks in
  // the original flow.
  Function *Extract(const string &name,
                    const std::vector<Variable *> &inputs,
                    const std::vector<Variable *> &outputs,
                    Flow *subflow);

  // Check flow graph consistency.
  bool IsConsistent() const;

 private:
  // Infer which variables are inputs and outputs to functions.
  void InferInputsAndOutputs();

  // Apply transformations to flow graph.
  void Transform(const Transformations &transformations);

  // Combine two op types to a single combined op type.
  bool Combine(const string &first,
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

// Component type for applying transformations to a flow.
class Transformer {
 public:
  virtual ~Transformer() = default;

  // Apply transformations to flow and return true is any transformations were
  // applied.
  virtual bool Transform(Flow *flow) = 0;
};

}  // namespace myelin
}  // namespace sling

#endif  // MYELIN_FLOW_H_

