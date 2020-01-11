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

#ifndef SLING_MYELIN_FLOW_H_
#define SLING_MYELIN_FLOW_H_

#include <string>
#include <unordered_map>
#include <vector>

#include "sling/base/status.h"
#include "sling/base/types.h"

namespace sling {
namespace myelin {

class Gradients;
class Transformations;

class Cell;
class Step;
class Tensor;

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
  TypeTraits(Type type, const char *name, int size,
             const char *ctype, const char *ptx, int cuda, const char *pytype,
             void *zero, void *one)
      : type_(type), name_(name), size_(size),
        ctype_(ctype), ptx_(ptx), cuda_(cuda), pytype_(pytype),
        zero_(zero), one_(one) {}

  Type type() const { return type_; }
  const string &name() const { return name_; }
  size_t size() const { return size_; }
  bool valid() const { return type_ != DT_INVALID; }
  const char *ctype() const { return ctype_; }
  const char *ptx() const { return ptx_; }
  int cuda() const { return cuda_; }
  const char *pytype() const { return pytype_; }

  // Return data formatted according to type.
  string str(const void *data) const;

  // Return data as number.
  double number(const void *data) const;

  // Binary representation of zero.
  const void *zero() const { return zero_; }

  // Binary representation of one.
  const void *one() const { return one_; }

  // Look up traits from type code.
  static const TypeTraits &of(Type type);

  // Look up traits from type name.
  static const TypeTraits &of(string &name);

 private:
  Type type_;           // basic type
  string name_;         // type name
  size_t size_;         // size in bytes
  const char *ctype_;   // C type
  const char *ptx_;     // CUDA PTX type
  int cuda_;            // CUDA CUBLAS type
  const char *pytype_;  // Python type

  const void *zero_;    // binary representation of zero for type
  const void *one_;     // binary representation of one for type
};

// Look up traits from type.
template<typename T> inline const TypeTraits &Traits();

#define TYPE_TRAIT(type, dt) \
  template<> inline const TypeTraits &Traits<type>() { \
    return TypeTraits::of(dt); \
  } \
  template<> inline const TypeTraits &Traits<type *>() { \
    return TypeTraits::of(dt); \
  } \

TYPE_TRAIT(float, DT_FLOAT);
TYPE_TRAIT(double, DT_DOUBLE);
TYPE_TRAIT(uint8_t, DT_UINT8);
TYPE_TRAIT(uint16_t, DT_UINT16);
TYPE_TRAIT(int8_t, DT_INT8);
TYPE_TRAIT(int16_t, DT_INT16);
TYPE_TRAIT(int32_t, DT_INT32);
TYPE_TRAIT(int64_t, DT_INT64);
TYPE_TRAIT(bool, DT_BOOL);

#undef TYPE_TRAIT

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

  // Set shape to [rank-1, ..., 0].
  void reverse(int rank) {
    dims_.resize(rank);
    for (int d = 0; d < rank; ++d) dims_[d] = rank - d - 1;
  }

  // Return transposed shape.
  Shape transposed() const {
    Shape t;
    for (int d = rank() - 1; d >= 0; --d) t.add(dim(d));
    return t;
  }

  // Return permuted shape.
  Shape permuted(const Shape &perm) const {
    CHECK_EQ(rank(), perm.rank());
    Shape p;
    for (int d = 0; d < rank(); ++d) p.add(dim(perm.dim(d)));
    return p;
  }

  // Return reduced shape.
  Shape reduced(int axis, bool keepdims = false) const {
    Shape r;
    if (axis >= 0) {
      for (int d = 0; d < rank(); ++d) {
        if (d != axis) {
          r.add(dim(d));
        } else if (keepdims) {
          r.add(1);
        }
      }
    }
    return r;
  }

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

  // Check if shape is fully defined, i.e. all dimensions have specified sizes.
  bool defined() const {
    for (int d : dims_) if (d <= 0) return false;
    return true;
  }

  // Check if shape is missing, i.e. some dimensions are zero.
  bool missing() const {
    for (int d : dims_) if (d == 0) return true;
    return false;
  }

  // Return the number of outer elements relative to dimension.
  int outer(int d) const {
    int n = 1;
    for (int i = 0; i < d; ++i) {
      n *= dims_[i];
      if (n < 0) return -1;
    }
    return n;
  }

  // Return the number of inner elements relative to dimension.
  int inner(int d) const {
    int n = 1;
    for (int i = d; i < dims_.size(); ++i) {
      n *= dims_[i];
      if (n < 0) return -1;
    }
    return n;
  }

  // Index operator.
  int &operator [](int d) { return dims_[d]; }
  const int &operator [](int d) const { return dims_[d]; }

  // Check if shape is the same as another shape. Undefined dimensions are
  // not compared.
  bool IsSameSize(const Shape &other) const;
  bool operator==(const Shape &other) const { return IsSameSize(other); }
  bool operator!=(const Shape &other) const { return !IsSameSize(other); }

  // Check if shape is broadcast compatible with another shape.
  bool IsCompatible(const Shape &other) const;

  // Return the common size between this shape and another shape. The common
  // size is the product of all the shared suffix dimensions.
  int CommonSize(const Shape &other) const;

  // Check if shape is a singular broadcast, i.e. all dimensions except the
  // last are the same, and the last dimension is scalar.
  bool IsSingleBroadcast(const Shape &other) const;

  // Return shape as string.
  string ToString() const;

  // Return dimensions.
  const std::vector<int> dims() const { return dims_; }

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
  const string &GetAttr(const string &name) const;
  int GetAttr(const string &name, int defval) const;
  bool GetAttr(const string &name, bool defval) const;
  float GetAttr(const string &name, float defval) const;
  bool GetAttr(const string &name, Shape *shape) const;

  // Check if attribute exists.
  bool HasAttr(const string &name) const;

  // Set attribute.
  void SetAttr(const string &name, const string &value);
  void SetAttr(const string &name, const char *value);
  void SetAttr(const string &name, int value);
  void SetAttr(const string &name, bool value);
  void SetAttr(const string &name, float value);
  void SetAttr(const string &name, const Shape &value);

  // Remove attribute.
  void RemoveAttr(const string &name);

  // Copy attributes from another attribute list.
  void CopyAttrsFrom(const Attributes &other);

  // Return attribute list.
  const std::vector<Attribute> attrs() const { return *this; }
};

// Flow graph for computation.
class Flow {
 public:
  struct Operation;
  struct Function;

  // Flow file version
  static const int VERSION = 6;
  static const int MAGIC = 0x776f6c66;

  // Flow artifact.
  template<typename T> struct Artifact {
    // Check if flag is set.
    bool is(uint32 flag) const { return (flag & flags) != 0; }

    // Set or clear flag.
    T *set(uint32 flag, bool enable = true) {
      if (enable) {
        flags |= flag;
      } else {
        flags &= ~flag;
      }
      return static_cast<T *>(this);
    }

    string name;         // artifact name
    uint32 flags = 0;    // artifact flags (meaning depends on artifact type)
  };

  // Flow variable.
  struct Variable : public Artifact<Variable>, public Attributes {
    // Variable flags.
    enum Flag {
      NONE       = 0,    // no flags
      IN         = 1,    // input variable
      OUT        = 2,    // output variable
      REF        = 4,    // reference variable
      LEARNABLE  = 8,    // learnable global variable
      UNIQUE     = 16,   // input with single gradient
      DYNAMIC    = 32,   // dynamically sized tensor channel
      ROW        = 64,   // request row-major order
      COL        = 128,  // request column-major order
      NOGRADIENT = 256,  // do not compute gradient for variable
    };

    // Initialization for learnable parameters.
    enum Initialization {
      INIT_ZERO    = 0,  // initialize to zero
      INIT_UNIFORM = 1,  // uniform random initialization
      INIT_NORMAL  = 2,  // normal-distributed initialization
      INIT_ORTHO   = 3,  // normal-distributed orthogonal initialization
    };

    // Add alias for variable.
    void AddAlias(const string &alias);

    // Return the rank of the variable tensor.
    int rank() const { return shape.rank(); }

    // Return size of dimension.
    int dim(int d) const { return shape.dim(d); }

    // Return the number of elements in the variable tensor.
    int elements() const { return shape.elements(); }

    // Return the number of usages of variable.
    int usages() const { return consumers.size(); }

    // Input variable flag.
    bool in() const { return is(IN); }
    Variable *set_in(bool enable = true) { return set(IN, enable); }

    // Output variable flag.
    bool out() const { return is(OUT); }
    Variable *set_out(bool enable = true) { return set(OUT, enable); }

    // Reference variable flag.
    bool ref() const { return is(REF); }
    Variable *set_ref(bool enable = true) { return set(REF, enable); }

    // Learnable variable flag.
    bool learnable() const { return is(LEARNABLE); }
    Variable *set_learnable(bool enable = true) {
      return set(LEARNABLE, enable);
    }

    // Dynamic size flag.
    bool dynamic() const { return is(DYNAMIC); }
    Variable *set_dynamic(bool enable = true) {
      return set(DYNAMIC, enable);
    }

    // Unique gradient flag.
    bool unique() const { return is(UNIQUE); }
    Variable *set_unique(bool enable = true) {
      return set(UNIQUE, enable);
    }

    // Check if variable is a constant.
    bool constant() const { return data != nullptr; }

    // Check if variable is a global variable. Global variables are either
    // constants or read/write learnable variables.
    bool global() const { return constant() || learnable(); }

    // Check if variable is a local variable.
    bool local() const { return !global(); }

    // Check if variable is detached, i.e. no producer or consumers.
    bool detached() const { return producer == nullptr && usages() == 0; }

    // Check if variable is a scalar.
    bool scalar() const { return elements() == 1; }

    // Return type as string.
    string TypeString() const;

    // Return data in text format.
    string DataString() const;

    // Set data for variable. The storage is not owned by the variable.
    void SetData(const void *buffer, int len) {
      data = const_cast<char *>(static_cast<const char *>(buffer));
      size = len;
    }

    // Get data as scalar. Return false if types do not match.
    template <typename T> bool GetData(T *value) const {
      if (data == nullptr) return false;
      auto &traits = Traits<T>();
      if (type != traits.type() || size != traits.size()) return false;
      *value = *reinterpret_cast<const T *>(data);
      return true;
    }

    // Get data as vector. Return false if types do not match.
    template <typename T> bool GetData(std::vector<T> *value) const {
      if (data == nullptr) return false;
      auto &traits = Traits<T>();
      if (type != traits.type()) return false;
      int elements = size / traits.size();
      if (elements * traits.size() != size) return false;
      const T *array = reinterpret_cast<const T *>(data);
      value->assign(array, array + elements);
      return true;
    }

    // Return scalar constant as number.
    double number() const {
      DCHECK(constant());
      DCHECK_EQ(elements(), 1);
      return traits().number(data);
    }

    // Return type traits for variable type.
    const TypeTraits &traits() const { return TypeTraits::of(type); }

    // Check if variable has a dependency on some operation.
    bool DependsOn(const Operation *op) const;

    std::vector<string> aliases;         // additional aliases for variable
    Type type = DT_INVALID;              // element type for variable
    Shape shape;                         // variable shape
    char *data = nullptr;                // data for constants (owned by flow)
    uint64_t size = 0;                   // size of data in bytes
    Initialization init = INIT_ZERO;     // initialization

    Operation *producer = nullptr;       // operation producing variable
    std::vector<Operation *> consumers;  // list of consumers of variable
    Tensor *tensor = nullptr;            // tensor for variable
  };

  // Flow operation.
  struct Operation : public Artifact<Operation>, public Attributes {
    // Variable flags.
    enum Flag {
      NONE       = 0,   // no flags
      NOGRADIENT = 1,   // do not compute gradient for op
    };

    // Add input to operation.
    void AddInput(Variable *var);

    // Add output to operation.
    void AddOutput(Variable *var);

    // Return input index for variable or -1 if variable is not an input.
    int InputIndex(const Variable *var) const;

    // Return output index for variable or -1 if variable is not an output.
    int OutputIndex(const Variable *var) const;

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

    // Replace input variable with another variable.
    void ReplaceInput(Variable *var, Variable *replacement);

    // Replace output variable with another variable.
    void ReplaceOutput(Variable *var, Variable *replacement);

    // Swap order of inputs.
    void SwapInputs(int first = 0, int second = 1);

    // Get prototype variable for operation. This is the biggest output from the
    // operation, unless this is a scalar or the operation does not have any
    // outputs. In that case, the biggest input is returned.
    Variable *GetPrototype() const;

    // Return in and out degree.
    int indegree() const { return inputs.size(); }
    int outdegree() const { return outputs.size(); }

    string type;                      // operation type
    std::vector<Variable *> inputs;   // input variables
    std::vector<Variable *> outputs;  // output variables
    Function *func = nullptr;         // function that operation belongs to

    int task = 0;                     // task id for operation for parallel op
    int priority = 3;                 // task priority for op compute ordering
    int order = -1;                   // placement in computation order
    int missing = 0;                  // number of inputs that are not yet ready
    Step *step = nullptr;             // step for operation
  };

  // Flow function.
  struct Function : public Artifact<Function> {
    // Variable flags.
    enum Flag {
      NONE     = 0,   // no flags
      TRAINING = 1,   // function only needed for training
      BACKPROP = 2,   // build gradient for function
    };

    // Add operation to function.
    void AddOperation(Operation *op);

    // Training function flag.
    bool training() const { return is(TRAINING); }
    Function *set_training(bool enable = true) {
      return set(TRAINING, enable);
    }

    // Back-propagation flag.
    bool backprop() const { return is(BACKPROP); }
    Function *set_backkprop(bool enable = true) {
      return set(BACKPROP, enable);
    }

    std::vector<Operation *> ops;     // ops for function in compute order
    std::vector<Variable *> unused;   // unused input/output variables
    Cell *cell = nullptr;             // cell for function
  };

  // Flow connector.
  struct Connector : public Artifact<Connector> {
    // Add linked variable to connector.
    Connector *AddLink(Variable *var);

    // Remove linked variable from connector. Return false if link was not
    // found.
    bool RemoveLink(Variable *var);

    // Replace linked variable with another variable. Return false if link was
    // not found.
    bool ReplaceLink(Variable *old, Variable *var);

    std::vector<Variable *> links;    // variables linked to connector
  };

  // Blob for storing auxiliary data blocks in flow files.
  struct Blob : public Artifact<Blob>, public Attributes {
    string type;                      // data block type
    const char *data = nullptr;       // data for blob
    uint64_t size = 0;                // size of data for blob
  };

  // Path in flow graph.
  struct Node {
    int input = 0;                    // operation input
    string type;                      // operation type
    int output = 0;                   // operation output
  };
  typedef std::vector<Node> Path;

  Flow();
  ~Flow();

  // Allocate memory that is owned by the flow.
  char *AllocateMemory(size_t size);

  // Allocate and initialize memory that is owned by the flow.
  char *AllocateMemory(const void *data, size_t size);
  char *AllocateMemory(const string &str) {
    return AllocateMemory(str.data(), str.size());
  }

  // Load flow from file.
  Status Load(const string &filename);

  // Read flow from buffer. This does not take ownership of the buffer and it
  // must outlive the flow.
  void Read(const char *data, size_t size);

  // Save flow to file.
  void Save(const string &filename, int version = VERSION) const;

  // Analyze flow.
  void Analyze(const Transformations &transformations);

  // Add variable.
  Variable *AddVariable(const string &name, Type type, const Shape &shape,
                        Variable::Flag flags = Variable::NONE);

  // Add learnable variable.
  Variable *AddWeights(const string &name, Type type, const Shape &shape) {
    return AddVariable(name, type, shape, Variable::LEARNABLE);
  }

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

  // Link variables using new connector.
  Connector *Connect(const std::vector<Variable *> &links);

  // Add data block.
  Blob *AddBlob(const string &name, const string &type);

  // Delete variable.
  void DeleteVariable(Variable *var);

  // Delete operation.
  void DeleteOperation(Operation *op);

  // Delete function.
  void DeleteFunction(Function *func);

  // Look up variable by name.
  Variable *Var(const string &name);

  // Look up operation by name.
  Operation *Op(const string &name);

  // Look up function by name.
  Function *Func(const string &name);

  // Look up connector by name.
  Connector *Cnx(const string &name);

  // Look up blob by name.
  Blob *DataBlock(const string &name);

  // Return corresponding gradient variable.
  Variable *GradientVar(Variable *var);

  // Return corresponding gradient function.
  Function *GradientFunc(Function *func);

  // Return primal variable for function.
  Variable *PrimalVar(Function *func);

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

  // Return all data blocks.
  const std::vector<Blob *> &blobs() const { return blobs_; }

  // Batch size.
  int batch_size() const { return batch_size_; }
  void set_batch_size(int batch_size) { batch_size_ = batch_size; }

  // Fuse two operations into a combined op.
  Operation *Fuse(Operation *first,
                  Operation *second,
                  const string &combined,
                  bool merge_inputs = false);

  // Remove operation from flow.
  void RemoveOperation(Operation *op);

  // Eliminate no-op from flow by moving input to output.
  void Eliminate(Operation *op);

  // Find sequences of ops in flow graph matching a path expression. A path
  // expression is a list of nodes separated by '|'. Each node is a node type
  // with optional input and ouput numbers, i.e. {<input>:}<type>{:<output>}.
  std::vector<Operation *> Find(const string &pathexpr);
  std::vector<Operation *> Find(const std::vector<string> &nodes);
  std::vector<Operation *> Find(std::initializer_list<string> nodes);
  std::vector<Operation *> Find(const Path &path);
  static void ParsePath(const string &pathexpr, Path *path);

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

  // Return ops and variables in dependency order for a function. This does not
  // change the internal order of the operations and variables.
  void Order(Function *func,
             std::vector<Operation *> *ops,
             std::vector<Variable *> *vars) const;

  // Return unique variable name with prefix.
  string VarName(const string &prefix);

  // Return unique operation name with prefix.
  string OpName(const string &prefix);

 private:
  // Infer which variables are inputs and outputs to functions.
  void InferInputsAndOutputs();

  // Apply transformations to flow graph. Returns false if no transformations
  // were applied.
  bool Transform(const Transformations &transformations);

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

  // Blobs.
  std::vector<Blob *> blobs_;

  // Data areas owned by flow.
  std::vector<char *> memory_;

  // Batch size.
  int batch_size_ = 1;
};

// Component type for inferring types and shapes of operation outputs.
class Typer {
 public:
  virtual ~Typer() = default;

  // Return descriptive name for typer.
  virtual string Name() = 0;

  // Return true if the type of the outputs of the operations has been
  // inferred.
  virtual bool InferTypes(Flow *flow, Flow::Operation *op) = 0;
};

// Component type for applying transformations to a flow.
class Transformer {
 public:
  virtual ~Transformer() = default;

  // Return descriptive name for transformer.
  virtual string Name() = 0;

  // Apply transformations to flow and return true is any transformations were
  // applied.
  virtual bool Transform(Flow *flow) = 0;
};

// Flow graph transformations.
class Transformations {
 public:
  ~Transformations();

  // Register flow transformation component. Transfers ownership from caller.
  void RegisterTransformer(Transformer *transformer) {
    transformers_.emplace_back(transformer);
  }

  // Register type inference component. Transfers ownership from caller.
  void RegisterTyper(Typer *typer) {
    typers_.emplace_back(typer);
  }

  // Flow transformation components.
  const std::vector<Transformer *> &transformers() const {
    return transformers_;
  }

  // Type inference components.
  const std::vector<Typer *> &typers() const {
    return typers_;
  }

 private:
  // Flow transformation components.
  std::vector<Transformer *> transformers_;

  // Type inference components.
  std::vector<Typer *> typers_;
};

// Return name of corresponding gradient variable.
string GradientVarName(const string &name);

// Return name of corresponding gradient function.
string GradientFuncName(const string &name);

// Return name of primal variable in gradient function.
string PrimalVarName(const string &name);

}  // namespace myelin
}  // namespace sling

#endif  // SLING_MYELIN_FLOW_H_

