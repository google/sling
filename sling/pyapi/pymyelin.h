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

#ifndef SLING_PYAPI_PYMYELIN_H_
#define SLING_PYAPI_PYMYELIN_H_

#include "sling/myelin/compiler.h"
#include "sling/pyapi/pybase.h"

namespace sling {

// Utility class for holding on to internal memory buffers defined in other
// Python objects. This uses the Python buffer interface to get direct access
// to the internal memory representation of other Python objects like
// memoryview and numpy arrays, so these do not need to be copied in the
// Myelin flows.
class PyBuffers {
 public:
  PyBuffers(myelin::Flow *flow) : flow_(flow) {}
  ~PyBuffers();
  char *GetData(PyObject *obj, myelin::Type type, size_t *size);
 private:
  myelin::Flow *flow_;
  std::vector<Py_buffer *> views_;
  std::vector<PyObject *> refs_;
};

// Python wrapper for Myelin compiler.
struct PyCompiler : public PyBase {
  // Initialize wrapper.
  int Init(PyObject *args, PyObject *kwds);

  // Deallocate wrapper.
  void Dealloc();

  // Compile flow.
  PyObject *Compile(PyObject *arg);

  // Import Python flow into Myelin flow.
  static bool ImportFlow(PyObject *pyflow, myelin::Flow *flow,
                         PyBuffers *buffers);

  // Import attributes for flow artifact.
  static bool ImportAttributes(PyObject *obj, myelin::Attributes *attrs);

  // Get string attribute for object.
  static const char *PyStrAttr(PyObject *obj, const char *name);

  // Get integer attribute for object.
  static int PyIntAttr(PyObject *obj, const char *name);

  // Get attribute for object. Returns new reference.
  static PyObject *PyAttr(PyObject *obj, const char *name);

  // Myelin compiler.
  myelin::Compiler *compiler;

  // Registration.
  static PyTypeObject type;
  static PyMethodTable methods;
  static void Define(PyObject *module);
};

// Python wrapper for Myelin network.
struct PyNetwork : public PyBase {
  // Initialize wrapper.
  int Init(myelin::Network *net);

  // Deallocate wrapper.
  void Dealloc();

  // Look up global tensor in network.
  PyObject *LookupTensor(PyObject *key);

  // Get global tensor value.
  PyObject *GetTensor(PyObject *key);

  // Assign value to tensor.
  int SetTensor(PyObject *key, PyObject *value);

  // Look up cell in network.
  PyObject *LookupCell(PyObject *key);

  // Return profile report if profiling is enabled.
  PyObject *Profile();

  // Find named tensor in cell or a global tensor if cell is null.
  myelin::Tensor *FindTensor(PyObject *key, const myelin::Cell *cell);

  // Myelin network.
  myelin::Network *net;

  // Registration.
  static PyTypeObject type;
  static PyMappingMethods mapping;
  static PyMethodTable methods;
  static void Define(PyObject *module);
};

// Python wrapper for Myelin cell.
struct PyCell : public PyBase {
  // Initialize wrapper.
  int Init(PyNetwork *pynet, myelin::Cell *cell);

  // Deallocate wrapper.
  void Dealloc();

  // Return new data instance for cell.
  PyObject *NewInstance();

  // Return new channel.
  PyObject *NewChannel(PyObject *args);

  // Return parameter tensor index. This can be used as a key for looking up
  // tensors in instances.
  PyObject *Index(PyObject *key);

  // Check if parameter is in cell.
  int Contains(PyObject *key);

  // Myelin cell.
  myelin::Cell *cell;

  // Network that owns the cell.
  PyNetwork *pynet;

  // Registration.
  static PyTypeObject type;
  static PySequenceMethods sequence;
  static PyMethodTable methods;
  static void Define(PyObject *module);
};

// Python wrapper for Myelin instance.
struct PyInstance : public PyBase {
  // Initialize wrapper.
  int Init(PyCell *pycell);

  // Deallocate wrapper.
  void Dealloc();

  // Look up local tensor in instance.
  PyObject *LookupTensor(PyObject *key);

  // Get tensor value.
  PyObject *GetTensor(PyObject *key);

  // Assign value to tensor.
  int SetTensor(PyObject *key, PyObject *value);

  // Connect channel element to reference tensor in instance.
  PyObject *Connect(PyObject *args);

  // Run cell computation on instance.
  PyObject *Compute();

  // Clear instance.
  PyObject *Clear();

  // Return data instance as string.
  PyObject *Str();

  // Myelin data instance.
  myelin::Instance *data;

  // Cell for the instance.
  PyCell *pycell;

  // Registration.
  static PyTypeObject type;
  static PyMappingMethods mapping;
  static PyMethodTable methods;
  static void Define(PyObject *module);
};

// Python wrapper for Myelin channel.
struct PyChannel : public PyBase {
  // Initialize wrapper.
  int Init(PyNetwork *pynet, myelin::Tensor *format, int size);

  // Deallocate wrapper.
  void Dealloc();

  // Return channel size.
  Py_ssize_t Size();

  // Return channel element.
  PyObject *Lookup(PyObject *key);

  // Resize channel.
  PyObject *Resize(PyObject *args);

  // Myelin channel data.
  myelin::Channel *channel;

  // Network for channel.
  PyNetwork *pynet;

  // Type checking.
  static bool TypeCheck(PyBase *object) {
    return PyBase::TypeCheck(object, &type);
  }
  static bool TypeCheck(PyObject *object) {
    return PyBase::TypeCheck(object, &type);
  }

  // Registration.
  static PyTypeObject type;
  static PyMappingMethods mapping;
  static PyMethodTable methods;
  static void Define(PyObject *module);
};

// Python wrapper for Myelin tensor data.
struct PyTensor : public PyBase {
  // Initialize wrapper.
  int Init(PyObject *owner, char *data, const myelin::Tensor *format);

  // Deallocate wrapper.
  void Dealloc();

  // Return tensor name.
  PyObject *Name();

  // Return tensor rank.
  PyObject *Rank();

  // Return number of elements in tensor.
  Py_ssize_t Size();

  // Return tensor shape.
  PyObject *Shape();

  // Return tensor data type.
  PyObject *Type();

  // Return tensor as string.
  PyObject *Str();

  // Get element from tensor.
  PyObject *GetElement(PyObject *index);

  // Assign value to tensor element.
  int SetElement(PyObject *index, PyObject *value);

  // Buffer interface for accessing tensor data.
  int GetBuffer(Py_buffer *view, int flags);
  void ReleaseBuffer(Py_buffer *view);

  // Get shape and stides. There are allocated lazily.
  Py_ssize_t *GetShape();
  Py_ssize_t *GetStrides();

  // Return tensor type as Python type format string.
  char *GetFormat() {
    return const_cast<char *>(myelin::TypeTraits::of(format->type()).pytype());
  }

  // Get address of element in tensor.
  char *GetAddress(PyObject *index);

  // Reference for keeping data alive.
  PyObject *owner;

  // Raw data for tensor.
  char *data;

  // Tensor format.
  const myelin::Tensor *format;

  // Shape and strides in Python format.
  Py_ssize_t *shape;
  Py_ssize_t *strides;

  // Registration.
  static PyTypeObject type;
  static PyMappingMethods mapping;
  static PyBufferProcs buffer;
  static PyMethodTable methods;
  static void Define(PyObject *module);
};

}  // namespace sling

#endif  // SLING_PYAPI_PYMYELIN_H_

