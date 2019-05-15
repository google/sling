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

#ifndef SLING_PYAPI_PYBASE_H_
#define SLING_PYAPI_PYBASE_H_

#ifdef SLING_GOOGLE3
#include <Python.h>
#include <structmember.h>
#elif PYVER==35
#include <python3.5/Python.h>
#include <python3.5/structmember.h>
#elif PYVER==36
#include <python3.6/Python.h>
#include <python3.6/structmember.h>
#elif PYVER==37
#include <python3.7/Python.h>
#include <python3.7/structmember.h>
#else
#include <Python.h>
#include <structmember.h>
#endif

#include <vector>

#include "sling/string/text.h"

namespace sling {

template <class T> struct MemberPtr { T ptr; ptrdiff_t adj; };

template <class Dest, class Source>
inline Dest method_cast(const Source &source) {
  union { Source s; MemberPtr<Dest> d; };
  s = source;
  return d.ptr;
}

// Method table for Python member functions.
class PyMethodTable {
 public:
  // Initialize method table.
  PyMethodTable();

  // Add method to method table.
  void Add(const char *name, PyCFunction method,  int flags);

  // Add member function with no arguments.
  template<class T> void Add(const char *name, PyObject *(T::*method)()) {
    Add(name, method_cast<PyCFunction>(method), METH_NOARGS);
  }

  // Add member function with argument list.
  template<class T> void Add(
      const char *name,
      PyObject *(T::*method)(PyObject *args)) {
    Add(name, method_cast<PyCFunction>(method), METH_VARARGS);
  }

  // Add member function with arguments and keywords.
  template<class T> void Add(
      const char *name,
      PyObject *(T::*method)(PyObject *args, PyObject *kw)) {
    Add(name, method_cast<PyCFunction>(method), METH_VARARGS | METH_KEYWORDS);
  }

  // Add member function with a single object argument.
  template<class T> void AddO(
      const char *name,
      PyObject *(T::*method)(PyObject *obj)) {
    Add(name, method_cast<PyCFunction>(method), METH_O);
  }

  // Return pointer to method table.
  PyMethodDef *table() { return table_.data(); }

 private:
  std::vector<PyMethodDef> table_;
};

// Base class for Python wrapper objects. This has the Python object header
// information for reference counting and type information.
struct PyBase : public PyVarObject {
  // Return wrapper as a python object.
  PyObject *AsObject() { return reinterpret_cast<PyObject *>(this); }

  // Free object.
  void Free() {
    PyObject *self = AsObject();
    Py_TYPE(self)->tp_free(self);
  }

  // Initialize type information.
  static void InitType(PyTypeObject *type,
                       const char *name,
                       size_t size,
                       bool instantiable);

  // Register type.
  static void RegisterType(PyTypeObject *type);

  // Register type and add it to the namespace of the module.
  static void RegisterType(PyTypeObject *type,
                           PyObject *module,
                           const char *name);

  // Register constant value in namespace of the module.
  static void RegisterEnum(PyObject *module,
                           const char *name,
                           int value);

  // Allocate string.
  static PyObject *AllocateString(Text text) {
    return PyUnicode_FromStringAndSize(text.data(), text.size());
  }

  // Get string from Python string or bytes.
  static const char *GetString(PyObject *obj) {
    if (PyBytes_Check(obj)) {
      return PyBytes_AsString(obj);
    } else {
      return PyUnicode_AsUTF8(obj);
    }
  }

  // Type checking.
  static bool TypeCheck(PyObject *object, PyTypeObject *type) {
    if (!PyObject_TypeCheck(object, type)) {
      PyErr_BadArgument();
      return false;
    } else {
      return true;
    }
  }
  static bool TypeCheck(PyBase *object, PyTypeObject *type) {
    return TypeCheck(object->AsObject(), type);
  }
};

}  // namespace sling

#endif  // SLING_PYAPI_PYBASE_H_

