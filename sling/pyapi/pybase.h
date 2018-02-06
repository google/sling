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

#include <python2.7/Python.h>

namespace sling {

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
};

}  // namespace sling

#endif  // SLING_PYAPI_PYBASE_H_

