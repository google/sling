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

#include "sling/pyapi/pybase.h"

namespace sling {

void PyBase::InitType(PyTypeObject *type,
                      const char *name,
                      size_t size,
                      bool instantiable) {
  type->tp_name = name;
  type->tp_basicsize = size;
  if (instantiable) type->tp_new = PyType_GenericNew;
  type->tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
}

void PyBase::RegisterType(PyTypeObject *type) {
  PyType_Ready(type);
  Py_INCREF(type);
}

void PyBase::RegisterType(PyTypeObject *type,
                          PyObject *module,
                          const char *name) {
  PyType_Ready(type);
  Py_INCREF(type);
  PyModule_AddObject(module, name, reinterpret_cast<PyObject *>(type));
}

void PyBase::RegisterEnum(PyObject *module,
                          const char *name,
                          int value) {
  PyModule_AddIntConstant(module, name, value);
}

}  // namespace sling

