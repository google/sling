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

#include "sling/pyapi/pywiki.h"

#include "sling/frame/object.h"
#include "sling/frame/reader.h"
#include "sling/frame/serialization.h"
#include "sling/frame/store.h"
#include "sling/stream/memory.h"

namespace sling {

// Python type declarations.
PyTypeObject PyWikiConverter::type;
PyMethodTable PyWikiConverter::methods;

void PyWikiConverter::Define(PyObject *module) {
  InitType(&type, "sling.WikiConverter", sizeof(PyWikiConverter), true);
  type.tp_init = method_cast<initproc>(&PyWikiConverter::Init);
  type.tp_dealloc = method_cast<destructor>(&PyWikiConverter::Dealloc);

  methods.Add("convert_wikidata", &PyWikiConverter::ConvertWikidata);
  type.tp_methods = methods.table();

  RegisterType(&type, module, "WikiConverter");
}

int PyWikiConverter::Init(PyObject *args, PyObject *kwds) {
  // Get store argument.
  pystore = nullptr;
  converter = nullptr;
  if (!PyArg_ParseTuple(args, "O", &pystore)) return -1;
  if (!PyObject_TypeCheck(pystore, &PyStore::type)) return -1;

  // Initialize converter.
  Py_INCREF(pystore);
  converter = new nlp::WikidataConverter(pystore->store, "");

  return 0;
}

void PyWikiConverter::Dealloc() {
  delete converter;
  if (pystore) Py_DECREF(pystore);
  Free();
}

PyObject *PyWikiConverter::ConvertWikidata(PyObject *args, PyObject *kw) {
  // Get store and Wikidata JSON string.
  PyStore *pystore = nullptr;
  const char *json = nullptr;
  if (!PyArg_ParseTuple(args, "Os", &pystore, &json)) return nullptr;
  if (!PyObject_TypeCheck(pystore, &PyStore::type)) return nullptr;

  // Parse JSON.
  ArrayInputStream stream(json, strlen(json));
  Input input(&stream);
  Reader reader(pystore->store, &input);
  reader.set_json(true);
  Object obj = reader.Read();
  if (reader.error()) {
    PyErr_SetString(PyExc_ValueError, reader.error_message().c_str());
    return nullptr;
  }
  if (!obj.valid() || !obj.IsFrame()) {
    PyErr_SetString(PyExc_ValueError, "Not a valid frame");
    return nullptr;
  }

  // Convert Wikidata JSON to SLING frame.
  const Frame &item = converter->Convert(obj.AsFrame());

  return pystore->PyValue(item.handle());
}

}  // namespace sling
