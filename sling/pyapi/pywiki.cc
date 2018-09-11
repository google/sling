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
#include "sling/pyapi/pyarray.h"
#include "sling/pyapi/pyframe.h"
#include "sling/stream/memory.h"

namespace sling {

// Python type declarations.
PyTypeObject PyWikiConverter::type;
PyMethodTable PyWikiConverter::methods;
PyTypeObject PyFactExtractor::type;
PyMethodTable PyFactExtractor::methods;

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
  pycommons = nullptr;
  converter = nullptr;
  if (!PyArg_ParseTuple(args, "O", &pycommons)) return -1;
  if (!PyObject_TypeCheck(pycommons, &PyStore::type)) return -1;

  // Initialize converter.
  Py_INCREF(pycommons);
  converter = new nlp::WikidataConverter(pycommons->store, "");

  return 0;
}

void PyWikiConverter::Dealloc() {
  delete converter;
  if (pycommons) Py_DECREF(pycommons);
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

void PyFactExtractor::Define(PyObject *module) {
  InitType(&type, "sling.FactExtractor", sizeof(PyFactExtractor), true);
  type.tp_init = method_cast<initproc>(&PyFactExtractor::Init);
  type.tp_dealloc = method_cast<destructor>(&PyFactExtractor::Dealloc);

  methods.Add("extract_facts", &PyFactExtractor::ExtractFacts);
  type.tp_methods = methods.table();

  RegisterType(&type, module, "FactExtractor");
}

int PyFactExtractor::Init(PyObject *args, PyObject *kwds) {
  // Get store argument.
  pycommons = nullptr;
  catalog = nullptr;
  if (!PyArg_ParseTuple(args, "O", &pycommons)) return -1;
  if (!PyObject_TypeCheck(pycommons, &PyStore::type)) return -1;

  // Initialize fact extractor catalog.
  Py_INCREF(pycommons);
  catalog = new nlp::FactCatalog();
  catalog->Init(pycommons->store);

  return 0;
}

void PyFactExtractor::Dealloc() {
  delete catalog;
  if (pycommons) Py_DECREF(pycommons);
  Free();
}

PyObject *PyFactExtractor::ExtractFacts(PyObject *args, PyObject *kw) {
  // Get store and Wikidata item.
  PyStore *pystore = nullptr;
  PyFrame *pyitem = nullptr;
  if (!PyArg_ParseTuple(args, "OO", &pystore, &pyitem)) return nullptr;
  if (!PyObject_TypeCheck(pystore, &PyStore::type)) return nullptr;
  if (!PyObject_TypeCheck(pyitem, &PyFrame::type)) return nullptr;

  // Extract facts.
  nlp::Facts facts(catalog, pystore->store);
  facts.Extract(pyitem->handle());

  // Return array of facts.
  const Handle *begin = facts.list().data();
  const Handle *end = begin + facts.list().size();
  return pystore->PyValue(pystore->store->AllocateArray(begin, end));
}

}  // namespace sling

