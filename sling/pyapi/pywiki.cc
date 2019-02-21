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
PyTypeObject PyTaxonomy::type;
PyMethodTable PyTaxonomy::methods;

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
  if (!PyStore::TypeCheck(pycommons)) return -1;

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
  if (!PyStore::TypeCheck(pystore)) return nullptr;

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

  methods.Add("facts", &PyFactExtractor::Facts);
  methods.Add("facts_for", &PyFactExtractor::FactsFor);
  methods.Add("types", &PyFactExtractor::Types);
  methods.Add("taxonomy", &PyFactExtractor::Taxonomy);
  methods.Add("subsumes", &PyFactExtractor::Subsumes);
  type.tp_methods = methods.table();

  RegisterType(&type, module, "FactExtractor");
}

int PyFactExtractor::Init(PyObject *args, PyObject *kwds) {
  // Get store argument.
  pycommons = nullptr;
  catalog = nullptr;
  if (!PyArg_ParseTuple(args, "O", &pycommons)) return -1;
  if (!PyStore::TypeCheck(pycommons)) return -1;

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

PyObject *PyFactExtractor::Facts(PyObject *args, PyObject *kw) {
  // Get store and Wikidata item.
  PyStore *pystore = nullptr;
  PyFrame *pyitem = nullptr;
  int closure = 1;
  if (!PyArg_ParseTuple(args, "OO|i", &pystore, &pyitem, &closure)) {
    return nullptr;
  }
  if (!PyStore::TypeCheck(pystore)) return nullptr;
  if (!PyFrame::TypeCheck(pyitem)) return nullptr;

  // Extract facts.
  nlp::Facts facts(catalog, pystore->store);
  facts.set_closure(closure == 1);
  facts.Extract(pyitem->handle());

  // Return array of facts.
  const Handle *begin = facts.list().data();
  const Handle *end = begin + facts.list().size();
  return pystore->PyValue(pystore->store->AllocateArray(begin, end));
}

PyObject *PyFactExtractor::FactsFor(PyObject *args, PyObject *kw) {
  // Get store and Wikidata item.
  PyStore *pystore = nullptr;
  PyFrame *pyitem = nullptr;
  PyObject *pyproperties = nullptr;
  int closure = 1;
  if (!PyArg_ParseTuple(
      args, "OOO|i", &pystore, &pyitem, &pyproperties, &closure)) {
    return nullptr;
  }
  if (!PyStore::TypeCheck(pystore)) return nullptr;
  if (!PyFrame::TypeCheck(pyitem)) return nullptr;
  if (!PyList_Check(pyproperties)) return nullptr;

  HandleSet properties;
  int size = PyList_Size(pyproperties);
  for (int i = 0; i < size; ++i) {
    PyObject *item = PyList_GetItem(pyproperties, i);
    if (!PyFrame::TypeCheck(item)) return nullptr;
    Handle handle = reinterpret_cast<PyFrame *>(item)->handle();
    properties.insert(handle);
  }

  // Extract facts.
  nlp::Facts facts(catalog, pystore->store);
  facts.set_closure(closure == 1);
  facts.ExtractFor(pyitem->handle(), properties);

  // Return array of facts.
  const Handle *begin = facts.list().data();
  const Handle *end = begin + facts.list().size();
  return pystore->PyValue(pystore->store->AllocateArray(begin, end));
}

PyObject *PyFactExtractor::Subsumes(PyObject *args, PyObject *kw) {
  // Get store and Wikidata item.
  PyStore *pystore = nullptr;
  PyFrame *pyproperty = nullptr;
  PyFrame *pycoarse = nullptr;
  PyFrame *pyfine = nullptr;
  if (!PyArg_ParseTuple(
    args, "OOOO", &pystore, &pyproperty, &pycoarse, &pyfine)) {
    return nullptr;
  }
  if (!PyStore::TypeCheck(pystore)) return nullptr;
  if (!PyFrame::TypeCheck(pyproperty)) return nullptr;
  if (!PyFrame::TypeCheck(pycoarse)) return nullptr;
  if (!PyFrame::TypeCheck(pyfine)) return nullptr;

  nlp::Facts facts(catalog, pystore->store);
  bool subsumes = facts.Subsumes(pyproperty->handle(),
    pycoarse->handle(), pyfine->handle());

  if (subsumes) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
}

PyObject *PyFactExtractor::Types(PyObject *args, PyObject *kw) {
  // Get store and Wikidata item.
  PyStore *pystore = nullptr;
  PyFrame *pyitem = nullptr;
  if (!PyArg_ParseTuple(args, "OO", &pystore, &pyitem)) return nullptr;
  if (!PyStore::TypeCheck(pystore)) return nullptr;
  if (!PyFrame::TypeCheck(pyitem)) return nullptr;

  // Extract types.
  nlp::Facts facts(catalog, pystore->store);
  Handles types(pystore->store);
  facts.ExtractItemTypes(pyitem->handle(), &types);

  // Return array of types.
  const Handle *begin = types.data();
  const Handle *end = begin + types.size();
  return pystore->PyValue(pystore->store->AllocateArray(begin, end));
}

PyObject *PyFactExtractor::Taxonomy(PyObject *args, PyObject *kw) {
  // Get type list from arguments.
  PyObject *pytypes = nullptr;
  if (!PyArg_ParseTuple(args, "|O", &pytypes)) return nullptr;

  // Return taxonomy.
  PyTaxonomy *taxonomy = PyObject_New(PyTaxonomy, &PyTaxonomy::type);
  taxonomy->Init(this, pytypes);
  return taxonomy->AsObject();
}

void PyTaxonomy::Define(PyObject *module) {
  InitType(&type, "sling.Taxonomy", sizeof(PyTaxonomy), false);
  type.tp_dealloc = method_cast<destructor>(&PyTaxonomy::Dealloc);

  methods.AddO("classify", &PyTaxonomy::Classify);
  type.tp_methods = methods.table();

  RegisterType(&type, module, "Taxonomy");
}

int PyTaxonomy::Init(PyFactExtractor *extractor, PyObject *typelist) {
  // Keep reference to extractor to keep fact catalog alive.
  Py_INCREF(extractor);
  pyextractor = extractor;
  taxonomy = nullptr;

  if (typelist == nullptr) {
    // Create default taxonomy.
    taxonomy = pyextractor->catalog->CreateDefaultTaxonomy();
  } else {
    // Build type list.
    if (!PyList_Check(typelist)) {
      PyErr_BadArgument();
      return -1;
    }
    int size = PyList_Size(typelist);
    std::vector<Text> types;
    for (int i = 0; i < size; ++i) {
      PyObject *item = PyList_GetItem(typelist, i);
      if (!PyString_Check(item)) {
        PyErr_BadArgument();
        return -1;
      }
      const char *name = PyString_AsString(item);
      if (name == nullptr) {
        PyErr_BadArgument();
        return -1;
      }
      types.emplace_back(name);
    }

    // Create taxonomy from type list.
    taxonomy = new nlp::Taxonomy(pyextractor->catalog, types);
  }

  return 0;
}

void PyTaxonomy::Dealloc() {
  delete taxonomy;
  if (pyextractor) Py_DECREF(pyextractor);
  Free();
}

PyObject *PyTaxonomy::Classify(PyObject *item) {
  // Get item frame.
  if (!PyFrame::TypeCheck(item)) return nullptr;
  PyFrame *pyframe = reinterpret_cast<PyFrame *>(item);

  // Classify item.
  Handle type = taxonomy->Classify(pyframe->AsFrame());

  return pyframe->pystore->PyValue(type);
}

}  // namespace sling

