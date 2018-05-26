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

#include "sling/pyapi/pyphrase.h"

#include "sling/pyapi/pystore.h"

namespace sling {

// Python type declarations.
PyTypeObject PyPhraseTable::type;

PyMethodDef PyPhraseTable::methods[] = {
  {"lookup", PYFUNC(PyPhraseTable::Lookup), METH_O, ""},
  {"query", PYFUNC(PyPhraseTable::Query), METH_O, ""},
  {nullptr}
};

void PyPhraseTable::Define(PyObject *module) {
  InitType(&type, "sling.api.PhraseTable", sizeof(PyPhraseTable), true);

  type.tp_init = method_cast<initproc>(&PyPhraseTable::Init);
  type.tp_dealloc = method_cast<destructor>(&PyPhraseTable::Dealloc);
  type.tp_methods = methods;

  RegisterType(&type, module, "PhraseTable");
}

int PyPhraseTable::Init(PyObject *args, PyObject *kwds) {
  // Get store and phrase table file name.
  const char *filename = nullptr;
  if (!PyArg_ParseTuple(args, "Os", &pystore, &filename)) return -1;
  if (!PyObject_TypeCheck(pystore, &PyStore::type)) return -1;
  Py_INCREF(pystore);

  // Load phrase table.
  phrase_table = new nlp::PhraseTable();
  phrase_table->Load(pystore->store, filename);

  // Initialize tokenizer.
  tokenizer = new nlp::PhraseTokenizer();

  return 0;
}

void PyPhraseTable::Dealloc() {
  delete tokenizer;
  delete phrase_table;
  Py_DECREF(pystore);
  Free();
}

PyObject *PyPhraseTable::Lookup(PyObject *obj) {
  // Get phrase.
  char *phrase = PyString_AsString(obj);
  if (phrase == nullptr) return nullptr;

  // Compute phrase fingerprint.
  uint64 fp = tokenizer->Fingerprint(phrase);

  // Get matching items.
  Handles matches(pystore->store);
  phrase_table->Lookup(fp, &matches);

  // Create list of matching items.
  PyObject *result = PyList_New(matches.size());
  for (int i = 0; i < matches.size(); ++i) {
    PyList_SetItem(result, i, pystore->PyValue(matches[i]));
  }

  return result;
}

PyObject *PyPhraseTable::Query(PyObject *obj) {
  // Get phrase.
  char *phrase = PyString_AsString(obj);
  if (phrase == nullptr) return nullptr;

  // Compute phrase fingerprint.
  uint64 fp = tokenizer->Fingerprint(phrase);

  // Get matching items.
  nlp::PhraseTable::MatchList matches;
  phrase_table->Lookup(fp, &matches);

  // Create list of matching items.
  PyObject *result = PyList_New(matches.size());
  for (int i = 0; i < matches.size(); ++i) {
    PyObject *match = PyTuple_New(2);
    PyTuple_SetItem(match, 0, pystore->PyValue(matches[i].first));
    PyTuple_SetItem(match, 1, PyInt_FromLong(matches[i].second));
    PyList_SetItem(result, i, match);
  }

  return result;
}

}  // namespace sling

