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
PyTypeObject PyPhraseMatch::type;
PyMethodTable PyPhraseMatch::methods;
PyTypeObject PyPhraseTable::type;
PyMethodTable PyPhraseTable::methods;

void PyPhraseMatch::Define(PyObject *module) {
  InitType(&type, "sling.api.PhraseMatch", sizeof(PyPhraseMatch), false);
  type.tp_dealloc = method_cast<destructor>(&PyPhraseMatch::Dealloc);

  methods.Add("id", &PyPhraseMatch::Id);
  methods.Add("item", &PyPhraseMatch::Item);
  methods.Add("form", &PyPhraseMatch::Form);
  methods.Add("count", &PyPhraseMatch::Count);
  methods.Add("reliable", &PyPhraseMatch::Reliable);
  type.tp_methods = methods.table();

  RegisterType(&type, module, "PhraseMatch");

  RegisterEnum(module, "CASE_INVALID", CASE_INVALID);
  RegisterEnum(module, "CASE_NONE", CASE_NONE);
  RegisterEnum(module, "CASE_UPPER", CASE_UPPER);
  RegisterEnum(module, "CASE_LOWER", CASE_LOWER);
  RegisterEnum(module, "CASE_TITLE", CASE_TITLE);
}

int PyPhraseMatch::Init(PyStore *pystore,
                        const nlp::PhraseTable::Match &match) {
  pyitem = pystore->PyValue(match.item);
  info = match;
  return 0;
}

void PyPhraseMatch::Dealloc() {
  Py_DECREF(pyitem);
  Free();
}

PyObject *PyPhraseMatch::Id() {
  return PyUnicode_FromStringAndSize(info.id.data(), info.id.size());
}

PyObject *PyPhraseMatch::Item() {
  Py_INCREF(pyitem);
  return pyitem;
}

PyObject *PyPhraseMatch::Form() {
  return PyLong_FromLong(info.form);
}

PyObject *PyPhraseMatch::Count() {
  return PyLong_FromLong(info.count);
}

PyObject *PyPhraseMatch::Reliable() {
  return PyBool_FromLong(info.reliable);
}

void PyPhraseTable::Define(PyObject *module) {
  InitType(&type, "sling.api.PhraseTable", sizeof(PyPhraseTable), true);

  type.tp_init = method_cast<initproc>(&PyPhraseTable::Init);
  type.tp_dealloc = method_cast<destructor>(&PyPhraseTable::Dealloc);

  methods.AddO("lookup", &PyPhraseTable::Lookup);
  methods.AddO("query", &PyPhraseTable::Query);
  methods.AddO("fingerprint", &PyPhraseTable::Fingerprint);
  methods.AddO("form", &PyPhraseTable::Form);
  type.tp_methods = methods.table();

  RegisterType(&type, module, "PhraseTable");
}

int PyPhraseTable::Init(PyObject *args, PyObject *kwds) {
  // Get store and phrase table file name.
  const char *filename = nullptr;
  if (!PyArg_ParseTuple(args, "Os", &pystore, &filename)) return -1;
  if (!PyStore::TypeCheck(pystore)) return -1;
  Py_INCREF(pystore);

  // Load phrase table.
  phrase_table = new nlp::PhraseTable();
  phrase_table->Load(pystore->store, filename);

  // Initialize tokenizer.
  tokenizer = new nlp::PhraseTokenizer();
  Normalization norm = ParseNormalization(phrase_table->normalization());
  tokenizer->set_normalization(norm);

  return 0;
}

void PyPhraseTable::Dealloc() {
  delete tokenizer;
  delete phrase_table;
  if (pystore) Py_DECREF(pystore);
  Free();
}

PyObject *PyPhraseTable::Lookup(PyObject *obj) {
  // Get phrase.
  const char *phrase = GetString(obj);
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
  const char *phrase = GetString(obj);
  if (phrase == nullptr) return nullptr;

  // Compute phrase fingerprint.
  uint64 fp = tokenizer->Fingerprint(phrase);

  // Get matching items.
  nlp::PhraseTable::MatchList matches;
  phrase_table->Lookup(fp, &matches);

  // Create list of matching items.
  PyObject *result = PyList_New(matches.size());
  for (int i = 0; i < matches.size(); ++i) {
    PyPhraseMatch *match = PyObject_New(PyPhraseMatch, &PyPhraseMatch::type);
    match->Init(pystore, matches[i]);
    PyList_SetItem(result, i, match->AsObject());
  }

  return result;
}

PyObject *PyPhraseTable::Fingerprint(PyObject *obj) {
  // Get phrase.
  const char *phrase = GetString(obj);
  if (phrase == nullptr) return nullptr;

  // Compute phrase fingerprint.
  uint64 fp = tokenizer->Fingerprint(phrase);

  // Return fingerprint.
  return PyLong_FromUnsignedLong(fp);
}

PyObject *PyPhraseTable::Form(PyObject *obj) {
  // Get phrase.
  const char *phrase = GetString(obj);
  if (phrase == nullptr) return nullptr;

  // Determine case form.
  uint64 fp;
  CaseForm form;
  tokenizer->FingerprintAndForm(phrase, &fp, &form);

  // Return case form.
  return PyLong_FromLong(form);
}

}  // namespace sling
