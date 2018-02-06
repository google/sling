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

#include "sling/pyapi/pyparser.h"

#include "sling/nlp/document/document.h"
#include "sling/nlp/document/document-tokenizer.h"
#include "sling/nlp/parser/parser.h"
#include "sling/pyapi/pyframe.h"
#include "sling/pyapi/pystore.h"

namespace sling {

// Python type declarations.
PyTypeObject PyTokenizer::type;
PyTypeObject PyParser::type;

PyMethodDef PyTokenizer::methods[] = {
  {"tokenize", (PyCFunction) &PyTokenizer::Tokenize, METH_VARARGS, ""},
  {nullptr}
};

void PyTokenizer::Define(PyObject *module) {
  InitType(&type, "sling.api.Tokenizer", sizeof(PyTokenizer), true);

  type.tp_init = reinterpret_cast<initproc>(&PyTokenizer::Init);
  type.tp_dealloc = reinterpret_cast<destructor>(&PyTokenizer::Dealloc);
  type.tp_methods = methods;

  RegisterType(&type, module, "Tokenizer");
}

int PyTokenizer::Init(PyObject *args, PyObject *kwds) {
  // Initialize tokenizer.
  tokenizer = new nlp::DocumentTokenizer();
  return 0;
}

void PyTokenizer::Dealloc() {
  delete tokenizer;
  Free();
}

PyObject *PyTokenizer::Tokenize(PyObject *args) {
  // Get arguments.
  PyStore *pystore;
  PyObject *text;
  if (!PyArg_ParseTuple(args, "OS", &pystore, &text)) return nullptr;
  if (!PyObject_TypeCheck(pystore, &PyStore::type)) return nullptr;
  if (!pystore->Writable()) return nullptr;

  // Get text.
  char *data;
  Py_ssize_t length;
  PyString_AsStringAndSize(text, &data, &length);

  // Initialize empty document.
  nlp::Document document(pystore->store);

  // Tokenize text.
  tokenizer->Tokenize(&document, Text(data, length));
  document.Update();

  // Create document frame wrapper.
  PyFrame *frame = PyObject_New(PyFrame, &PyFrame::type);
  frame->Init(pystore, document.top().handle());
  return frame->AsObject();
}

PyMethodDef PyParser::methods[] = {
  {"parse", (PyCFunction) &PyParser::Parse, METH_VARARGS, ""},
  {nullptr}
};

void PyParser::Define(PyObject *module) {
  InitType(&type, "sling.api.Parser", sizeof(PyParser), true);

  type.tp_init = reinterpret_cast<initproc>(&PyParser::Init);
  type.tp_dealloc = reinterpret_cast<destructor>(&PyParser::Dealloc);
  type.tp_methods = methods;

  RegisterType(&type, module, "Parser");
}

int PyParser::Init(PyObject *args, PyObject *kwds) {
  // Get arguments.
  PyStore *pystore;
  char *filename;
  if (!PyArg_ParseTuple(args, "Os", &pystore, &filename)) return -1;
  if (!PyObject_TypeCheck(pystore, &PyStore::type)) return -1;
  if (!pystore->Writable()) return -1;

  // Save reference to store to keep it alive.
  this->pystore = pystore;
  Py_INCREF(pystore);

  // Load parser.
  parser = new nlp::Parser();
  parser->Load(pystore->store, filename);

  return 0;
}

void PyParser::Dealloc() {
  // Delete parser.
  delete parser;

  // Release reference to store.
  Py_DECREF(pystore);

  // Free object.
  Free();
}

PyObject *PyParser::Parse(PyObject *args) {
  // Get arguments.
  PyFrame *pyframe;
  if (!PyArg_ParseTuple(args, "O", &pyframe)) return nullptr;
  if (!PyObject_TypeCheck(pyframe, &PyFrame::type)) return nullptr;
  if (!pyframe->pystore->Writable()) return nullptr;

  // Initialize document.
  Frame top(pyframe->pystore->store, pyframe->handle());
  nlp::Document document(top);

  // Parse document.
  parser->Parse(&document);
  document.Update();

  Py_RETURN_NONE;
}

}  // namespace sling

