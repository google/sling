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

#ifndef SLING_PYAPI_PYPARSER_H_
#define SLING_PYAPI_PYPARSER_H_

#include "sling/frame/store.h"
#include "sling/nlp/document/annotator.h"
#include "sling/nlp/document/document.h"
#include "sling/nlp/document/document-tokenizer.h"
#include "sling/nlp/parser/parser.h"
#include "sling/pyapi/pybase.h"
#include "sling/pyapi/pystore.h"

namespace sling {

// Python wrapper for tokenizer.
struct PyTokenizer : public PyBase {
  // Initialize tokenizer wrapper.
  int Init(PyObject *args, PyObject *kwds);

  // Deallocate tokenizer wrapper.
  void Dealloc();

  // Tokenize text and return document frame with tokens.
  PyObject *Tokenize(PyObject *args);

  // Parse LEX-encoded text and return document with tokens, spans, and frames.
  PyObject *Lex(PyObject *args);

  // Document tokenizer.
  nlp::DocumentTokenizer *tokenizer;

  // Registration.
  static PyTypeObject type;
  static PyMethodTable methods;
  static void Define(PyObject *module);
};

// Python wrapper for parser.
struct PyParser : public PyBase {
  // Initialize parser wrapper.
  int Init(PyObject *args, PyObject *kwds);

  // Deallocate parser wrapper.
  void Dealloc();

  // Parse document.
  PyObject *Parse(PyObject *args);

  // Document parser.
  nlp::Parser *parser;

  // Commons store for parser.
  PyStore *pystore;

  // Registration.
  static PyTypeObject type;
  static PyMethodTable methods;
  static void Define(PyObject *module);
};

// Python wrapper for analyzer.
struct PyAnalyzer : public PyBase {
  // Initialize analyzer wrapper.
  int Init(PyObject *args, PyObject *kwds);

  // Deallocate analyzer wrapper.
  void Dealloc();

  // Annotate document.
  PyObject *Annotate(PyObject *obj);

  // Commons store for analyzer.
  PyStore *pystore;

  // Document analyzer.
  nlp::DocumentAnnotation *analyzer;

  // Document schema.
  nlp::DocumentNames *docschema;

  // Registration.
  static PyTypeObject type;
  static PyMethodTable methods;
  static void Define(PyObject *module);
};

// Return document frame in LEX-encoded format.
PyObject *PyToLex(PyObject *self, PyObject *args);

// Evaluate frame annotations.
PyObject *PyEvaluateFrames(PyObject *self, PyObject *args);

}  // namespace sling

#endif  // SLING_PYAPI_PYPARSER_H_

