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

#ifndef SLING_PYAPI_PYWIKI_H_
#define SLING_PYAPI_PYWIKI_H_

#include "sling/nlp/wiki/wikidata-converter.h"
#include "sling/pyapi/pybase.h"
#include "sling/pyapi/pystore.h"

namespace sling {

// Python wrapper for Wiki converter.
struct PyWikiConverter : public PyBase {
  int Init(PyObject *args, PyObject *kwds);
  void Dealloc();

  // Convert Wikidata JSON to SLING frame.
  PyObject *ConvertWikidata(PyObject *args, PyObject *kw);

  // Commons store for converter.
  PyStore *pystore;

  // Wikidata converter.
  nlp::WikidataConverter *converter;

  // Registration.
  static PyTypeObject type;
  static PyMethodTable methods;
  static void Define(PyObject *module);
};

}  // namespace sling

#endif  // SLING_PYAPI_PYWIKI_H_
