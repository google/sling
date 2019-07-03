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

#include "sling/nlp/embedding/plausibility-model.h"
#include "sling/nlp/kb/facts.h"
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
  PyStore *pycommons;

  // Wikidata converter.
  nlp::WikidataConverter *converter;

  // Registration.
  static PyTypeObject type;
  static PyMethodTable methods;
  static void Define(PyObject *module);
};

// Python wrapper for Wiki fact extractor.
struct PyFactExtractor : public PyBase {
  int Init(PyObject *args, PyObject *kwds);
  void Dealloc();

  // Extract list of facts from item.
  PyObject *Facts(PyObject *args, PyObject *kw);

  // Extract list of facts from item for certain properties.
  PyObject *FactsFor(PyObject *args, PyObject *kw);

  // Returns if a given item is in a property-specific closure of another.
  PyObject *InClosure(PyObject *args, PyObject *kw);

  // Get types for item.
  PyObject *Types(PyObject *args, PyObject *kw);

  // Create new taxonomy based on type list.
  PyObject *Taxonomy(PyObject *args, PyObject *kw);

  // Commons store for converter.
  PyStore *pycommons;

  // Fact extractor.
  nlp::FactCatalog *catalog;

  // Type checking.
  static bool TypeCheck(PyBase *object) {
    return PyBase::TypeCheck(object, &type);
  }
  static bool TypeCheck(PyObject *object) {
    return PyBase::TypeCheck(object, &type);
  }

  // Registration.
  static PyTypeObject type;
  static PyMethodTable methods;
  static void Define(PyObject *module);
};

// Python wrapper for taxonomy.
struct PyTaxonomy : public PyBase {
  int Init(PyFactExtractor *extractor, PyObject *typelist);
  void Dealloc();

  // Classify type for item according to taxonomy.
  PyObject *Classify(PyObject *item);

  // Fact extractor for taxonomy.
  PyFactExtractor *pyextractor;

  // Taxonomy.
  nlp::Taxonomy *taxonomy;

  // Registration.
  static PyTypeObject type;
  static PyMethodTable methods;
  static void Define(PyObject *module);
};

// Python wrapper for plausibility model.
struct PyPlausibility : public PyBase {
  int Init(PyObject *args, PyObject *kwds);
  void Dealloc();

  // Return fact plausibility score for item.
  PyObject *Score(PyObject *args);

  // Fact extractor for plausibility model.
  PyFactExtractor *pyextractor;

  // Plausibility model.
  nlp::PlausibilityModel *model;

  // Registration.
  static PyTypeObject type;
  static PyMethodTable methods;
  static void Define(PyObject *module);
};

}  // namespace sling

#endif  // SLING_PYAPI_PYWIKI_H_
