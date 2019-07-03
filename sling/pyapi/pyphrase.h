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

#ifndef SLING_PYAPI_PYPHRASE_H_
#define SLING_PYAPI_PYPHRASE_H_

#include "sling/nlp/document/phrase-tokenizer.h"
#include "sling/nlp/kb/phrase-table.h"
#include "sling/pyapi/pybase.h"
#include "sling/pyapi/pystore.h"

namespace sling {

// Python wrapper for phrase match.
struct PyPhraseMatch : public PyBase {
  // Initialize wrapper.
  int Init(PyStore *pystore, const nlp::PhraseTable::Match &match);

  // Deallocate wrapper.
  void Dealloc();

  // Field accessors.
  PyObject *Id();
  PyObject *Item();
  PyObject *Form();
  PyObject *Count();
  PyObject *Reliable();

  // Matched item.
  PyObject *pyitem;

  // Phrase match info.
  nlp::PhraseTable::Match info;

  // Registration.
  static PyTypeObject type;
  static PyMethodTable methods;
  static void Define(PyObject *module);
};

// Python wrapper for phrase table.
struct PyPhraseTable : public PyBase {
  // Initialize phrase table wrapper.
  int Init(PyObject *args, PyObject *kwds);

  // Deallocate phrase table wrapper.
  void Dealloc();

  // Look up entities in phrase table.
  PyObject *Lookup(PyObject *obj);

  // Look up entities in phrase table returning entities and counts.
  PyObject *Query(PyObject *obj);

  // Compute phrase fingerprint.
  PyObject *Fingerprint(PyObject *obj);

  // Compute phrase case form.
  PyObject *Form(PyObject *obj);

  // Phrase tokenizer.
  nlp::PhraseTokenizer *tokenizer;

  // Phrase table.
  nlp::PhraseTable *phrase_table;

  // Store for items.
  PyStore *pystore = nullptr;

  // Registration.
  static PyTypeObject type;
  static PyMethodTable methods;
  static void Define(PyObject *module);
};

}  // namespace sling

#endif  // SLING_PYAPI_PYPHRASE_H_

