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

#ifndef SLING_PYAPI_PYARRAY_H_
#define SLING_PYAPI_PYARRAY_H_

#include "sling/frame/store.h"
#include "sling/pyapi/pybase.h"
#include "sling/pyapi/pystore.h"

namespace sling {

// Python wrapper for array.
struct PyArray : public PyBase, public Root {
  // Initialize array wrapper.
  void Init(PyStore *pystore, Handle handle);

  // Deallocate array wrapper.
  void Dealloc();

  // Return the number of elements.
  Py_ssize_t Size();

  // Get element from array.
  PyObject *GetItem(Py_ssize_t index);

  // Set element in array.
  int SetItem(Py_ssize_t index, PyObject *value);

  // Return iterator for all items in array.
  PyObject *Items();

  // Return handle as hash value for array.
  long Hash();

  // Check if array contains value.
  int Contains(PyObject *key);

  // Return store for array.
  PyObject *GetStore();

  // Return array as string.
  PyObject *Str();

  // Return array in ascii or binary encoding.
  PyObject *Data(PyObject *args, PyObject *kw);

  // Check if array can be modified.
  bool Writable();

  // Return handle for array.
  Handle handle() const { return handle_; }

  // Dereference array reference.
  ArrayDatum *array() { return pystore->store->Deref(handle())->AsArray(); }

  // Store for frame.
  PyStore *pystore;

  // Registration.
  static PyTypeObject type;
  static PySequenceMethods sequence;
  static PyMethodTable methods;
  static void Define(PyObject *module);
};

// Python wrapper for array item iterator.
struct PyItems : public PyBase {
  // Initialize array item iterator.
  void Init(PyArray *pyarray);

  // Deallocate array item iterator.
  void Dealloc();

  // Return next item.
  PyObject *Next();

  // Return self.
  PyObject *Self();

  // Array that is being iterated.
  PyArray *pyarray;

  // Current item.
  int current;

  // Registration.
  static PyTypeObject type;
  static void Define(PyObject *module);
};

}  // namespace sling

#endif  // SLING_PYAPI_PYARRAY_H_

