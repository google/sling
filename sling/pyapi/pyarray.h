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
  // A slice represents a subset of the elements in an array.
  struct Slice {
    // Initialize slice from Python slice object.
    int Init(PyObject *slice, Py_ssize_t size) {
      int rc = PySlice_GetIndicesEx(slice, size, &start, &stop, &step, &length);
      stop = start + step * length;
      return rc;
    }

    // Combine slice with a base slice.
    void Combine(Slice *base) {
      start = base->pos(start);
      stop = base->pos(stop);
      step *= base->step;
      DCHECK_EQ(stop, start + step * length);
    }

    // Position of index in underlying array, where index is between 0 and
    // length.
    int pos(int index) const {
      DCHECK_GE(index, 0);
      DCHECK_LT(index, length);
      return start + step * index;
    }

    Py_ssize_t start;
    Py_ssize_t stop;
    Py_ssize_t step;
    Py_ssize_t length;
  };

  // Initialize array wrapper. Takes ownership of the slice.
  void Init(PyStore *pystore, Handle handle, Slice *slice = nullptr);

  // Deallocate array wrapper.
  void Dealloc();

  // Return the number of elements.
  Py_ssize_t Size();

  // Get element from array.
  PyObject *GetItem(Py_ssize_t index);

  // Get element from array with options.
  PyObject *Get(PyObject *args, PyObject *kw);

  // Get elements from array.
  PyObject *GetItems(PyObject *key);

  // Set element in array.
  int SetItem(Py_ssize_t index, PyObject *value);

  // Return iterator for all items in array.
  PyObject *Items();

  // Return handle as hash value for array.
  long Hash();

  // Check if array is equal to another object.
  PyObject *Compare(PyObject *other, int op);

  // Check if array contains value.
  int Contains(PyObject *key);

  // Return store for array.
  PyObject *GetStore();

  // Return array as string.
  PyObject *Str();

  // Return array in text or binary encoding.
  PyObject *Data(PyObject *args, PyObject *kw);

  // Check if another array belongs to a compatible store.
  bool CompatibleStore(PyArray *other);

  // Check if array can be modified.
  bool Writable();

  // Return handle for array. If it is an array slice, this will create a new
  // array with the sliced elements.
  Handle AsValue();

  // Return length of array (or slice).
  int length() {
    return slice == nullptr ? array()->length() : slice->length;
  }

  // Return element position in the underlying array.
  int pos(int index) {
    return slice == nullptr ? index : slice->pos(index);
  }

  // Return handle for array.
  Handle handle() const { return handle_; }

  // Dereference array reference.
  ArrayDatum *array() { return pystore->store->Deref(handle())->AsArray(); }

  // Store for array.
  PyStore *pystore;

  // Array slice or the whole array if the slice is null.
  Slice *slice;

  // Registration.
  static PyTypeObject type;
  static PyMappingMethods mapping;
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

