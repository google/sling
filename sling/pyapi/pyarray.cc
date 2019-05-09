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

#include "sling/pyapi/pyarray.h"

#include "sling/pyapi/pystore.h"

namespace sling {

// Python type declarations.
PyTypeObject PyArray::type;
PyMappingMethods PyArray::mapping;
PySequenceMethods PyArray::sequence;
PyMethodTable PyArray::methods;
PyTypeObject PyItems::type;

void PyArray::Define(PyObject *module) {
  InitType(&type, "sling.Array", sizeof(PyArray), false);
  type.tp_dealloc = method_cast<destructor>(&PyArray::Dealloc);
  type.tp_str = method_cast<reprfunc>(&PyArray::Str);
  type.tp_iter = method_cast<getiterfunc>(&PyArray::Items);
  type.tp_hash = method_cast<hashfunc>(&PyArray::Hash);
  type.tp_richcompare = method_cast<richcmpfunc>(&PyArray::Compare);

  type.tp_as_mapping = &mapping;
  mapping.mp_length = method_cast<lenfunc>(&PyArray::Size);
  mapping.mp_subscript = method_cast<binaryfunc>(&PyArray::GetItems);

  type.tp_as_sequence = &sequence;
  sequence.sq_length = method_cast<lenfunc>(&PyArray::Size);
  sequence.sq_item = method_cast<ssizeargfunc>(&PyArray::GetItem);
  sequence.sq_ass_item = method_cast<ssizeobjargproc>(&PyArray::SetItem);
  sequence.sq_contains = method_cast<objobjproc>(&PyArray::Contains);

  methods.Add("get", &PyArray::Get);
  methods.Add("store", &PyArray::GetStore);
  methods.Add("data", &PyArray::Data);
  type.tp_methods = methods.table();

  RegisterType(&type, module, "Array");
}

void PyArray::Init(PyStore *pystore, Handle handle, Slice *slice) {
  // Add array as root object for store to keep it alive in the store.
  InitRoot(pystore->store, handle);
  this->slice = slice;

  // Add reference to store to keep it alive.
  this->pystore = pystore;
  Py_INCREF(pystore);
}

void PyArray::Dealloc() {
  // Unlock tracking of handle in store.
  Unlink();

  // Release reference to store.
  Py_DECREF(pystore);

  // Free object.
  delete slice;
  Free();
}

Py_ssize_t PyArray::Size() {
  return length();
}

PyObject *PyArray::GetItem(Py_ssize_t index) {
  // Check array bounds.
  ArrayDatum *arr = array();
  if (index < 0) index = length() + index;
  if (index < 0 || index >= length()) {
    PyErr_SetString(PyExc_IndexError, "Array index out of bounds");
    return nullptr;
  }

  // Return array element.
  return pystore->PyValue(arr->get(pos(index)));
}

PyObject *PyArray::Get(PyObject *args, PyObject *kw) {
  static const char *kwlist[] = {"index", "binary", nullptr};
  int index = 0;
  bool binary = false;
  if (!PyArg_ParseTupleAndKeywords(args, kw, "i|b",
          const_cast<char **>(kwlist), &index, &binary)) return nullptr;

  // Check array bounds.
  ArrayDatum *arr = array();
  if (index < 0) index = length() + index;
  if (index < 0 || index >= length()) {
    PyErr_SetString(PyExc_IndexError, "Array index out of bounds");
    return nullptr;
  }

  // Return array element.
  return pystore->PyValue(arr->get(pos(index)), binary);
}

PyObject *PyArray::GetItems(PyObject *key) {
  if (PyLong_Check(key)) {
    // Simple integer index.
    return GetItem(PyLong_AS_LONG(key));
  } else if (PySlice_Check(key)) {
    // Get index slice.
    Slice *subset = new Slice();
    if (subset->Init(key, length()) == -1) {
      delete subset;
      return nullptr;
    }

    // Combine with existing slice when making a slice of a slice.
    if (slice) subset->Combine(slice);

    // Create sliced array wrapper.
    PyArray *sliced = PyObject_New(PyArray, &PyArray::type);
    sliced->Init(pystore, handle(), subset);
    return sliced->AsObject();
  } else {
    PyErr_SetString(PyExc_IndexError, "Integer or slice expected");
    return nullptr;
  }
}

int PyArray::SetItem(Py_ssize_t index, PyObject *value) {
  // Check that array is writable.
  if (!Writable()) return -1;

  // Check array bounds.
  if (index < 0) index = length() + index;
  if (index < 0 || index >= length()) {
    PyErr_SetString(PyExc_IndexError, "Array index out of bounds");
    return -1;
  }

  // Set array element.
  Handle handle = pystore->Value(value);
  if (handle.IsError()) return -1;
  *array()->at(pos(index)) = handle;
  return 0;
}

PyObject *PyArray::Items() {
  PyItems *iter = PyObject_New(PyItems, &PyItems::type);
  iter->Init(this);
  return iter->AsObject();
}

long PyArray::Hash() {
  if (slice == nullptr) {
    return pystore->store->Fingerprint(handle(), true);
  } else {
    return pystore->store->Fingerprint(array(),
                                       slice->start,
                                       slice->stop,
                                       slice->step);
  }
}

PyObject *PyArray::Compare(PyObject *other, int op) {
  // Only equality check is supported.
  if (op != Py_EQ && op != Py_NE) {
    PyErr_SetString(PyExc_TypeError, "Unsupported array comparison");
    return nullptr;
  }

  // Check if other object is an array.
  bool match = false;
  if (PyObject_TypeCheck(other, &PyArray::type)) {
    PyArray *pyother = reinterpret_cast<PyArray *>(other);
    if (CompatibleStore(pyother)) {
      // Check if arrays are equal. Arrays are compared by value, except that
      // elements that are frames are compared by reference.
      if (slice == nullptr && pyother->slice == nullptr) {
        match = pystore->store->Equal(handle(), pyother->handle(), true);
      } else {
        int len = length();
        if (len == pyother->length()) {
          ArrayDatum *arr = array();
          ArrayDatum *other = pyother->array();
          match = true;
          for (int i = 0; i < len && match; ++i) {
            match = pystore->store->Equal(arr->get(pos(i)),
                                          other->get(pyother->pos(i)), true);
          }
        }
      }
    }
  }

  if (op == Py_NE) match = !match;
  return PyBool_FromLong(match);
}

int PyArray::Contains(PyObject *key) {
  // Get handle for key.
  Handle handle = pystore->Value(key);
  if (handle.IsError()) return -1;

  // Check if value is contained in array.
  ArrayDatum *arr = array();
  if (slice == nullptr) {
    for (int idx = 0; idx < arr->length(); ++idx) {
      if (pystore->store->Equal(arr->get(idx), handle), true) return true;
    }
  } else {
    for (int idx = slice->start; idx != slice->stop; idx += slice->step) {
      if (pystore->store->Equal(arr->get(idx), handle, true)) return true;
    }
  }
  return false;
}

PyObject *PyArray::GetStore() {
  Py_INCREF(pystore);
  return pystore->AsObject();
}

PyObject *PyArray::Str() {
  Handle h = AsValue();
  if (h.IsError()) return nullptr;
  StringPrinter printer(pystore->store);
  printer.Print(h);
  const string &text = printer.text();
  return PyUnicode_FromStringAndSize(text.data(), text.size());
}

PyObject *PyArray::Data(PyObject *args, PyObject *kw) {
  // Get arguments.
  SerializationFlags flags(pystore->store);
  if (!flags.ParseFlags(args, kw)) return nullptr;

  // Serialize frame.
  Handle h = AsValue();
  if (h.IsError()) return nullptr;
  if (flags.binary) {
    StringEncoder encoder(pystore->store);
    flags.InitEncoder(encoder.encoder());
    encoder.Encode(h);
    const string &buffer = encoder.buffer();
    return PyUnicode_FromStringAndSize(buffer.data(), buffer.size());
  } else {
    StringPrinter printer(pystore->store);
    flags.InitPrinter(printer.printer());
    printer.Print(h);
    const string &text = printer.text();
    return PyUnicode_FromStringAndSize(text.data(), text.size());
  }
}

bool PyArray::Writable() {
  if (pystore->store->frozen() || !pystore->store->Owned(handle())) {
    PyErr_SetString(PyExc_ValueError, "Array is not writable");
    return false;
  }
  return true;
}

Handle PyArray::AsValue() {
  if (slice == nullptr) return handle();
  if (pystore->store->frozen()) {
    PyErr_SetString(PyExc_ValueError, "Cannot clone array slice");
    return Handle::error();
  }
  Handle sliced = pystore->store->AllocateArray(slice->length);
  Handle *data = pystore->store->GetArray(sliced)->begin();
  ArrayDatum *arr = array();
  for (int idx = slice->start; idx != slice->stop; idx += slice->step) {
    *data++ = arr->get(idx);
  }
  return sliced;
}

bool PyArray::CompatibleStore(PyArray *other) {
  // Arrays are compatible if they are in the same store.
  if (pystore->store == other->pystore->store) return true;

  if (handle().IsLocalRef()) {
    // A local store is also compatible with its global store.
    return pystore->pyglobals->store == other->pystore->store;
  } else if (other->handle().IsLocalRef()) {
    // A global store is also compatible with a local store based on it.
    return pystore->store == other->pystore->pyglobals->store;
  } else {
    // Arrays belong to different global stores.
    return false;
  }
}

void PyItems::Define(PyObject *module) {
  InitType(&type, "sling.Items", sizeof(PyItems), false);
  type.tp_dealloc = method_cast<destructor>(&PyItems::Dealloc);
  type.tp_iter = method_cast<getiterfunc>(&PyItems::Self);
  type.tp_iternext = method_cast<iternextfunc>(&PyItems::Next);
  RegisterType(&type);
}

void PyItems::Init(PyArray *pyarray) {
  current = 0;
  this->pyarray = pyarray;
  Py_INCREF(pyarray);
}

void PyItems::Dealloc() {
  // Release reference to array.
  Py_DECREF(pyarray);

  // Free object.
  Free();
}

PyObject *PyItems::Next() {
  // Check bounds.
  if (current == pyarray->length()) {
    PyErr_SetNone(PyExc_StopIteration);
    return nullptr;
  }

  // Get next item in array.
  int index = pyarray->pos(current++);
  return pyarray->pystore->PyValue(pyarray->array()->get(index));
}

PyObject *PyItems::Self() {
  Py_INCREF(this);
  return AsObject();
}

}  // namespace sling
