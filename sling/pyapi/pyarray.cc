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
PySequenceMethods PyArray::sequence;
PyTypeObject PyItems::type;

PyMethodDef PyArray::methods[] = {
  {"store", PYFUNC(PyArray::GetStore), METH_NOARGS, ""},
  {"data", PYFUNC(PyArray::Data), METH_KEYWORDS, ""},
  {nullptr}
};

void PyArray::Define(PyObject *module) {
  InitType(&type, "sling.Array", sizeof(PyArray), false);
  type.tp_dealloc = method_cast<destructor>(&PyArray::Dealloc);
  type.tp_str = method_cast<reprfunc>(&PyArray::Str);
  type.tp_iter = method_cast<getiterfunc>(&PyArray::Items);
  type.tp_hash = method_cast<hashfunc>(&PyArray::Hash);
  type.tp_methods = methods;

  type.tp_as_sequence = &sequence;
  sequence.sq_length = method_cast<lenfunc>(&PyArray::Size);
  sequence.sq_item = method_cast<ssizeargfunc>(&PyArray::GetItem);
  sequence.sq_ass_item = method_cast<ssizeobjargproc>(&PyArray::SetItem);
  sequence.sq_contains = method_cast<objobjproc>(&PyArray::Contains);

  RegisterType(&type, module, "Array");
}

void PyArray::Init(PyStore *pystore, Handle handle) {
  // Add array as root object for store to keep it alive in the store.
  InitRoot(pystore->store, handle);

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
  Free();
}

Py_ssize_t PyArray::Size() {
  return array()->length();
}

PyObject *PyArray::GetItem(Py_ssize_t index) {
  // Check array bounds.
  ArrayDatum *arr = array();
  if (index < 0) index = arr->length() + index;
  if (index < 0 || index >= arr->length()) {
    PyErr_SetString(PyExc_IndexError, "Array index out of bounds");
    return nullptr;
  }

  // Return array element.
  return pystore->PyValue(arr->get(index));
}

int PyArray::SetItem(Py_ssize_t index, PyObject *value) {
  // Check that array is writable.
  if (!Writable()) return -1;

  // Check array bounds.
  if (index < 0) index = array()->length() + index;
  if (index < 0 || index >= array()->length()) {
    PyErr_SetString(PyExc_IndexError, "Array index out of bounds");
    return -1;
  }

  // Set array element.
  Handle handle = pystore->Value(value);
  if (handle.IsError()) return -1;
  *array()->at(index) = handle;
  return 0;
}

PyObject *PyArray::Items() {
  PyItems *iter = PyObject_New(PyItems, &PyItems::type);
  iter->Init(this);
  return iter->AsObject();
}

long PyArray::Hash() {
  return handle().bits;
}

int PyArray::Contains(PyObject *key) {
  // Get handle for key.
  Handle handle = pystore->Value(key);
  if (handle.IsError()) return -1;

  // Check if value is contained in array.
  ArrayDatum *arr = array();
  for (int i = 0; i < arr->length(); ++i) {
    if (arr->get(i) == handle) return true;
  }
  return false;
}

PyObject *PyArray::GetStore() {
  Py_INCREF(pystore);
  return pystore->AsObject();
}

PyObject *PyArray::Str() {
  StringPrinter printer(pystore->store);
  printer.Print(handle());
  const string &text = printer.text();
  return PyString_FromStringAndSize(text.data(), text.size());
}

PyObject *PyArray::Data(PyObject *args, PyObject *kw) {
  // Get arguments.
  SerializationFlags flags(pystore->store);
  if (!flags.ParseFlags(args, kw)) return nullptr;

  // Serialize frame.
  if (flags.binary) {
    StringEncoder encoder(pystore->store);
    flags.InitEncoder(encoder.encoder());
    encoder.Encode(handle());
    const string &buffer = encoder.buffer();
    return PyString_FromStringAndSize(buffer.data(), buffer.size());
  } else {
    StringPrinter printer(pystore->store);
    flags.InitPrinter(printer.printer());
    printer.Print(handle());
    const string &text = printer.text();
    return PyString_FromStringAndSize(text.data(), text.size());
  }
}

bool PyArray::Writable() {
  if (pystore->store->frozen() || !pystore->store->Owned(handle())) {
    PyErr_SetString(PyExc_ValueError, "Array is not writable");
    return false;
  }
  return true;
}

void PyItems::Define(PyObject *module) {
  InitType(&type, "sling.Items", sizeof(PyItems), false);
  type.tp_dealloc = method_cast<destructor>(&PyItems::Dealloc);
  type.tp_iter = method_cast<getiterfunc>(&PyItems::Self);
  type.tp_iternext = method_cast<iternextfunc>(&PyItems::Next);
  RegisterType(&type);
}

void PyItems::Init(PyArray *pyarray) {
  current = -1;
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
  ArrayDatum *arr = pyarray->array();
  if (++current >= arr->length()) {
    PyErr_SetNone(PyExc_StopIteration);
    return nullptr;
  }

  // Get next item in array.
  return pyarray->pystore->PyValue(arr->get(current));
}

PyObject *PyItems::Self() {
  Py_INCREF(this);
  return AsObject();
}

}  // namespace sling

