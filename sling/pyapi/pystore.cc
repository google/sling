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

#include "sling/pyapi/pystore.h"

#include "sling/pyapi/pyarray.h"
#include "sling/pyapi/pyframe.h"
#include "sling/stream/file.h"
#include "sling/stream/unix-file.h"

namespace sling {

// Python type declarations.
PyTypeObject PyStore::type;
PyMappingMethods PyStore::mapping;
PySequenceMethods PyStore::sequence;
PyTypeObject PySymbols::type;

PyMethodDef PyStore::methods[] = {
  {"freeze", (PyCFunction) &PyStore::Freeze, METH_NOARGS, ""},
  {"load", (PyCFunction) &PyStore::Load, METH_VARARGS | METH_KEYWORDS, ""},
  {"save", (PyCFunction) &PyStore::Save, METH_VARARGS | METH_KEYWORDS, ""},
  {"parse", (PyCFunction) &PyStore::Parse, METH_VARARGS| METH_KEYWORDS, ""},
  {"frame", (PyCFunction) &PyStore::NewFrame, METH_O, ""},
  {"array", (PyCFunction) &PyStore::NewArray, METH_O, ""},
  {"globals", (PyCFunction) &PyStore::Globals, METH_NOARGS, ""},
  {nullptr}
};

void PyStore::Define(PyObject *module) {
  InitType(&type, "sling.Store", sizeof(PyStore), true);

  type.tp_init = reinterpret_cast<initproc>(&PyStore::Init);
  type.tp_dealloc = reinterpret_cast<destructor>(&PyStore::Dealloc);
  type.tp_iter = &PyStore::Symbols;
  type.tp_methods = methods;

  type.tp_as_mapping = &mapping;
  mapping.mp_length = &PyStore::Size;
  mapping.mp_subscript = &PyStore::Lookup;

  type.tp_as_sequence = &sequence;
  sequence.sq_contains = &PyStore::Contains;

  RegisterType(&type, module, "Store");
}

int PyStore::Init(PyObject *args, PyObject *kwds) {
  // Get optional globals argument.
  PyStore *globals = nullptr;
  if (!PyArg_ParseTuple(args, "|O", &globals)) return -1;

  // Create new store.
  if (globals != nullptr) {
    // Check that argument is a store.
    if (!PyObject_TypeCheck(globals, &type)) return -1;

    // Check that global has been frozen.
    if (!globals->store->frozen()) {
      PyErr_SetString(PyExc_ValueError, "Global store is not frozen");
      return -1;
    }

    // Create local store.
    store = new Store(globals->store);

    // Save reference to global store.
    pyglobals = globals;
    Py_INCREF(pyglobals);
  } else {
    // Create global store.
    store = new Store();
    pyglobals = nullptr;
  }

  // Make new store shared.
  store->Share();

  return 0;
}

void PyStore::Dealloc() {
  store->Release();
  if (pyglobals != nullptr) Py_DECREF(pyglobals);
  Free();
}

PyObject *PyStore::Freeze() {
  if (store->globals() != nullptr) {
    PyErr_SetString(PyExc_ValueError, "Local store cannot be frozen");
    return nullptr;
  }
  store->Freeze();
  Py_RETURN_NONE;
}

PyObject *PyStore::Load(PyObject *args, PyObject *kw) {
  // Parse arguments.
  static const char *kwlist[] = {"file", "binary", nullptr};
  PyObject *file = nullptr;
  bool force_binary = false;
  bool ok = PyArg_ParseTupleAndKeywords(
                args, kw, "O|b", const_cast<char **>(kwlist),
                &file, &force_binary);
  if (!ok) return nullptr;

  // Check that store is writable.
  if (!Writable()) return nullptr;

  // Read frames from file.
  if (PyFile_Check(file)) {
    // Load store from file object.
    StdFileInputStream stream(PyFile_AsFile(file), false);
    InputParser parser(store, &stream, force_binary);
    store->LockGC();
    Object result = parser.ReadAll();
    store->UnlockGC();
    if (parser.error()) {
      PyErr_SetString(PyExc_IOError, parser.error_message().c_str());
      return nullptr;
    }
    return PyValue(result.handle());
  } else if (PyString_Check(file)) {
    // Load store store from file. First, open input file.
    File *f;
    Status st = File::Open(PyString_AsString(file), "r", &f);
    if (!st.ok()) {
      PyErr_SetString(PyExc_IOError, st.message());
      return nullptr;
    }

    // Load frames from file.
    FileInputStream stream(f);
    InputParser parser(store, &stream, force_binary);
    store->LockGC();
    Object result = parser.ReadAll();
    store->UnlockGC();
    if (parser.error()) {
      PyErr_SetString(PyExc_IOError, parser.error_message().c_str());
      return nullptr;
    }
    return PyValue(result.handle());
  } else {
    PyErr_SetString(PyExc_ValueError, "File or string argument expected");
    return nullptr;
  }
}

PyObject *PyStore::Save(PyObject *args, PyObject *kw) {
  // Get arguments.
  SerializationFlags flags(store);
  PyObject *file = flags.ParseArgs(args, kw);

  // Get output stream.
  OutputStream *stream;
  if (PyFile_Check(file)) {
    // Create stream from stdio file.
    stream = new StdFileOutputStream(PyFile_AsFile(file), false);
  } else if (PyString_Check(file)) {
    // Open output file.
    File *f;
    Status st = File::Open(PyString_AsString(file), "w", &f);
    if (!st.ok()) {
      PyErr_SetString(PyExc_IOError, st.message());
      return nullptr;
    }
    stream = new FileOutputStream(f);
  } else {
    PyErr_SetString(PyExc_ValueError, "File or string argument expected");
    return nullptr;
  }

  // Write frames to output.
  Output output(stream);
  if (flags.binary) {
    Encoder encoder(store, &output);
    flags.InitEncoder(&encoder);
    encoder.EncodeAll();
  } else {
    Printer printer(store, &output);
    flags.InitPrinter(&printer);
    printer.PrintAll();
  }

  output.Flush();
  delete stream;
  Py_RETURN_NONE;
}

PyObject *PyStore::Parse(PyObject *args, PyObject *kw) {
  // Parse arguments.
  static const char *kwlist[] = {"data", "binary", nullptr};
  PyObject *object = nullptr;
  bool force_binary = false;
  bool ok = PyArg_ParseTupleAndKeywords(
                args, kw, "S|b", const_cast<char **>(kwlist),
                  &object, &force_binary);
  if (!ok) return nullptr;

  // Check that store is writable.
  if (!Writable()) return nullptr;

  // Get data buffer.
  char *data;
  Py_ssize_t length;
  PyString_AsStringAndSize(object, &data, &length);

  // Load frames from memory buffer.
  ArrayInputStream stream(data, length);
  InputParser parser(store, &stream, force_binary);
  Object result = parser.ReadAll();
  if (parser.error()) {
    PyErr_SetString(PyExc_IOError, parser.error_message().c_str());
    return nullptr;
  }
  return PyValue(result.handle());
}

Py_ssize_t PyStore::Size() {
  return store->num_symbols();
}

PyObject *PyStore::Lookup(PyObject *key) {
  // Get symbol name.
  char *name = PyString_AsString(key);
  if (name == nullptr) return nullptr;

  // Lookup name in symbol table.
  Handle handle = store->Lookup(name);
  return PyValue(handle);
}

int PyStore::Contains(PyObject *key) {
  // Get symbol name.
  char *name = PyString_AsString(key);
  if (name == nullptr) return -1;

  // Lookup name in symbol table.
  Handle handle = store->LookupExisting(name);
  return !handle.IsNil();
}

PyObject *PyStore::Symbols() {
  PySymbols *iter = PyObject_New(PySymbols, &PySymbols::type);
  iter->Init(this);
  return iter->AsObject();
}

PyObject *PyStore::NewFrame(PyObject *arg) {
  // Check that store is writable.
  if (!Writable()) return nullptr;

  // Parse data into slot list.
  GCLock lock(store);
  std::vector<Slot> slots;
  if (!SlotList(arg, &slots)) return nullptr;

  // Allocate new frame.
  Slot *begin = slots.data();
  Slot *end = slots.data() + slots.size();
  Handle handle = store->AllocateFrame(begin, end);

  // Return new frame wrapper for handle.
  PyFrame *frame = PyObject_New(PyFrame, &PyFrame::type);
  frame->Init(this, handle);
  return frame->AsObject();
}

PyObject *PyStore::NewArray(PyObject *arg) {
  // Check that store is writable.
  if (!Writable()) return nullptr;

  GCLock lock(store);
  Handle handle;
  if (PyList_Check(arg)) {
    // Inialize new array from Python list.
    int size = PyList_Size(arg);
    handle = store->AllocateArray(size);
    ArrayDatum *array = store->Deref(handle)->AsArray();
    for (int i = 0; i < size; ++i) {
      PyObject *item = PyList_GetItem(arg, i);
      Handle value = Value(item);
      if (value.IsError()) return nullptr;
      *array->at(i) = value;
    }
  } else {
    int size = PyInt_AsLong(arg);
    if (size < 0) return nullptr;
    handle = store->AllocateArray(size);
  }

  // Return array wrapper for new array.
  PyArray *array = PyObject_New(PyArray, &PyArray::type);
  array->Init(this, handle);
  return array->AsObject();
}

bool PyStore::Writable() {
  if (store->frozen()) {
    PyErr_SetString(PyExc_ValueError, "Frame store is not writable");
    return false;
  }
  return true;
}

PyObject *PyStore::Globals() {
  // Return None if store is not a local store.
  if (pyglobals == nullptr) Py_RETURN_NONE;

  Py_INCREF(pyglobals);
  return pyglobals->AsObject();
}

PyObject *PyStore::PyValue(Handle handle) {
  switch (handle.tag()) {
    case Handle::kGlobal:
    case Handle::kLocal: {
      // Return None for nil.
      if (handle.IsNil()) Py_RETURN_NONE;

      // Get datum for object.
      Datum *datum = store->Deref(handle);

      if (datum->IsFrame()) {
        // Return new frame wrapper for handle.
        PyFrame *frame = PyObject_New(PyFrame, &PyFrame::type);
        frame->Init(this, handle);
        return frame->AsObject();
      } else if (datum->IsString()) {
        // Return string object.
        StringDatum *str = datum->AsString();
        return PyString_FromStringAndSize(str->data(), str->size());
      } else if (datum->IsArray()) {
        // Return new frame array for handle.
        PyArray *array = PyObject_New(PyArray, &PyArray::type);
        array->Init(this, handle);
        return array->AsObject();
      } else if (datum->IsSymbol()) {
        // Return symbol name.
        SymbolDatum *symbol = datum->AsSymbol();
        StringDatum *str = store->Deref(symbol->name)->AsString();
        return PyString_FromStringAndSize(str->data(), str->size());
      } else {
        // Unsupported type.
        PyErr_SetString(PyExc_ValueError, "Unsupported object type");
        return nullptr;
      }
    }

    case Handle::kIntTag:
      // Return integer object.
      return PyInt_FromLong(handle.AsInt());

    case Handle::kFloatTag:
      // Return floating point number object.
      return PyFloat_FromDouble(handle.AsFloat());
  }

  return nullptr;
}

Handle PyStore::Value(PyObject *object) {
  if (object == Py_None) {
    return Handle::nil();
  } else if (PyObject_TypeCheck(object, &PyFrame::type)) {
    // Return handle for frame.
    PyFrame *frame = reinterpret_cast<PyFrame *>(object);
    if (frame->pystore->store != store &&
        frame->pystore->store != store->globals()) {
      PyErr_SetString(PyExc_ValueError, "Frame does not belong to this store");
      return Handle::error();
    }
    return frame->handle();
  } else if (PyString_Check(object)) {
    // Create string and return handle.
    if (!Writable()) return Handle::error();
    char *data;
    Py_ssize_t length;
    PyString_AsStringAndSize(object, &data, &length);
    return  store->AllocateString(Text(data, length));
  } else if (PyInt_Check(object)) {
    // Return integer handle.
    return Handle::Integer(PyInt_AsLong(object));
  } else if (PyFloat_Check(object)) {
    // Return floating point number handle.
    return Handle::Float(PyFloat_AsDouble(object));
  } else if (PyObject_TypeCheck(object, &PyArray::type)) {
    // Return handle for array.
    PyArray *array = reinterpret_cast<PyArray *>(object);
    if (array->pystore->store != store &&
        array->pystore->store != store->globals()) {
      PyErr_SetString(PyExc_ValueError, "Array does not belong to this store");
      return Handle::error();
    }
    return array->handle();
  } else if (PyDict_Check(object)) {
    // Build frame from dictionary.
    if (!Writable()) return Handle::error();
    GCLock lock(store);
    PyObject *k;
    PyObject *v;
    Py_ssize_t pos = 0;
    std::vector<Slot> slots;
    slots.reserve(PyDict_Size(object));
    while (PyDict_Next(object, &pos, &k, &v)) {
      // Get slot name.
      Handle name = RoleValue(k);
      if (name.IsError()) return Handle::error();

      // Get slot value.
      Handle value;
      if (name.IsId() && PyString_Check(v)) {
        value = SymbolValue(v);
      } else {
        value = Value(v);
      }
      if (value.IsError()) return Handle::error();

      // Add slot.
      slots.emplace_back(name, value);
    }

    // Allocate new frame.
    Slot *begin = slots.data();
    Slot *end = slots.data() + slots.size();
    return store->AllocateFrame(begin, end);
  } else if (PyList_Check(object)) {
    // Build array from list.
    if (!Writable()) return Handle::error();
    GCLock lock(store);
    int size = PyList_Size(object);
    Handle handle = store->AllocateArray(size);
    ArrayDatum *array = store->Deref(handle)->AsArray();
    for (int i = 0; i < size; ++i) {
      PyObject *item = PyList_GetItem(object, i);
      Handle value = Value(item);
      if (value.IsError()) return Handle::error();
      *array->at(i) = value;
    }
    return handle;
  } else {
    PyErr_SetString(PyExc_ValueError, "Unsupported frame value type");
    return Handle::error();
  }
}

Handle PyStore::RoleValue(PyObject *object, bool existing) {
  if (PyString_Check(object)) {
    char *name = PyString_AsString(object);
    if (name == nullptr) return Handle::error();
    if (existing) {
      return store->LookupExisting(name);
    } else {
      return store->Lookup(name);
    }
  } else {
    return Value(object);
  }
}

Handle PyStore::SymbolValue(PyObject *object) {
  if (PyString_Check(object)) {
    char *name = PyString_AsString(object);
    if (name == nullptr) return Handle::error();
    return store->Symbol(name);
  } else {
    return Value(object);
  }
}

bool PyStore::SlotList(PyObject *object, std::vector<Slot> *slots) {
  if (PyDict_Check(object)) {
    // Build slots from key/value pairs in dictionary.
    PyObject *k;
    PyObject *v;
    Py_ssize_t pos = 0;
    slots->reserve(slots->size() + PyDict_Size(object));
    while (PyDict_Next(object, &pos, &k, &v)) {
      // Get slot name.
      Handle name = RoleValue(k);
      if (name.IsError()) return false;

      // Get slot value.
      Handle value;
      if (name.IsId() && PyString_Check(v)) {
        value = SymbolValue(v);
      } else {
        value = Value(v);
      }
      if (value.IsError()) return false;

      // Add slot.
      slots->emplace_back(name, value);
    }
  } else if (PyList_Check(object)) {
    // Build slots from list of 2-tuples.
    int size = PyList_Size(object);
    slots->reserve(slots->size() + size);
    for (int i = 0; i < size; ++i) {
      // Check that item is a 2-tuple.
      PyObject *item = PyList_GetItem(object, i);
      if (!PyTuple_Check(item) ||  PyTuple_Size(item) != 2) {
        PyErr_SetString(PyExc_ValueError, "Slot list must contain 2-tuples");
        return false;
      }

      // Get slot name.
      PyObject *k = PyTuple_GetItem(item, 0);
      Handle name = RoleValue(k);
      if (name.IsError()) return false;

      // Get slot value.
      PyObject *v = PyTuple_GetItem(item, 1);
      Handle value;
      if (name.IsId() && PyString_Check(v)) {
        value = SymbolValue(v);
      } else {
        value = Value(v);
      }
      if (value.IsError()) return false;

      // Add slot.
      slots->emplace_back(name, value);
    }
  } else {
    return false;
  }
  return true;
}

void PySymbols::Define(PyObject *module) {
  InitType(&type, "sling.Symbols", sizeof(PySymbols), false);
  type.tp_dealloc = reinterpret_cast<destructor>(&PySymbols::Dealloc);
  type.tp_iter = &PySymbols::Self;
  type.tp_iternext = &PySymbols::Next;
  RegisterType(&type, module, "Symbols");
}

void PySymbols::Init(PyStore *pystore) {
  // Initialize iterator.
  bucket = -1;
  current = Handle::nil();

  // Add reference to store to keep it alive.
  this->pystore = pystore;
  Py_INCREF(pystore);
}

void PySymbols::Dealloc() {
  Py_DECREF(pystore);
  Free();
}

PyObject *PySymbols::Next() {
  // Get next bucket if needed.
  if (current.IsNil()) {
    MapDatum *symbols = pystore->store->GetMap(pystore->store->symbols());
    while (current.IsNil()) {
      if (++bucket >= symbols->length()) {
        PyErr_SetNone(PyExc_StopIteration);
        return nullptr;
      }
      current = symbols->get(bucket);
    }
  }

  // Get next symbol in bucket.
  SymbolDatum *symbol = pystore->store->Deref(current)->AsSymbol();
  current = symbol->next;
  return pystore->PyValue(symbol->value);
}

PyObject *PySymbols::Self() {
  Py_INCREF(this);
  return AsObject();
}

SerializationFlags::SerializationFlags(Store *store) {
  if (store->globals() == 0) global = true;
}

void SerializationFlags::InitEncoder(Encoder *encoder) {
  encoder->set_shallow(shallow);
  encoder->set_global(global);
}

void SerializationFlags::InitPrinter(Printer *printer) {
  printer->set_indent(pretty ? 2 : 0);
  printer->set_shallow(shallow);
  printer->set_global(global);
  printer->set_byref(byref);
}

PyObject *SerializationFlags::ParseArgs(PyObject *args, PyObject *kw) {
  static const char *kwlist[] = {
    "file", "binary", "global", "shallow", "byref", "pretty", nullptr
  };
  PyObject *file = nullptr;
  bool ok = PyArg_ParseTupleAndKeywords(
                args, kw, "O|bbbbb", const_cast<char **>(kwlist),
                &file, &binary, &global, &shallow, &byref, &pretty);
  if (!ok) return nullptr;
  return file;
}

bool SerializationFlags::ParseFlags(PyObject *args, PyObject *kw) {
  static const char *kwlist[] = {
    "binary", "global", "shallow", "byref", "pretty", nullptr
  };
  return PyArg_ParseTupleAndKeywords(
      args, kw, "|bbbbb", const_cast<char **>(kwlist),
      &binary, &global, &shallow, &byref, &pretty);
}

}  // namespace sling

