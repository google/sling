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

#include "sling/frame/snapshot.h"
#include "sling/frame/turtle.h"
#include "sling/frame/xml.h"
#include "sling/pyapi/pyarray.h"
#include "sling/pyapi/pyframe.h"
#include "sling/stream/file.h"

namespace sling {

// Python type declarations.
PyTypeObject PyStore::type;
PyMappingMethods PyStore::mapping;
PySequenceMethods PyStore::sequence;
PyMethodTable PyStore::methods;
PyTypeObject PySymbols::type;

void PyStore::Define(PyObject *module) {
  InitType(&type, "sling.Store", sizeof(PyStore), true);

  type.tp_init = method_cast<initproc>(&PyStore::Init);
  type.tp_dealloc = method_cast<destructor>(&PyStore::Dealloc);
  type.tp_iter = method_cast<getiterfunc>(&PyStore::Symbols);

  methods.Add("freeze", &PyStore::Freeze);
  methods.Add("load", &PyStore::Load);
  methods.Add("save", &PyStore::Save);
  methods.Add("parse", &PyStore::Parse);
  methods.AddO("frame", &PyStore::NewFrame);
  methods.AddO("array", &PyStore::NewArray);
  methods.AddO("resolve", &PyStore::Resolve);
  methods.Add("globals", &PyStore::Globals);
  methods.Add("lockgc", &PyStore::LockGC);
  methods.Add("unlockgc", &PyStore::UnlockGC);
  type.tp_methods = methods.table();

  type.tp_as_mapping = &mapping;
  mapping.mp_length = method_cast<lenfunc>(&PyStore::Size);
  mapping.mp_subscript = method_cast<binaryfunc>(&PyStore::Lookup);

  type.tp_as_sequence = &sequence;
  sequence.sq_contains = method_cast<objobjproc>(&PyStore::Contains);

  RegisterType(&type, module, "Store");
}

int PyStore::Init(PyObject *args, PyObject *kwds) {
  // Get optional globals argument.
  PyStore *globals = nullptr;
  if (!PyArg_ParseTuple(args, "|O", &globals)) return -1;

  // Create new store.
  if (globals != nullptr) {
    // Check that argument is a store.
    if (!PyStore::TypeCheck(globals)) return -1;

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
  if (store != nullptr) store->Release();
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
  static const char *kwlist[] = {"filename", "binary", "snapshot", nullptr};
  char *filename = nullptr;
  bool force_binary = false;
  bool snapshot = true;
  bool ok = PyArg_ParseTupleAndKeywords(
                args, kw, "s|bb", const_cast<char **>(kwlist),
                &filename, &force_binary, &snapshot);
  if (!ok) return nullptr;

  // Check that store is writable.
  if (!Writable()) return nullptr;

  // Read frames from file.
  if (snapshot && store->Pristine() && Snapshot::Valid(filename)) {
    // Load store from snapshot.
    Status st = Snapshot::Read(store, filename);
    if (!st.ok()) {
      PyErr_SetString(PyExc_IOError, st.message());
      return nullptr;
    }
    Py_RETURN_NONE;
  } else {
    // Load store store from file. First, open input file.
    File *f;
    Status st = File::Open(filename, "r", &f);
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
  }
}

PyObject *PyStore::Save(PyObject *args, PyObject *kw) {
  // Get arguments.
  SerializationFlags flags(store);
  char *filename = flags.ParseArgs(args, kw);
  if (filename == nullptr) return nullptr;

  // Get output stream.
  File *f;
  Status st = File::Open(filename, "w", &f);
  if (!st.ok()) {
    PyErr_SetString(PyExc_IOError, st.message());
    return nullptr;
  }

  // Write frames to output.
  FileOutputStream stream(f);
  Output output(&stream);
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
  Py_RETURN_NONE;
}

PyObject *PyStore::Parse(PyObject *args, PyObject *kw) {
  // Parse arguments.
  static const char *kwlist[] = {
    "data", "binary", "json", "xml", "ttl", nullptr
  };
  PyObject *object = nullptr;
  bool force_binary = false;
  bool json = false;
  bool xml = false;
  bool ttl = false;
  bool ok = PyArg_ParseTupleAndKeywords(
                args, kw, "O|bbbb", const_cast<char **>(kwlist),
                &object, &force_binary, &json, &xml, &ttl);
  if (!ok) return nullptr;

  // Check that store is writable.
  if (!Writable()) return nullptr;

  // Get data buffer.
  const char *data;
  Py_ssize_t length;
  if (PyUnicode_Check(object)) {
    data = PyUnicode_AsUTF8AndSize(object, &length);
    if (data == nullptr) return nullptr;
  } else {
    char *ptr;
    if (PyBytes_AsStringAndSize(object, &ptr, &length) == -1) return nullptr;
    data = ptr;
  }
  ArrayInputStream stream(data, length);

  if (xml) {
    // Parse input as XML.
    Input input(&stream);
    XMLReader reader(store, &input);
    Frame result = reader.Read();
    if (result.IsNil()) {
      PyErr_SetString(PyExc_IOError, "XML error");
      return nullptr;
    }
    return PyValue(result.handle());
  } else if (ttl) {
    // Parse input as TTL.
    Input input(&stream);
    TurtleParser parser(store, &input);
    Object result = parser.ReadAll();
    if (parser.error()) {
      PyErr_SetString(PyExc_IOError, parser.error_message().c_str());
      return nullptr;
    }
    return PyValue(result.handle());
  } else {
    // Load frames from memory buffer.
    InputParser parser(store, &stream, force_binary, json);
    Object result = parser.ReadAll();
    if (parser.error()) {
      PyErr_SetString(PyExc_IOError, parser.error_message().c_str());
      return nullptr;
    }
    return PyValue(result.handle());
  }
}

Py_ssize_t PyStore::Size() {
  return store->num_symbols();
}

PyObject *PyStore::Lookup(PyObject *key) {
  // Get symbol name.
  const char *name = GetString(key);
  if (name == nullptr) return nullptr;

  // Lookup name in symbol table.
  Handle handle = store->Lookup(name);
  return PyValue(handle);
}

PyObject *PyStore::Resolve(PyObject *object) {
  if (PyObject_TypeCheck(object, &PyFrame::type)) {
    PyFrame *pyframe = reinterpret_cast<PyFrame *>(object);
    Handle handle = pyframe->handle();
    Handle qua = store->Resolve(handle);
    if (qua != handle) return PyValue(qua);
  }
  Py_INCREF(object);
  return object;
}

int PyStore::Contains(PyObject *key) {
  // Get symbol name.
  const char *name = GetString(key);
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

  GCLock lock(store);
  std::vector<Slot> slots;

  // If the argument is a string, create a frame with that id.
  if (PyUnicode_Check(arg)) {
    slots.emplace_back(Handle::id(), SymbolValue(arg));
  } else {
    // Parse data into slot list.
    if (!SlotList(arg, &slots)) return nullptr;
  }

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
    // Initialize new array from Python list.
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
    int size = PyLong_AsLong(arg);
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

PyObject *PyStore::LockGC() {
  store->LockGC();
  Py_RETURN_NONE;
}

PyObject *PyStore::UnlockGC() {
  store->UnlockGC();
  Py_RETURN_NONE;
}

PyObject *PyStore::PyValue(Handle handle, bool binary) {
  switch (handle.tag()) {
    case Handle::kGlobal:
    case Handle::kLocal: {
      // Return None for nil.
      if (handle.IsNil()) Py_RETURN_NONE;

      // Get datum for object.
      Datum *datum = store->Deref(handle);

      // Convert SLING object to Python object.
      if (datum->IsFrame()) {
        // Return new frame wrapper for handle.
        PyFrame *frame = PyObject_New(PyFrame, &PyFrame::type);
        frame->Init(this, handle);
        return frame->AsObject();
      } else if (datum->IsString()) {
        StringDatum *str = datum->AsString();
        PyObject *pystr;
        if (binary) {
          // Return string as bytes.
          pystr = PyBytes_FromStringAndSize(str->data(), str->size());
        } else {
          // Return unicode string object.
          pystr = PyUnicode_FromStringAndSize(str->data(), str->size());
          if (pystr == nullptr) {
            // Fall back to bytes if string is not valid UTF8.
            PyErr_Clear();
            pystr = PyBytes_FromStringAndSize(str->data(), str->size());
          }
        }
        return pystr;
      } else if (datum->IsArray()) {
        // Return new frame array for handle.
        PyArray *array = PyObject_New(PyArray, &PyArray::type);
        array->Init(this, handle);
        return array->AsObject();
      } else if (datum->IsSymbol()) {
        // Return symbol name.
        SymbolDatum *symbol = datum->AsSymbol();
        StringDatum *str = store->Deref(symbol->name)->AsString();
        return PyUnicode_FromStringAndSize(str->data(), str->size());
      } else {
        // Unsupported type.
        PyErr_SetString(PyExc_ValueError, "Unsupported object type");
        return nullptr;
      }
    }

    case Handle::kIntTag:
      // Return integer object.
      return PyLong_FromLong(handle.AsInt());

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
  } else if (PyUnicode_Check(object)) {
    // Create string and return handle.
    if (!Writable()) return Handle::error();
    Py_ssize_t length;
    const char *data = PyUnicode_AsUTF8AndSize(object, &length);
    return  store->AllocateString(Text(data, length));
  } else if (PyBytes_Check(object)) {
    // Create string from bytes and return handle.
    if (!Writable()) return Handle::error();
    char *data;
    Py_ssize_t length;
    PyBytes_AsStringAndSize(object, &data, &length);
    return store->AllocateString(Text(data, length));
  } else if (PyLong_Check(object)) {
    // Return integer handle.
    return Handle::Integer(PyLong_AsLong(object));
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
    return array->AsValue();
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
      if (name.IsId() && PyUnicode_Check(v)) {
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
  if (PyUnicode_Check(object)) {
    const char *name = GetString(object);
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
  if (PyUnicode_Check(object)) {
    const char *name = GetString(object);
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
      if (name.IsId() && PyUnicode_Check(v)) {
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
      if (name.IsId() && PyUnicode_Check(v)) {
        value = SymbolValue(v);
      } else {
        value = Value(v);
      }
      if (value.IsError()) return false;

      // Add slot.
      slots->emplace_back(name, value);
    }
  } else {
    PyErr_SetString(PyExc_ValueError, "Invalid slot list");
    return false;
  }
  return true;
}

void PySymbols::Define(PyObject *module) {
  InitType(&type, "sling.Symbols", sizeof(PySymbols), false);
  type.tp_dealloc = method_cast<destructor>(&PySymbols::Dealloc);
  type.tp_iter = method_cast<getiterfunc>(&PySymbols::Self);
  type.tp_iternext = method_cast<iternextfunc>(&PySymbols::Next);
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
  printer->set_utf8(utf8);
}

char *SerializationFlags::ParseArgs(PyObject *args, PyObject *kw) {
  static const char *kwlist[] = {
    "filename", "binary", "global", "shallow", "byref", "pretty",
    "utf8", "json", nullptr
  };
  char *filename = nullptr;
  bool ok = PyArg_ParseTupleAndKeywords(
      args, kw, "s|bbbbbbb", const_cast<char **>(kwlist),
      &filename, &binary, &global, &shallow, &byref, &pretty, &utf8, &json);
  if (!ok) return nullptr;
  return filename;
}

bool SerializationFlags::ParseFlags(PyObject *args, PyObject *kw) {
  static const char *kwlist[] = {
    "binary", "global", "shallow", "byref", "pretty",
    "utf8", "json", nullptr
  };
  return PyArg_ParseTupleAndKeywords(
      args, kw, "|bbbbbbb", const_cast<char **>(kwlist),
      &binary, &global, &shallow, &byref, &pretty, &utf8, &json);
}

}  // namespace sling

