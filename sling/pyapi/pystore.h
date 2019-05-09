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

#ifndef SLING_PYAPI_PYSTORE_H_
#define SLING_PYAPI_PYSTORE_H_

#include "sling/frame/store.h"
#include "sling/frame/serialization.h"
#include "sling/pyapi/pybase.h"

namespace sling {

// Python wrapper for frame store.
struct PyStore : public PyBase {
  // Initialize new store.
  int Init(PyObject *args, PyObject *kwds);

  // Deallocate store.
  void Dealloc();

  // Freeze store.
  PyObject *Freeze();

  // Load frames from file.
  PyObject *Load(PyObject *args, PyObject *kw);

  // Save frames to file.
  PyObject *Save(PyObject *args, PyObject *kw);

  // Parse string as binary or ascii encoded frames.
  PyObject *Parse(PyObject *args, PyObject *kw);

  // Resolves the given object.
  PyObject *Resolve(PyObject *object);

  // Return the number of objects in the symbol table.
  Py_ssize_t Size();

  // Look up object in symbol table.
  PyObject *Lookup(PyObject *key);

  // Check if symbol is in store.
  int Contains(PyObject *key);

  // Return iterator for all symbols in symbol table.
  PyObject *Symbols();

  // Create new frame.
  PyObject *NewFrame(PyObject *arg);

  // Create new array.
  PyObject *NewArray(PyObject *arg);

  // Return global store for local store.
  PyObject *Globals();

  // Lock/unlock garbage collection for the store.
  PyObject *LockGC();
  PyObject *UnlockGC();

  // Create new Python object for handle value.
  PyObject *PyValue(Handle handle, bool binary=false);

  // Check if store can be modified.
  bool Writable();

  // Get handle value for Python object. Returns Handle::error() if the value
  // could not be converted.
  Handle Value(PyObject *object);

  // Get role handle value for Python object. This is similar to Value() except
  // that strings are considered to be symbol names. If existing=true then
  // nil will be returned if the symbol does not already exist.
  Handle RoleValue(PyObject *object, bool existing = false);

  // Get symbol handle value for Python object.
  Handle SymbolValue(PyObject *object);

  // Convert Python object to slot list. The Python object can either be a
  // dict or a list of 2-tuples.
  bool SlotList(PyObject *object, std::vector<Slot> *slots);

  // Underlying frame store.
  Store *store;

  // Global store or null if this is not a local store.
  PyStore *pyglobals;

  // Type checking.
  static bool TypeCheck(PyBase *object) {
    return PyBase::TypeCheck(object, &type);
  }

  // Registration.
  static PyTypeObject type;
  static PyMappingMethods mapping;
  static PySequenceMethods sequence;
  static PyMethodTable methods;
  static void Define(PyObject *module);
};

// Python wrapper for symbol iterator.
struct PySymbols : public PyBase {
  // Initialize symbol iterator.
  void Init(PyStore *pystore);

  // Deallocate symbol iterator.
  void Dealloc();

  // Return next symbol.
  PyObject *Next();

  // Return self.
  PyObject *Self();

  // Store for symbols.
  PyStore *pystore;

  // Current symbol table bucket.
  int bucket;

  // Current symbol handle.
  Handle current;

  // Registration.
  static PyTypeObject type;
  static void Define(PyObject *module);
};

// Flags for serializing frames.
struct SerializationFlags {
  // Initialize serialization flags for store.
  SerializationFlags(Store *store);

  // Set flags for encoder.
  void InitEncoder(Encoder *encoder);

  // Set flags for printer.
  void InitPrinter(Printer *printer);

  // Parse arguments for methods taking a filename argument.
  char *ParseArgs(PyObject *args, PyObject *kw);

  // Parse arguments for methods taking no fixed arguments.
  bool ParseFlags(PyObject *args, PyObject *kw);

  bool binary = false;  // output in binary encoding
  bool global = false;  // output frames in the global store by value
  bool shallow = true;  // output frames with ids by reference
  bool byref = true;    // output anonymous frames by reference using index ids
  bool pretty = false;  // pretty print with indentation
  bool utf8 = false;    // output strings in utf-8 encoding
  bool json = false;    // output in JSON notation
};

}  // namespace sling

#endif  // SLING_PYAPI_PYSTORE_H_

