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

#include "sling/pyapi/pyframe.h"

#include "sling/frame/json.h"
#include "sling/pyapi/pystore.h"

namespace sling {

// Python type declarations.
PyTypeObject PyFrame::type;
PyMappingMethods PyFrame::mapping;
PySequenceMethods PyFrame::sequence;
PyMethodTable PyFrame::methods;
PyTypeObject PySlots::type;

void PyFrame::Define(PyObject *module) {
  InitType(&type, "sling.Frame", sizeof(PyFrame), false);
  type.tp_dealloc = method_cast<destructor>(&PyFrame::Dealloc);
  type.tp_getattro = method_cast<getattrofunc>(&PyFrame::GetAttr);
  type.tp_setattro = method_cast<setattrofunc>(&PyFrame::SetAttr);
  type.tp_str = method_cast<reprfunc>(&PyFrame::Str);
  type.tp_repr = method_cast<reprfunc>(&PyFrame::Repr);
  type.tp_iter = method_cast<getiterfunc>(&PyFrame::Slots);
  type.tp_call = method_cast<ternaryfunc>(&PyFrame::Find);
  type.tp_hash = method_cast<hashfunc>(&PyFrame::Hash);
  type.tp_richcompare = method_cast<richcmpfunc>(&PyFrame::Compare);

  type.tp_as_mapping = &mapping;
  mapping.mp_length = method_cast<lenfunc>(&PyFrame::Size);
  mapping.mp_subscript = method_cast<binaryfunc>(&PyFrame::Lookup);
  mapping.mp_ass_subscript = method_cast<objobjargproc>(&PyFrame::Assign);

  type.tp_as_sequence = &sequence;
  sequence.sq_contains = method_cast<objobjproc>(&PyFrame::Contains);

  methods.Add("get", &PyFrame::Get);
  methods.Add("data", &PyFrame::Data);
  methods.Add("append", &PyFrame::Append);
  methods.AddO("extend", &PyFrame::Extend);
  methods.Add("store", &PyFrame::GetStore);
  methods.AddO("isa", &PyFrame::IsA);
  methods.Add("islocal", &PyFrame::IsLocal);
  methods.Add("isglobal", &PyFrame::IsGlobal);
  methods.Add("isanonymous", &PyFrame::IsAnonymous);
  methods.Add("ispublic", &PyFrame::IsPublic);
  methods.Add("isnamed", &PyFrame::IsPublic);  // legacy
  methods.Add("resolve", &PyFrame::Resolve);
  type.tp_methods = methods.table();

  RegisterType(&type, module, "Frame");
}

void PyFrame::Init(PyStore *pystore, Handle handle) {
  // Add reference to store to keep it alive.
  if (handle.IsGlobalRef() && pystore->pyglobals != nullptr) {
    this->pystore = pystore->pyglobals;
  } else {
    this->pystore = pystore;
  }
  Py_INCREF(this->pystore);

  // Add frame as root object for store to keep it alive in the store.
  InitRoot(this->pystore->store, handle);
}

void PyFrame::Dealloc() {
  // Unlock tracking of handle in store.
  Unlink();

  // Release reference to store.
  Py_DECREF(pystore);

  // Free object.
  Free();
}

Py_ssize_t PyFrame::Size() {
  return frame()->slots();
}

long PyFrame::Hash() {
  return handle().bits;
}

PyObject *PyFrame::Compare(PyObject *other, int op) {
  // Only equality check is supported.
  if (op != Py_EQ && op != Py_NE) {
    PyErr_SetString(PyExc_TypeError, "Invalid frame comparison");
    return nullptr;
  }

  // Check if other object is a frame.
  bool match = false;
  if (PyObject_TypeCheck(other, &PyFrame::type)) {
    PyFrame *pyother = reinterpret_cast<PyFrame *>(other);

    // Frames are compared by reference, so check if the stores and handles are
    // the same.
    match = CompatibleStore(pyother) && pyother->handle() == handle();
  }

  if (op == Py_NE) match = !match;
  return PyBool_FromLong(match);
}

PyObject *PyFrame::GetStore() {
  Py_INCREF(pystore);
  return pystore->AsObject();
}

PyObject *PyFrame::Lookup(PyObject *key) {
  // Look up role.
  Handle role = pystore->RoleValue(key, true);
  if (role.IsError()) return nullptr;

  // Return None if the role name does not exist.
  if (role.IsNil()) Py_RETURN_NONE;

  // Look up (first) value for role.
  Handle value = frame()->get(role);
  return pystore->PyValue(value);
}

PyObject *PyFrame::Get(PyObject *args, PyObject *kw) {
  static const char *kwlist[] = {"role", "binary", nullptr};
  PyObject *key = nullptr;
  bool binary = false;
  if (!PyArg_ParseTupleAndKeywords(args, kw, "O|b",
          const_cast<char **>(kwlist), &key, &binary)) return nullptr;

  // Look up role.
  Handle role = pystore->RoleValue(key, true);
  if (role.IsError()) return nullptr;

  // Return None if the role name does not exist.
  if (role.IsNil()) Py_RETURN_NONE;

  // Look up (first) value for role.
  Handle value = frame()->get(role);
  return pystore->PyValue(value, binary);
}

int PyFrame::Assign(PyObject *key, PyObject *v) {
  // Check that frame is writable.
  if (!Writable()) return -1;

  // Look up role.
  GCLock lock(pystore->store);
  Handle role = pystore->RoleValue(key);
  if (role.IsError()) return -1;

  // Check role.
  if (role.IsNil()) {
    PyErr_SetString(PyExc_IndexError, "Role not defined");
    return -1;
  };
  if (role == Handle::id()) {
    PyErr_SetString(PyExc_IndexError, "Frame id cannot be changed");
    return -1;
  };

  if (v == nullptr) {
    // Delete all slots with name.
    pystore->store->Delete(handle(), role);
  } else {
    // Get new frame role value.
    Handle value = pystore->Value(v);
    if (value.IsError()) return -1;

    // Set frame slot value.
    pystore->store->Set(handle(), role, value);
  }

  return 0;
}

int PyFrame::Contains(PyObject *key) {
  // Look up role.
  Handle role = pystore->RoleValue(key, true);
  if (role.IsError()) return -1;

  // Check if frame has slot with role.
  if (role.IsNil()) return 0;
  return frame()->has(role);
}

PyObject *PyFrame::GetAttr(PyObject *key) {
  // Resolve methods.
  PyObject *method = PyObject_GenericGetAttr(AsObject(), key);
  if (method != nullptr) return method;
  PyErr_Clear();

  // Get attribute name.
  const char *name = GetString(key);
  if (name == nullptr) return nullptr;

  // Lookup role.
  Handle role = pystore->store->LookupExisting(name);
  if (role.IsNil()) Py_RETURN_NONE;

  // Get role value for frame.
  Handle value = frame()->get(role);
  return pystore->PyValue(value);
}

int PyFrame::SetAttr(PyObject *key, PyObject *v) {
  // Check that frame is writable.
  if (!Writable()) return -1;

  // Get role name.
  const char *name = GetString(key);
  if (name == nullptr) return -1;

  // Lookup role.
  GCLock lock(pystore->store);
  Handle role = pystore->store->Lookup(name);
  if (role.IsNil()) {
    PyErr_SetString(PyExc_IndexError, "Role not defined");
    return -1;
  };
  if (role == Handle::id()) {
    PyErr_SetString(PyExc_IndexError, "Frame id cannot be changed");
    return -1;
  };

  if (v == nullptr) {
    // Delete all slots with name.
    pystore->store->Delete(handle(), role);
  } else {
    // Get role value.
    Handle value = pystore->Value(v);
    if (value.IsError())  return -1;

    // Set role value for frame.
    pystore->store->Set(handle(), role, value);
  }

  return 0;
}

PyObject *PyFrame::Append(PyObject *args) {
  // Check that frame is writable.
  if (!Writable()) return nullptr;

  // Get name and value arguments.
  PyObject *pyname;
  PyObject *pyvalue;
  if (!PyArg_ParseTuple(args, "OO", &pyname, &pyvalue)) return nullptr;

  // Get role.
  GCLock lock(pystore->store);
  Handle role = pystore->RoleValue(pyname);
  if (role.IsError()) return nullptr;

  // Get value.
  Handle value = pystore->Value(pyvalue);
  if (value.IsError()) return nullptr;

  // Add new slot to frame.
  pystore->store->Add(handle(), role, value);
  Py_RETURN_NONE;
}

PyObject *PyFrame::Extend(PyObject *arg) {
  // Check that frame is writable.
  if (!Writable()) return nullptr;

  // Get existing slots for frame.
  GCLock lock(pystore->store);
  std::vector<Slot> slots(frame()->begin(), frame()->end());

  // Append slots from data to slot list.
  if (!pystore->SlotList(arg, &slots)) return nullptr;

  // Reallocate frame.
  Slot *begin = slots.data();
  Slot *end = slots.data() + slots.size();
  handle_ = pystore->store->AllocateFrame(begin, end, handle_);

  Py_RETURN_NONE;
}

PyObject *PyFrame::Slots() {
  PySlots *iter = PyObject_New(PySlots, &PySlots::type);
  iter->Init(this, Handle::nil());
  return iter->AsObject();
}

PyObject *PyFrame::Find(PyObject *args, PyObject *kw) {
  // Get role argument.
  PyObject *pyrole;
  if (!PyArg_ParseTuple(args, "O", &pyrole)) return nullptr;
  Handle role = pystore->RoleValue(pyrole);
  if (role.IsError()) return nullptr;

  // Create iterator for finding all slot with the role.
  PySlots *iter = PyObject_New(PySlots, &PySlots::type);
  iter->Init(this, role);
  return iter->AsObject();
}

PyObject *PyFrame::Str() {
  StringPrinter printer(pystore->store);
  printer.Print(handle());
  const string &text = printer.text();
  return PyUnicode_FromStringAndSize(text.data(), text.size());
}

PyObject *PyFrame::Repr() {
  FrameDatum *f = frame();
  if (f->IsPublic()) {
    // Return frame id.
    Handle id = f->get(Handle::id());
    SymbolDatum *symbol = pystore->store->Deref(id)->AsSymbol();
    StringDatum *name = pystore->store->GetString(symbol->name);
    return PyUnicode_FromStringAndSize(name->data(), name->size());
  } else {
    // Return frame as text.
    StringPrinter printer(pystore->store);
    printer.Print(handle());
    const string &text = printer.text();
    return PyUnicode_FromStringAndSize(text.data(), text.size());
  }
}

PyObject *PyFrame::Data(PyObject *args, PyObject *kw) {
  // Get arguments.
  SerializationFlags flags(pystore->store);
  if (!flags.ParseFlags(args, kw)) return nullptr;

  // Serialize frame.
  if (flags.binary) {
    StringEncoder encoder(pystore->store);
    flags.InitEncoder(encoder.encoder());
    encoder.Encode(handle());
    const string &buffer = encoder.buffer();
    return PyBytes_FromStringAndSize(buffer.data(), buffer.size());
  } else if (flags.json) {
    string json;
    StringOutputStream stream(&json);
    Output output(&stream);
    JSONWriter writer(pystore->store, &output);
    writer.set_indent(flags.pretty ? 2 : 0);
    writer.set_shallow(flags.shallow);
    writer.set_global(flags.global);
    writer.set_byref(flags.byref);
    writer.Write(handle());
    output.Flush();
    return PyUnicode_FromStringAndSize(json.data(), json.size());
  } else {
    StringPrinter printer(pystore->store);
    flags.InitPrinter(printer.printer());
    printer.Print(handle());
    const string &text = printer.text();
    return PyUnicode_FromStringAndSize(text.data(), text.size());
  }
}

PyObject *PyFrame::IsA(PyObject *arg) {
  // Look up type.
  Handle type = pystore->RoleValue(arg, true);
  if (type.IsError()) return nullptr;

  // Return False if the type name does not exist.
  if (type.IsNil()) Py_RETURN_FALSE;

  // Check type.
  return PyBool_FromLong(frame()->isa(type));
}

PyObject *PyFrame::IsLocal() {
  return PyBool_FromLong(handle().IsLocalRef());
}

PyObject *PyFrame::IsGlobal() {
  return PyBool_FromLong(handle().IsGlobalRef());
}

PyObject *PyFrame::IsAnonymous() {
  return PyBool_FromLong(frame()->IsAnonymous());
}

PyObject *PyFrame::IsPublic() {
  return PyBool_FromLong(frame()->IsPublic());
}

PyObject *PyFrame::Resolve() {
  Handle qua = pystore->store->Resolve(handle());
  if (qua == handle()) {
    Py_INCREF(this);
    return AsObject();
  } else {
    return pystore->PyValue(qua);
  }
}

bool PyFrame::Writable() {
  if (pystore->store->frozen() || !pystore->store->Owned(handle())) {
    PyErr_SetString(PyExc_ValueError, "Frame is not writable");
    return false;
  }
  return true;
}

bool PyFrame::CompatibleStore(PyFrame *other) {
  // Frames are compatible if they are in the same store.
  if (pystore->store == other->pystore->store) return true;

  if (handle().IsLocalRef()) {
    // A local store is also compatible with its global store.
    return pystore->pyglobals->store == other->pystore->store;
  } else if (other->handle().IsLocalRef()) {
    // A global store is also compatible with a local store based on it.
    return pystore->store == other->pystore->pyglobals->store;
  } else {
    // Frames belong to different global stores.
    return false;
  }
}

void PySlots::Define(PyObject *module) {
  InitType(&type, "sling.Slots", sizeof(PySlots), false);
  type.tp_dealloc = method_cast<destructor>(&PySlots::Dealloc);
  type.tp_iter = method_cast<getiterfunc>(&PySlots::Self);
  type.tp_iternext = method_cast<iternextfunc>(&PySlots::Next);
  RegisterType(&type, module, "Slots");
}

void PySlots::Init(PyFrame *pyframe, Handle role) {
  current = -1;
  this->pyframe = pyframe;
  this->role = role;
  Py_INCREF(pyframe);
}

void PySlots::Dealloc() {
  Py_DECREF(pyframe);
  Free();
}

PyObject *PySlots::Next() {
  // Check if there are any more slots.
  FrameDatum *f = pyframe->frame();
  while (++current < f->slots()) {
    // Check for role match.
    Slot *slot = f->begin() + current;
    if (role.IsNil()) {
      // Create two-tuple for name and value.
      PyObject *name = pyframe->pystore->PyValue(slot->name);
      PyObject *value = pyframe->pystore->PyValue(slot->value);
      PyObject *pair = PyTuple_Pack(2, name, value);
      Py_DECREF(name);
      Py_DECREF(value);
      return pair;
    } else if (role == slot->name) {
      return pyframe->pystore->PyValue(slot->value);
    }
  }

  // No more slots.
  PyErr_SetNone(PyExc_StopIteration);
  return nullptr;
}

PyObject *PySlots::Self() {
  Py_INCREF(this);
  return AsObject();
}

}  // namespace sling

