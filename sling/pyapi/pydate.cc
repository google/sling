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

#include "sling/pyapi/pydate.h"

#include <stddef.h>

#include "sling/pyapi/pyframe.h"

namespace sling {

// Python type declarations.
PyTypeObject PyDate::type;
PyMethodTable PyDate::methods;
PyTypeObject PyCalendar::type;
PyMethodTable PyCalendar::methods;

PyMemberDef PyDate::members[] = {
  {"year", T_INT, offsetof(struct PyDate, date.year), 0, ""},
  {"month", T_INT, offsetof(struct PyDate, date.month), 0, ""},
  {"day", T_INT, offsetof(struct PyDate, date.day), 0, ""},
  {"precision", T_INT, offsetof(struct PyDate, date.precision), 0, ""},
  {nullptr}
};

void PyDate::Define(PyObject *module) {
  InitType(&type, "sling.Date", sizeof(PyDate), true);
  type.tp_init = method_cast<initproc>(&PyDate::Init);
  type.tp_dealloc = method_cast<destructor>(&PyDate::Dealloc);
  type.tp_str = method_cast<reprfunc>(&PyDate::Str);
  type.tp_members = members;

  methods.Add("iso", &PyDate::ISO);
  methods.Add("value", &PyDate::Value);
  type.tp_methods = methods.table();

  RegisterType(&type, module, "Date");

  RegisterEnum(module, "MILLENNIUM", nlp::Date::MILLENNIUM);
  RegisterEnum(module, "CENTURY", nlp::Date::CENTURY);
  RegisterEnum(module, "DECADE", nlp::Date::DECADE);
  RegisterEnum(module, "YEAR", nlp::Date::YEAR);
  RegisterEnum(module, "MONTH", nlp::Date::MONTH);
  RegisterEnum(module, "DAY", nlp::Date::DAY);
}

int PyDate::Init(PyObject *args, PyObject *kwds) {
  // Get arguments.
  if (PyTuple_Size(args) > 1) {
    date.year = date.month = date.day = 0;
    date.precision = nlp::Date::DAY;
    if (!PyArg_ParseTuple(args, "ii|i", &date.year, &date.month, &date.day)) {
      return -1;
    }
    if (date.day == 0) date.precision = nlp::Date::MONTH;
    if (date.month == 0) date.precision = nlp::Date::YEAR;
    if (date.year == 0) date.precision = nlp::Date::NONE;
  } else {
    PyObject *time = nullptr;
    if (!PyArg_ParseTuple(args, "|O", &time)) return -1;

    if (time == nullptr) {
      // Return empty date.
      date.year = date.month = date.day = 0;
      date.precision = nlp::Date::NONE;
    } else if (PyObject_TypeCheck(time, &PyFrame::type)) {
      // Parse date from object.
      PyFrame *frame = reinterpret_cast<PyFrame *>(time);
      Store *store = frame->pystore->store;
      date.Init(Object(store, store->Resolve(frame->handle())));
    } else if (PyUnicode_Check(time)) {
      // Parse date from string.
      Py_ssize_t length;
      const char *data = PyUnicode_AsUTF8AndSize(time, &length);
      date.ParseFromString(Text(data, length));
    } else if (PyLong_Check(time)) {
      // Parse date from number.
      date.ParseFromNumber(PyLong_AsLong(time));
    } else {
      PyErr_SetString(PyExc_ValueError, "Cannot create date from value");
      return -1;
    }
  }

  return 0;
}

void PyDate::Dealloc() {
  Free();
}

PyObject *PyDate::Str() {
  return AllocateString(date.AsString());
}

PyObject *PyDate::Value() {
  int number = date.AsNumber();
  if (number != -1) return PyLong_FromLong(number);
  return AllocateString(date.AsString());
}

PyObject *PyDate::ISO() {
  return AllocateString(date.ISO8601());
}

void PyCalendar::Define(PyObject *module) {
  InitType(&type, "sling.Calendar", sizeof(PyCalendar), true);
  type.tp_init = method_cast<initproc>(&PyCalendar::Init);
  type.tp_dealloc = method_cast<destructor>(&PyCalendar::Dealloc);

  methods.AddO("str", &PyCalendar::Str);
  methods.AddO("day", &PyCalendar::Day);
  methods.AddO("month", &PyCalendar::Month);
  methods.AddO("year", &PyCalendar::Year);
  methods.AddO("decade", &PyCalendar::Decade);
  methods.AddO("century", &PyCalendar::Century);
  methods.AddO("millennium", &PyCalendar::Millennium);
  type.tp_methods = methods.table();

  RegisterType(&type, module, "Calendar");
}

int PyCalendar::Init(PyObject *args, PyObject *kwds) {
  // Get store argument.
  pystore = nullptr;
  if (!PyArg_ParseTuple(args, "O", &pystore)) return -1;
  if (!PyStore::TypeCheck(pystore)) return -1;

  // Initialize calendar.
  Py_INCREF(pystore);
  calendar = new nlp::Calendar();
  calendar->Init(pystore->store);

  return 0;
}

void PyCalendar::Dealloc() {
  delete calendar;
  if (pystore) Py_DECREF(pystore);
  Free();
}

PyObject *PyCalendar::Str(PyObject *obj) {
  PyDate *pydate = GetDate(obj);
  if (pydate == nullptr) return nullptr;
  return AllocateString(calendar->DateAsString(pydate->date));
}

PyObject *PyCalendar::Day(PyObject *obj) {
  PyDate *pydate = GetDate(obj);
  if (pydate == nullptr) return nullptr;
  return pystore->PyValue(calendar->Day(pydate->date));
}

PyObject *PyCalendar::Month(PyObject *obj) {
  PyDate *pydate = GetDate(obj);
  if (pydate == nullptr) return nullptr;
  return pystore->PyValue(calendar->Month(pydate->date));
}

PyObject *PyCalendar::Year(PyObject *obj) {
  PyDate *pydate = GetDate(obj);
  if (pydate == nullptr) return nullptr;
  return pystore->PyValue(calendar->Year(pydate->date));
}

PyObject *PyCalendar::Decade(PyObject *obj) {
  PyDate *pydate = GetDate(obj);
  if (pydate == nullptr) return nullptr;
  return pystore->PyValue(calendar->Decade(pydate->date));
}

PyObject *PyCalendar::Century(PyObject *obj) {
  PyDate *pydate = GetDate(obj);
  if (pydate == nullptr) return nullptr;
  return pystore->PyValue(calendar->Century(pydate->date));
}

PyObject *PyCalendar::Millennium(PyObject *obj) {
  PyDate *pydate = GetDate(obj);
  if (pydate == nullptr) return nullptr;
  return pystore->PyValue(calendar->Millennium(pydate->date));
}

PyDate *PyCalendar::GetDate(PyObject *obj) {
  if (PyObject_TypeCheck(obj, &PyDate::type)) {
    PyDate *pydate = reinterpret_cast<PyDate *>(obj);
    return pydate;
  } else {
    PyErr_SetString(PyExc_TypeError, "sling.Date object expected");
    return nullptr;
  }
}

}  // namespace sling

