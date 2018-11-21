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

#ifndef SLING_PYAPI_PYDATE_H_
#define SLING_PYAPI_PYDATE_H_

#include "sling/nlp/kb/calendar.h"
#include "sling/pyapi/pybase.h"
#include "sling/pyapi/pystore.h"

namespace sling {

// Python wrapper for date.
struct PyDate : public PyBase {
  // Initialize date wrapper.
  int Init(PyObject *args, PyObject *kwds);

  // Deallocate date wrapper.
  void Dealloc();

  // Convert date to string (YYYY-MM-DD).
  PyObject *Str();

  // Date in ISO 8601 format.
  PyObject *ISO();

  // Convert date to string or integer value.
  PyObject *Value();

  // Date object.
  nlp::Date date;

  // Registration.
  static PyTypeObject type;
  static PyMemberDef members[];
  static PyMethodTable methods;
  static void Define(PyObject *module);
};

// Python wrapper for calendar.
struct PyCalendar : public PyBase {
  // Initialize calendar wrapper.
  int Init(PyObject *args, PyObject *kwds);

  // Deallocate record reader wrapper.
  void Dealloc();

  // Convert date to human-readable string.
  PyObject *Str(PyObject *obj);

  // Return frames for date parts.
  PyObject *Day(PyObject *obj);
  PyObject *Month(PyObject *obj);
  PyObject *Year(PyObject *obj);
  PyObject *Decade(PyObject *obj);
  PyObject *Century(PyObject *obj);
  PyObject *Millennium(PyObject *obj);

  // Get date object.
  PyDate *GetDate(PyObject *obj);

  // Store for calendar frames.
  PyStore *pystore;

  // Calendar.
  nlp::Calendar *calendar;

  // Registration.
  static PyTypeObject type;
  static PyMethodTable methods;
  static void Define(PyObject *module);
};

}  // namespace sling

#endif  // SLING_PYAPI_PYDATE_H_

