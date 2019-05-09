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

#include "sling/pyapi/pymisc.h"

#include <vector>

#include "sling/base/flags.h"
#include "sling/base/logging.h"

namespace sling {

PyObject *PyGetFlags() {
  // Build list of registered flags.
  std::vector<Flag *> flags;
  for (Flag *flag = Flag::head; flag != nullptr; flag = flag->next) {
    if (strcmp(flag->name, "help") == 0) continue;
    flags.push_back(flag);
  }

  // Create a list of tuples with the flags. Each tuple contains name, help,
  // and default value.
  PyObject *list = PyList_New(flags.size());
  for (int i = 0; i < flags.size(); ++i) {
    Flag *flag = flags[i];

    // Get name and help string.
    PyObject *name = PyUnicode_FromString(flag->name);
    PyObject *help = PyUnicode_FromString(flag->help);

    // Get default flag value.
    PyObject *defval = nullptr;
    switch (flag->type) {
      case Flag::BOOL:
        defval = PyBool_FromLong(flag->value<bool>());
        break;
      case Flag::INT32:
        defval = PyLong_FromLong(flag->value<int32>());
        break;
      case Flag::UINT32:
        defval = PyLong_FromLong(flag->value<uint32>());
        break;
      case Flag::INT64:
        defval = PyLong_FromLongLong(flag->value<int64>());
        break;
      case Flag::UINT64:
        defval = PyLong_FromUnsignedLongLong(flag->value<uint64>());
        break;
      case Flag::DOUBLE:
        defval = PyFloat_FromDouble(flag->value<double>());
        break;
      case Flag::STRING:
        defval = PyUnicode_FromStringAndSize(
            flag->value<string>().data(), flag->value<string>().size());
        break;
    }

    // Add tuple to list.
    PyList_SetItem(list, i, PyTuple_Pack(3, name, help, defval));
  }

  return list;
}

PyObject *PySetFlag(PyObject *self, PyObject *args) {
  // Get flag name and new value.
  const char *name = nullptr;
  PyObject *value = nullptr;
  if (!PyArg_ParseTuple(args, "sO", &name, &value)) return nullptr;

  // Find flag.
  Flag *flag = Flag::head;
  while (flag != nullptr && strcmp(flag->name, name) != 0) flag = flag->next;
  if (flag == nullptr) {
    PyErr_SetString(PyExc_ValueError, "Unknown flag");
    return nullptr;
  }

  switch (flag->type) {
    case Flag::BOOL:
      flag->value<bool>() = (value == Py_True);
      break;
    case Flag::INT32:
      flag->value<int32>() = PyLong_AsLong(value);
      break;
    case Flag::UINT32:
      flag->value<uint32>() = PyLong_AsLong(value);
      break;
    case Flag::INT64:
      flag->value<int64>() = PyLong_AsLongLong(value);
      break;
    case Flag::UINT64:
      flag->value<uint64>() = PyLong_AsUnsignedLongLong(value);
      break;
    case Flag::DOUBLE:
      flag->value<double>() = PyFloat_AsDouble(value);
      break;
    case Flag::STRING:
      flag->value<string>() = PyUnicode_AsUTF8(value);
      break;
  }

  Py_RETURN_NONE;
}

PyObject *PyLogMessage(PyObject *self, PyObject *args) {
  // Get severity, file, line, and message.
  int severity;
  const char *filename;
  int line;
  const char *message;
  if (!PyArg_ParseTuple(args, "isis", &severity, &filename, &line, &message)) {
    return nullptr;
  }

  // Output log message.
  LogMessage(filename, line, severity) << message;

  Py_RETURN_NONE;
}

}  // namespace sling

