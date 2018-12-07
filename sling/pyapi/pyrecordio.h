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

#ifndef SLING_PYAPI_PYRECORDIO_H_
#define SLING_PYAPI_PYRECORDIO_H_

#include "sling/file/recordio.h"
#include "sling/pyapi/pybase.h"

namespace sling {

// Python wrapper for record reader.
struct PyRecordReader : public PyBase {
  // Initialize record reader wrapper.
  int Init(PyObject *args, PyObject *kwds);

  // Deallocate record reader wrapper.
  void Dealloc();

  // Close record reader.
  PyObject *Close();

  // Check for end of file.
  PyObject *Done();

  // Read next record from file returning key and value.
  PyObject *Read();

  // Return file position.
  PyObject *Tell();

  // Set file position.
  PyObject *Seek(PyObject *arg);

  // Seek to first record.
  PyObject *Rewind();

  // Return next record as 2-tuple with key and value.
  PyObject *Next();

  // Return self as iterator.
  PyObject *Self();

  // Record reader.
  RecordReader *reader;

  // Registration.
  static PyTypeObject type;
  static PyMethodTable methods;
  static void Define(PyObject *module);
};

// Python wrapper for record database.
struct PyRecordDatabase : public PyBase {
  // Initialize record database wrapper.
  int Init(PyObject *args, PyObject *kwds);

  // Deallocate record database wrapper.
  void Dealloc();

  // Look up record in database.
  PyObject *Lookup(PyObject *obj);

  // Close record database.
  PyObject *Close();

  // Return next record as 2-tuple with key and value.
  PyObject *Next();

  // Return self as iterator.
  PyObject *Self();

  // Record reader.
  RecordDatabase *db;

  // Registration.
  static PyTypeObject type;
  static PyMappingMethods mapping;
  static PyMethodTable methods;
  static void Define(PyObject *module);
};

// Python wrapper for record writer.
struct PyRecordWriter : public PyBase {
  // Initialize record writer wrapper.
  int Init(PyObject *args, PyObject *kwds);

  // Deallocate record writer wrapper.
  void Dealloc();

  // Close record writer.
  PyObject *Close();

  // Write record to file.
  PyObject *Write(PyObject *args);

  // Return file position.
  PyObject *Tell();

  // Record writer.
  RecordWriter *writer;

  // Registration.
  static PyTypeObject type;
  static PyMethodTable methods;
  static void Define(PyObject *module);
};

}  // namespace sling

#endif  // SLING_PYAPI_PYRECORDIO_H_

