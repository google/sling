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

#include "sling/pyapi/pyrecordio.h"

#include "sling/file/file.h"
#include "sling/file/recordio.h"

namespace sling {

// Python type declarations.
PyTypeObject PyRecordReader::type;
PyMethodTable PyRecordReader::methods;
PyTypeObject PyRecordDatabase::type;
PyMappingMethods PyRecordDatabase::mapping;
PyMethodTable PyRecordDatabase::methods;
PyTypeObject PyRecordWriter::type;
PyMethodTable PyRecordWriter::methods;

// Check status.
static bool CheckIO(Status status) {
  bool ok = status.ok();
  if (!ok) PyErr_SetString(PyExc_IOError, status.message());
  return ok;
}

void PyRecordReader::Define(PyObject *module) {
  InitType(&type, "sling.RecordReader", sizeof(PyRecordReader), true);

  type.tp_init = method_cast<initproc>(&PyRecordReader::Init);
  type.tp_dealloc = method_cast<destructor>(&PyRecordReader::Dealloc);
  type.tp_iter = method_cast<getiterfunc>(&PyRecordReader::Self);
  type.tp_iternext = method_cast<iternextfunc>(&PyRecordReader::Next);

  methods.Add("close", &PyRecordReader::Close);
  methods.Add("read", &PyRecordReader::Read);
  methods.Add("tell", &PyRecordReader::Tell);
  methods.AddO("seek", &PyRecordReader::Seek);
  methods.Add("rewind", &PyRecordReader::Rewind);
  methods.Add("done", &PyRecordReader::Done);
  type.tp_methods = methods.table();

  RegisterType(&type, module, "RecordReader");
}

int PyRecordReader::Init(PyObject *args, PyObject *kwds) {
  // Get arguments.
  char *filename;
  RecordFileOptions options;
  if (!PyArg_ParseTuple(args, "s|i", &filename, &options.buffer_size)) {
    return -1;
  }

  // Open file.
  File *f;
  if (!CheckIO(File::Open(filename, "r", &f))) return -1;

  // Create record reader.
  reader = new RecordReader(f, options);
  return 0;
}

void PyRecordReader::Dealloc() {
  delete reader;
  Free();
}

PyObject *PyRecordReader::Close() {
  if (!CheckIO(reader->Close())) return nullptr;
  Py_RETURN_NONE;
}

PyObject *PyRecordReader::Done() {
  return PyBool_FromLong(reader->Done());
}

PyObject *PyRecordReader::Read() {
  // Read next record.
  Record record;
  if (!CheckIO(reader->Read(&record))) return nullptr;

  // Create key and value tuple.
  PyObject *k = Py_None;
  PyObject *v = Py_None;
  if (!record.key.empty()) {
    k = PyBytes_FromStringAndSize(record.key.data(), record.key.size());
  }
  if (!record.value.empty()) {
    v = PyBytes_FromStringAndSize(record.value.data(), record.value.size());
  }
  PyObject *pair = PyTuple_Pack(2, k, v);
  if (k != Py_None) Py_DECREF(k);
  if (v != Py_None) Py_DECREF(v);

  return pair;
}

PyObject *PyRecordReader::Tell() {
  return PyLong_FromSsize_t(reader->Tell());
}

PyObject *PyRecordReader::Seek(PyObject *arg) {
  // Get position argument.
  Py_ssize_t pos = PyLong_AsSsize_t(arg);
  if (pos == -1) return nullptr;

  // Seek to position.
  if (!CheckIO(reader->Seek(pos))) return nullptr;
  Py_RETURN_NONE;
}

PyObject *PyRecordReader::Rewind() {
  // Seek to first record.
  if (!CheckIO(reader->Rewind())) return nullptr;
  Py_RETURN_NONE;
}

PyObject *PyRecordReader::Next() {
  // Check if there are more records.
  if (reader->Done()) {
    PyErr_SetNone(PyExc_StopIteration);
    return nullptr;
  }

  // Return next record.
  return Read();
}

PyObject *PyRecordReader::Self() {
  Py_INCREF(this);
  return AsObject();
}

void PyRecordDatabase::Define(PyObject *module) {
  InitType(&type, "sling.RecordDatabase", sizeof(PyRecordDatabase), true);

  type.tp_init = method_cast<initproc>(&PyRecordDatabase::Init);
  type.tp_dealloc = method_cast<destructor>(&PyRecordDatabase::Dealloc);
  type.tp_iter = method_cast<getiterfunc>(&PyRecordDatabase::Self);
  type.tp_iternext = method_cast<iternextfunc>(&PyRecordDatabase::Next);

  type.tp_as_mapping = &mapping;
  mapping.mp_subscript = method_cast<binaryfunc>(&PyRecordDatabase::Lookup);

  methods.Add("close", &PyRecordDatabase::Close);
  methods.AddO("lookup", &PyRecordDatabase::Lookup);
  type.tp_methods = methods.table();

  RegisterType(&type, module, "RecordDatabase");
}

int PyRecordDatabase::Init(PyObject *args, PyObject *kwds) {
  // Get arguments.
  static const char *kwlist[] = {"filename", "bufsize", "cache", nullptr};
  char *pattern = nullptr;
  RecordFileOptions options;
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "s|ii",
          const_cast<char **>(kwlist),
          &pattern, &options.buffer_size,
          &options.index_cache_size)) return -1;

  // Find matching file names.
  std::vector<string> filenames;
  if (!CheckIO(File::Match(pattern, &filenames))) return -1;
  if (filenames.empty()) {
    PyErr_SetString(PyExc_IOError, "No matching files");
    return -1;
  }

  // Create record database.
  db = new RecordDatabase(filenames, options);
  return 0;
}

void PyRecordDatabase::Dealloc() {
  delete db;
  Free();
}

PyObject *PyRecordDatabase::Close() {
  delete db;
  db = nullptr;
  Py_RETURN_NONE;
}

PyObject *PyRecordDatabase::Lookup(PyObject *obj) {
  // Get key.
  const char *key;
  if (PyUnicode_Check(obj)) {
    key = PyUnicode_AsUTF8(obj);
  } else {
    key = PyBytes_AsString(obj);
  }
  if (key == nullptr) return nullptr;

  // Look up record.
  CHECK(db != nullptr);
  Record record;
  if (!db->Lookup(key, &record)) Py_RETURN_NONE;
  return PyBytes_FromStringAndSize(record.value.data(), record.value.size());
}

PyObject *PyRecordDatabase::Next() {
  // Check for end of record database.
  if (db->Done()) {
    PyErr_SetNone(PyExc_StopIteration);
    return nullptr;
  }

  // Read next record.
  Record record;
  if (!db->Next(&record)) {
    PyErr_SetString(PyExc_IOError, "Error reading record");
    return nullptr;
  }

  // Create key and value tuple.
  PyObject *k = Py_None;
  PyObject *v = Py_None;
  if (!record.key.empty()) {
    k = PyBytes_FromStringAndSize(record.key.data(), record.key.size());
  }
  if (!record.value.empty()) {
    v = PyBytes_FromStringAndSize(record.value.data(), record.value.size());
  }
  PyObject *pair = PyTuple_Pack(2, k, v);
  if (k != Py_None) Py_DECREF(k);
  if (v != Py_None) Py_DECREF(v);

  return pair;
}

PyObject *PyRecordDatabase::Self() {
  Py_INCREF(this);
  return AsObject();
}

void PyRecordWriter::Define(PyObject *module) {
  InitType(&type, "sling.RecordWriter", sizeof(PyRecordWriter), true);

  type.tp_init = method_cast<initproc>(&PyRecordWriter::Init);
  type.tp_dealloc = method_cast<destructor>(&PyRecordWriter::Dealloc);

  methods.Add("close", &PyRecordWriter::Close);
  methods.Add("write", &PyRecordWriter::Write);
  methods.Add("tell", &PyRecordWriter::Tell);
  type.tp_methods = methods.table();

  RegisterType(&type, module, "RecordWriter");
}

int PyRecordWriter::Init(PyObject *args, PyObject *kwds) {
  // Get arguments.
  static const char *kwlist[] = {
      "filename", "bufsize", "chunksize", "compression", "index", nullptr};
  char *filename = nullptr;
  RecordFileOptions options;
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "s|iiib",
          const_cast<char **>(kwlist),
          &filename, &options.buffer_size, &options.chunk_size,
          &options.compression, &options.indexed)) return -1;

  // Open file.
  File *f;
  if (!CheckIO(File::Open(filename, "w", &f))) return -1;

  // Create record writer.
  writer = new RecordWriter(f, options);

  return 0;
}

void PyRecordWriter::Dealloc() {
  delete writer;
  Free();
}

PyObject *PyRecordWriter::Close() {
  if (!CheckIO(writer->Close())) return nullptr;
  Py_RETURN_NONE;
}

PyObject *PyRecordWriter::Write(PyObject *args) {
  // Get key and value argument.
  PyObject *pykey;
  PyObject *pyvalue;
  if (!PyArg_ParseTuple(args, "OO", &pykey, &pyvalue)) return nullptr;

  // Get key and value data buffers.
  Slice key;
  Slice value;
  if (pykey != Py_None) {
    if (PyUnicode_Check(pykey)) {
      Py_ssize_t length;
      const char *data = PyUnicode_AsUTF8AndSize(pykey, &length);
      if (data == nullptr) return nullptr;
      key = Slice(data, length);
    } else {
      char *data;
      Py_ssize_t length;
      if (PyBytes_AsStringAndSize(pykey, &data, &length)) return nullptr;
      key = Slice(data, length);
    }
  }
  if (pyvalue != Py_None) {
    if (PyUnicode_Check(pyvalue)) {
      Py_ssize_t length;
      const char *data = PyUnicode_AsUTF8AndSize(pyvalue, &length);
      if (data == nullptr) return nullptr;
      value = Slice(data, length);
    } else {
      char *data;
      Py_ssize_t length;
      if (PyBytes_AsStringAndSize(pyvalue, &data, &length)) return nullptr;
      value = Slice(data, length);
    }
  }

  // Write record.
  if (!CheckIO(writer->Write(key, value))) return nullptr;
  Py_RETURN_NONE;
}

PyObject *PyRecordWriter::Tell() {
  return PyLong_FromSsize_t(writer->Tell());
}

}  // namespace sling

