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

#include <python2.7/Python.h>

#include "sling/base/init.h"
#include "sling/pyapi/pyarray.h"
#include "sling/pyapi/pydate.h"
#include "sling/pyapi/pyframe.h"
#include "sling/pyapi/pymisc.h"
#include "sling/pyapi/pyparser.h"
#include "sling/pyapi/pyphrase.h"
#include "sling/pyapi/pyrecordio.h"
#include "sling/pyapi/pystore.h"
#include "sling/pyapi/pytask.h"

namespace sling {

static PyMethodDef py_funcs[] = {
  {"get_flags", (PyCFunction) PyGetFlags, METH_NOARGS, ""},
  {"set_flag", (PyCFunction) PySetFlag, METH_VARARGS, ""},
  {"log_message", (PyCFunction) PyLogMessage, METH_VARARGS, ""},
  {"start_task_monitor", (PyCFunction) StartTaskMonitor, METH_VARARGS, ""},
  {"get_job_statistics", (PyCFunction) GetJobStatistics, METH_NOARGS, ""},
  {"finalize_dashboard", (PyCFunction) FinalizeDashboard, METH_NOARGS, ""},
  {"tolex", (PyCFunction) PyToLex, METH_VARARGS, ""},
  {nullptr, nullptr, 0, nullptr}
};

static void RegisterPythonModule() {
  PyObject *module = Py_InitModule3("pysling", py_funcs, "SLING");
  PyStore::Define(module);
  PySymbols::Define(module);
  PyFrame::Define(module);
  PySlots::Define(module);
  PyArray::Define(module);
  PyItems::Define(module);
  PyTokenizer::Define(module);
  PyPhraseTable::Define(module);
  PyParser::Define(module);
  PyRecordReader::Define(module);
  PyRecordWriter::Define(module);
  PyJob::Define(module);
  PyCalendar::Define(module);
  PyDate::Define(module);
}

}  // namespace sling

extern "C" void initpysling() {
  sling::InitSharedLibrary();
  sling::RegisterPythonModule();
}

