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

#include "sling/pyapi/pytask.h"

#include <string.h>

#include "sling/http/http-server.h"
#include "sling/task/dashboard.h"

using namespace sling::task;

namespace sling {

// Task monitor.
static HTTPServer *http = nullptr;
static task::Dashboard *dashboard = nullptr;

// Python type declarations.
PyTypeObject PyJob::type;
PyMethodTable PyJob::methods;
PyTypeObject PyResource::type;
PyTypeObject PyTask::type;
PyMethodTable PyTask::methods;

void PyJob::Define(PyObject *module) {
  InitType(&type, "sling.api.Job", sizeof(PyJob), true);

  type.tp_init = method_cast<initproc>(&PyJob::Init);
  type.tp_dealloc = method_cast<destructor>(&PyJob::Dealloc);

  methods.Add("start", &PyJob::Start);
  methods.Add("wait", &PyJob::Wait);
  methods.Add("done", &PyJob::Done);
  methods.AddO("wait_for", &PyJob::WaitFor);
  methods.Add("counters", &PyJob::Counters);
  type.tp_methods = methods.table();

  RegisterType(&type, module, "Job");
}

int PyJob::Init(PyObject *args, PyObject *kwds) {
  // Create new job.
  job = new Job();
  running = false;

  // Get python job argument.
  PyObject *pyjob = nullptr;
  const char *name = nullptr;
  if (!PyArg_ParseTuple(args, "Os", &pyjob, &name)) return -1;
  job->set_name(name);

  // Get resources.
  ResourceMapping resource_mapping;
  PyObject *resources = PyAttr(pyjob, "resources");
  for (int i = 0; i < PyList_Size(resources); ++i) {
    PyObject *pyresource = PyList_GetItem(resources, i);
    PyObject *pyformat = PyAttr(pyresource, "format");
    PyObject *pyshard = PyAttr(pyresource, "shard");

    const char *name = PyStrAttr(pyresource, "name");
    Format format = PyGetFormat(pyformat);
    Shard shard = PyGetShard(pyshard);

    Resource *resource = job->CreateResource(name, format, shard);
    resource_mapping[pyresource] = resource;

    Py_DECREF(pyformat);
    Py_DECREF(pyshard);
  }
  Py_DECREF(resources);

  // Get tasks.
  TaskMapping task_mapping;
  PyObject *tasks = PyAttr(pyjob, "tasks");
  for (int i = 0; i < PyList_Size(tasks); ++i) {
    PyObject *pytask = PyList_GetItem(tasks, i);
    const char *type = PyStrAttr(pytask, "type");
    const char *name = PyStrAttr(pytask, "name");

    PyObject *pyshard = PyAttr(pytask, "shard");
    Shard shard = PyGetShard(pyshard);
    Py_DECREF(pyshard);

    Task *task = job->CreateTask(type, name, shard);
    task_mapping[pytask] = task;

    // Get task parameters.
    PyObject *params = PyAttr(pytask, "params");
    Py_ssize_t pos = 0;
    PyObject *k;
    PyObject *v;
    while (PyDict_Next(params, &pos, &k, &v)) {
      const char *key = PyUnicode_AsUTF8(k);
      const char *value = PyUnicode_AsUTF8(v);
      task->AddParameter(key, value);
    }
    Py_DECREF(params);

    // Get annotators.
    PyObject *annotators = PyAttr(pytask, "annotators");
    for (int i = 0; i < PyList_Size(annotators); ++i) {
      PyObject *annotator = PyList_GetItem(annotators, i);
      task->AddAnnotator(PyUnicode_AsUTF8(annotator));
    }
    Py_DECREF(annotators);

    // Bind inputs.
    PyObject *inputs = PyAttr(pytask, "inputs");
    for (int i = 0; i < PyList_Size(inputs); ++i) {
      PyObject *pybinding = PyList_GetItem(inputs, i);
      const char *name = PyStrAttr(pybinding, "name");
      PyObject *pyresource = PyAttr(pybinding, "resource");
      Resource *resource = resource_mapping[pyresource];
      CHECK(resource != nullptr);
      job->BindInput(task, resource, name);
      Py_DECREF(pyresource);
    }
    Py_DECREF(inputs);

    // Bind outputs.
    PyObject *outputs = PyAttr(pytask, "outputs");
    for (int i = 0; i < PyList_Size(outputs); ++i) {
      PyObject *pybinding = PyList_GetItem(outputs, i);
      const char *name = PyStrAttr(pybinding, "name");
      PyObject *pyresource = PyAttr(pybinding, "resource");
      Resource *resource = resource_mapping[pyresource];
      CHECK(resource != nullptr);
      job->BindOutput(task, resource, name);
      Py_DECREF(pyresource);
    }
    Py_DECREF(outputs);
  }
  Py_DECREF(tasks);

  // Get channels.
  PyObject *channels = PyAttr(pyjob, "channels");
  for (int i = 0; i < PyList_Size(channels); ++i) {
    PyObject *pychannel = PyList_GetItem(channels, i);

    PyObject *pyformat = PyAttr(pychannel, "format");
    Format format = PyGetFormat(pyformat);
    Py_DECREF(pyformat);

    PyObject *pyproducer = PyAttr(pychannel, "producer");
    Port producer = PyGetPort(pyproducer, task_mapping);
    Py_DECREF(pyproducer);

    PyObject *pyconsumer = PyAttr(pychannel, "consumer");
    Port consumer = PyGetPort(pyconsumer, task_mapping);
    Py_DECREF(pyconsumer);

    job->Connect(producer, consumer, format);
  }
  Py_DECREF(channels);

  return 0;
}

void PyJob::Dealloc() {
  CHECK(!running) << "Job is still running";
  delete job;
  Free();
}

PyObject *PyJob::Start() {
  if (!running) {
    // Add self-reference count to job to keep it alive while the job is
    // running. This reference is not released until the job has completed.
    Py_INCREF(this);
    running = true;

    // Register job in dashboard.
    if (dashboard != nullptr) {
      job->RegisterMonitor(dashboard);
    }

    // Start job.
    Py_BEGIN_ALLOW_THREADS;
    job->Start();
    Py_END_ALLOW_THREADS;
  }
  Py_RETURN_NONE;
}

PyObject *PyJob::Done() {
  bool done = job->Done();
  if (done && running) {
    running = false;
    Py_DECREF(this);
  }
  return PyBool_FromLong(done);
}

PyObject *PyJob::Wait() {
  Py_BEGIN_ALLOW_THREADS;
  job->Wait();
  Py_END_ALLOW_THREADS;
  if (running) {
    running = false;
    Py_DECREF(this);
  }
  Py_RETURN_NONE;
}

PyObject *PyJob::WaitFor(PyObject *timeout) {
  int ms = PyNumber_AsSsize_t(timeout, nullptr);
  bool done;
  Py_BEGIN_ALLOW_THREADS;
  done = job->Wait(ms);
  Py_END_ALLOW_THREADS;
  if (done && running) {
    running = false;
    Py_DECREF(this);
  }
  return PyBool_FromLong(done);
}

PyObject *PyJob::Counters() {
  // Create Python dictionary for counter values.
  PyObject *counters = PyDict_New();
  if (counters == nullptr) return nullptr;

  // Gather current counter values.
  job->IterateCounters([counters](const string &name, Counter *counter) {
    PyObject *key = PyUnicode_FromStringAndSize(name.data(), name.size());
    PyObject *val = PyLong_FromLong(counter->value());
    PyDict_SetItem(counters, key, val);
    Py_DECREF(key);
    Py_DECREF(val);
  });

  return counters;
}

Port PyJob::PyGetPort(PyObject *obj, const TaskMapping &mapping) {
  const char *name = PyStrAttr(obj, "name");

  PyObject *pyshard = PyAttr(obj, "shard");
  Shard shard = PyGetShard(pyshard);
  Py_DECREF(pyshard);

  PyObject *pytask = PyAttr(obj, "task");
  Task *task = mapping.at(pytask);
  Py_DECREF(pytask);

  return Port(task, name, shard);
}

Format PyJob::PyGetFormat(PyObject *obj) {
  const char *file = PyStrAttr(obj, "file");
  const char *key = PyStrAttr(obj, "key");
  const char *value = PyStrAttr(obj, "value");
  return Format(file, key, value);
}

Shard PyJob::PyGetShard(PyObject *obj) {
  if (obj == Py_None) return Shard();
  int part = PyIntAttr(obj, "part");
  int total = PyIntAttr(obj, "total");
  return Shard(part, total);
}

const char *PyJob::PyStrAttr(PyObject *obj, const char *name) {
  PyObject *attr = PyAttr(obj, name);
  const char *str = attr == Py_None ? "" : PyUnicode_AsUTF8(attr);
  CHECK(str != nullptr) << name;
  Py_DECREF(attr);
  return str;
}

int PyJob::PyIntAttr(PyObject *obj, const char *name) {
  PyObject *attr = PyAttr(obj, name);
  int value = PyNumber_AsSsize_t(attr, nullptr);
  Py_DECREF(attr);
  return value;
}

PyObject *PyJob::PyAttr(PyObject *obj, const char *name) {
  PyObject *attr = PyObject_GetAttrString(obj, name);
  CHECK(attr != nullptr) << name;
  return attr;
}

PyMemberDef PyResource::members[] = {
  {"name", T_OBJECT, offsetof(struct PyResource, name), READONLY, ""},
  {"format", T_OBJECT, offsetof(struct PyResource, format), READONLY, ""},
  {"part", T_INT, offsetof(struct PyResource, part), READONLY, ""},
  {"of", T_INT, offsetof(struct PyResource, of), READONLY, ""},
  {nullptr}
};

void PyResource::Define(PyObject *module) {
  InitType(&type, "sling.api.Resource", sizeof(PyResource), false);
  type.tp_init = method_cast<initproc>(&PyResource::Init);
  type.tp_dealloc = method_cast<destructor>(&PyResource::Dealloc);
  type.tp_members = members;

  RegisterType(&type, module, "Resource");
}

int PyResource::Init(task::Resource *resource) {
  name = AllocateString(resource->name());
  format = AllocateString(resource->format().ToString());
  part = resource->shard().part();
  of = resource->shard().total();
  return 0;
}

void PyResource::Dealloc() {
  if (name) Py_DECREF(name);
  if (format) Py_DECREF(format);
  Free();
}

void PyTask::Define(PyObject *module) {
  InitType(&type, "sling.api.Task", sizeof(PyTask), false);

  type.tp_init = method_cast<initproc>(&PyTask::Init);
  type.tp_dealloc = method_cast<destructor>(&PyTask::Dealloc);

  methods.Add("name", &PyTask::GetName);
  methods.Add("input", &PyTask::GetInput);
  methods.Add("inputs", &PyTask::GetInputs);
  methods.Add("output", &PyTask::GetOutput);
  methods.Add("outputs", &PyTask::GetOutputs);
  methods.Add("param", &PyTask::GetParameter);
  methods.Add("increment", &PyTask::Increment);
  type.tp_methods = methods.table();

  RegisterType(&type, module, "Task");
}

int PyTask::Init(Task *task) {
  task->AddRef();
  this->task = task;
  return 0;
}

void PyTask::Dealloc() {
  task->Release();
  Free();
}

PyObject *PyTask::Resource(Binding *binding) {
  if (binding == nullptr) Py_RETURN_NONE;
  PyResource *pyres = PyObject_New(PyResource, &PyResource::type);
  pyres->Init(binding->resource());
  return pyres->AsObject();
}

PyObject *PyTask::Resources(std::vector<Binding *> &bindings) {
  PyObject *list = PyList_New(bindings.size());
  if (list == nullptr) return nullptr;
  for (int i = 0; i < bindings.size(); ++i) {
    Binding *binding = bindings[i];
    PyResource *pyres = PyObject_New(PyResource, &PyResource::type);
    pyres->Init(binding->resource());
    PyList_SetItem(list, i, pyres->AsObject());
  }
  return list;
}

PyObject *PyTask::GetName() {
  return AllocateString(task->name());
}

PyObject *PyTask::GetInput(PyObject *args) {
  // Get arguments.
  const char *name;
  if (!PyArg_ParseTuple(args, "s", &name)) return nullptr;

  // Get input.
  return Resource(task->GetInput(name));
}

PyObject *PyTask::GetInputs(PyObject *args) {
  // Get arguments.
  const char *name;
  if (!PyArg_ParseTuple(args, "s", &name)) return nullptr;

  // Get inputs.
  std::vector<task::Binding *> inputs = task->GetInputs(name);
  return Resources(inputs);
}

PyObject *PyTask::GetOutput(PyObject *args) {
  // Get arguments.
  const char *name;
  if (!PyArg_ParseTuple(args, "s", &name)) return nullptr;

  // Get output.
  return Resource(task->GetOutput(name));
}

PyObject *PyTask::GetOutputs(PyObject *args) {
  // Get arguments.
  const char *name;
  if (!PyArg_ParseTuple(args, "s", &name)) return nullptr;

  // Get outputs.
  std::vector<task::Binding *> outputs = task->GetOutputs(name);
  return Resources(outputs);
}

PyObject *PyTask::GetParameter(PyObject *args) {
  // Get arguments.
  const char *name;
  PyObject *pydefval = nullptr;
  if (!PyArg_ParseTuple(args, "s|O", &name, &pydefval)) return nullptr;

  if (pydefval == nullptr) {
    return AllocateString(task->Get(name, ""));
  } else if (PyUnicode_Check(pydefval)) {
    const char *defval = PyUnicode_AsUTF8(pydefval);
    return AllocateString(task->Get(name, defval));
  } else if (PyLong_Check(pydefval)) {
    int64 defval = PyLong_AsLong(pydefval);
    return PyLong_FromLong(task->Get(name, defval));
  } else if (PyFloat_Check(pydefval)) {
    double defval = PyFloat_AsDouble(pydefval);
    return PyFloat_FromDouble(task->Get(name, defval));
  } else if (PyBool_Check(pydefval)) {
    bool defval = pydefval == Py_True;
    return PyBool_FromLong(task->Get(name, defval));
  } else {
    PyErr_SetString(PyExc_ValueError, "Unknown default value type");
    return nullptr;
  }
}

PyObject *PyTask::Increment(PyObject *args) {
  // Get arguments.
  const char *name;
  uint64 delta = 1;
  if (!PyArg_ParseTuple(args, "s|l", &name, &delta)) return nullptr;

  // Update counter.
  task->GetCounter(name)->Increment(delta);

  Py_RETURN_NONE;
}

void PyProcessor::Run(Task *task) {
  // Acquire Python global interpreter lock.
  PyGILState_STATE gstate = PyGILState_Ensure();

  // Create Python task object.
  PyObject *pyproc = PyObject_CallObject(pycls_, nullptr);
  CHECK(pyproc != nullptr);

  // Create task wrapper.
  PyTask *pytask = PyObject_New(PyTask, &PyTask::type);
  pytask->Init(task);

  // Call run() method to execute task.
  PyObject *ret = PyObject_CallMethod(pyproc, "run", "O", pytask->AsObject());
  if (ret == nullptr) {
    if (PyErr_Occurred()) PyErr_Print();
    LOG(FATAL) << "Error occured in task " << task->name();
  }
  Py_DECREF(ret);
  Py_DECREF(pyproc);
  Py_DECREF(pytask);

  // Release global interpreter lock.
  PyGILState_Release(gstate);
}

PyObject *PyRegisterTask(PyObject *self, PyObject *args) {
  // Get task name and class.
  const char *name;
  PyObject *cls;
  if (!PyArg_ParseTuple(args, "sO", &name, &cls)) return nullptr;

  // Check for type object.
  if (!PyType_Check(cls)) {
    PyErr_SetString(PyExc_ValueError, "Type object expected");
    return nullptr;
  }

  // Keep reference to class object to allow instance to be created in the task
  // factory.
  Py_INCREF(cls);

  // Initialize and acquire the global interpreter lock so the registered
  // Python tasks can be run in separate threads.
  PyEval_InitThreads();

  // Factory for creating new task instances.
  auto *factory = new Processor::Factory([cls]() {
    return new PyProcessor(cls);
  });

  // Dynamically register task processor in registry.
  Processor::Register(strdup(name), strdup(PyEval_GetFuncName(cls)),
                      "python", 0, factory);
  Py_RETURN_NONE;
}

PyObject *PyStartTaskMonitor(PyObject *self, PyObject *args) {
  // Get port number.
  int port;
  if (!PyArg_ParseTuple(args, "i", &port)) return nullptr;

  // Start HTTP server.
  bool start_http_server = false;
  if (http == nullptr) {
    LOG(INFO) << "Start HTTP server on port " << port;
    HTTPServerOptions options;
    http = new HTTPServer(options, port);
    start_http_server = true;
  }

  // Start dashboard.
  if (dashboard == nullptr) {
    dashboard = new task::Dashboard();
    dashboard->Register(http);
  }

  if (start_http_server) http->Start();

  Py_RETURN_NONE;
}

PyObject *PyGetJobStatistics() {
  if (dashboard == nullptr) Py_RETURN_NONE;
  string stats = dashboard->GetStatus();
  return PyUnicode_FromStringAndSize(stats.data(), stats.size());
}

PyObject *PyFinalizeDashboard() {
  if (dashboard != nullptr) dashboard->Finalize(60);
  Py_RETURN_NONE;
}

}  // namespace sling

