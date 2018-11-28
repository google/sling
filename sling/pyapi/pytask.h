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

#ifndef SLING_PYAPI_PYTASK_H_
#define SLING_PYAPI_PYTASK_H_

#include <unordered_map>

#include "sling/task/job.h"
#include "sling/task/process.h"
#include "sling/pyapi/pybase.h"

namespace sling {

// Python wrapper for job.
struct PyJob : public PyBase {
  // Mappings.
  typedef std::unordered_map<PyObject *, task::Resource *> ResourceMapping;
  typedef std::unordered_map<PyObject *, task::Task *> TaskMapping;

  // Initialize job wrapper.
  int Init(PyObject *args, PyObject *kwds);

  // Deallocate job wrapper.
  void Dealloc();

  // Start job.
  PyObject *Start();

  // Check if job has completed.
  PyObject *Done();

  // Wait for job to complete.
  PyObject *Wait();

  // Wait for job to complete with timeout.
  PyObject *WaitFor(PyObject *timeout);

  // Get current counter values.
  PyObject *Counters();

  // Convert Port object.
  static task::Port PyGetPort(PyObject *obj, const TaskMapping &mapping);

  // Convert Format object.
  static task::Format PyGetFormat(PyObject *obj);

  // Convert Shard object.
  static task::Shard PyGetShard(PyObject *obj);

  // Get string attribute for object.
  static const char *PyStrAttr(PyObject *obj, const char *name);

  // Get integer attribute for object.
  static int PyIntAttr(PyObject *obj, const char *name);

  // Get attribute for object. Returns new reference.
  static PyObject *PyAttr(PyObject *obj, const char *name);

  // Job object for runnning job.
  task::Job *job;

  // Whether job is currently running.
  bool running;

  // Registration.
  static PyTypeObject type;
  static PyMethodTable methods;
  static void Define(PyObject *module);
};

// Python wrapper for task resource.
struct PyResource : public PyBase {
  // Initialize resource wrapper.
  int Init(task::Resource *resource);

  // Deallocate job wrapper.
  void Dealloc();

  // Resource properties.
  PyObject *name;
  PyObject *format;
  int part;
  int of;

  // Registration.
  static PyTypeObject type;
  static PyMemberDef members[];
  static void Define(PyObject *module);
};

// Python wrapper for task.
struct PyTask : public PyBase {
  // Initialize task wrapper.
  int Init(task::Task *task);

  // Deallocate task wrapper.
  void Dealloc();

  // Get task name.
  PyObject *GetName();

  // Get input resource. Return None if resource not found.
  PyObject *GetInput(PyObject *args);

  // Get list of input resources.
  PyObject *GetInputs(PyObject *args);

  // Get output resource. Return None if resource not found.
  PyObject *GetOutput(PyObject *args);

  // Get list of output resources.
  PyObject *GetOutputs(PyObject *args);

  // Get task parameter with optional default value.
  PyObject *GetParameter(PyObject *args);

  // Increment counter.
  PyObject *Increment(PyObject *args);

  // Return resource for binding.
  PyObject *Resource(task::Binding *binding);

  // Return resources for bindings.
  PyObject *Resources(std::vector<task::Binding *> &bindings);

  // Task object.
  task::Task *task;

  // Registration.
  static PyTypeObject type;
  static PyMethodTable methods;
  static void Define(PyObject *module);
};

// Task processor that wraps a python task object. The wrapper is a task process
// so the python task can be run in a separate thread.
class PyProcessor : public task::Process {
 public:
  PyProcessor(PyObject *pycls) : pycls_(pycls) {}

  void Run(task::Task *task) override;

 private:
  PyObject *pycls_;
};

// Register task processor in task system.
PyObject *PyRegisterTask(PyObject *self, PyObject *args);

// Start task monitor dashboard.
PyObject *PyStartTaskMonitor(PyObject *self, PyObject *args);

// Get stats for running and completed jobs.
PyObject *PyGetJobStatistics();

// Wait until final status has been sent by dashboard.
PyObject *PyFinalizeDashboard();

}  // namespace sling

#endif  // SLING_PYAPI_PYTASK_H_

