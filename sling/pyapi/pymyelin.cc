// Copyright 2018 Google Inc.
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

#include "sling/pyapi/pymyelin.h"

#include "sling/myelin/flow.h"

namespace sling {

using namespace myelin;

// Python type declarations.
PyTypeObject PyCompiler::type;
PyMethodTable PyCompiler::methods;

PyTypeObject PyNetwork::type;
PyMappingMethods PyNetwork::mapping;
PyMethodTable PyNetwork::methods;

PyTypeObject PyCell::type;
PyMethodTable PyCell::methods;

PyTypeObject PyInstance::type;
PyMappingMethods PyInstance::mapping;
PyMethodTable PyInstance::methods;

PyTypeObject PyChannel::type;
PyMappingMethods PyChannel::mapping;
PyMethodTable PyChannel::methods;

PyTypeObject PyTensor::type;
PyMappingMethods PyTensor::mapping;
PyBufferProcs PyTensor::buffer;
PyMethodTable PyTensor::methods;

void PyCompiler::Define(PyObject *module) {
  InitType(&type, "sling.Compiler", sizeof(PyCompiler), true);
  type.tp_init = method_cast<initproc>(&PyCompiler::Init);
  type.tp_dealloc = method_cast<destructor>(&PyCompiler::Dealloc);

  methods.AddO("compile", &PyCompiler::Compile);
  type.tp_methods = methods.table();

  RegisterType(&type, module, "Compiler");
}

int PyCompiler::Init(PyObject *args, PyObject *kwds) {
  // Initialize compiler.
  compiler = new Compiler();
  compiler->set_perf_flopctr(false);

  return 0;
}

void PyCompiler::Dealloc() {
  delete compiler;
  Free();
}

PyObject *PyCompiler::Compile(PyObject *arg) {
  // Import Python-based flow into a Myelin flow.
  Flow flow;
  PyBuffers buffers(&flow);
  if (!ImportFlow(arg, &flow, &buffers)) return nullptr;

  // Compile flow to network.
  Network *net = new Network();
  compiler->Compile(&flow, net);

  // Return compiled network.
  PyNetwork *pynet = PyObject_New(PyNetwork, &PyNetwork::type);
  pynet->Init(net);
  return pynet->AsObject();
}

bool PyCompiler::ImportFlow(PyObject *pyflow, Flow *flow, PyBuffers *buffers) {
  // Get variables.
  PyObject *pyvars = PyAttr(pyflow, "vars");
  std::unordered_map<PyObject *, Flow::Variable *> varmap;
  Py_ssize_t pos = 0;
  PyObject *pyvar;
  while (PyDict_Next(pyvars, &pos, nullptr, &pyvar)) {
    const char *name = PyStrAttr(pyvar, "name");
    string type = PyStrAttr(pyvar, "type");
    auto &t = TypeTraits::of(type);

    PyObject *pyshape = PyAttr(pyvar, "shape");
    Shape shape;
    for (int i = 0; i < PyList_Size(pyshape); ++i) {
      int dim = PyInt_AsLong(PyList_GetItem(pyshape, i));
      if (dim == -1) dim = 1;
      shape.add(dim);
    }
    Py_DECREF(pyshape);

    auto *var = flow->AddVariable(name, t.type(), shape);
    var->flags = PyIntAttr(pyvar, "flags");
    varmap[pyvar] = var;

    PyObject *pydata = PyAttr(pyvar, "data");
    if (pydata != Py_None) {
      var->data = buffers->GetData(pydata, &var->size);
      if (var->data == nullptr) return false;
    }
    Py_DECREF(pydata);
  }
  Py_DECREF(pyvars);

  // Get operations.
  PyObject *pyops = PyAttr(pyflow, "ops");
  std::unordered_map<PyObject *, Flow::Operation *> opmap;
  pos = 0;
  PyObject *pyop;
  while (PyDict_Next(pyops, &pos, nullptr, &pyop)) {
    const char *name = PyStrAttr(pyop, "name");
    const char *type = PyStrAttr(pyop, "type");

    auto *op = flow->AddOperation(name, type);
    op->flags = PyIntAttr(pyop, "flags");
    opmap[pyop] = op;

    PyObject *pyinputs = PyAttr(pyop, "inputs");
    for (int i = 0; i < PyList_Size(pyinputs); ++i) {
      Flow::Variable *input = varmap[PyList_GetItem(pyinputs, i)];
      CHECK(input != nullptr);
      op->AddInput(input);
    }
    Py_DECREF(pyinputs);

    PyObject *pyoutputs = PyAttr(pyop, "outputs");
    for (int i = 0; i < PyList_Size(pyoutputs); ++i) {
      Flow::Variable *output = varmap[PyList_GetItem(pyoutputs, i)];
      CHECK(output != nullptr);
      op->AddOutput(output);
    }
    Py_DECREF(pyoutputs);

    if (!ImportAttributes(pyop, op)) return false;
  }
  Py_DECREF(pyops);

  // Get functions.
  PyObject *pyfuncs = PyAttr(pyflow, "funcs");
  pos = 0;
  PyObject *pyfunc;
  while (PyDict_Next(pyfuncs, &pos, nullptr, &pyfunc)) {
    const char *name = PyStrAttr(pyfunc, "name");

    auto *func = flow->AddFunction(name);
    func->flags = PyIntAttr(pyfunc, "flags");

    PyObject *pyops = PyAttr(pyfunc, "ops");
    for (int i = 0; i < PyList_Size(pyops); ++i) {
      Flow::Operation *op = opmap[PyList_GetItem(pyops, i)];
      CHECK(op != nullptr);
      func->AddOperation(op);
    }
    Py_DECREF(pyops);
  }
  Py_DECREF(pyfuncs);

  // Get connectors.
  PyObject *pycnxs = PyAttr(pyflow, "cnxs");
  pos = 0;
  PyObject *pycnx;
  while (PyDict_Next(pycnxs, &pos, nullptr, &pycnx)) {
    const char *name = PyStrAttr(pycnx, "name");

    auto *cnx = flow->AddConnector(name);
    cnx->flags = PyIntAttr(pycnx, "flags");

    PyObject *pylinks = PyAttr(pycnx, "links");
    for (int i = 0; i < PyList_Size(pylinks); ++i) {
      Flow::Variable *var = varmap[PyList_GetItem(pylinks, i)];
      CHECK(var != nullptr);
      cnx->AddLink(var);
    }
    Py_DECREF(pylinks);
  }
  Py_DECREF(pycnxs);

  // Get blobs.
  PyObject *pyblobs = PyAttr(pyflow, "blobs");
  pos = 0;
  PyObject *pyblob;
  while (PyDict_Next(pyblobs, &pos, nullptr, &pyblob)) {
    const char *name = PyStrAttr(pyblob, "name");
    const char *type = PyStrAttr(pyblob, "type");

    auto *blob = flow->AddBlob(name, type);
    blob->flags = PyIntAttr(pyblob, "flags");

    PyObject *pydata = PyAttr(pyblob, "data");
    if (pydata != Py_None) {
      blob->data = buffers->GetData(pydata, &blob->size);
      if (blob->data == nullptr) return false;
    }
    Py_DECREF(pydata);

    if (!ImportAttributes(pyblob, blob)) return false;
  }
  Py_DECREF(pyblobs);

  return true;
}

bool PyCompiler::ImportAttributes(PyObject *obj, Attributes *attrs) {
  PyObject *pyattrs = PyAttr(obj, "attrs");
  Py_ssize_t pos = 0;
  PyObject *pyname;
  PyObject *pyvalue;
  while (PyDict_Next(pyattrs, &pos, &pyname, &pyvalue)) {
    const char *name = PyString_AsString(pyname);
    if (name == nullptr) return false;
    const char *value = PyString_AsString(pyvalue);
    if (value == nullptr) return false;
    attrs->SetAttr(name, value);
  }

  return true;
}

const char *PyCompiler::PyStrAttr(PyObject *obj, const char *name) {
  PyObject *attr = PyAttr(obj, name);
  const char *str = attr == Py_None ? "" : PyString_AsString(attr);
  CHECK(str != nullptr) << name;
  Py_DECREF(attr);
  return str;
}

int PyCompiler::PyIntAttr(PyObject *obj, const char *name) {
  PyObject *attr = PyAttr(obj, name);
  int value = PyNumber_AsSsize_t(attr, nullptr);
  Py_DECREF(attr);
  return value;
}

PyObject *PyCompiler::PyAttr(PyObject *obj, const char *name) {
  PyObject *attr = PyObject_GetAttrString(obj, name);
  CHECK(attr != nullptr) << name;
  return attr;
}

void PyNetwork::Define(PyObject *module) {
  InitType(&type, "sling.Network", sizeof(PyNetwork), false);
  type.tp_init = method_cast<initproc>(&PyNetwork::Init);
  type.tp_dealloc = method_cast<destructor>(&PyNetwork::Dealloc);

  type.tp_as_mapping = &mapping;
  mapping.mp_subscript = method_cast<binaryfunc>(&PyNetwork::LookupTensor);

  methods.AddO("cell", &PyNetwork::LookupCell);
  methods.Add("profile", &PyNetwork::Profile);
  type.tp_methods = methods.table();

  RegisterType(&type, module, "Network");
}

int PyNetwork::Init(Network *net) {
  this->net = net;
  return 0;
}

void PyNetwork::Dealloc() {
  delete net;
  Free();
}

PyObject *PyNetwork::LookupTensor(PyObject *key) {
  // Look up tensor in network.
  Tensor *tensor = GetTensor(key, nullptr);
  if (tensor == nullptr) return nullptr;

  // Get tensor data buffer.
  if (tensor->placement() == DEVICE) Py_RETURN_NONE;
  char *ptr = tensor->data();
  if (ptr == nullptr) Py_RETURN_NONE;
  if (tensor->ref()) {
    if (tensor->ref_placement() == DEVICE) Py_RETURN_NONE;
    ptr = *reinterpret_cast<char **>(ptr);
  }
  if (ptr == nullptr) Py_RETURN_NONE;

  // Return tensor data.
  PyTensor *pytensor = PyObject_New(PyTensor, &PyTensor::type);
  pytensor->Init(this->AsObject(), ptr, tensor);
  return pytensor->AsObject();
}

PyObject *PyNetwork::LookupCell(PyObject *key) {
  // Get cell name.
  const char *name = PyString_AsString(key);
  if (name == nullptr) return nullptr;

  // Look up cell in network.
  Cell *cell = net->LookupCell(name);
  if (cell == nullptr) {
    PyErr_SetString(PyExc_TypeError, "Unknown cell");
    return nullptr;
  }

  // Return cell wrapper.
  PyCell *pycell = PyObject_New(PyCell, &PyCell::type);
  pycell->Init(this, cell);
  return pycell->AsObject();
}

PyObject *PyNetwork::Profile() {
  return AllocateString(ProfileReport(*net));
}

Tensor *PyNetwork::GetTensor(PyObject *key, const Cell *cell) {
  // Get tensor name. If the key is a string, this used for looking up the
  // tensor by name. If key is an integer, it is used as an index into the
  // parameter array of the network. Otherwise, Otherwise, the repr() method
  // is used for computing the name of the tensor.
  Tensor *tensor;
  if (PyInt_Check(key)) {
    int index = PyInt_AsLong(key);
    auto &params = net->parameters();
    if (index < 0 || index >= params.size()) {
      PyErr_SetString(PyExc_IndexError, "Invalid parameter tensor index");
      return nullptr;
    }
    tensor = params[index];
  } else if (PyString_Check(key)) {
    const char *name = PyString_AsString(key);
    if (name == nullptr) return nullptr;
    tensor = net->LookupParameter(name);
  } else {
    PyObject *repr = PyObject_Repr(key);
    if (repr == nullptr) return nullptr;
    const char *name = PyString_AsString(repr);
    if (name == nullptr) {
      Py_DECREF(repr);
      return nullptr;
    }
    tensor = net->LookupParameter(name);
    Py_DECREF(repr);
  }

  if (tensor == nullptr) {
    PyErr_SetString(PyExc_ValueError, "Unknown tensor");
    return nullptr;
  }

  if (tensor->cell() != cell) {
    if (cell == nullptr) {
      PyErr_SetString(PyExc_TypeError, "Tensor is not a global tensor");
    } else {
      PyErr_SetString(PyExc_TypeError, "Tensor does not belong to cell");
    }
    return nullptr;
  }

  return tensor;
}

void PyCell::Define(PyObject *module) {
  InitType(&type, "sling.Cell", sizeof(PyCell), false);
  type.tp_init = method_cast<initproc>(&PyCell::Init);
  type.tp_dealloc = method_cast<destructor>(&PyCell::Dealloc);

  methods.Add("instance", &PyCell::NewInstance);
  methods.Add("channel", &PyCell::NewChannel);
  methods.AddO("index", &PyCell::Index);
  type.tp_methods = methods.table();

  RegisterType(&type, module, "Cell");
}

int PyCell::Init(PyNetwork *pynet, myelin::Cell *cell) {
  this->cell = cell;
  this->pynet = pynet;
  Py_INCREF(pynet);
  return 0;
}

void PyCell::Dealloc() {
  Py_DECREF(pynet);
  Free();
}

PyObject *PyCell::NewInstance() {
  PyInstance *pyinstance = PyObject_New(PyInstance, &PyInstance::type);
  pyinstance->Init(this);
  return pyinstance->AsObject();
}

PyObject *PyCell::NewChannel(PyObject *args) {
  // Get tensor name for channel and optionally size.
  PyObject *key = nullptr;
  int size = 0;
  if (!PyArg_ParseTuple(args, "O|i", &key, &size)) return nullptr;

  // Look up tensor in network.
  Tensor *tensor = pynet->GetTensor(key, cell);
  if (tensor == nullptr) return nullptr;

  // Create new channel.
  PyChannel *pychannel = PyObject_New(PyChannel, &PyChannel::type);
  pychannel->Init(pynet, tensor, size);
  return pychannel->AsObject();
}

PyObject *PyCell::Index(PyObject *key) {
  // Look up tensor in network.
  Tensor *tensor = pynet->GetTensor(key, cell);
  if (tensor == nullptr) return nullptr;

  // Find parameter tensor index.
  int index = -1;
  auto &params = pynet->net->parameters();
  for (int i = 0; i < params.size(); ++i) {
    if (params[i] == tensor) {
      index = i;
      break;
    }
  }
  return PyInt_FromLong(index);
}

void PyInstance::Define(PyObject *module) {
  InitType(&type, "sling.Instance", sizeof(PyInstance), false);
  type.tp_init = method_cast<initproc>(&PyInstance::Init);
  type.tp_dealloc = method_cast<destructor>(&PyInstance::Dealloc);
  type.tp_str = method_cast<reprfunc>(&PyInstance::Str);
  type.tp_repr = method_cast<reprfunc>(&PyInstance::Str);

  type.tp_as_mapping = &mapping;
  mapping.mp_subscript = method_cast<binaryfunc>(&PyInstance::LookupTensor);

  methods.Add("compute", &PyInstance::Compute);
  methods.Add("clear", &PyInstance::Clear);
  methods.Add("connect", &PyInstance::Connect);
  type.tp_methods = methods.table();

  RegisterType(&type, module, "Instance");
}

int PyInstance::Init(PyCell *pycell) {
  this->pycell = pycell;
  Py_INCREF(pycell);
  data = new Instance(pycell->cell);
  data->Clear();
  return 0;
}

void PyInstance::Dealloc() {
  delete data;
  Py_DECREF(pycell);
  Free();
}

PyObject *PyInstance::LookupTensor(PyObject *key) {
  // Look up tensor in network.
  Tensor *tensor = pycell->pynet->GetTensor(key, data->cell());
  if (tensor == nullptr) return nullptr;

  // Get tensor data buffer.
  if (tensor->placement() == DEVICE) Py_RETURN_NONE;
  char *ptr = data->GetAddress(tensor);
  if (ptr == nullptr) Py_RETURN_NONE;
  if (tensor->ref()) {
    if (tensor->ref_placement() == DEVICE) Py_RETURN_NONE;
    ptr = *reinterpret_cast<char **>(ptr);
  }
  if (ptr == nullptr) Py_RETURN_NONE;

  // Return tensor data.
  PyTensor *pytensor = PyObject_New(PyTensor, &PyTensor::type);
  pytensor->Init(this->AsObject(), ptr, tensor);
  return pytensor->AsObject();
}

PyObject *PyInstance::Connect(PyObject *args) {
  // Get arguments: tensor name, channel, index.
  PyObject *key;
  PyChannel *pychannel;
  int index;
  if (!PyArg_ParseTuple(args, "OOi", &key, &pychannel, &index)) return nullptr;
  if (!PyChannel::TypeCheck(pychannel)) return nullptr;

  // Look up tensor in network.
  Tensor *tensor = pycell->pynet->GetTensor(key, data->cell());
  if (tensor == nullptr) return nullptr;

  // Check index.
  if (index < 0 || index >= pychannel->channel->size()) {
    PyErr_SetString(PyExc_IndexError, "Invalid channel element index");
    return nullptr;
  }

  // Set reference tensor to element in channel.
  data->Set(tensor, pychannel->channel, index);

  Py_RETURN_NONE;
}

PyObject *PyInstance::Compute() {
  data->Compute();
  Py_RETURN_NONE;
}

PyObject *PyInstance::Clear() {
  data->Clear();
  Py_RETURN_NONE;
}

PyObject *PyInstance::Str() {
  return AllocateString(data->ToString());
}

void PyChannel::Define(PyObject *module) {
  InitType(&type, "sling.Channel", sizeof(PyChannel), false);
  type.tp_init = method_cast<initproc>(&PyChannel::Init);
  type.tp_dealloc = method_cast<destructor>(&PyChannel::Dealloc);

  type.tp_as_mapping = &mapping;
  mapping.mp_length = method_cast<lenfunc>(&PyChannel::Size);
  mapping.mp_subscript = method_cast<binaryfunc>(&PyChannel::Lookup);

  methods.Add("resize", &PyChannel::Resize);
  type.tp_methods = methods.table();

  RegisterType(&type, module, "Channel");
}

int PyChannel::Init(PyNetwork *pynet, Tensor *format, int size) {
  this->pynet = pynet;
  Py_INCREF(pynet);
  channel = new Channel(format);
  if (size > 0) channel->resize(size);
  return 0;
}

void PyChannel::Dealloc() {
  delete channel;
  Py_DECREF(pynet);
  Free();
}

PyObject *PyChannel::Size() {
  return PyInt_FromLong(channel->size());
}

PyObject *PyChannel::Lookup(PyObject *key) {
  // Get index.
  int index = PyInt_AsLong(key);
  if (index == -1 && PyErr_Occurred()) return nullptr;
  if (index < 0 || index >= channel->size()) {
    PyErr_SetString(PyExc_IndexError, "Invalid channel element index");
    return nullptr;
  }

  // Cannot access channel elements in device.
  if (channel->placement() == DEVICE) Py_RETURN_NONE;

  // Return element as tensor.
  char *ptr = channel->at(index);
  PyTensor *pytensor = PyObject_New(PyTensor, &PyTensor::type);
  pytensor->Init(this->AsObject(), ptr, channel->format());
  return pytensor->AsObject();
}

PyObject *PyChannel::Resize(PyObject *args) {
  // Get new channel size.
  int size = 0;
  if (!PyArg_ParseTuple(args, "i", &size)) return nullptr;
  if (size < 0) size = 0;

  // Resize channel.
  channel->resize(size);

  Py_RETURN_NONE;
}

void PyTensor::Define(PyObject *module) {
  InitType(&type, "sling.Tensor", sizeof(PyTensor), false);
  type.tp_init = method_cast<initproc>(&PyTensor::Init);
  type.tp_dealloc = method_cast<destructor>(&PyTensor::Dealloc);
  type.tp_str = method_cast<reprfunc>(&PyTensor::Str);
  type.tp_repr = method_cast<reprfunc>(&PyTensor::Str);

  type.tp_as_mapping = &mapping;
  mapping.mp_subscript = method_cast<binaryfunc>(&PyTensor::GetElement);
  mapping.mp_ass_subscript = method_cast<objobjargproc>(&PyTensor::SetElement);

  type.tp_as_buffer = &buffer;
  type.tp_flags |= Py_TPFLAGS_HAVE_NEWBUFFER;
  buffer.bf_getbuffer =
      method_cast<getbufferproc>(&PyTensor::GetBuffer);
  buffer.bf_releasebuffer =
      method_cast<releasebufferproc>(&PyTensor::ReleaseBuffer);

  methods.Add("name", &PyTensor::Name);
  methods.Add("rank", &PyTensor::Rank);
  methods.Add("shape", &PyTensor::Shape);
  methods.Add("type", &PyTensor::Type);
  type.tp_methods = methods.table();

  RegisterType(&type, module, "Tensor");
}

int PyTensor::Init(PyObject *owner, char *data, const Tensor *format) {
  this->owner = owner;
  this->data = data;
  this->format = format;
  if (owner) Py_INCREF(owner);
  shape = nullptr;
  strides = nullptr;
  return 0;
}

void PyTensor::Dealloc() {
  if (shape) free(shape);
  if (strides) free(strides);
  if (owner) Py_DECREF(owner);
  Free();
}

PyObject *PyTensor::Name() {
  return AllocateString(format->name());
}

PyObject *PyTensor::Rank() {
  return PyInt_FromLong(format->rank());
}

PyObject *PyTensor::Shape() {
  PyObject *dims = PyList_New(format->rank());
  for (int d = 0; d < format->rank(); ++d) {
    PyList_SetItem(dims, d, PyInt_FromLong(format->dim(d)));
  }
  return dims;
}

PyObject *PyTensor::Type() {
  return AllocateString(TypeTraits::of(format->type()).name());
}

PyObject *PyTensor::Str() {
  return AllocateString(format->ToString(data, false));
}

PyObject *PyTensor::GetElement(PyObject *index) {
  // Get reference to tensor element.
  char *ptr = GetAddress(index);
  if (ptr == nullptr) return nullptr;

  // Return element.
  switch (format->type()) {
    case DT_FLOAT:
      return PyFloat_FromDouble(*reinterpret_cast<float *>(ptr));
    case DT_DOUBLE:
      return PyFloat_FromDouble(*reinterpret_cast<double *>(ptr));
    case DT_INT32:
      return PyInt_FromLong(*reinterpret_cast<int32 *>(ptr));
    case DT_UINT8:
      return PyInt_FromLong(*reinterpret_cast<uint8 *>(ptr));
    case DT_INT16:
      return PyInt_FromLong(*reinterpret_cast<int16 *>(ptr));
    case DT_INT8:
      return PyInt_FromLong(*reinterpret_cast<int8 *>(ptr));
    case DT_INT64:
      return PyLong_FromLongLong(*reinterpret_cast<int64 *>(ptr));
    case DT_BOOL:
      return PyBool_FromLong(*reinterpret_cast<bool *>(ptr));
    default:
      PyErr_SetString(PyExc_ValueError, "Unsupported element type");
      return nullptr;
  }
}

int PyTensor::SetElement(PyObject *index, PyObject *value) {
  // Elements cannot be deleted.
  if (value == nullptr) {
    PyErr_SetString(PyExc_ValueError, "Cannot delete values from tensor");
    return -1;
  }

  // Get reference to tensor element.
  char *ptr = GetAddress(index);
  if (ptr == nullptr) return -1;

  // Return element.
  switch (format->type()) {
    case DT_FLOAT: {
      float v = PyFloat_AsDouble(value);
      if (v == -1.0 && PyErr_Occurred()) return -1;
      *reinterpret_cast<float *>(ptr) = v;
      break;
    }
    case DT_DOUBLE: {
      double v = PyFloat_AsDouble(value);
      if (v == -1.0 && PyErr_Occurred()) return -1;
      *reinterpret_cast<double *>(ptr) = v;
      break;
    }
    case DT_INT32: {
      int v = PyInt_AsLong(value);
      if (v == -1 && PyErr_Occurred()) return -1;
      *reinterpret_cast<int32 *>(ptr) = v;
      break;
    }
    case DT_UINT8: {
      int v = PyInt_AsLong(value);
      if (v == -1 && PyErr_Occurred()) return -1;
      *reinterpret_cast<uint8 *>(ptr) = v;
      break;
    }
    case DT_INT16: {
      int v = PyInt_AsLong(value);
      if (v == -1 && PyErr_Occurred()) return -1;
      *reinterpret_cast<int16 *>(ptr) = v;
      break;
    }
    case DT_INT8: {
      int v = PyInt_AsLong(value);
      if (v == -1 && PyErr_Occurred()) return -1;
      *reinterpret_cast<int8 *>(ptr) = v;
      break;
    }
    case DT_INT64: {
      int64 v = PyLong_AsLongLong(value);
      if (v == -1 && PyErr_Occurred()) return -1;
      *reinterpret_cast<int64 *>(ptr) = v;
      break;
    }
    case DT_BOOL: {
      int v = PyObject_IsTrue(value);
      if (v == -1) return -1;
      *reinterpret_cast<bool *>(ptr) = v;
      break;
    }
    default:
      PyErr_SetString(PyExc_ValueError, "Unsupported element type");
      return -1;
  }

  return 0;
}

char *PyTensor::GetAddress(PyObject *index) {
  int rank = format->rank();
  if (rank == 0) {
    // Ignore index for scalars.
    return data;
  } else if (rank == 1) {
    // Get single-dimensional index.
    int idx = PyInt_AsLong(index);
    if (idx == -1 &&  PyErr_Occurred()) return nullptr;
    if (idx < 0) idx += format->dim(0);
    if (idx < 0 || idx >= format->dim(0)) {
      PyErr_SetString(PyExc_IndexError, "Invalid tensor index");
      return nullptr;
    }
    return data + format->offset(idx);
  } else if (PyTuple_Check(index)) {
    // Get multi-dimensional index.
    int size =  PyTuple_Size(index);
    if (size != rank) {
      PyErr_SetString(PyExc_IndexError, "Wrong number of indices");
      return nullptr;
    }
    size_t ofs = 0;
    for (int d = 0; d < rank; ++d) {
      int idx = PyInt_AsLong(PyTuple_GetItem(index, d));
      if (idx == -1 &&  PyErr_Occurred()) return nullptr;
      if (idx < 0) idx += format->dim(d);
      if (idx < 0 || idx >= format->dim(d)) {
        PyErr_SetString(PyExc_IndexError, "Invalid tensor index");
        return nullptr;
      }
      ofs += idx * format->stride(d);
    }
    return data + ofs;
  } else {
    PyErr_SetString(PyExc_IndexError, "Invalid tensor index");
    return nullptr;
  }
}

int PyTensor::GetBuffer(Py_buffer *view, int flags) {
  memset(view, 0, sizeof(Py_buffer));
  view->buf = data;
  view->obj = AsObject();
  view->len = format->size();
  view->readonly = 0;

  if (flags != PyBUF_SIMPLE) {
    int dims = format->rank();
    view->itemsize = format->element_size();

    if (flags & PyBUF_FORMAT) {
      view->format = GetFormat();
    }

    if (flags & PyBUF_ND) {
      view->ndim = dims;
      if (dims > 0) view->shape = GetShape();
    }

    if (flags & PyBUF_STRIDES) {
      if ((flags & PyBUF_C_CONTIGUOUS) == PyBUF_C_CONTIGUOUS) {
        if (format->order() != ROW_MAJOR) {
          PyErr_SetString(PyExc_TypeError, "Buffer is not row-major");
          return -1;
        }
      }
      if ((flags & PyBUF_F_CONTIGUOUS) == PyBUF_F_CONTIGUOUS) {
        if (format->order() != COLUMN_MAJOR) {
          PyErr_SetString(PyExc_TypeError, "Buffer is not column-major");
          return -1;
        }
      }

      if (dims > 0) view->strides = GetStrides();
    }
  }

  Py_INCREF(view->obj);
  return 0;
}

void PyTensor::ReleaseBuffer(Py_buffer *view) {
}

Py_ssize_t *PyTensor::GetShape() {
  if (shape == nullptr) {
    int dims = format->rank();
    shape = static_cast<Py_ssize_t *>(malloc(dims * sizeof(Py_ssize_t)));
    for (int d = 0; d < dims; ++d) shape[d] = format->dim(d);
  }
  return shape;
}

Py_ssize_t *PyTensor::GetStrides() {
  if (strides == nullptr) {
    int dims = format->rank();
    strides = static_cast<Py_ssize_t *>(malloc(dims * sizeof(Py_ssize_t)));
    for (int d = 0; d < dims; ++d) strides[d] = format->stride(d);
  }
  return strides;
}

PyBuffers::~PyBuffers() {
  for (auto *view : views_) {
    PyBuffer_Release(view);
    delete view;
  }
  for (auto *ref : refs_) {
    Py_DECREF(ref);
  }
}

char *PyBuffers::GetData(PyObject *obj, size_t *size) {
  if (PyObject_CheckBuffer(obj)) {
    // Get data using Python buffer protocol.
    Py_buffer *view = new Py_buffer;
    if (PyObject_GetBuffer(obj, view, PyBUF_C_CONTIGUOUS) == -1) {
      delete view;
      return nullptr;
    }
    views_.push_back(view);
    *size = view->len;
    return static_cast<char *>(view->buf);
  } else if (PyString_Check(obj)) {
    // Get string buffer.
    char *data;
    Py_ssize_t length;
    if (PyString_AsStringAndSize(obj, &data, &length) == -1) return nullptr;
    Py_INCREF(obj);
    refs_.push_back(obj);
    *size = length;
    return data;
  } else if (PyFloat_Check(obj)) {
    float v = PyFloat_AsDouble(obj);
    *size = sizeof(float);
    return flow_->AllocateMemory(&v, sizeof(float));
  } else if (PyInt_Check(obj)) {
    int v = PyInt_AsLong(obj);
    *size = sizeof(int);
    return flow_->AllocateMemory(&v, sizeof(int));
  } else {
    PyErr_SetString(PyExc_TypeError, "Cannot get data from object");
    return nullptr;
  }
}

}  // namespace sling

