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

#ifndef SLING_PYAPI_PYMISC_H_
#define SLING_PYAPI_PYMISC_H_

#include "sling/pyapi/pybase.h"

namespace sling {

// Get list of registered command-line flags.
PyObject *PyGetFlags();

// Set value of command-line flag.
PyObject *PySetFlag(PyObject *self, PyObject *args);

// Log message.
PyObject *PyLogMessage(PyObject *self, PyObject *args);

}  // namespace sling

#endif  // SLING_PYAPI_PYMISC_H_

