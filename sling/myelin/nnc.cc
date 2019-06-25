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

// Neural network compiler that compiles a flow file with a neural network model
// to an ELF object file that can be linked into a binary. It also generates a
// C++ header file for accessing the compiled model from other C++ modules. The
// model data can either be linked into the object file or stored in an external
// data file that can be loaded at run time.

#include <string>

#include "sling/base/init.h"
#include "sling/base/flags.h"
#include "sling/base/logging.h"
#include "sling/base/types.h"
#include "sling/file/file.h"
#include "sling/myelin/aot-linker.h"
#include "sling/myelin/compiler.h"

DEFINE_string(flow, "", "Myelin flow file");
DEFINE_string(o, "", "ELF object output file for generated code");
DEFINE_string(hdr, "", "C++ header file for accessing model");
DEFINE_string(data, "", "Separate data file for storing parameters");
DEFINE_string(ns, "", "C++ name space for generated code");
DEFINE_bool(upper, false, "Uppercase class names");
DEFINE_bool(pic, false, "Generate position-independent code");

using namespace sling;
using namespace sling::myelin;

// Return the base name of a file name.
string basename(const string &name) {
  string base = name;
  int begin = base.rfind('/');
  if (begin != -1) base = base.substr(begin + 1);
  int end = base.find('.');
  if (end != -1) base = base.substr(0, end);
  return base;
}

int main(int argc, char *argv[]) {
  InitProgram(&argc, &argv);

  // Load flow.
  Flow flow;
  CHECK(flow.Load(FLAGS_flow));

  // Set up AOT linker.
  AOTLinker::Options linker_opts;
  if (!FLAGS_ns.empty()) {
    linker_opts.ns = FLAGS_ns;
  } else {
    linker_opts.ns = basename(FLAGS_flow);
  }
  if (!FLAGS_data.empty()) linker_opts.external_data = true;
  linker_opts.uppercase_names = FLAGS_upper;
  linker_opts.flow_file = FLAGS_flow;
  AOTLinker linker(linker_opts);

  // Compile flow.
  Network net;
  net.set_linker(&linker);
  net.options().aot = true;
  net.options().pic = FLAGS_pic;
  Compiler compiler;
  compiler.set_perf_flopctr(false);
  compiler.Compile(&flow, &net);

  // Add channels.
  for (Flow::Connector *cnx : flow.cnxs()) {
    if (cnx->links.empty()) continue;
    Tensor *format = net.GetParameter(cnx->links[0]->name);
    linker.AddChannel(cnx->name, format);
  }

  // Write ELF object file.
  if (!FLAGS_o.empty()) {
    linker.Link();
    linker.Write(FLAGS_o);
  }

  // Write header file.
  if (!FLAGS_hdr.empty()) {
    linker.WriteHeader(FLAGS_hdr);
  }

  // Write parameter data file.
  if (!FLAGS_data.empty()) {
    linker.WriteData(FLAGS_data);
  }

  return 0;
}

