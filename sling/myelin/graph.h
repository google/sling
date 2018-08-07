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

#ifndef SLING_MYELIN_GRAPH_H_
#define SLING_MYELIN_GRAPH_H_

#include <map>
#include <string>

#include "sling/base/types.h"
#include "sling/myelin/flow.h"

namespace sling {
namespace myelin {

// Options for node in Graphviz DOT graph.
struct GraphNodeOptions {
  const char *shape = nullptr;
  const char *style = nullptr;
  const char *color = nullptr;
  const char *fillcolor = nullptr;
  const char *fontname = nullptr;
  int penwidth = 0;

  // Append attributes to string.
  void Append(string *str, const char *delim = " ") const;
};

// Options for Graphviz DOT graph.
struct GraphOptions {
  GraphOptions();

  // Graph generation options.
  const char *fontname = "arial";
  const char *direction = "BT";
  const char *splines = "spline";
  bool op_type_as_label = true;
  bool types_in_labels = true;
  bool include_constants = true;
  bool include_intermediates = false;
  bool cluster_functions = true;
  int max_value_size = 8;
  int edge_thickness_scalar = 0;

  // Options for operations, inputs, outputs, and constants.
  GraphNodeOptions ops;
  GraphNodeOptions inputs;
  GraphNodeOptions outputs;
  GraphNodeOptions vars;
  GraphNodeOptions consts;
  GraphNodeOptions globals;
  GraphNodeOptions funcs;

  // Options for individual op nodes.
  std::map<string, GraphNodeOptions> custom_ops;

  // Options for individual var nodes.
  std::map<string, GraphNodeOptions> custom_vars;
};

// Convert flow to DOT graph.
string FlowToDotGraph(const Flow &flow, const GraphOptions &options);

// Write DOT graph file for flow.
void FlowToDotGraphFile(const Flow &flow,
                        const GraphOptions &options,
                        const string &filename);

}  // namespace myelin
}  // namespace sling

#endif  // SLING_MYELIN_GRAPH_H_

