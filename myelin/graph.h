#ifndef MYELIN_GRAPH_H_
#define MYELIN_GRAPH_H_

#include <map>
#include <string>

#include "base/types.h"
#include "myelin/flow.h"

namespace sling {
namespace myelin {

// Options for node in Graphviz DOT graph.
struct GraphNodeOptions {
  const char *shape = nullptr;
  const char *style = nullptr;
  const char *color = nullptr;
  const char *fillcolor = nullptr;
  int penwidth = 0;

  // Append attributes to string.
  void Append(string *str) const;
};

// Options for Graphviz DOT graph.
struct GraphOptions {
  GraphOptions();

  // Graph generation options.
  const char *fontname = "arial";
  bool op_type_as_label = true;
  bool types_in_labels = true;
  bool include_constants = true;
  int max_value_size = 16;
  float edge_thickness_scalar = 0.0;

  // Options for operations, inputs, outputs, and constants.
  GraphNodeOptions ops;
  GraphNodeOptions inputs;
  GraphNodeOptions outputs;
  GraphNodeOptions consts;

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

#endif  // MYELIN_GRAPH_H_

