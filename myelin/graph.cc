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

#include "myelin/graph.h"

#include <math.h>

#include "base/types.h"
#include "file/file.h"
#include "string/printf.h"

namespace sling {
namespace myelin {

static void AppendOpId(string *str, const Flow::Operation *op) {
  str->push_back('"');
  str->append(op->name);
  str->push_back('"');
}

static void AppendVarId(string *str, const Flow::Variable *var) {
  str->push_back('"');
  str->append("v:");
  str->append(var->name);
  str->push_back('"');
}

static void AppendPenWidth(string *str,
                           const Flow::Variable *var,
                           const GraphOptions &options) {
  if (options.edge_thickness_scalar == 0) return;
  size_t size = TypeTraits::of(var->type).size() * var->shape.elements();
  int width = log(size) * options.edge_thickness_scalar;
  if (width < 1) width = 1;
  StringAppendF(str, "penwidth=%d", width);
}

void GraphNodeOptions::Append(string *str) const {
  if (shape != nullptr) StringAppendF(str, "shape=%s ", shape);
  if (style != nullptr) StringAppendF(str, "style=\"%s\" ", style);
  if (color != nullptr) StringAppendF(str, "color=\"%s\" ", color);
  if (fillcolor != nullptr) StringAppendF(str, "fillcolor=\"%s\" ", fillcolor);
  if (penwidth != 0) StringAppendF(str, "penwidth=%d ", penwidth);
}

GraphOptions::GraphOptions() {
  ops.shape = "box";
  ops.style = "rounded,filled";
  ops.color = "#A79776";
  ops.fillcolor = "#EFD8A9";

  inputs.shape = "ellipse";
  inputs.style = "filled";
  inputs.color = "#899E7F";
  inputs.fillcolor = "#C5E2B6";

  outputs.shape = "ellipse";
  outputs.style = "filled";
  outputs.color = "#828A9A";
  outputs.fillcolor = "#bbc6dd";

  consts.shape = "box";
  consts.style = "filled";
  consts.color = "#A6A6A6";
  consts.fillcolor = "#EEEEEE";
}

string FlowToDotGraph(const Flow &flow, const GraphOptions &options) {
  string str;

  // Output DOT graph header.
  str.append("digraph flow {\n");
  StringAppendF(&str, "graph [rankdir=%s]\n", options.direction);
  StringAppendF(&str, "node [fontname=\"%s\"]\n", options.fontname);

  // Output DOT graph nodes for ops.
  for (Flow::Operation *op : flow.ops()) {
    AppendOpId(&str, op);
    str.append(" [");

    str.append("label=\"");
    if (options.op_type_as_label) {
      if (op->HasAttr("expr")) {
        str.append(op->GetAttr("expr"));
      } else {
        str.append(op->type);
      }
    } else {
      str.append(op->name);
    }
    if (options.types_in_labels && op->outdegree() >= 1) {
      str.append("\\n");
      str.append(op->outputs[0]->TypeString());
    }
    str.append("\" ");
    auto f = options.custom_ops.find(op->name);
    if (f != options.custom_ops.end()) {
      f->second.Append(&str);
    } else {
      options.ops.Append(&str);
    }
    str.append("];\n");
  }

  // Output DOT graph edges between ops.
  for (Flow::Operation *op : flow.ops()) {
    for (Flow::Variable *input : op->inputs) {
      if (input->producer != nullptr) {
        AppendOpId(&str, input->producer);
        str.append(" -> ");
        AppendOpId(&str, op);
        str.append(" [");
        str.append("tooltip=\"");
        str.append(input->name);
        str.append("\" ");
        AppendPenWidth(&str, input, options);
        str.append("];\n");
      }
    }
  }

  // Output DOT graph nodes and edges for inputs, outputs, and constants.
  for (Flow::Variable *var : flow.vars()) {
    if (!options.include_constants && var->data != nullptr) continue;
    if (var->in || var->out) {
      AppendVarId(&str, var);
      str.append(" [");
      str.append("label=\"");
      size_t slash = var->name.rfind('/');
      if (slash != string::npos) {
        str.append(var->name.substr(slash + 1));
      } else {
        str.append(var->name);
      }
      if (options.types_in_labels) {
        str.append("\\n");
        str.append(var->TypeString());
      }
      if (options.max_value_size > 0 && var->data != nullptr) {
        int elements = var->elements();
        if (elements > 0 && elements <= options.max_value_size) {
          str.append("\\n");
          str.append(var->DataString());
        }
      }
      str.append("\" ");

      auto f = options.custom_vars.find(var->name);
      if (f != options.custom_vars.end()) {
        f->second.Append(&str);
      } else if (var->data != nullptr) {
        options.consts.Append(&str);
      } else if (var->in) {
        options.inputs.Append(&str);
      } else {
        options.outputs.Append(&str);
      }
      str.append("];\n");
    }
    if (var->in) {
      for (Flow::Operation *consumer : var->consumers) {
        AppendVarId(&str, var);
        str.append(" -> ");
        AppendOpId(&str, consumer);
        str.append(" [");
        str.append("tooltip=\"");
        str.append(var->name);
        str.append("\" ");
        AppendPenWidth(&str, var, options);
        str.append("];\n");
      }
    }
    if (var->out && var->producer != nullptr) {
        AppendOpId(&str, var->producer);
        str.append(" -> ");
        AppendVarId(&str, var);
        str.append(" [");
        str.append("tooltip=\"");
        str.append(var->name);
        str.append("\" ");
        AppendPenWidth(&str, var, options);
        str.append("];\n");
    }
  }

  // Output DOT graph footer.
  str.append("}\n");

  return str;
}

void FlowToDotGraphFile(const Flow &flow,
                        const GraphOptions &options,
                        const string &filename) {
  string dot = FlowToDotGraph(flow, options);
  CHECK(File::WriteContents(filename, dot));
}

}  // namespace myelin
}  // namespace sling

