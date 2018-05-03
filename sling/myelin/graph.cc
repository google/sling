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

#include "sling/myelin/graph.h"

#include <math.h>
#include <set>

#include "sling/base/types.h"
#include "sling/file/file.h"
#include "sling/string/printf.h"

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

void GraphNodeOptions::Append(string *str, const char *delim) const {
  if (shape != nullptr) {
    StringAppendF(str, "shape=%s%s", shape, delim);
  }
  if (style != nullptr) {
    StringAppendF(str, "style=\"%s\"%s", style, delim);
  }
  if (color != nullptr) {
    StringAppendF(str, "color=\"%s\"%s", color, delim);
  }
  if (fillcolor != nullptr) {
    StringAppendF(str, "fillcolor=\"%s\"%s", fillcolor, delim);
  }
  if (fontname != nullptr) {
    StringAppendF(str, "fontname=\"%s\"%s", fontname, delim);
  }
  if (penwidth != 0) {
    StringAppendF(str, "penwidth=%d%s", penwidth, delim);
  }
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
  outputs.fillcolor = "#BBC6DD";

  vars.shape = "ellipse";
  vars.style = "filled";
  vars.color = "#89998A";
  vars.fillcolor = "#C4DAC5";

  consts.shape = "box";
  consts.style = "filled";
  consts.color = "#A6A6A6";
  consts.fillcolor = "#EEEEEE";

  globals.shape = "box";
  globals.style = "filled";
  globals.color = "#A6A6A6";
  globals.fillcolor = "#EEEEEE";

  funcs.shape = "box";
  funcs.style = "rounded,filled";
  funcs.fillcolor = "#FCFCFC";
  funcs.fontname = fontname;
}

static void AppendOp(string *str,
                     Flow::Operation *op,
                     const GraphOptions &options) {
  AppendOpId(str, op);
  str->append(" [");

  str->append("label=\"");
  if (options.op_type_as_label) {
    if (op->HasAttr("expr")) {
      if (op->type == "Assign") str->append("&#8612; ");
      str->append(op->GetAttr("expr"));
    } else if (op->HasAttr("var")) {
      str->append("&#10132; ");
      str->append(op->GetAttr("var"));
    } else {
      str->append(op->type);
    }
  } else {
    str->append(op->name);
  }
  if (options.types_in_labels && op->outdegree() >= 1) {
    str->append("\\n");
    str->append(op->outputs[0]->TypeString());
  }
  str->append("\" ");
  auto f = options.custom_ops.find(op->name);
  if (f != options.custom_ops.end()) {
    f->second.Append(str);
  } else {
    options.ops.Append(str);
  }

  bool calculate = (op->type == "Calculate" || op->type == "Assign");
  str->append(" tooltip=\"");
  StringAppendF(str, "name: %s&#10;", op->name.c_str());
  StringAppendF(str, "type: %s&#10;", op->type.c_str());
  if (!op->inputs.empty()) {
    str->append("input:&#10;");
    for (int i = 0; i < op->inputs.size(); ++i) {
      Flow::Variable *var = op->inputs[i];
      if (calculate) StringAppendF(str, "  %%%d ", i);
      StringAppendF(str, "  %s: %s",
        var->name.c_str(), var->TypeString().c_str());
      str->append("&#10;");
    }
  }
  if (!op->outputs.empty()) {
    str->append("output:&#10;");
    for (int i = 0; i < op->outputs.size(); ++i) {
      Flow::Variable *var = op->outputs[i];
      if (calculate) StringAppendF(str, "  @%d ", i);
      StringAppendF(str, "  %s: %s",
        var->name.c_str(), var->TypeString().c_str());
      str->append("&#10;");
    }
  }
  if (!op->attrs().empty()) {
    str->append("attr:&#10;");
    for (const auto &attr : op->attrs()) {
      StringAppendF(str, "  %s = %s&#10;",
        attr.name.c_str(), attr.value.c_str());
    }
  }
  str->append("\"");

  str->append("];\n");
}

static void AppendVar(string *str,
                      Flow::Variable *var,
                      const GraphOptions &options) {
  if (var->in() || var->out() || var->global()) {
    AppendVarId(str, var);
    str->append(" [");
    str->append("label=\"");
    size_t slash = var->name.rfind('/');
    if (slash != string::npos) {
      str->append(var->name.substr(slash + 1));
    } else {
      str->append(var->name);
    }
    if (options.types_in_labels) {
      str->append("\\n");
      str->append(var->TypeString());
    }
    if (options.max_value_size > 0 && var->data != nullptr) {
      int elements = var->elements();
      if (elements > 0 && elements <= options.max_value_size) {
        str->append("\\n");
        str->append(var->DataString());
      }
    }
    str->append("\" ");

    str->append(" tooltip=\"");
    if (var->constant()) str->append("const ");
    if (var->learnable()) str->append("learnable ");
    if (var->in()) str->append("in ");
    if (var->out()) str->append("out ");
    if (var->unique()) str->append("unique ");
    str->append("var ");
    str->append(var->name);
    if (!var->aliases.empty()) {
      str->append("&#10;alias:");
      for (const string &alias : var->aliases) {
        str->append("&#10;  ");
        str->append(alias);
      }
    }
    str->append("\" ");

    auto f = options.custom_vars.find(var->name);
    if (f != options.custom_vars.end()) {
      f->second.Append(str);
    } else if (var->constant()) {
      options.consts.Append(str);
    } else if (var->global()) {
      options.globals.Append(str);
    } else if (var->out() && !var->in()) {
      options.outputs.Append(str);
    } else if (var->in() && !var->out()) {
      options.inputs.Append(str);
    } else {
      options.vars.Append(str);
    }
    str->append("];\n");

    if (var->producer != nullptr) {
      AppendOpId(str, var->producer);
      str->append(" -> ");
      AppendVarId(str, var);
      str->append(" [");
      str->append("tooltip=\"");
      str->append(var->name);
      if (var->producer->outputs.size() > 1) {
        StringAppendF(str, " (@%d)", var->producer->OutputIndex(var));
      }
      str->append("\" ");
      AppendPenWidth(str, var, options);
      str->append("];\n");
    }

    for (Flow::Operation *consumer : var->consumers) {
      AppendVarId(str, var);
      str->append(" -> ");
      AppendOpId(str, consumer);
      str->append(" [");
      str->append("tooltip=\"");
      if (consumer->inputs.size() > 1) {
        StringAppendF(str, "%%%d = ", consumer->InputIndex(var));
      }
      str->append(var->name);
      str->append("\" ");
      AppendPenWidth(str, var, options);
      str->append("];\n");
    }
  }
}

static bool Exclusive(Flow::Variable *var, Flow::Function *func) {
  if ((var->producer != nullptr && var->producer->func == func) ||
      !var->consumers.empty()) {
    for (Flow::Operation *consumer : var->consumers) {
      if (consumer->func != func) return false;
    }
    return true;
  } else {
    return false;
  }
}

string FlowToDotGraph(const Flow &flow, const GraphOptions &options) {
  string str;

  // Output DOT graph header.
  str.append("digraph flow {\n");
  StringAppendF(&str, "graph [rankdir=%s;splines=%s]\n",
               options.direction,
               options.splines);
  StringAppendF(&str, "node [fontname=\"%s\"]\n", options.fontname);

  // Output DOT graph nodes for ops.
  auto funcs = flow.ops();
  funcs.push_back(nullptr);
  int cluster_id = 0;
  std::set<Flow::Variable *> exclusive;
  for (Flow::Function *func : flow.funcs()) {
    // Optionally make a cluster for each function.
    if (options.cluster_functions && func != nullptr) {
      StringAppendF(&str, "subgraph cluster_%d {\n", cluster_id++);
      options.funcs.Append(&str, ";\n");
      StringAppendF(&str, "label=\"%s\";\n", func->name.c_str());
      StringAppendF(&str, "tooltip=\"%s%s\";\n",
                    func->name.c_str(),
                    func->training() ? " (train)" : "");
    }

    // Output all ops in function.
    for (Flow::Operation *op : flow.ops()) {
      if (op->func == func) {
        AppendOp(&str, op, options);
      }
    }

    // Output variables that are only used by ops in the function.
    for (Flow::Variable *var : flow.vars()) {
      if (Exclusive(var, func)) {
        if (options.include_constants || var->data == nullptr) {
          AppendVar(&str, var, options);
        }
        exclusive.insert(var);
      }
    }

    if (options.cluster_functions && func != nullptr) str.append("}\n");
  }

  // Output DOT graph edges between ops.
  for (Flow::Operation *op : flow.ops()) {
    for (int i = 0; i < op->inputs.size(); ++i) {
      Flow::Variable *input = op->inputs[i];
      if (input->producer != nullptr && !input->in() && !input->out()) {
        AppendOpId(&str, input->producer);
        str.append(" -> ");
        AppendOpId(&str, op);
        str.append(" [");
        str.append("tooltip=\"");
        if (op->inputs.size() > 1) {
          StringAppendF(&str, "%%%d = ", i);
        }
        str.append(input->name);
        if (input->producer->outputs.size() > 1) {
          StringAppendF(&str, " (@%d)", input->producer->OutputIndex(input));
        }
        if (!input->aliases.empty()) {
          str.append("&#10;alias:");
          for (const string &alias : input->aliases) {
            str.append("&#10;  ");
            str.append(alias);
          }
        }
        str.append("\" ");
        AppendPenWidth(&str, input, options);
        str.append("];\n");
      }
    }
  }

  // Output shared variables.
  for (Flow::Variable *var : flow.vars()) {
    if (exclusive.count(var) == 0) {
      if (options.include_constants || !var->constant()) {
        AppendVar(&str, var, options);
      }
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

