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

static bool Intermediate(Flow::Variable *var) {
  if (var->in() || var->out() || var->global()) return true;
  if (var->producer == nullptr) return true;
  for (Flow::Operation *consumer : var->consumers) {
    if (consumer->func != var->producer->func) return true;
  }
  return false;
}

static bool Exclusive(Flow::Variable *var, Flow::Function *func) {
  if (var->producer == nullptr) {
    if (var->usages() == 0) return false;
  } else {
    if (var->producer->func != func) return false;
  }
  for (Flow::Operation *consumer : var->consumers) {
    if (consumer->func != func) return false;
  }
  return true;
}

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
  const Shape shape = var->shape;
  int width;
  if (shape.elements() == 1 || !shape.defined()) {
    width = 1;
  } else if (shape.rank() == 1) {
    width = 2;
  } else if (shape.rank() == 2) {
    if (shape.dim(0) == 1 || shape.dim(0) == 1) {
      width = 2;
    } else {
      width = 3;
    }
  } else {
    width = 4;
  }
  StringAppendF(str, "penwidth=%d", width * options.edge_thickness_scalar);
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

static void AppendOp(string *str,
                     Flow::Operation *op,
                     const GraphOptions &options) {
  AppendOpId(str, op);
  str->append(" [");

  str->append("label=\"");
  if (options.op_type_as_label) {
    if (op->HasAttr("expr")) {
      if (op->type == "Assign") str->append("&#8612; ");
      string expr = op->GetAttr("expr");
      for (char c : expr) {
        str->push_back(c);
        if (c == ';') str->append("&#10;");
      }
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
  if (op->func != nullptr) {
    StringAppendF(str, "func: %s&#10;", op->func->name.c_str());
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
  if (var->is(Flow::Variable::ROW)) str->append("row ");
  if (var->is(Flow::Variable::COL)) str->append("col ");
  if (var->learnable()) str->append("learnable ");
  if (var->in()) str->append("in ");
  if (var->out()) str->append("out ");
  if (var->unique()) str->append("unique ");
  if (var->is(Flow::Variable::NOGRADIENT)) str->append("nograd ");
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
  int cluster_id = 0;
  std::set<Flow::Variable *> exclusive;
  for (Flow::Function *func : flow.funcs()) {
    // Optionally make a cluster for each function.
    if (options.cluster_functions) {
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
      if (!options.include_constants && var->constant()) continue;
      if (Exclusive(var, func)) {
        if (options.include_intermediates || Intermediate(var)) {
          AppendVar(&str, var, options);
        }
        exclusive.insert(var);
      }
    }

    if (options.cluster_functions) str.append("}\n");
  }

  // Output shared ops.
  for (Flow::Operation *op : flow.ops()) {
    if (op->func == nullptr) {
      AppendOp(&str, op, options);
    }
  }

  // Output shared variables.
  for (Flow::Variable *var : flow.vars()) {
    if (exclusive.count(var) > 0) continue;
    if (!options.include_constants && var->constant()) continue;
    AppendVar(&str, var, options);
  }


  // Output DOT graph edges between ops and variables.
  for (Flow::Variable *var : flow.vars()) {
    if (!options.include_constants && var->constant()) continue;
    if (options.include_intermediates || Intermediate(var)) {
      if (var->producer != nullptr) {
        // Output edge between producer and variable.
        AppendOpId(&str, var->producer);
        str.append(" -> ");
        AppendVarId(&str, var);
        str.append(" [");
        str.append("tooltip=\"");
        str.append(var->name);
        if (var->producer->outputs.size() > 1) {
          StringAppendF(&str, " (@%d)", var->producer->OutputIndex(var));
        }
        str.append("\" ");
        AppendPenWidth(&str, var, options);
        str.append("];\n");
      }

      // Output edges between variable and consumers.
      for (Flow::Operation *consumer : var->consumers) {
        AppendVarId(&str, var);
        str.append(" -> ");
        AppendOpId(&str, consumer);
        str.append(" [");
        str.append("tooltip=\"");
        if (consumer->inputs.size() > 1) {
          StringAppendF(&str, "%%%d = ", consumer->InputIndex(var));
        }
        str.append(var->name);
        str.append("\" ");
        AppendPenWidth(&str, var, options);
        str.append("];\n");
      }
    } else if (var->producer != nullptr) {
      // Output edges between producer and consumers.
      Flow::Operation *producer = var->producer;
      for (Flow::Operation *consumer : var->consumers) {
        AppendOpId(&str, producer);
        str.append(" -> ");
        AppendOpId(&str, consumer);
        str.append(" [");
        str.append("tooltip=\"");
        if (consumer->inputs.size() > 1) {
          StringAppendF(&str, "%%%d = ", consumer->InputIndex(var));
        }
        str.append(var->name);
        if (producer->outputs.size() > 1) {
          StringAppendF(&str, " (@%d)", producer->OutputIndex(var));
        }
        if (!var->aliases.empty()) {
          str.append("&#10;alias:");
          for (const string &alias : var->aliases) {
            str.append("&#10;  ");
            str.append(alias);
          }
        }
        str.append("\" ");
        AppendPenWidth(&str, var, options);
        str.append("];\n");
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

