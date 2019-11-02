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

#include <inttypes.h>
#include <math.h>
#include <algorithm>
#include <map>

#include "sling/myelin/profile.h"

#include "sling/base/types.h"
#include "sling/string/printf.h"
#include "third_party/jit/cpu.h"

namespace sling {
namespace myelin {

static const char *divider = "+---------+---------+--------------+----------"
                             "+-------------------------------"
                             "+---+------------------------\n";

static const char *header = "| percent |  accum% |      time    |   gflops |"
                            " kernel"
                            "                        | t | step\n";

static float max_giga_flops = 100000;

static string TimeStr(float us) {
  if (us < 1000000) {
    return StringPrintf("%10.3f μs", us);
  } else {
    return StringPrintf("%10.3f ms", us / 1000);
  }
}

Profile::Profile(ProfileSummary *summary, Order order)
    : cell_(summary->cell()) {
  Initialize(summary->data(), order);
}

Profile::Profile(Instance *instance, Order order) : cell_(instance->cell()) {
  if (cell_->profile() != nullptr) {
    Initialize(instance->Get<int64>(cell_->profile()), order);
  }
}

void Profile::Initialize(int64 *data, Order order) {
  // First element is evocation count followed by the overhead counter and
  // then one cycle counter for each step.
  invocations_ = data[0];
  overhead_ = data[1];
  timing_ = data + 2;
  total_ = overhead_;
  total_complexity_ = 0;
  tasks_ = reinterpret_cast<TaskTiming *>(timing_ + steps());

  steps_.resize(cell()->steps().size());
  for (int i = 0; i < steps(); ++i) {
    const Step *step = cell()->steps()[i];
    StepInfo &s = steps_[i];
    s.index = i;
    s.step = step;
    switch (order) {
      case POSITION:
        break;
      case TIME:
        s.sort_value = time(i);
        break;
      case GFLOPS:
        s.sort_value = gigaflops(i);
        break;
      case COMPLEXITY:
        s.sort_value = complexity(i);
        break;
      case KERNEL:
        s.sort_name = step->type() +  step->GetAttr("expr") + step->variant();
        break;
      case NAME:
        s.sort_name = step->name();
        break;
      case TASK:
        s.sort_value = step->task_index();
        break;
    }

    total_ += timing_[i];
    total_complexity_ += complexity(i);
  }
  sort(steps_.begin(), steps_.end());
}

string Profile::ASCIIReport() const {
  // Check if profiling has been enabled.
  if (!enabled()) return "No profile";

  // Output title.
  jit::ProcessorInformation cpu;
  string report;
  StringAppendF(&report,
      "Profile for %" PRId64 " invocations of %s with %" PRId64 " operations\n",
      invocations_,
      cell()->name().c_str(),
      complexity());
  StringAppendF(&report, "CPU model: %s\n", cpu.brand());
  StringAppendF(&report,
      "CPU architecture: %s (family %02x model %02x stepping %02x), %d cores",
      cpu.architecture(),
      cpu.family(), cpu.model(), cpu.stepping(),
      jit::CPU::Processors());
  if (jit::CPU::MemorySize() > 0) {
    StringAppendF(&report,", %" PRId64 "G RAM",
                  jit::CPU::MemorySize() / 1073741824);
  }
  if (jit::CPU::L1CacheSize() > 0) {
    StringAppendF(&report,", %dK L1", jit::CPU::L1CacheSize() / 1024);
  }
  if (jit::CPU::L2CacheSize() > 0) {
    StringAppendF(&report,", %dK L2", jit::CPU::L2CacheSize() / 1024);
  }
  if (jit::CPU::L3CacheSize() > 0) {
    StringAppendF(&report,", %dK L3", jit::CPU::L3CacheSize() / 1024);
  }
  report.append("\n");

  report.append("CPU features:");
  if (jit::CPU::Enabled(jit::MMX)) report.append(" MMX");
  if (jit::CPU::Enabled(jit::SSE)) report.append(" SSE");
  if (jit::CPU::Enabled(jit::SSE2)) report.append(" SSE2");
  if (jit::CPU::Enabled(jit::SSE3)) report.append(" SSE3");
  if (jit::CPU::Enabled(jit::SSE4_1)) report.append(" SSE4.1");
  if (jit::CPU::Enabled(jit::SSE4_2)) report.append(" SSE4.2");
  if (jit::CPU::Enabled(jit::F16C)) report.append(" F16C");
  if (jit::CPU::Enabled(jit::AVX)) report.append(" AVX");
  if (jit::CPU::Enabled(jit::AVX2)) report.append(" AVX2");
  if (jit::CPU::Enabled(jit::AVX512F)) report.append(" AVX512F");
  if (jit::CPU::Enabled(jit::FMA3)) report.append(" FMA3");
  report.append("\n");
  string runtime_info = cell()->runtime()->Description();
  if (!runtime_info.empty()) {
    report.append("Runtime: ");
    report.append(runtime_info);
    report.append("\n");
  }
  report.append("\n");

  // Output header.
  report.append(divider);
  report.append(header);
  report.append(divider);

  // Output profile for each step.
  float accum = 0;
  for (int i = 0; i < steps(); ++i) {
    if (step(i)->noop()) continue;
    string tid;
    if (step(i)->task_index() != -1) {
      tid = StringPrintf("%2d", step(i)->cell()->task(step(i)->task_index()));
    }
    string name = step(i)->kernel()->Name();
    if (!step(i)->variant().empty()) {
      name.push_back('[');
      name.append(step(i)->variant());
      name.push_back(']');
    }
    float gflops = gigaflops(i);
    if (gflops >= max_giga_flops) gflops = 0;
    accum += percent(i);
    StringAppendF(&report,
                  "| %6.2f%% | %6.2f%% |%s |%9.3f | %-30s|%-2s | %s",
                  percent(i), accum, TimeStr(time(i)).c_str(), gflops,
                  name.c_str(),
                  tid.c_str(),
                  step(i)->name().c_str());

    const string &expr = step(i)->GetAttr("expr");
    if (!expr.empty()) {
      StringAppendF(&report, " [%s]", expr.c_str());
    }
    report.push_back('\n');
  }

  // Output overhead.
  if (overhead_ > 0) {
    StringAppendF(&report,
                  "| %6.2f%% | %6.2f%% |%s |%9.3f | %-30s|%-2s | %s\n",
                  overhead_percent(),
                  100.0,
                  TimeStr(overhead_time()).c_str(), 0.0, "", "",
                  "Entry & Exit");
  }

  // Output totals.
  float gflops = gigaflops();
  if (gflops >= max_giga_flops) gflops = 0;

  report.append(divider);
  StringAppendF(&report,
                "| 100.00%% | 100.00%% |%s |%9.3f | %-30s|   |\n",
                TimeStr(time()).c_str(), gflops, "TOTAL");
  report.append(divider);

  // Output task timing.
  if (tasks() > 0) {
    double total_start = 0.0;
    double total_wait = 0.0;
    report.append("\n");
    report.append("+-------|---------------+---------------+\n");
    report.append("|  task |    start time |     wait time |\n");
    report.append("+-------|---------------+---------------+\n");
    for (int i = 0; i < tasks(); ++i) {
      total_start += start_time(i);
      total_wait += wait_time(i);
      StringAppendF(&report, "| %5d | %s | %s |\n",
                    cell()->task(i),
                    TimeStr(start_time(i)).c_str(),
                    TimeStr(wait_time(i)).c_str());
    }
    report.append("+-------|---------------+---------------+\n");
    StringAppendF(&report, "| TOTAL | %s | %s |\n",
                  TimeStr(total_start).c_str(),
                  TimeStr(total_wait).c_str());
    report.append("+-------|---------------+---------------+\n");

    double compute_time = total_start + total_wait;
    for (int i = 0; i < steps(); ++i) {
      if (step(i)->task_index() == -1) {
        compute_time += time(i);
      }
    }

    double parallelism = time() / compute_time;
    double efficiency = parallelism / (tasks() + 1);
    double rate = 1.0 / (compute_time / 1e6);
    double gflops = complexity() / compute_time / 1e3;
    StringAppendF(&report,
                  "\n%.3f μs/invocation, %.0f Hz, parallelism %.3f, "
                  "%.2f%% efficiency, %.3f GFLOPS\n",
                  compute_time,
                  rate,
                  parallelism,
                  efficiency * 100.0,
                  gflops);
  }

  return report;
}

int64 Profile::Complexity(const Step *step) {
  // Check if the kernel can compute the number of operations for step.
  int64 ops = step->complexity();

  // If the kernel does not support complexity calculation the number of
  // operations are estimated using the largest input or output.
  if (ops == -1) {
    ops = 1;
    for (auto &input : step->inputs()) {
      int size = input->elements();
      if (size > ops) ops = size;
    }
    for (auto &output : step->outputs()) {
      int size = output->elements();
      if (size > ops) ops = size;
    }
    VLOG(8) << "Estimated complexity for step " << step->name();
  }

  return ops;
}

string ProfileOverview::ASCIIReport() const {
  static const char *divider =
      "+---------+--------------+-------------+--------------+"
      "--------------------------------------------------\n";
  static const char *header =
      "| percent |    time/call | invocations |   total time | cell\n";
  string report;
  if (!cells_.empty()) {
    report.append(divider);
    report.append(header);
    report.append(divider);
    for (const CellInfo &ci : cells_) {
      double time = ci.time * ci.invocations;
      double percent = total_time_ > 0 ? time / total_time_ * 100.0 : 0.0;
      StringAppendF(&report,
                   "| %6.2f%% |%s | %11" PRId64 " |%s | %s\n",
                   percent, TimeStr(ci.time).c_str(), ci.invocations,
                   TimeStr(time).c_str(), ci.cell->name().c_str());
    }
    report.append(divider);
  }
  return report;
}

static bool CompareTensorOrder(Tensor *a, Tensor *b) {
  if (a->first() == b->first()) {
    // Inputs are sorted before outputs.
    for (auto *op : a->consumers()) {
      if (op == b->producer()) return true;
    }
    for (auto *op : b->consumers()) {
      if (op == a->producer()) return false;
    }
  }
  return a->first() < b->first();
}

static string Escape(const string &str) {
  string escaped;
  const char *p = str.data();
  const char *end = p + str.size();
  while (p < end) {
    char ch = *p++;
    switch (ch) {
      case '&':  escaped.append("&amp;"); break;
      case '<':  escaped.append("&lt;"); break;
      case '>':  escaped.append("&gt;"); break;
      case '"':  escaped.append("&quot;"); break;
      case '\'': escaped.append("&#39;");  break;
      default: escaped.push_back(ch);
    }
  }
  return escaped;
}

static string Rainbow(float value) {
  static int rainbow[7][3] = {
    {255, 0 , 0},   // red
    {255, 127, 0},  // orange
    {255, 255, 0},  // yellow
    {0, 255, 0},    // green
    {0, 0, 255},    // blue
    {75, 0, 130},   // indigo
    {148, 0, 211},  // violet
  };

  float div = value * 5.999;
  int index = static_cast<int>(div);
  float mix1 = fmodf(div, 1.0);
  float mix0 = 1.0 - mix1;

  int r = rainbow[index][0] * mix0 + rainbow[index + 1][0] * mix1;
  int g = rainbow[index][1] * mix0 + rainbow[index + 1][1] * mix1;
  int b = rainbow[index][2] * mix0 + rainbow[index + 1][2] * mix1;
  return StringPrintf("#%02x%02x%02x", r, g, b);
}

string DataProfile::AsSVG() {
  // Compute width and height.
  const float data_width = 3000;
  const float label_width = 1000;
  const float label_dx = 10;
  const float label_dy = 15;
  const float step_height = 25;
  const float width = data_width + label_width;
  const float height = step_height * cell_->steps().size();

  // Write SVG header.
  string svg;
  svg.append("<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"no\"?>\n"
             "<!DOCTYPE svg PUBLIC \"-//W3C//DTD SVG 1.1//EN\""
             " \"http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd\">\n");
  StringAppendF(&svg, "<svg width=\"%0.f\" height=\"%0.f\" "
                "xmlns=\"http://www.w3.org/2000/svg\" "
                "xmlns:xlink=\"http://www.w3.org/1999/xlink\" "
                "style=\"font-family:arial;\">\n",
                width, height);

  // Output steps.
  StringAppendF(&svg,
      "<rect x=\"0\" y=\"0\" width=\"%0.f\" height=\"%0.f\" "
      "fill=\"#EEEEEE\"/>\n",
      data_width, height);

  std::map<Step *, int> stepmap;
  for (int i = 0; i < cell_->steps().size(); ++i) {
    Step *step = cell_->steps()[i];
    stepmap[step] = i;
    StringAppendF(&svg,
        "<text x=\"%0.f\" y=\"%0.f\">%s (%s)</text>\n",
        data_width + label_dx, i * step_height + label_dy,
        step->name().c_str(), step->type().c_str());
    if (i > 0) {
      StringAppendF(&svg,
          "<line x1=\"%0.f\" y1=\"%0.f\" x2=\"%0.f\" y2=\"%0.f\" "
          "stroke-dasharray=\"2,2\" "
          "style=\"stroke:#FFFFFF\"/>\n",
          0.0, i * step_height, data_width, i * step_height);
    }
  }

  // Sort instance tensors for cell.
  std::vector<Tensor *> tensors;
  for (Tensor *t : cell_->network()->parameters()) {
    if (t->cell() == cell_) tensors.push_back(t);
  }
  sort(tensors.begin(), tensors.end(), CompareTensorOrder);

  // Output tensors with allocation and lifetime.
  float color_range = 1.0 / tensors.size();
  float byte_width = data_width / cell_->instance_size();
  for (int i = 0; i < tensors.size(); ++i) {
    Tensor *t = tensors[i];
    string color = Rainbow(i * color_range);

    Step *f = cell_->network()->steps()[t->first()];
    Step *l = cell_->network()->steps()[t->last()];
    auto ff = stepmap.find(f);
    int first = ff != stepmap.end() ? ff->second : 0;
    auto fl = stepmap.find(l);
    int last = fl != stepmap.end() ? fl->second : cell_->steps().size() - 1;

    float x1 = t->offset() * byte_width;
    float x2 = (t->offset() + t->space()) * byte_width;
    float y1 = (first + (t->in() ? 0.0 : 0.5)) * step_height;
    float y2 = (last + (t->out() ? 1.0 : 0.5)) * step_height;

    // Title and tile.
    svg.append("<g>\n");
    float tile_width = x2 - x1;
    if (tile_width > 1.0) tile_width -= 1.0;
    float tile_height = y2 - y1;
    if (tile_height > 1.0) tile_height -= 1.0;
    string type = t->TypeString();
    if (t->in()) type.append(" in");
    if (t->out()) type.append(" out");
    StringAppendF(&svg,
        "<title>%s\n%s\noffset: %lu\nsize: %lu\nalign: %d</title>\n"
        "<rect x=\"%f\" y=\"%f\" width=\"%f\" height=\"%f\" "
        "fill=\"%s\" stroke=\"%s\"/>\n",
        Escape(t->name()).c_str(), Escape(type).c_str(),
        t->offset(), t->space(), t->byte_alignment(),
        x1, y1, tile_width, tile_height, color.c_str(), color.c_str());

    // Upper and lower tile shadow.
    if (tile_width > 3) {
      float border_width = x2 - x1 > 5 ? 2 : 1;
      StringAppendF(&svg,
          "<path d=\"M%f %f H%f L%f %f H%f V%f L%f %f Z\" "
          "fill=\"#FFFFFF\" opacity=\"0.5\"/>\n",
          x1 - 0.5, y1 - 0.5,
          x2 - 0.5,
          x2 - border_width - 0.5, y1 + border_width - 0.5,
          x1 + border_width - 0.5,
          y2 - border_width - 0.5,
          x1 - 0.5, y2 - 0.5);
      StringAppendF(&svg,
          "<path d=\"M%f %f V%f H%f L%f %f H%f V%fZ\" "
          "fill=\"#000000\" opacity=\"0.5\"/>\n",
          x2 - 0.5, y1 - 0.5,
          y2 - 0.5,
          x1 - 0.5,
          x1 + border_width - 0.5, y2 - border_width - 0.5,
          x2 - border_width - 0.5,
          y1 + border_width - 0.5);
    }
    svg.append("</g>\n");
  }

  // Write footer.
  svg.append("</svg>\n");
  return svg;
}

void LogProfile(const Network &net) {
  if (net.options().global_profiler) {
    LOG(INFO) << "Profiling report:\n" << ProfileReport(net);
  }
}

string ProfileReport(const Network &net) {
  string report;
  if (net.options().global_profiler) {
    ProfileOverview overview;
    for (const Cell *cell : net.cells()) {
      Profile profile(cell->profile_summary());
      report.append(profile.ASCIIReport());
      report.append("\n");
      overview.Add(profile);
    }
    report.append("Summary:\n");
    report.append(overview.ASCIIReport());
    report.append("\n");
  }
  return report;
}

}  // namespace myelin
}  // namespace sling

