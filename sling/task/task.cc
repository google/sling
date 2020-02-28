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

#include "sling/task/task.h"

#include "sling/base/logging.h"
#include "sling/string/numbers.h"

REGISTER_COMPONENT_REGISTRY("task processor", sling::task::Processor);

namespace sling {
namespace task {

string Shard::ToString() const {
  string str;
  if (!singleton()) {
    str.append("[");
    str.append(SimpleItoa(part()));
    str.append("/");
    str.append(SimpleItoa(total()));
    str.append("]");
  }
  return str;
}

Format::Format(const string &format) {
  size_t slash = format.find('/');
  if (slash != -1) {
    file_ = format.substr(0, slash);
    size_t colon = format.find(':', slash + 1);
    if (colon != -1) {
      key_ = format.substr(slash + 1, colon - slash + 1);
      value_ = format.substr(colon + 1);
    } else {
      value_ = format.substr(slash + 1);
    }
  } else {
    file_ = format;
  }
}

Format::Format(const string &file, const string &record) {
  file_ = file;
  size_t colon = record.find(':');
  if (colon != -1) {
    key_ = record.substr(0, colon);
    value_ = record.substr(colon + 1);
  } else {
    value_ = record;
  }
}

string Format::ToString() const {
  string fmt = file_;
  if (!key_.empty() || !value_.empty()) {
    fmt.append("/");
    if (!key_.empty()) {
      fmt.append(key_);
      fmt.append(":");
    }
    fmt.append(value_);
  }
  return fmt;
}

string Port::ToString() const {
  string str;
  str.append(task_->name());
  str.append(task_->shard().ToString());
  str.append(".");
  str.append(name_);
  str.append(shard_.ToString());
  return str;
}

void Channel::ConnectProducer(const Port &port) {
  // Get producer task.
  Task *task = port.task();

  // Store channel producer info.
  producer_ = port;

  // Connect channel to sink in producer.
  task->ConnectSink(this);

  // Create input counters.
  string arg = task->name() + "." + port.name();
  task->GetCounter("output_shards[" + arg + "]")->Increment();
  input_shards_done_ = task->GetCounter("output_shards_done[" + arg + "]");
  input_messages_ = task->GetCounter("output_messages[" + arg + "]");
  input_key_bytes_ = task->GetCounter("output_key_bytes[" + arg + "]");
  input_value_bytes_ = task->GetCounter("output_value_bytes[" + arg + "]");
}

void Channel::ConnectConsumer(const Port &port) {
  // Get consumer task.
  Task *task = port.task();

  // Store channel consumer info.
  consumer_ = port;

  // Connect channel to source in consumer.
  task->ConnectSource(this);

  // Create output counters.
  string arg = task->name() + "." + port.name();
  task->GetCounter("input_shards[" + arg + "]")->Increment();
  output_shards_done_ = task->GetCounter("input_shards_done[" + arg + "]");
  output_messages_ = task->GetCounter("input_messages[" + arg + "]");
  output_key_bytes_ = task->GetCounter("input_key_bytes[" + arg + "]");
  output_value_bytes_ = task->GetCounter("input_value_bytes[" + arg + "]");
}

void Channel::Send(Message *message) {
  // Messages cannot be sent after channel has been closed.
  CHECK(!closed_);

  // Update statistics.
  size_t keylen = message->key().size();
  size_t vallen = message->value().size();
  input_messages_->Increment();
  output_messages_->Increment();
  input_key_bytes_->Increment(keylen);
  output_key_bytes_->Increment(keylen);
  input_value_bytes_->Increment(vallen);
  output_value_bytes_->Increment(vallen);

  // Send message to consumer.
  consumer_.task()->OnReceive(this, message);
}

void Channel::Close() {
  // Mark channel as closed.
  CHECK(!closed_);
  closed_ = true;

  // Update statistics.
  output_shards_done_->Increment();
  input_shards_done_->Increment();

  // Notify container.
  consumer_.task()->env()->ChannelCompleted(this);
}

void Processor::Init(Task *task) {
}

void Processor::AttachInput(Binding *resource) {
}

void Processor::AttachOutput(Binding *resource) {
}

void Processor::ConnectSource(Channel *channel) {
}

void Processor::ConnectSink(Channel *channel) {
}

void Processor::Start(Task *task) {
}

void Processor::Receive(Channel *channel, Message *message) {
  delete message;
}

void Processor::Close(Channel *channel) {
}

void Processor::Done(Task *task) {
}

Task::Task(Environment *env, int id, const string &type,
           const string &name, Shard shard)
    : env_(env), id_(id), name_(name), shard_(shard) {
  processor_ = Processor::Create(type);
}

Task::Task(Environment *env) : env_(env), id_(0), processor_(nullptr) {}

Task::~Task() {
  delete processor_;
  for (auto b : inputs_) delete b;
  for (auto b : outputs_) delete b;
}

string Task::ToString() const {
  string str;
  str.append(name_);
  str.append(shard_.ToString());
  return str;
}

Binding *Task::GetInput(const string &name) {
  for (auto *input : inputs_) {
    if (input->name() == name) return input;
  }
  return nullptr;
}

const string &Task::GetInputFile(const string &name) {
  Binding *input = GetInput(name);
  if (input == nullptr) {
    LOG(FATAL) << "Input " << name << " is missing for task " << ToString();
  }
  return input->resource()->name();
}

Binding *Task::GetOutput(const string &name) {
  for (auto *output : outputs_) {
    if (output->name() == name) return output;
  }
  return nullptr;
}

const string &Task::GetOutputFile(const string &name) {
  Binding *output = GetOutput(name);
  if (output == nullptr) {
    LOG(FATAL) << "Output " << name << " is missing for task " << ToString();
  }
  return output->resource()->name();
}

std::vector<Binding *> Task::GetInputs(const string &name) {
  std::vector<Binding *> bindings;
  for (auto *input : inputs_) {
    if (input->name() == name) bindings.push_back(input);
  }
  return bindings;
}

std::vector<string> Task::GetInputFiles(const string &name) {
  std::vector<string> files;
  for (auto *input : inputs_) {
    if (input->name() == name) files.push_back(input->resource()->name());
  }
  return files;
}

std::vector<Binding *> Task::GetOutputs(const string &name) {
  std::vector<Binding *> bindings;
  for (auto *output : outputs_) {
    if (output->name() == name) bindings.push_back(output);
  }
  return bindings;
}

std::vector<string> Task::GetOutputFiles(const string &name) {
  std::vector<string> files;
  for (auto *output : outputs_) {
    if (output->name() == name) files.push_back(output->resource()->name());
  }
  return files;
}

Channel *Task::GetSource(const string &name) {
  for (auto *source : sources_) {
    if (source->consumer().name() == name) return source;
  }
  return nullptr;
}

Channel *Task::GetSink(const string &name) {
  for (auto *sink : sinks_) {
    if (sink->producer().name() == name) return sink;
  }
  return nullptr;
}

std::vector<Channel *> Task::GetSources(const string &name) {
  std::vector<Channel *> channels;
  for (auto *source : sources_) {
    if (source->consumer().name() == name) channels.push_back(source);
  }
  return channels;
}

std::vector<Channel *> Task::GetSinks(const string &name) {
  std::vector<Channel *> channels;
  for (auto *sink : sinks_) {
    if (sink->producer().name() == name) channels.push_back(sink);
  }
  return channels;
}

const string &Task::Get(const string &name, const string &defval) {
  for (const auto &p : parameters_) {
    if (p.name == name) return p.value;
  }
  return defval;
}

string Task::Get(const string &name, const char *defval) {
  for (const auto &p : parameters_) {
    if (p.name == name) return p.value;
  }
  return defval;
}

int32 Task::Get(const string &name, int32 defval) {
  static const string empty("");
  const string &value = Get(name, empty);
  if (value.empty()) return defval;
  int32 v;
  CHECK(safe_strto32(value.c_str(), &v)) << value;
  return v;
}

int64 Task::Get(const string &name, int64 defval) {
  static const string empty("");
  const string &value = Get(name, empty);
  if (value.empty()) return defval;
  int64 v;
  CHECK(safe_strto64(value.c_str(), &v)) << value;
  return v;
}

float Task::Get(const string &name, float defval) {
  static const string empty = string("");
  const string &value = Get(name, empty);
  if (value.empty()) return defval;
  float v;
  CHECK(safe_strtof(value.c_str(), &v)) << value;
  return v;
}

double Task::Get(const string &name, double defval) {
  static const string empty = string("");
  const string &value = Get(name, empty);
  if (value.empty()) return defval;
  double v;
  CHECK(safe_strtod(value.c_str(), &v)) << value;
  return v;
}

bool Task::Get(const string &name, bool defval) {
  static const string empty = string("");
  const string &value = Get(name, empty);
  if (value.empty()) return defval;
  return value == "true" || value == "1";
}

std::vector<string> Task::Get(const string &name,
                              const std::vector<string> &defval) {
  static const string empty = string("");
  const string &str = Get(name, empty);
  if (str.empty()) return defval;

  const char *p = str.c_str();
  if (*p == 0) return defval;
  std::vector<string> values;
  if (*p == '[') p++;
  while (*p == ' ') p++;
  while (*p != 0 && *p != ']') {
    while (*p == ' ') p++;
    if (!values.empty() && *p++ != ',') return defval;
    while (*p == ' ') p++;
    string value;
    while (*p != 0 && *p != ',' && *p != ']') value.push_back(*p++);
    values.push_back(value);
  }
  if (*p == ']') p++;
  if (*p != 0) return defval;
  return values;
}


std::vector<int> Task::Get(const string &name, const std::vector<int> &defval) {
  static const std::vector<string> empty;
  const std::vector<string> values = Get(name, empty);
  if (values.empty()) return defval;

  std::vector<int> list;
  for (const string &value : values) {
    int v;
    CHECK(safe_strto32(value.c_str(), &v)) << value;
    list.push_back(v);
  }
  return list;
}

void Task::Fetch(const string &name, string *value) {
  *value = Get(name, *value);
}

void Task::Fetch(const string &name, int32 *value) {
  *value = Get(name, *value);
}

void Task::Fetch(const string &name, int64 *value) {
  *value = Get(name, *value);
}

void Task::Fetch(const string &name, float *value) {
  *value = Get(name, *value);
}

void Task::Fetch(const string &name, double *value) {
  *value = Get(name, *value);
}

void Task::Fetch(const string &name, bool *value) {
  *value = Get(name, *value);
}

void Task::Fetch(const string &name, std::vector<string> *value) {
  *value = Get(name, *value);
}

void Task::Fetch(const string &name, std::vector<int> *value) {
  *value = Get(name, *value);
}

void Task::AddParameter(const string &name, const string &value) {
  parameters_.emplace_back(name, value);
}

void Task::AddParameter(const string &name, const char *value) {
  AddParameter(name, string(value));
}

void Task::AddParameter(const string &name, int32 value) {
  AddParameter(name, SimpleItoa(value));
}

void Task::AddParameter(const string &name, int64 value) {
  AddParameter(name, SimpleItoa(value));
}

void Task::AddParameter(const string &name, double value) {
  AddParameter(name, SimpleDtoa(value));
}

void Task::AddParameter(const string &name, float value) {
  AddParameter(name, SimpleFtoa(value));
}

void Task::AddParameter(const string &name, bool value) {
  AddParameter(name, value ? "true" : "false");
}

void Task::AddAnnotator(const string &annotator) {
  annotators_.push_back(annotator);
}

Counter *Task::GetCounter(const string &name) {
  return env_->GetCounter(name);
}

void Task::AttachInput(Binding *resource) {
  inputs_.push_back(resource);
  if (processor_ != nullptr) processor_->AttachInput(resource);
}

void Task::AttachOutput(Binding *resource) {
  outputs_.push_back(resource);
  if (processor_ != nullptr) processor_->AttachOutput(resource);
}

void Task::ConnectSource(Channel *channel) {
  // Add source channel.
  sources_.push_back(channel);
  if (processor_ != nullptr) processor_->ConnectSource(channel);

  // Add reference count for input channel.
  AddRef();
}

void Task::ConnectSink(Channel *channel) {
  sinks_.push_back(channel);
  if (processor_ != nullptr) processor_->ConnectSink(channel);
}

void Task::OnReceive(Channel *channel, Message *message) {
  // Send message to processor.
  if (processor_ != nullptr) {
    AddRef();
    processor_->Receive(channel, message);
    Release();
  }
}

void Task::OnClose(Channel *channel) {
  // Notify processor.
  if (processor_ != nullptr) processor_->Close(channel);

  // Release reference count for channel.
  Release();
}

void Task::Init() {
  if (processor_ != nullptr) processor_->Init(this);
}

void Task::Start() {
  if (processor_ != nullptr) {
    AddRef();
    processor_->Start(this);
    Release();
  }
}

void Task::Done() {
  // Check if task has already been marked as done.
  bool expected = false;
  if (done_.compare_exchange_strong(expected, true)) {
    // Notify processor.
    if (processor_ != nullptr) processor_->Done(this);

    // The processor is no longer needed after it is done, so the processor is
    // deleted here to free up resources.
    delete processor_;
    processor_ = nullptr;

    // Close any remaining output channels.
    for (Channel *channel : sinks_) {
      if (!channel->closed()) channel->Close();
    }

    // Dispose of shared assets managed by task.
    DisposeAssets();
  }
}

void Task::AddRef() {
  refs_.fetch_add(1);
}

void Task::Release() {
  if (refs_.fetch_sub(1) == 1) {
    // Notify container that task has completed.
    env_->TaskCompleted(this);
  }
}

void Processor::Register(const char *name, const char *clsname,
                         const char *filename, int line,
                         Factory *factory) {
  new Registry::Registrar(registry(), name, clsname, filename, line, factory);
}

}  // namespace task
}  // namespace sling

