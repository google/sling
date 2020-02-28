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

#ifndef SLING_TASK_TASK_H_
#define SLING_TASK_TASK_H_

#include <atomic>
#include <functional>
#include <string>
#include <unordered_map>
#include <vector>

#include "sling/base/registry.h"
#include "sling/base/types.h"
#include "sling/task/environment.h"
#include "sling/task/message.h"
#include "sling/util/asset.h"

namespace sling {
namespace task {

class Task;
class Processor;
class Stage;

// Format specifier.
class Format {
 public:
  // Create format for format specifier:
  //   <file format>
  //   <file format>/<value format>
  //   <file format>/<key format>:<value format>
  Format(const string &file, const string &key, const string &value)
      : file_(file), key_(key), value_(value) {}
  Format(const string &format);
  Format(const string &file, const string &record);

  // File format.
  const string &file() const { return file_; }
  void set_file(const string &file) { file_ = file; }

  // Record key format.
  const string &key() const { return key_; }
  void set_key(const string &key) { key_ = key; }

  // Record value format.
  const string &value() const { return value_; }
  void set_value(const string &value) { value_ = value; }

  // Returns format with new file format.
  Format As(const string &file) const {
    return Format(file, key_, value_);
  }
  Format AsMessage() const { return Format("message", key_, value_); }

  // Return format name as string.
  string ToString() const;

 private:
  // File format.
  string file_;

  // Key format.
  string key_;

  // Value format.
  string value_;
};

// A shard is one part of a multi-part resource or channel.
class Shard {
 public:
  Shard() : part_(0), total_(0) {}
  Shard(int part, int total) : part_(part), total_(total) {}

  // Whether this is a singleton.
  bool singleton() const { return total_ == 0; }

  // Shard number and total number of shards.
  int part() const { return part_; }
  int total() const { return total_; }

  // Return shard as string.
  string ToString() const;

 private:
  // Part number in multi-part data.
  int part_;

  // Total number of parts. This is zero for singletons.
  int total_;
};

// A resource is an external persistent resource, like a file or a set of
// sharded files.
class Resource {
 public:
  Resource(int id, const string &name, const Shard &shard, const Format &format)
    : id_(id), name_(name), shard_(shard), format_(format) {}

  Resource(int id, const string &name, const Format &format)
    : id_(id), name_(name), shard_(), format_(format) {}

  int id() const { return id_; }
  const string &name() const { return name_; }
  const Shard &shard() const { return shard_; }
  const Format &format() const { return format_; }

 private:
  // Resource id.
  int id_;

  // Resource name, e.g. a file name.
  string name_;

  // Resource shard.
  Shard shard_;

  // Format for resource.
  Format format_;
};

// A binding connects a resource to a task input or output.
class Binding {
 public:
  Binding(const string &name, Resource *resource)
      : name_(name), resource_(resource) {}

  const string &name() const { return name_; }
  Resource *resource() const { return resource_; }
  const string &filename() const { return resource_->name(); }

 private:
  // Input or output name.
  string name_;

  // Resource that the input or output is bound to.
  Resource *resource_;
};

// A port connects a channel to a source or sink for a task.
class Port {
 public:
  Port() {}
  Port(Task *task, const string &name, Shard shard)
      : task_(task), name_(name), shard_(shard) {}
  Port(Task *task, const string &name)
      : task_(task), name_(name) {}

  Task *task() const { return task_; }
  const string &name() const { return name_; }
  Shard shard() const { return shard_; }

  // Return port as string.
  string ToString() const;

 private:
  Task *task_;
  string name_;
  Shard shard_;
};

// A channel connects an output port of the producer task (the source) with an
// input port of the consumer task (the sink), and the channel can then be used
// for sending messages from the source to the sink.
class Channel {
 public:
  // Initialize new channel.
  Channel(int id, const Format &format) : id_(id), format_(format) {}

  // Connect producer sink port for channel input.
  void ConnectProducer(const Port &port);

  // Set consumer source port for channel output.
  void ConnectConsumer(const Port &port);

  // Return channel id.
  int id() const { return id_; }

  // Format of messages sent through the channel.
  const Format &format() const { return format_; }

  // Return channel producer and consumer ports.
  const Port &producer() const { return producer_; }
  const Port &consumer() const { return consumer_; }

  // Check whether channel is closed.
  bool closed() const { return closed_; }

  // Send message to channel consumer. The caller relinquishes ownership of
  // message.
  void Send(Message *message);

  // Close channel so no more messages can be sent on channel.
  void Close();

 private:
  // Channel id.
  int id_;

  // Format of message transmitted on channel.
  Format format_;

  // Sink port for sending messages on channel.
  Port producer_;

  // Source port for receiving messages on channel.
  Port consumer_;

  // Whether the channel is closed for transmitting messages.
  bool closed_ = false;

  // Statistics counters.
  Counter *input_shards_done_ = nullptr;
  Counter *input_messages_ = nullptr;
  Counter *input_key_bytes_ = nullptr;
  Counter *input_value_bytes_ = nullptr;
  Counter *output_shards_done_ = nullptr;
  Counter *output_messages_ = nullptr;
  Counter *output_key_bytes_ = nullptr;
  Counter *output_value_bytes_ = nullptr;
};

// A task processor reads input from input resources and messages from source
// channels and produces output resources and messages on sink channels.
class Processor : public Component<Processor> {
 public:
  virtual ~Processor() = default;

  // Initialize task. This is called before resources are attached and channels
  // are connected but after parameters have been initialized.
  virtual void Init(Task *task);

  // Attach input and output resources.
  virtual void AttachInput(Binding *resource);
  virtual void AttachOutput(Binding *resource);

  // Connect source and sink channels.
  virtual void ConnectSource(Channel *channel);
  virtual void ConnectSink(Channel *channel);

  // Start task. This is called after resources have been attached and channels
  // have been connected but before any messages are received.
  virtual void Start(Task *task);

  // Receive message on channel. This transfers ownership of the message to the
  // processor.
  virtual void Receive(Channel *channel, Message *message);

  // Notify that an input channel has been closed. This implies that no more
  // messages will be received on this channel.
  virtual void Close(Channel *channel);

  // Called when all messages from input channels have been processed. Messages
  // can still be sent on the output channels, but any remaining active output
  // channels are closed after Done() has completed.
  virtual void Done(Task *task);

  // Dynamically register task processor component.
  static void Register(const char *name, const char *clsname,
                       const char *filename, int line,
                       Factory *factory);
};

#define REGISTER_TASK_PROCESSOR(type, component) \
    REGISTER_COMPONENT_TYPE(sling::task::Processor, type, component)

// A task is a node in the job computation graph. A processor is used for
// processing the input data and producing the output data.
class Task : public AssetManager {
 public:
  // Parameter with name and value.
  struct Parameter {
    Parameter(const string &n, const string &v) : name(n), value(v) {}
    string name;
    string value;
  };

  // Create a task with the specified type of processor.
  Task(Environment *env, int id, const string &type,
       const string &name, Shard shard);

  // Create task with no processor (for annotation pipeline).
  Task(Environment *env);

  // Delete task.
  ~Task();

  // Return environment for task.
  Environment *env() const { return env_; }

  // Return task id.
  int id() const { return id_; }

  // Return task name.
  const string &name() const { return name_; }

  // Return task shard.
  const Shard &shard() const { return shard_; }

  // Return identifier string for task.
  string ToString() const;

  // Check if task is done.
  bool done() const { return done_; }

  // Get the resource binding for singleton input. Return null if not bound.
  Binding *GetInput(const string &name);
  const string &GetInputFile(const string &name);

  // Get the resource binding for singleton output. Return null if not bound.
  Binding *GetOutput(const string &name);
  const string &GetOutputFile(const string &name);

  // Get resource bindings for input.
  std::vector<Binding *> GetInputs(const string &name);
  std::vector<string> GetInputFiles(const string &name);

  // Get resource bindings for output.
  std::vector<Binding *> GetOutputs(const string &name);
  std::vector<string> GetOutputFiles(const string &name);

  // Get all inputs.
  const std::vector<Binding *> &inputs() const { return inputs_; }

  // Get all outputs.
  const std::vector<Binding *> &outputs() const { return outputs_; }

  // Get singleton inbound source channel.
  Channel *GetSource(const string &name);

  // Get singleton outbound sink channel.
  Channel *GetSink(const string &name);

  // Get inbound source channels.
  std::vector<Channel *> GetSources(const string &name);

  // Get outbound sink channels.
  std::vector<Channel *> GetSinks(const string &name);

  // Get source channels.
  const std::vector<Channel *> &sources() const { return sources_; }

  // Get sink channels.
  const std::vector<Channel *> &sinks() const { return sinks_; }

  // Get task parameters.
  const std::vector<Parameter> &parameters() const { return parameters_; }

  // Get task parameter value.
  const string &Get(const string &name, const string &defval);
  string Get(const string &name, const char *defval);
  int32 Get(const string &name, int32 defval);
  int64 Get(const string &name, int64 defval);
  double Get(const string &name, double defval);
  float Get(const string &name, float defval);
  bool Get(const string &name, bool defval);

  std::vector<string> Get(const string &name,
                          const std::vector<string> &defval);
  std::vector<int> Get(const string &name, const std::vector<int> &defval);

  // Fetch task parameter value.
  void Fetch(const string &name, string *value);
  void Fetch(const string &name, int32 *value);
  void Fetch(const string &name, int64 *value);
  void Fetch(const string &name, double *value);
  void Fetch(const string &name, float *value);
  void Fetch(const string &name, bool *value);

  void Fetch(const string &name, std::vector<string> *value);
  void Fetch(const string &name, std::vector<int> *value);

  // Add task parameter.
  void AddParameter(const string &name, const string &value);
  void AddParameter(const string &name, const char *value);
  void AddParameter(const string &name, int32 value);
  void AddParameter(const string &name, int64 value);
  void AddParameter(const string &name, double value);
  void AddParameter(const string &name, float value);
  void AddParameter(const string &name, bool value);

  // Add annotator to task.
  void AddAnnotator(const string &annotator);

  // Return list of annotators.
  const std::vector<string> &annotators() const { return annotators_; }

  // Get statistics counter.
  Counter *GetCounter(const string &name);

  // Attach input and output resources. Takes ownership of binding.
  void AttachInput(Binding *resource);
  void AttachOutput(Binding *resource);

  // Connect source and sink channels.
  void ConnectSource(Channel *channel);
  void ConnectSink(Channel *channel);

  // Notification when message for task has been received.
  void OnReceive(Channel *channel, Message *message);

  // Notification that input channel has been closed.
  void OnClose(Channel *channel);

  // State management.
  void Init();
  void Start();
  void Done();

  // Reference counting. When all references have been released the task is
  // marked as done.
  void AddRef();
  void Release();

  // Get and set stage for task.
  Stage *stage() const { return stage_; }
  void set_stage(Stage *stage) { stage_ = stage; }

 private:
  // Environment owning the task.
  Environment *env_;

  // Stage for task.
  Stage *stage_ = nullptr;

  // Task id.
  int id_;

  // Task name and shard.
  string name_;
  Shard shard_;

  // Inputs and outputs for task.
  std::vector<Binding *> inputs_;
  std::vector<Binding *> outputs_;

  // Sources and sinks for task.
  std::vector<Channel *> sources_;
  std::vector<Channel *> sinks_;

  // Task parameters.
  std::vector<Parameter> parameters_;

  // Annotators for pipeline.
  std::vector<string> annotators_;

  // Processor for executing task.
  Processor *processor_;

  // Flag to indicate that the task is done.
  std::atomic<bool> done_{false};

  // Reference count for keeping task alive.
  std::atomic<int> refs_{0};
};

}  // namespace task
}  // namespace sling

#endif  // SLING_TASK_TASK_H_

