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

#include "sling/task/frames.h"

#include "sling/base/logging.h"
#include "sling/frame/encoder.h"
#include "sling/frame/decoder.h"
#include "sling/frame/object.h"
#include "sling/frame/reader.h"
#include "sling/frame/serialization.h"
#include "sling/frame/store.h"
#include "sling/frame/wire.h"
#include "sling/stream/file.h"
#include "sling/stream/memory.h"
#include "sling/task/task.h"

namespace sling {
namespace task {

void FrameProcessor::Start(Task *task) {
  // Create commons store.
  commons_ = new Store();

  // Load commons store from file.
  for (Binding *binding : task->GetInputs("commons")) {
    LoadStore(binding->resource()->name(), commons_);
  }

  // Get output channel (optional).
  output_ = task->GetSink("output");

  // Bind names.
  InitCommons(task);
  CHECK(names_.Bind(commons_));

  // Start up processor.
  Startup(task);

  // Freeze commons store.
  commons_->Freeze();

  // Update statistics for common store.
  MemoryUsage usage;
  commons_->GetMemoryUsage(&usage, true);
  task->GetCounter("commons_memory")->Increment(usage.memory_used());
  task->GetCounter("commons_handles")->Increment(usage.used_handles());
  task->GetCounter("commons_symbols")->Increment(usage.num_symbols());
  task->GetCounter("commons_gcs")->Increment(usage.num_gcs);
  task->GetCounter("commons_gctime")->Increment(usage.gc_time);

  // Get counters for frame stores.
  frame_memory_ = task->GetCounter("frame_memory");
  frame_handles_ = task->GetCounter("frame_handles");
  frame_symbols_ = task->GetCounter("frame_symbols");
  frame_gcs_ = task->GetCounter("frame_gcs");
  frame_gctime_ = task->GetCounter("frame_gctime");
}

void FrameProcessor::Receive(Channel *channel, Message *message) {
  // Create store for frame.
  Store store(commons_);

  // Decode frame from message.
  Frame frame = DecodeMessage(&store, message);
  CHECK(frame.valid());

  // Process frame.
  Process(message->key(), frame);

  // Update statistics.
  MemoryUsage usage;
  store.GetMemoryUsage(&usage, true);
  frame_memory_->Increment(usage.memory_used());
  frame_handles_->Increment(usage.used_handles());
  frame_symbols_->Increment(usage.num_symbols());
  frame_gcs_->Increment(usage.num_gcs);
  frame_gctime_->Increment(usage.gc_time);

  // Delete input message.
  delete message;
}

void FrameProcessor::Done(Task *task) {
  // Flush output.
  Flush(task);

  // Delete commons store.
  delete commons_;
  commons_ = nullptr;
}

void FrameProcessor::Output(Text key, const Object &value) {
  CHECK(output_ != nullptr);
  output_->Send(CreateMessage(key, value));
}

void FrameProcessor::Output(const Frame &value) {
  CHECK(output_ != nullptr);
  output_->Send(CreateMessage(value));
}

void FrameProcessor::OutputShallow(Text key, const Object &value) {
  CHECK(output_ != nullptr);
  output_->Send(CreateMessage(key, value, true));
}

void FrameProcessor::OutputShallow(const Frame &value) {
  CHECK(output_ != nullptr);
  output_->Send(CreateMessage(value, true));
}

void FrameProcessor::InitCommons(Task *task) {}
void FrameProcessor::Startup(Task *task) {}
void FrameProcessor::Process(Slice key, const Frame &frame) {}
void FrameProcessor::Flush(Task *task) {}

Message *CreateMessage(Text key, const Object &object, bool shallow) {
  ArrayOutputStream stream;
  Output output(&stream);
  Encoder encoder(object.store(), &output);
  encoder.set_shallow(shallow);
  encoder.Encode(object);
  output.Flush();
  return new Message(Slice(key.data(), key.size()), stream.data());
}

Message *CreateMessage(const Frame &frame, bool shallow) {
  return CreateMessage(frame.Id(), frame, shallow);
}

Frame DecodeMessage(Store *store, Message *message) {
  ArrayInputStream stream(message->value().data(), message->value().size());
  Input input(&stream);
  if (input.Peek() == WIRE_BINARY_MARKER) {
    Decoder decoder(store, &input);
    return decoder.Decode().AsFrame();
  } else {
    Reader reader(store, &input);
    return reader.Read().AsFrame();
  }
}

}  // namespace task
}  // namespace sling

