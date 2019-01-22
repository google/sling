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

#ifndef SLING_TASK_FRAMES_H_
#define SLING_TASK_FRAMES_H_

#include "sling/frame/object.h"
#include "sling/task/message.h"
#include "sling/task/task.h"

namespace sling {
namespace task {

// Task processor for receiving and sending frames.
class FrameProcessor : public Processor {
 public:
  ~FrameProcessor() { delete commons_; }

  // Task processor implementation.
  void Start(Task *task) override;
  void Receive(Channel *channel, Message *message) override;
  void Done(Task *task) override;

  // Called to initialize commons store.
  virtual void InitCommons(Task *task);

  // Called to initialize frame processor.
  virtual void Startup(Task *task);

  // Called for each frame received on input.
  virtual void Process(Slice key, const Frame &frame);

  // Called when all frames have been received.
  virtual void Flush(Task *task);

  // Output object to output.
  void Output(Text key, const Object &value);

  // Output frame to output using frame id as key.
  void Output(const Frame &frame);

  // Output shallow encoding of frame to output.
  void OutputShallow(Text key, const Object &value);
  void OutputShallow(const Frame &frame);

  // Return output channel.
  Channel *output() const { return output_; }

 protected:
  // Commons store for messages.
  Store *commons_ = nullptr;

  // Name bindings.
  Names names_;

  // Output channel (optional).
  Channel *output_;

  // Statistics.
  Counter *frame_memory_;
  Counter *frame_handles_;
  Counter *frame_symbols_;
  Counter *frame_gcs_;
  Counter *frame_gctime_;
};

// Create message from object.
Message *CreateMessage(Text key, const Object &Object, bool shallow = false);

// Create message with encoded frame using frame id as key.
Message *CreateMessage(const Frame &frame, bool shallow = false);

// Decode message as frame.
Frame DecodeMessage(Store *store, Message *message);

}  // namespace task
}  // namespace sling

#endif  // SLING_TASK_FRAMES_H_

