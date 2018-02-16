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

#include "sling/task/documents.h"

namespace sling {
namespace task {

void DocumentProcessor::InitCommons(Task *task) {
  // Bind document names.
  docnames_ = new nlp::DocumentNames(commons_);
}

void DocumentProcessor::Start(Task *task) {
  // Initialize frame processor.
  FrameProcessor::Start(task);

  // Statistics.
  num_document_ = task->GetCounter("num_documents");
  num_tokens_ = task->GetCounter("num_tokens");
  num_spans_ = task->GetCounter("num_spans");
}

void DocumentProcessor::Process(Slice key, const Frame &frame) {
  // Create document from frame.
  nlp::Document document(frame, docnames_);
  num_document_->Increment();
  num_tokens_->Increment(document.num_tokens());
  num_spans_->Increment(document.num_spans());

  // Process document.
  Process(key, document);
}

void DocumentProcessor::Process(Slice key, const nlp::Document &document) {}

void DocumentProcessor::Output(Text key, const nlp::Document &document) {
  FrameProcessor::Output(key, document.top());
}

void DocumentProcessor::Output(const nlp::Document &document) {
  FrameProcessor::Output(document.top());
}

}  // namespace task
}  // namespace sling

