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

#ifndef SLING_TASK_DOCUMENTS_H_
#define SLING_TASK_DOCUMENTS_H_

#include "sling/nlp/document/annotator.h"
#include "sling/nlp/document/document.h"
#include "sling/task/frames.h"

namespace sling {
namespace task {

// Task processor for receiving and sending documents.
class DocumentProcessor : public FrameProcessor {
 public:
  ~DocumentProcessor() { if (docnames_) docnames_->Release(); }

  void Process(Slice key, const Frame &frame) override;

  // Initialize commons store with document symbols.
  void InitCommons(Task *task) override;

  // Initialize document processor.
  void Start(Task *task) override;

  // Called for each document received on input.
  virtual void Process(Slice key, const nlp::Document &document);

  // Output document to output.
  void Output(Text key, const nlp::Document &document);

  // Output document to output using document id as key.
  void Output(const nlp::Document &document);

  // Document schema.
  const nlp::DocumentNames *docnames() const { return docnames_; }

 private:
  // Document symbol names.
  const nlp::DocumentNames *docnames_ = nullptr;

  // Document annotator pipeline for preprocessing incoming documents.
  nlp::Pipeline pipeline_;

  // Statistics.
  Counter *num_documents_;
  Counter *num_tokens_;
  Counter *num_spans_;
};

}  // namespace task
}  // namespace sling

#endif  // SLING_TASK_DOCUMENTS_H_

