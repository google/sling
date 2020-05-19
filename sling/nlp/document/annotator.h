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

#ifndef SLING_NLP_DOCUMENT_ANNOTATOR_H_
#define SLING_NLP_DOCUMENT_ANNOTATOR_H_

#include <string>
#include <vector>

#include "sling/base/registry.h"
#include "sling/base/types.h"
#include "sling/nlp/document/document.h"
#include "sling/task/task.h"

namespace sling {
namespace nlp {

// Document annotation component interface.
class Annotator : public Component<Annotator> {
 public:
  virtual ~Annotator() = default;

  // Initialize document annotator.
  virtual void Init(task::Task *task, Store *commons);

  // Annotate document.
  virtual void Annotate(Document *document) = 0;
};

#define REGISTER_ANNOTATOR(type, component) \
    REGISTER_COMPONENT_TYPE(sling::nlp::Annotator, type, component)

// Document annotation pipeline.
class Pipeline {
 public:
  ~Pipeline();

  // Initialize document annotation pipeline.
  void Init(task::Task *task, Store *commons);

  // Annotate document.
  void Annotate(Document *document);

  // Check for no-op pipeline.
  bool empty() const { return annotators_.empty(); }

 private:
  // Document annotators.
  std::vector<Annotator *> annotators_;
};

class DocumentAnnotation : public task::Environment {
 public:
  DocumentAnnotation();
  ~DocumentAnnotation();

  // Initialize document annotation pipeline from task specification.
  void Init(Store *commons, const Frame &config);
  void Init(Store *commons, const string &spec);

  // Annotate document.
  void Annotate(Document *document);

  // Environment interface.
  task::Counter *GetCounter(const string &name) override;
  void ChannelCompleted(task::Channel *channel) override;
  void TaskCompleted(task::Task *task) override;

 private:
  // Initialize task from configuration frame.
  void InitTaskFromConfig(const Frame &config);

  // Pipeline for annotating documents.
  Pipeline pipeline_;

  // Task specification for document annotation pipeline.
  task::Task task_;
  std::vector<task::Resource *> resources_;

  // Dummy counter.
  task::Counter dummy_;
};

}  // namespace nlp
}  // namespace sling

#endif  // SLING_NLP_DOCUMENT_ANNOTATOR_H_

