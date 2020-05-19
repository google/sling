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

#include "sling/nlp/document/annotator.h"

#include "sling/base/logging.h"
#include "sling/file/file.h"
#include "sling/frame/object.h"
#include "sling/frame/serialization.h"

REGISTER_COMPONENT_REGISTRY("document annotator", sling::nlp::Annotator);

namespace sling {
namespace nlp {

using namespace sling::task;

void Annotator::Init(Task *task, Store *commons) {}

Pipeline::~Pipeline() {
  for (Annotator *a : annotators_) delete a;
}

void Pipeline::Init(Task *task, Store *commons) {
  // Create annotators.
  for (const string &type : task->annotators()) {
    annotators_.push_back(Annotator::Create(type));
  }

  // Initialize annotators.
  for (Annotator *a : annotators_) a->Init(task, commons);
}

void Pipeline::Annotate(Document *document) {
  for (Annotator *a : annotators_) a->Annotate(document);
}

DocumentAnnotation::DocumentAnnotation() : task_(this) {}

DocumentAnnotation::~DocumentAnnotation() {
  for (auto *r : resources_) delete r;
}

void DocumentAnnotation::Init(Store *commons, const Frame &config) {
  // Initialize pipeline annotation task from configuration frame.
  InitTaskFromConfig(config);

  // Load commons store from file.
  for (Binding *binding : task_.GetInputs("commons")) {
    LoadStore(binding->resource()->name(), commons);
  }

  // Initialize annotators.
  pipeline_.Init(&task_, commons);
}

void DocumentAnnotation::Init(Store *commons, const string &spec) {
  // Parse specification.
  Store store;
  Frame config = Frame::nil();
  if (!spec.empty()) {
    Object obj = FromText(&store, spec);
    CHECK(obj.IsFrame());
    config = obj.AsFrame();
  }

  // Initialize pipeline annotation task from specification.
  Init(commons, config);
}

void DocumentAnnotation::Annotate(Document *document) {
  if (!pipeline_.empty()) {
    pipeline_.Annotate(document);
    document->Update();
  }
}

Counter *DocumentAnnotation::GetCounter(const string &name) { return &dummy_; }
void DocumentAnnotation::ChannelCompleted(Channel *channel) {}
void DocumentAnnotation::TaskCompleted(Task *task) {}

void DocumentAnnotation::InitTaskFromConfig(const Frame &config) {
  // Ignore empty configuration.
  if (!config.valid()) return;

  // Set up task specification from configuration frame.
  Store *store = config.store();
  Handle n_annotator = store->Lookup("annotator");
  Handle n_inputs = store->Lookup("inputs");
  Handle n_file = store->Lookup("file");
  Handle n_format = store->Lookup("format");
  Handle n_parameters = store->Lookup("parameters");

  int rid = 0;
  for (const Slot &s : config) {
    if (s.name == n_annotator) {
      // Add annotator.
      String annotator(store, s.value);
      task_.AddAnnotator(annotator.value());
    } else if (s.name == n_inputs) {
      // Add inputs.
      Frame inputs(store, s.value);
      for (const Slot &si : inputs) {
        string name = Frame(store, si.name).Id().str();
        Frame input(store, si.value);
        string pattern = input.GetString(n_file);
        std::vector<string> files;
        File::Match(pattern, &files);
        Format format(input.GetString(n_format));
        if (files.empty()) {
          Resource *r = new Resource(rid++, pattern,  Shard(), format);
          resources_.push_back(r);
          task_.AttachInput(new Binding(name, r));
        } else {
          int parts = files.size();
          for (int i = 0; i < parts; ++i) {
            Shard shard =  parts == 1 ? Shard(i, parts) : Shard();
            Resource *r = new Resource(rid++, files[i],  shard, format);
            resources_.push_back(r);
            task_.AttachInput(new Binding(name, r));
          }
        }
      }
    } else if (s.name == n_parameters) {
      // Add parameters.
      Frame parameters(store, s.value);
      for (const Slot &sp : parameters) {
        string name = Frame(store, sp.name).Id().str();
        Object value(store, sp.value);
        if (value.IsString()) {
          task_.AddParameter(name, value.AsString().value());
        } else if (value.IsInt()) {
          task_.AddParameter(name, value.AsInt());
        } else if (value.IsFloat()) {
          task_.AddParameter(name, value.AsFloat());
        } else {
          LOG(WARNING) << "Unknown value type for parameter: " << name;
        }
      }
    }
  }
}

}  // namespace nlp
}  // namespace sling

