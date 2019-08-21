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

#include "sling/nlp/parser/parser.h"
#include "sling/nlp/document/annotator.h"

namespace sling {
namespace nlp {

// Document annotator for adding semantic parse annotations to document.
class ParserAnnotator : public Annotator {
 public:
  void Init(task::Task *task, Store *commons) override {
    // Load parser model.
    string model = task->GetInputFile("parser");
    LOG(INFO) << "Loading parser model from " << model;
    parser_.Load(commons, model);
  }

  void Annotate(Document *document) override {
    // Parse document.
    parser_.Parse(document);
  }

 private:
  // Parser model.
  Parser parser_;
};

REGISTER_ANNOTATOR("parser", ParserAnnotator);

// Document annotator for adding names to frame based on first mention.
class MentionNameAnnotator : public Annotator {
 public:
  void Init(task::Task *task, Store *commons) override {
    names_.Bind(commons);
  }

  void Annotate(Document *document) override {
    Handles evoked(document->store());
    for (Span *span : document->spans()) {
      span->AllEvoked(&evoked);
      for (Handle h : evoked) {
        Frame f(document->store(), h);
        if (!f.Has(n_name_)) {
          f.Add(n_name_, span->GetText());
        }
      }
    }
  }

 private:
  Names names_;
  Name n_name_{names_, "name"};
};

REGISTER_ANNOTATOR("mention-name", MentionNameAnnotator);


}  // namespace nlp
}  // namespace sling
