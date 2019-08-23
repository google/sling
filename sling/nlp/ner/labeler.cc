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
#include "sling/nlp/ner/annotators.h"

namespace sling {
namespace nlp {

using namespace task;

// Add NER annotations to documents and resolve entity mentions.
class NamedEntityAnnotator : public Annotator {
 public:
  void Init(Task *task, Store *commons) override {
    // Initialize span annotator.
    SpanAnnotator::Resources resources;
    resources.aliases = task->GetInputFile("aliases");
    resources.dictionary = task->GetInputFile("dictionary");
    resources.resolve = task->Get("resolve", false);
    resources.language = task->Get("language", "en");

    annotator_.Init(commons, resources);
  }

  void Annotate(Document *document) override {
    // Make a copy of the input document and clear annotations for output.
    Document original(*document);
    document->ClearAnnotations();

    // Annotate document.
    annotator_.Annotate(original, document);
  }

 private:
  // Span annotator for labeling documents.
  SpanAnnotator annotator_;
};

REGISTER_ANNOTATOR("ner", NamedEntityAnnotator);

}  // namespace nlp
}  // namespace sling

