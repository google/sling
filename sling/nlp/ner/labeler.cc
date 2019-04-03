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

#include "sling/nlp/ner/annotators.h"
#include "sling/task/documents.h"

namespace sling {
namespace nlp {

using namespace task;

// Run NER annotators on documents and resolve entity mentions.
class DocumentNERLabeler : public DocumentProcessor {
 public:
  void Startup(Task *task) override {
    // Initialize span annotator.
    SpanAnnotator::Resources resources;
    resources.aliases = task->GetInputFile("aliases");
    resources.dictionary = task->GetInputFile("dictionary");
    resources.resolve = task->Get("resolve", false);
    annotator_.Init(commons_, resources);

    // Add stop words.
    // TODO: make this configurable.
    std::vector<string> stop_words = {
      ".", ",", "-", ":", ";", "(", ")", "``", "''", "'", "--", "/", "&", "?",
      "the", "a", "an", "'s", "is", "was", "and",
      "in", "of", "by", "to", "at", "as",
    };
    annotator_.AddStopWords(stop_words);
  }

  void Process(Slice key, const Document &document) override {
    // Create unannotated output document.
    Document output(document);
    output.ClearAnnotations();

    // Annotate document.
    annotator_.Annotate(document, &output);

    // Output annotated document.
    Output(key, output);
  }

 private:
  // Span annotator for labeling documents.
  SpanAnnotator annotator_;
};

REGISTER_TASK_PROCESSOR("document-ner-labeler", DocumentNERLabeler);

}  // namespace nlp
}  // namespace sling

