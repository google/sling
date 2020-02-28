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

#include <string>
#include <vector>

#include "sling/nlp/document/annotator.h"
#include "sling/nlp/document/document.h"
#include "sling/nlp/kb/facts.h"

namespace sling {
namespace nlp {

using namespace task;

// Annotate entity types for resolved frames.
class TypeAnnotator : public Annotator {
 public:
  ~TypeAnnotator() { delete taxonomy_; }

  void Init(Task *task, Store *commons) override {
    catalog_.Init(commons);
    taxonomy_ = catalog_.CreateEntityTaxonomy();
  }

  // Annotate types for all evoked frames in document.
  void Annotate(Document *document) override {
    Store *store = document->store();
    Handles evoked(store);
    for (Span *span : document->spans()) {
      span->AllEvoked(&evoked);
      for (Handle h : evoked) {
        Handle resolved = store->Resolve(h);
        if (resolved == h) continue;
        if (!store->IsFrame(resolved)) continue;

        Frame f(store, resolved);
        Handle type = taxonomy_->Classify(f);
        if (type.IsNil()) continue;

        Builder(store, h).AddIsA(type).Update();
      }
    }
  }

 private:
  // Fact catalog for fact extraction.
  FactCatalog catalog_;

  // Entity type taxonomy.
  Taxonomy *taxonomy_ = nullptr;
};

REGISTER_ANNOTATOR("types", TypeAnnotator);

// Document annotator for deleting references to other frames (i.e. is: slots).
class ClearReferencesAnnotator : public Annotator {
 public:
  void Init(task::Task *task, Store *commons) override {
    names_.Bind(commons);
  }

  void Annotate(Document *document) override {
    Store *store = document->store();
    Handles evoked(store);
    for (Span *span : document->spans()) {
      span->AllEvoked(&evoked);
      for (Handle h : evoked) {
        Frame f(store, h);
        if (f.Has(Handle::is())) {
          // Delete all is: slots.
          Builder(f).Delete(Handle::is()).Update();
        }
      }
    }
  }

 private:
  Names names_;
  Name n_name_{names_, "name"};
};

REGISTER_ANNOTATOR("clear-references", ClearReferencesAnnotator);

}  // namespace nlp
}  // namespace sling

