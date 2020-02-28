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
#include "sling/nlp/document/document.h"

namespace sling {
namespace nlp {

using namespace task;

// Document annotator for pruning nominals that are not targets of other frames.
class NominalPruningAnnotator : public Annotator {
 public:
  void Init(Task *task, Store *commons) override {
    task->Fetch("keep_typed_nominals", &keep_typed_nominals_);
  }

  void Annotate(Document *document) override {
    // Collect all the local targets for slots in evoked frames.
    Store *store = document->store();
    Handles evoked(store);
    HandleSet targets;
    for (Span *span : document->spans()) {
      if (IsNominal(span)) continue;
      span->AllEvoked(&evoked);
      for (Handle h : evoked) {
        for (const Slot &s : Frame(store, h)) {
          if (s.value.IsLocalRef()) targets.insert(s.value);
        }
      }
    }

    // Delete all nominal mentions that are not targets.
    for (Span *span : document->spans()) {
      if (!IsNominal(span)) continue;
      span->AllEvoked(&evoked);
      bool is_target = false;
      for (Handle h : evoked) {
        if (targets.count(h) > 0) is_target = true;
      }
      if (!is_target) document->DeleteSpan(span);
    }
  }

  bool IsNominal(Span *span) const {
    CaseForm form = span->Form();
    if (keep_typed_nominals_) {
      Frame frame = span->Evoked();
      if (!frame.valid()) return false;
      if (frame.Has(Handle::isa())) return false;
      if (span->initial() && form == CASE_NONE) return true;
    }
    return form == CASE_LOWER;
  }

 private:
  bool keep_typed_nominals_ = true;
};

REGISTER_ANNOTATOR("prune-nominals", NominalPruningAnnotator);

}  // namespace nlp
}  // namespace sling

