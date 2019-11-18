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

// Annotate relations between resolved mentions.
class RelationAnnotator : public Annotator {
 public:
  void Init(Task *task, Store *commons) override {
    // Initialize fact extractor.
    catalog_.Init(commons);

    // Set up property priorities.
    std::vector<const char *> priority_order = {
      "P27",    // country of citizenship
      "P17",    // country
      "P19",    // place of birth
      "P20",    // place of death
      "P119",   // place of burial
    };
    for (int i = 0; i < priority_order.size(); ++i) {
      Handle property = commons->Lookup(priority_order[i]);
      priority_[property] = priority_order.size() - i;
    }

    std::vector<const char *> blocked_properties = {
      "P461",   // opposite of
      "P205",   // basin country
      "P530",   // diplomatic relation
      "P47",    // shares border with
      "P206",   // located in or next to body of water
      "P706",   // located on terrain feature
      "P1589",  // lowest point
      "P1376",  // capital of
      "P150",   // contains administrative territorial entity
      "P190",   // twinned administrative body
    };
    for (const char *prop : blocked_properties) {
      Handle property = commons->Lookup(prop);
      priority_[property] = -1;
    }
  }

  // Annotate relations in document.
  void Annotate(Document *document) override {
    // Process each sentence separately so we do not annotate relations between
    // mentions in different sentences.
    Store *store = document->store();
    for (SentenceIterator s(document); s.more(); s.next()) {
      // Find all resolved spans in sentence.
      std::vector<Mention> mentions;
      HandleSet targets;
      for (int t = s.begin(); t < s.end(); ++t) {
        // Get spans starting on this token.
        for (Span *span = document->GetSpanAt(t); span; span = span->parent()) {
          // Discard spans  we already have.
          bool existing = false;
          for (Mention &m : mentions) {
            if (m.span == span) {
              existing = true;
              break;
            }
          }
          if (existing) continue;

          // TEST: only add top-level mentions.
          if (span->parent() != nullptr) continue;

          // Add new mention.
          Mention mention;
          mention.span = span;
          mention.outer = span;
          while (mention.outer->parent() != nullptr) {
            mention.outer = mention.outer->parent();
          }
          mention.frame = span->evoked();
          if (mention.frame.IsNil()) continue;
          mention.item = store->Resolve(mention.frame);
          mentions.emplace_back(mention);
          targets.insert(mention.item);
        }
      }

      // Find facts for each mention that match a target in the sentence.
      for (Mention &source : mentions) {
        // Only consider top-level subjects for now.
        if (source.span != source.outer) continue;

        // Get facts for mention.
        if (!source.item.IsGlobalRef()) continue;
        Facts facts(&catalog_);
        facts.set_numeric_dates(true);
        facts.Extract(source.item);

        // Try to find mentions of the fact targets.
        int matches = 0;
        for (int i = 0; i < facts.size(); ++i) {
          // Only search for simple facts for now.
          if (!facts.simple(i)) continue;

          // Check if the fact target is mentioned in sentence.
          Handle value = facts.last(i);
          if (targets.count(value) == 0) continue;

          // Find closest mention of fact target.
          Mention *target = nullptr;
          for (Mention &t : mentions) {
            if (t.item != value) continue;

            // Source and target should not be in the same top-level span. These
            // relations are handled by the phrase annotator.
            if (t.outer == source.outer) continue;

            // Select target with the smallest distance to the source mention.
            if (target == nullptr) {
              target = &t;
            } else {
              int current_distance = Distance(source.span, target->span);
              int new_distance = Distance(source.span, t.span);
              if (new_distance < current_distance) {
                target = &t;
              }
            }
          }
          if (target == nullptr) continue;

          // Ignore self-relations.
          if (target->item == source.item) continue;

          // Skip match if property is blocked.
          Handle property = facts.first(i);
          int priority = Priority(property);
          if (priority < 0) continue;

          // Clear property matches on first match.
          if (matches++ == 0) {
            for (Mention &m : mentions) m.property = Handle::nil();
          }

          // Check if this is the best match for target.
          if (target->property.IsNil() ||
              priority > Priority(target->property)) {
            target->property = property;
          }
        }

        // Add relations to source mention.
        if (matches > 0) {
          Frame existing(store, source.frame);
          if (existing.IsPublic()) {
            Builder b(store);
            b.AddIs(existing);
            for (const Mention &m : mentions) {
              if (!m.property.IsNil()) {
                b.Add(m.property, m.frame);
              }
            }
            source.span->Replace(existing, b.Create());
            source.frame = b.handle();
          } else {
            Builder b(existing);
            for (const Mention &m : mentions) {
              if (!m.property.IsNil()) {
                b.Add(m.property, m.frame);
              }
            }
            b.Update();
          }
        }
      }
    }
  }

 private:
  // Compute distance between two spans. It is assumed that the spans are not
  // overlapping.
  static int Distance(const Span *s1, const Span *s2) {
    if (s1->begin() < s2->begin()) {
      return s2->begin() - s1->end();
    } else {
      return s1->begin() - s2->end();
    }
  }

  // Get priority for property.
  int Priority(Handle property) const {
    auto f = priority_.find(property);
    return f != priority_.end() ? f->second : 0;
  }

  // Entity mention in sentence.
  struct Mention {
    Handle frame;     // frame annotations for entity
    Handle item;      // item describing entity
    Span *span;       // Span evoking frame
    Span *outer;      // top-most containing span
    Handle property;  // property for match
  };

  // Fact catalog for fact extraction.
  FactCatalog catalog_;

  // Priority for properties. Priority -1 means that the property is blocked.
  HandleMap<int> priority_;
};

REGISTER_ANNOTATOR("relations", RelationAnnotator);

}  // namespace nlp
}  // namespace sling

