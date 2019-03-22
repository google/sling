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

#include "sling/nlp/ner/resolver.h"

namespace sling {
namespace nlp {

Resolver::Resolver(Store *store,
                   const PhraseTable *aliases,
                   const ResolverNames *names)
    : store_(store), aliases_(aliases) {
  if (names_ == nullptr) {
    names_ = new ResolverNames(store);
  } else {
    names_->AddRef();
  }
}

void Resolver::AddTopic(Handle entity) {
  // Add entity to context model.
  context_[entity] += 1.0;
}

void Resolver::AddEntity(Handle entity) {
  // Add entity to context model.
  Frame item(store_, entity);
  float popularity = item.GetInt(names_->n_popularity, 1);
  context_[entity] += mention_weight_ / popularity;

  // Add outbound links to context model.
  Frame links = item.GetFrame(names_->n_links);
  if (links.valid()) {
    for (const Slot &s : links) {
      Frame link(store_, s.name);
      float popularity = link.GetInt(names_->n_popularity, 1);
      float count = s.value.AsInt();
      context_[link.handle()] += count / popularity;
    }
  }
}

void Resolver::Score(uint64 fp, CaseForm form, Candidates *candidates) const {
  // Get candidates from phrase table.
  PhraseTable::MatchList matches;
  aliases_->Lookup(fp, &matches);

  // Score candidates.
  candidates->clear();
  for (auto &m : matches) {
    // Compute score for candidate.
    float score = ContextScore(m.item, base_context_score);

    // Add scores from outbound links for candidate.
    Frame item(store_, m.item);
    if (item.valid()) {
      Frame links = item.GetFrame(names_->n_links);
      if (links.valid()) {
        for (const Slot &s : links) {
          score += ContextScore(s.name) * s.value.AsInt();
        }
      }
    }

    // Add case form penalty.
    if (form != CASE_NONE && m.form != CASE_NONE && form != m.form) {
      score *= case_form_penalty;
    }

    // Add name prior to score.
    score *= m.count;

    // Add final score to top-k candidate list.
    candidates->push(Candidate(m.item, score));
  }
}

Handle Resolver::Resolve(uint64 fp, CaseForm form) const {
  Candidates best(1);
  Score(fp, form, &best);
  return best.empty() ? Handle::nil() : best[0].entity;
}

}  // namespace nlp
}  // namespace sling

