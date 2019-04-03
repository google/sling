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

void EntityResolver::Init(Store *commons, const PhraseTable *aliases) {
  aliases_ = aliases;
  CHECK(names_.Bind(commons));
}

void ResolverContext::AddTopic(Handle entity) {
  // Add entity to context model.
  context_[entity] += 1.0;
}

void ResolverContext::AddEntity(Handle entity) {
  // Add entity to context model.
  Frame item(store_, entity);
  float popularity = item.GetInt(resolver_->n_popularity, 1);
  context_[entity] += resolver_->mention_weight_ / popularity;

  // Add outbound links to context model.
  Frame links = item.GetFrame(resolver_->n_links);
  if (links.valid()) {
    for (const Slot &s : links) {
      Frame link(store_, s.name);
      float popularity = link.GetInt(resolver_->n_popularity, 1);
      float count = s.value.AsInt();
      context_[link.handle()] += count / popularity;
    }
  }
}

void ResolverContext::AddMention(uint64 fp, CaseForm form,
                                 Handle entity, int count) {
  // Only the first resolved entity for the mention phrase is kept.
  Mention &mention = mentions_[fp];
  if (mention.entity.IsNil()) {
    mention.entity = entity;
    mention.form = form;
    mention.count = count * resolver_->mention_boost_;
  } else if (entity == mention.entity) {
    mention.count += count * resolver_->mention_boost_;
  }
}

void ResolverContext::Score(uint64 fp, CaseForm form,
                            Candidates *candidates) const {
  // Get candidates from phrase table.
  PhraseTable::MatchList matches;
  resolver_->aliases_->Lookup(fp, &matches);

  // Add phrase match from mention model.
  auto f = mentions_.find(fp);
  if (f != mentions_.end()) {
    // Add local mention if it is not already a candidate.
    const Mention &mention = f->second;
    bool found = false;
    for (auto &c : matches) {
      if (c.item == mention.entity) {
        c.count += mention.count;
        found = true;
        break;
      }
    }
    if (!found) {
      PhraseTable::Match m;
      m.item = mention.entity;
      m.count = mention.count;
      m.form = mention.form;
      m.reliable = true;
      matches.emplace_back(m);
    }
  }

  // Score candidates.
  candidates->clear();
  for (auto &m : matches) {
    // Compute score for candidate.
    float context = ContextScore(m.item, resolver_->base_context_score);

    // Add scores from outbound links for candidate.
    Frame item(store_, m.item);
    if (item.valid()) {
      Frame links = item.GetFrame(resolver_->n_links);
      if (links.valid()) {
        for (const Slot &s : links) {
          context += ContextScore(s.name) * s.value.AsInt();
        }
      }
    }

    // Combine context score with name prior.
    float score = context;
    score *= m.count;

    // Apply case form penalty.
    if (form != CASE_NONE && m.form != CASE_NONE && form != m.form) {
      score *= resolver_->case_form_penalty;
    }

    // Add final score to top-k candidate list. Local match has empty id.
    candidates->push(Candidate(m.item, score, m.count, context, m.id.empty()));
  }
}

Handle ResolverContext::Resolve(uint64 fp, CaseForm form) const {
  Candidates best(1);
  Score(fp, form, &best);
  return best.empty() ? Handle::nil() : best[0].entity;
}

int ResolverContext::GetPopularity(Handle entity) const {
  Frame item(store_, entity);
  return item.GetInt(resolver_->n_popularity, 1);
}

}  // namespace nlp
}  // namespace sling

