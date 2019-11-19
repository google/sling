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

#include "sling/nlp/kb/resolver.h"

namespace sling {
namespace nlp {

void EntityResolver::Init(Store *commons, const PhraseTable *aliases) {
  aliases_ = aliases;
  CHECK(names_.Bind(commons));
}

ResolverContext::ResolverContext(Store *store, const EntityResolver *resolver)
    : store_(store), resolver_(resolver), tracking_(store),
      n_popularity_(resolver->n_popularity_.handle()),
      n_links_(resolver->n_links_.handle()) {}

void ResolverContext::AddTopic(Handle entity) {
  // Add entity to context model.
  context_[entity] += resolver_->topic_weight_;
  Track(entity);
}

void ResolverContext::AddEntity(Handle entity) {
  // Add entity to context model.
  float popularity = GetPopularity(entity);
  context_[entity] += resolver_->mention_weight_ / popularity;
  Track(entity);

  // Add outbound links to context model.
  FrameDatum *item = store_->GetFrame(entity);
  Handle item_links = item->get(n_links_);
  if (!item_links.IsNil()) {
    FrameDatum *links = store_->GetFrame(item_links);
    for (const Slot *s = links->begin(); s != links->end(); ++s) {
      Handle link = s->name;
      float count = s->value.AsInt();
      float popularity = GetPopularity(link);
      context_[link] += count / popularity;
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
    Track(entity);
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
    if (m.item.IsNil()) continue;

    // Compute score for candidate.
    float context = ContextScore(m.item, resolver_->base_context_score);

    // Add scores from outbound links for candidate.
    if (!m.item.IsNil()) {
      FrameDatum *item = store_->GetFrame(m.item);
      Handle item_links = item->get(n_links_);
      if (!item_links.IsNil()) {
        FrameDatum *links = store_->GetFrame(item_links);
        for (const Slot *s = links->begin(); s != links->end(); ++s) {
          context += ContextScore(s->name) * s->value.AsInt();
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

}  // namespace nlp
}  // namespace sling

