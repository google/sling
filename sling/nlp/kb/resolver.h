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

#ifndef SLING_NLP_KB_RESOLVER_H_
#define SLING_NLP_KB_RESOLVER_H_

#include <utility>

#include "sling/base/types.h"
#include "sling/frame/object.h"
#include "sling/frame/store.h"
#include "sling/nlp/kb/phrase-table.h"
#include "sling/util/top.h"
#include "sling/util/unicode.h"

namespace sling {
namespace nlp {

class ResolverContext;

// Entity resolver.
class EntityResolver {
 public:
  // Initialize entity resolver.
  void Init(Store *commons, const PhraseTable *aliases);

 private:
  // Phrase table with aliases for entities.
  const PhraseTable *aliases_ = nullptr;

  // Symbols
  Names names_;
  Name n_popularity_{names_, "/w/item/popularity"};
  Name n_links_{names_, "/w/item/links"};

  // Hyperparameters.
  float topic_weight_ = 100.0;
  float mention_weight_ = 500.0;
  float base_context_score = 1e-3;
  float case_form_penalty = 0.1;
  int mention_boost_ = 30;

  friend class ResolverContext;
};

// Entity resolver context. It uses a phrase table for alias candidates and the
// link graph for incrementally scoring entity mentions. It builds up a mention
// and context representation to model the discourse.
class ResolverContext {
 public:
  // Entity resolution candidate.
  struct Candidate {
    Candidate(Handle entity, float score, int count, float context, bool local)
        : entity(entity), score(score), count(count), context(context),
          local(local) {}

    // Candidate comparison operator.
    bool operator >(const Candidate &other) const {
      return score > other.score;
    }

    Handle entity;       // item for candidate entity
    float score;         // score for candidate
    int count;           // entity prior frequency
    float context;       // context score
    bool local;          // local mention
  };

  // List of top candidates with scores.
  typedef Top<Candidate> Candidates;

  // Initialize resolver context.
  ResolverContext(Store *store, const EntityResolver *resolver);

  // Add entity topic to context.
  void AddTopic(Handle entity);

  // Add entity and output-bound links to context.
  void AddEntity(Handle entity);

  // Add mention to mention model.
  void AddMention(uint64 fp, CaseForm form, Handle entity, int count);

  // Score candidates for alias. The alias is specified using the fingerprint
  // and case form. Returns the top-k entities with the highest score.
  void Score(uint64 fp, CaseForm form, Candidates *candidates) const;

  // Resolve entity in context. Return the highest scoring entity or nil if
  // no matching entity is found.
  Handle Resolve(uint64 fp, CaseForm form) const;

  // Get entity popularity.
  int GetPopularity(Handle entity) const {
    Handle popularity = store_->GetFrame(entity)->get(n_popularity_);
    return popularity.IsNil() ? 1 : popularity.AsInt();
  }

 private:
  // Add tracking of anonymous frame to prevent it from being reclaimed.
  void Track(Handle h) {
    if (h.IsRef() && !h.IsNil() && store_->IsAnonymous(h)) {
      tracking_.push_back(h);
    }
  }

  // Resolved mention phrase.
  struct Mention {
    Handle entity = Handle::nil();  // resolved entity
    CaseForm form = CASE_INVALID;   // case form for mention phrase
    int count = 0;                  // number of mentions
  };

  // Look up context score for entity.
  float ContextScore(Handle entity, float defval = 0.0) const {
    auto f = context_.find(entity);
    return f != context_.end() ? f->second : defval;
  }

  // Frame store for resolved entities.
  Store *store_;

  // Entity resolver.
  const EntityResolver *resolver_;

  // Scores for entities in resolver context.
  HandleMap<float> context_;

  // Local mention phrases mapping from phrase fingerprint to entity.
  std::unordered_map<uint64, Mention> mentions_;

  // Tracked frame handles.
  Handles tracking_;

  // Symbols.
  Handle n_popularity_;
  Handle n_links_;
};

}  // namespace nlp
}  // namespace sling

#endif  // SLING_NLP_KB_RESOLVER_H_
