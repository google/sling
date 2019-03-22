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

#ifndef SLING_NLP_NER_RESOLVER_H_
#define SLING_NLP_NER_RESOLVER_H_

#include <utility>

#include "sling/base/types.h"
#include "sling/frame/object.h"
#include "sling/frame/store.h"
#include "sling/nlp/kb/phrase-table.h"
#include "sling/util/top.h"
#include "sling/util/unicode.h"

namespace sling {
namespace nlp {

// Symbol names for entity resolver.
struct ResolverNames : public SharedNames {
  ResolverNames(Store *store) { CHECK(Bind(store)); }

  Name n_popularity{*this, "/w/item/popularity"};
  Name n_links{*this, "/w/item/links"};
};

// Entity resolver using a phrase table for alias candidates and the link
// graph for context scoring.
class Resolver {
 public:
  // Entity resolution candidate.
  struct Candidate {
    Candidate(Handle entity, float score) : entity(entity), score(score) {}

    // Candidate comparison operator.
    bool operator >(const Candidate &other) const {
      return score > other.score;
    }

    Handle entity;  // item for candidate entity
    float score;    // score for candidate
  };

  // List of top candidates with scores.
  typedef Top<Candidate> Candidates;

  // Initialize resolver.
  Resolver(Store *store,
           const PhraseTable *aliases,
           const ResolverNames *names = nullptr);
  ~Resolver() { if (names_) names_->Release(); }

  // Add entity topic to context.
  void AddTopic(Handle entity);

  // Add entity and output-bound links to context.
  void AddEntity(Handle entity);

  // Score candidates for alias. The alias is specified using the fingerprint
  // and case form. Returns a the top-k entities with the highest score.
  void Score(uint64 fp, CaseForm form, Candidates *candidates) const;

  // Resolve entity in context. Return the highest scoring entity or nil if
  // no matching entity is found.
  Handle Resolve(uint64 fp, CaseForm form) const;

 private:
  float ContextScore(Handle entity, float defval = 0.0) const {
    auto f = context_.find(entity);
    return f != context_.end() ? f->second : defval;
  }

  // Knowledge base with link graph.
  Store *store_;

  // Phrase table for looking up aliases.
  const PhraseTable *aliases_;

  // Symbols for resolver.
  const ResolverNames *names_ = nullptr;

  // Context scores for entities in resolver.
  HandleMap<float> context_;

  // Hyperparameters.
  float mention_weight_ = 100.0;
  float base_context_score = 1e-3;
  float case_form_penalty = 0.1;
};

}  // namespace nlp
}  // namespace sling

#endif  // SLING_NLP_NER_RESOLVER_H_
