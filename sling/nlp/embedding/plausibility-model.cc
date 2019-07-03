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

#include "sling/nlp/embedding/plausibility-model.h"

#include "sling/myelin/flow.h"
#include "sling/myelin/compiler.h"
#include "sling/frame/serialization.h"

namespace sling {
namespace nlp {

using namespace myelin;

void PlausibilityModel::Load(Store *store, const string &filename) {
  // Load model from flow file.
  Flow flow;
  flow.Load(filename);

  // Compile model.
  Compiler compiler;
  compiler.Compile(&flow, &model_);

  // Read fact lexicon into store.
  Flow::Blob *facts = flow.DataBlock("facts");
  if (facts != nullptr) {
    // Decode fact lexicon.
    fact_lexicon_ = Decode(store, Text(facts->data, facts->size)).AsArray();

    // Build fact mapping.
    num_facts_ = fact_lexicon_.length();
    for (int i = 0; i < num_facts_; ++i) {
      uint64 fp = store->Fingerprint(fact_lexicon_.get(i));
      fact_mapping_[fp] = i;
    }

    // Get model inputs/outputs.
    scorer_ = model_.GetCell("scorer");
    premise_ = scorer_->GetParameter("scorer/premise");
    hypothesis_ = scorer_->GetParameter("scorer/hypothesis");
    probs_ = scorer_->GetParameter("scorer/probs");
    max_features_ = premise_->dim(1);
  }
}

float PlausibilityModel::Score(const Facts &premise,
                               const Facts &hypothesis) const {
  Instance scorer(scorer_);

  // Set premise.
  if (!CopyFeatures(premise, scorer.Get<int>(premise_))) {
    return EMPTY_PREMISE;
  }

  // Set hypothesis.
  if (!CopyFeatures(hypothesis, scorer.Get<int>(hypothesis_))) {
    return EMPTY_HYPOTHESIS;
  }

  // Compute plausibility score.
  scorer.Compute();
  return scorer.Get<float>(probs_)[1];
}

bool PlausibilityModel::CopyFeatures(const Facts &facts, int *features) const {
  int *p = features;
  int *end = features + max_features_;
  for (int i = 0; i < facts.size(); ++i) {
    int f = Lookup(facts.fingerprint(i));
    if (f == -1) continue;
    *p++ = f;
    if (p == end) break;
  }
  if (p < end) *p = -1;
  return p != features;
}

int PlausibilityModel::Lookup(uint64 fp) const {
  auto f = fact_mapping_.find(fp);
  if (f == fact_mapping_.end()) return -1;
  return f->second;
}

int PlausibilityModel::Lookup(const Array &fact) const {
  uint64 fp = fact.store()->Fingerprint(fact.handle());
  return Lookup(fp);
}

}  // namespace nlp
}  // namespace sling

