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

#ifndef SLING_NLP_EMBEDDING_PLAUSIBILITY_MODEL_H_
#define SLING_NLP_EMBEDDING_PLAUSIBILITY_MODEL_H_

#include <vector>

#include "sling/frame/object.h"
#include "sling/frame/store.h"
#include "sling/myelin/compute.h"
#include "sling/nlp/kb/facts.h"

namespace sling {
namespace nlp {

class PlausibilityModel {
 public:
  // Pseudo-scores for empty premise or hypothesis.
  static constexpr float EMPTY_PREMISE = -1.0;
  static constexpr float EMPTY_HYPOTHESIS = -2.0;

  // Load model from file.
  void Load(Store *store, const string &filename);

  // Predict how likely a fact (hypothesis) is to be true about an item given
  // a set of known facts about the item (premise).
  float Score(const Facts &premise, const Facts &hypothesis) const;

  // Look up fact in lexicon. Return -1 for unknown fact.
  int Lookup(const Array &fact) const;

  // Look up fact fingerprint in lexicon. Return -1 for unknown fact.
  int Lookup(uint64 fp) const;

  // Return fact lexicon.
  const Array &lexicon() const { return fact_lexicon_; }

 private:
  // Copy facts to feature vector.
  bool CopyFeatures(const Facts &facts, int *features) const;

  // Plausibility model.
  myelin::Network model_;
  myelin::Cell *scorer_ = nullptr;
  myelin::Tensor *premise_ = nullptr;
  myelin::Tensor *hypothesis_ = nullptr;
  myelin::Tensor *probs_ = nullptr;

  // Fact lexicon.
  Array fact_lexicon_;

  // Mapping from fact fingerprints to fact ids.
  std::unordered_map<uint64, int> fact_mapping_;

  // Model parameters.
  int num_facts_ = 0;
  int max_features_ = 0;
};

}  // namespace nlp
}  // namespace sling

#endif  // SLING_NLP_EMBEDDING_PLAUSIBILITY_MODEL_H_

