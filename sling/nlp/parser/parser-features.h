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

#ifndef SLING_NLP_PARSER_PARSER_FEATURES_H_
#define SLING_NLP_PARSER_PARSER_FEATURES_H_

#include <vector>

#include "sling/myelin/compute.h"
#include "sling/nlp/parser/parser-state.h"
#include "sling/nlp/parser/roles.h"

namespace sling {
namespace nlp {

class ParserFeatureExtractor;

// Feature model for parser decoder.
class ParserFeatureModel {
 public:
  // Initialize feature model.
  void Init(myelin::Cell *cell,
            const RoleSet *roles,
            int frame_limit);

  // Return output with activation for current step.
  const myelin::Tensor *activation() const { return activation_; }

 private:
  // Get parameter tensor in decoder cell.
  myelin::Tensor *GetParam(const string &name, bool optional = false);

  // Parser decoder cell.
  myelin::Cell *cell_ = nullptr;

  // Set of roles considered.
  const RoleSet *roles_;

  // Maximum frame attention index considered (exclusive).
  int frame_limit_;

  // Features.
  myelin::Tensor *token_feature_;             // current token feature

  myelin::Tensor *attention_tokens_feature_;  // token attention feature
  myelin::Tensor *attention_steps_feature_;   // step attention feature

  myelin::Tensor *history_feature_;           // history feature

  myelin::Tensor *mark_tokens_feature_;       // mark tokens feature
  myelin::Tensor *mark_steps_feature_;        // mark steps feature

  myelin::Tensor *out_roles_feature_;         // out roles feature
  myelin::Tensor *in_roles_feature_;          // in roles feature
  myelin::Tensor *unlabeled_roles_feature_;   // unlabeled roles feature
  myelin::Tensor *labeled_roles_feature_;     // labeled roles feature

  // Feature dimensions.
  int mark_depth_ = 0;                        // mark stack depth to use
  int history_size_ = 0;                      // number of history features
  int out_roles_size_ = 0;                    // max number of out roles
  int in_roles_size_ = 0;                     // max number of in roles
  int labeled_roles_size_ = 0;                // max number of unlabeled roles
  int unlabeled_roles_size_ = 0;              // max number of labeled roles

  // Channel links.
  myelin::Tensor *tokens_;                    // link to token encodings
  myelin::Tensor *steps_;                     // link to step activations


  // Output with activation for current step.
  myelin::Tensor *activation_;

  friend class ParserFeatureExtractor;
};

// Parser feature extractor for extracting features from a parse state.
class ParserFeatureExtractor {
 public:
  // Initialize parser feature extractor for parser state.
  ParserFeatureExtractor(const ParserFeatureModel *features,
                         const ParserState *state)
      : features_(features), state_(state) {}

  // Attach instance to input and output channels.
  void Attach(myelin::Channel *encodings,
              myelin::Channel *activations,
              myelin::Instance *instance);

  // Extract features from current state and add these to the data instance.
  void Extract(myelin::Instance *data);

 private:
  // Wrapper for data instance for looking up feature input tensors.
  class Data {
   public:
    Data(myelin::Instance *instance) : instance_(instance) {}

    int *Get(const myelin::Tensor *type) {
      return type ? instance_->Get<int>(type) : nullptr;
    }

   private:
    myelin::Instance *instance_;
  };

  // Feature extractor model.
  const ParserFeatureModel *features_;

  // Current parser state.
  const ParserState *state_;
};

}  // namespace nlp
}  // namespace sling

#endif  // SLING_NLP_PARSER_PARSER_FEATURES_H_

