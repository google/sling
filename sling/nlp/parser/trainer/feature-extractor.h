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

#ifndef SLING_NLP_PARSER_TRAINER_FEATURE_EXTRACTOR_H_
#define SLING_NLP_PARSER_TRAINER_FEATURE_EXTRACTOR_H_

#include <functional>
#include <vector>

#include "dragnn/protos/spec.pb.h"
#include "sling/nlp/document/lexicon.h"
#include "sling/nlp/parser/trainer/shared-resources.h"
#include "sling/nlp/parser/trainer/transition-state.h"

namespace sling {
namespace nlp {

// Feature extractor for fixed features. Working assumptions:
// - Exactly one feature per channel.
// - The feature can output multiple ids per firing, up to a max specified
//   in the channel's spec. The extractor doesn't enforce this check though,
//   and violating this check can overwrite other feature ids in the
//   preallocated memory block.
class FixedFeatureExtractor {
 public:
  // Initializes all the channels.
  void Init(syntaxnet::dragnn::ComponentSpec &spec, SharedResources *resources);

  // Precomputes any features for 'state'. This is useful, for instance, for
  // computing lexical features for all tokens in one shot.
  void Preprocess(SemparState *state);

  // Outputs feature id(s) for 'state' and given 'channel' into the array from
  // 'output' onwards.
  void Extract(int channel, SemparState *state, int64 *output) const;

  // Reports the maximum number of feature ids for 'channel'.
  int MaxNumIds(int channel) const { return max_num_ids_[channel]; }

 private:
  // Parses frame limit for role features from 'param'.
  void ParseFrameLimit(const string &param);

  // Whether any lexical document features are present.
  bool has_document_features_ = false;

  // Frame limit for role features. All role features are required to share the
  // same role limit.
  int role_frame_limit_ = -1;

  // Lexicon for document features. Not owned.
  const Lexicon *lexicon_ = nullptr;

  // Feature functions, one per channel.
  std::vector<std::function<void(SemparState *, int64 *)>> functions_;

  // Maximum number of feature ids, one per channel.
  std::vector<int> max_num_ids_;
};

// Extractor for link features. Working assumptions:
// - Like fixed features, each channel has one feature type,
//   but multiple values of that feature can fire in that channel.
//   fire in that channel, e.g. the history feature can report the last, second
//   last, third last actions etc. The number of values is given apriori.
// - If a feature value is missing then -1 is used instead.
// - The values have a fixed order. Thus the kth output should also be the
//   kth value (or -1). E.g. for the history feature, the kth value should
//   always be the kth last step.
class LinkFeatureExtractor {
 public:
  // Initializes all linked feature channels.
  void Init(syntaxnet::dragnn::ComponentSpec &spec, SharedResources *resources);

  // From 'state', computes linked features for 'channel', and reports the
  // output from 'output' onwards.
  void Extract(int channel, SemparState *state, int *output) const;

  // Reports the channel size (i.e. number of feature values) for 'channel'.
  int ChannelSize(int channel) const { return channel_size_[channel]; }

 private:
  // Feature functions, one per channel.
  std::vector<std::function<void(SemparState *, int *)>> functions_;

  // Channel size, one per channel.
  std::vector<int> channel_size_;

  // Limits for various link features.
  int history_limit_ = 0;
  int frame_creation_limit_ = 0;
  int frame_focus_limit_ = 0;
  int frame_end_limit_ = 0;
};

}  // namespace nlp
}  // namespace sling

#endif // SLING_NLP_PARSER_TRAINER_FEATURE_EXTRACTOR_H_
