// Copyright 2017 Google Inc. All Rights Reserved.
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
// =============================================================================

#ifndef SYNTAXNET_DRAGNN_CORE_INTERFACES_TRANSITION_STATE_H_
#define SYNTAXNET_DRAGNN_CORE_INTERFACES_TRANSITION_STATE_H_

#include <memory>
#include <vector>

#include "base/types.h"

namespace syntaxnet {
namespace dragnn {

// TransitionState interface that represents a batch item under processing.
class TransitionState {
 public:
  virtual ~TransitionState() {}

  // Get the score associated with this transition state.
  virtual const float GetScore() const = 0;

  // Set the score associated with this transition state.
  virtual void SetScore(const float score) = 0;
};

}  // namespace dragnn
}  // namespace syntaxnet

#endif  // SYNTAXNET_DRAGNN_CORE_INTERFACES_TRANSITION_STATE_H_
