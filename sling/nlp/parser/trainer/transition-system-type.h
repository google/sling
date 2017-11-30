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

#ifndef SLING_NLP_PARSER_TRAINER_SEMPAR_TRANSITION_SYSTEM_TYPE_H_
#define SLING_NLP_PARSER_TRAINER_SEMPAR_TRANSITION_SYSTEM_TYPE_H_

namespace sling {
namespace nlp {

// Types of transition systems supported.
enum TransitionSystemType {
  SHIFT_ONLY,
  SEMPAR
};

}  // namespace nlp
}  // namespace sling

#endif  // SLING_NLP_PARSER_TRAINER_SEMPAR_TRANSITION_SYSTEM_TYPE_H_
