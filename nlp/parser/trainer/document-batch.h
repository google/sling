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

#ifndef NLP_PARSER_TRAINER_DOCUMENT_BATCH_H_
#define NLP_PARSER_TRAINER_DOCUMENT_BATCH_H_

#include <string>
#include <utility>
#include <vector>

#include "dragnn/core/interfaces/input_batch.h"
#include "frame/store.h"
#include "nlp/parser/trainer/sempar-instance.h"

namespace sling {
namespace nlp {

// InputBatch implementation for SLING document batches.
class DocumentBatch : public syntaxnet::dragnn::InputBatch {
 public:
  // Translates from a vector of serialized Document frames.
  void SetData(const std::vector<string> &data) override;

  // Translates to a vector of serialized Document frames.
  const std::vector<string> GetSerializedData() const override;

  // Returns the size of the batch.
  int size() const { return items_.size(); }
  SemparInstance *item(int i) { return &items_[i]; }

  // Decodes the documents in the batch. 'global' is used to construct the
  // local stores.
  void Decode(Store *global);

 private:
  // Document batch.
  std::vector<SemparInstance> items_;
};

}  // namespace nlp
}  // namespace sling

#endif  // NLP_PARSER_TRAINER_DOCUMENT_BATCH_H_
