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

#include "sling/nlp/document/document.h"
#include "sling/task/documents.h"
#include "sling/util/fingerprint.h"

namespace sling {
namespace nlp {

using namespace task;

// Split document corpus into training and evaluation data sets. The training
// data is shuffled based on the contents of the document.
class CorpusSplitter : public DocumentProcessor {
 public:
  void Startup(Task *task) override {
    // Get output chanels.
    train_ = task->GetSink("train");
    eval_ = task->GetSink("eval");
    CHECK(train_ != nullptr) << "train channel missing";
    CHECK(eval_ != nullptr) << "eval channel missing";

    // Get parameters.
    task->Fetch("split_ratio", &split_ratio_);
  }

  void Process(Slice key, const Document &document) override {
    uint64 fp = Fingerprint(document.text().data(), document.text().size());
    if (fp % split_ratio_ == (split_ratio_ - 1)) {
      // Output evaluation document.
      eval_->Send(CreateMessage(key, document.top()));
    } else {
      // Output training document.
      train_->Send(CreateMessage(std::to_string(fp), document.top()));
    }
  }

 private:
  // Channels for training and evaluation documents.
  Channel *train_ = nullptr;
  Channel *eval_ = nullptr;

  // Corpus split ratio, i.e. a corpus split ratio of 10 means that one in ten
  // documents is added to the evaluation set (90% train, 10% eval).
  int split_ratio_ = 10;
};

REGISTER_TASK_PROCESSOR("corpus-split", CorpusSplitter);

}  // namespace nlp
}  // namespace sling

