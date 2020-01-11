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

#include "sling/nlp/parser/action-table.h"
#include "sling/nlp/parser/parser.h"

namespace sling {
namespace nlp {

using namespace myelin;

// Deletegate for fixed action classification.
class MultiClassDelegate : public Delegate {
 public:
  void Initialize(const Network &network, const Frame &spec) override {
    cell_ = network.GetCell(spec.GetString("cell"));
    input_ = cell_->GetParameter(cell_->name() + "/input");
    output_ = cell_->GetParameter(cell_->name() + "/output");
    actions_.Read(spec);
  }

  DelegateInstance *CreateInstance() override {
    return new MultiClassDelegateInstance(this);
  }

  // Multi-class delegate instance.
  class MultiClassDelegateInstance : public DelegateInstance {
   public:
    MultiClassDelegateInstance(MultiClassDelegate *delegate)
        : delegate_(delegate),
          data_(delegate->cell_) {}

    void Predict(float *activation, ParserAction *action) override {
      // Predict action from activations.
      data_.SetReference(delegate_->input_, activation);
      data_.Compute();
      int argmax = *data_.Get<int>(delegate_->output_);
      *action = delegate_->actions_.Action(argmax);
    }

   private:
    MultiClassDelegate *delegate_;
    Instance data_;
  };

 private:
  ActionTable actions_;        // action table for multi-class classification

  Cell *cell_ = nullptr;       // cell for computation
  Tensor *input_ = nullptr;    // input for activations
  Tensor *output_ = nullptr;   // output prediction
};

REGISTER_DELEGATE("multiclass", MultiClassDelegate);

}  // namespace nlp
}  // namespace sling

