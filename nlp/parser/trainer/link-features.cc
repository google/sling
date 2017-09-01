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

#include "nlp/parser/trainer/feature.h"

#include "string/strcat.h"

namespace sling {
namespace nlp {

using syntaxnet::dragnn::ComponentSpec;

class ConstantFeature : public SemparFeature {
 public:
  int TrainFinish(ComponentSpec *spec) override { return argument() + 1; }

  string FeatureToString(int64 id) const override {
    return StrCat(name(), "=", id);
  }

  void Extract(Args *args) override {
    args->Output(argument());
  }
};

REGISTER_SEMPAR_FEATURE("constant", ConstantFeature);

class FrameCreationStepFeature : public SemparFeature {
 public:
  int TrainFinish(ComponentSpec *spec) override { return 1000; }

  string FeatureToString(int64 id) const override {
    return StrCat(name(), "(", argument(), ")=", id);
  }

  void Extract(Args *args) override {
    CHECK(!args->state->shift_only());
    if (argument() >= args->parser_state()->AttentionSize()) return;
    if (argument() < 0) return;
    args->Output(args->state->CreationStep(argument()));
  }
};

REGISTER_SEMPAR_FEATURE("frame-creation", FrameCreationStepFeature);

class FrameFocusStepFeature : public SemparFeature {
 public:
  int TrainFinish(ComponentSpec *spec) override { return 1000; }

  string FeatureToString(int64 id) const override {
    return StrCat(name(), "(", argument(), ")=", id);
  }

  void Extract(Args *args) override {
    CHECK(!args->state->shift_only());
    if (argument() >= args->parser_state()->AttentionSize()) return;
    if (argument() < 0) return;
    args->Output(args->state->FocusStep(argument()));
  }
};

REGISTER_SEMPAR_FEATURE("frame-focus", FrameFocusStepFeature);

class FrameEndStepFeature : public SemparFeature {
 public:
  int TrainFinish(ComponentSpec *spec) override { return 1000; }

  string FeatureToString(int64 id) const override {
    return StrCat(name(), "(", argument(), ")=", id);
  }

  void Extract(Args *args) override {
    CHECK(!args->state->shift_only());
    if (argument() >= args->parser_state()->AttentionSize()) return;
    if (argument() < 0) return;
    int frame = args->parser_state()->Attention(argument());
    args->Output(args->parser_state()->FrameEvokeEnd(frame) - 1);
  }
};

REGISTER_SEMPAR_FEATURE("frame-end", FrameEndStepFeature);

class CurrentTokenFeature : public SemparFeature {
 public:
  int TrainFinish(ComponentSpec *spec) override { return 1000; }

  string FeatureToString(int64 id) const override {
    return StrCat(name(), "(", argument(), ")=", id);
  }

  void Extract(Args *args) override {
    CHECK(!args->state->shift_only());
    int index = args->state->current() + argument();
    if (index < 0 || index >= args->state->end()) return;
    args->Output(index);
  }
};

REGISTER_SEMPAR_FEATURE("current-token", CurrentTokenFeature);

}  // namespace nlp
}  // namespace sling
