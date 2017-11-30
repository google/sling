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

#include "sling/nlp/parser/trainer/feature-extractor.h"

#include <functional>

#include "sling/base/logging.h"
#include "sling/base/macros.h"
#include "sling/nlp/document/features.h"

namespace sling {
namespace nlp {

void FixedFeatureExtractor::ParseFrameLimit(const string &param) {
  CHECK(!param.empty());
  char *end = nullptr;
  int i = strtol(param.c_str(), &end, 10);
  CHECK(end == param.c_str() + param.size()) << param;

  if (role_frame_limit_ == -1) {
    role_frame_limit_ = i;
  } else {
    CHECK_EQ(i, role_frame_limit_)
        << "Frame limits should be the same for all role features";
  }
}

namespace {

// Functions for various role features. These are curried via their first
// arguments and used in FixedFeatureExtractor.
void InRole(int max, SemparState *state, int64 *output) {
  int64 *end = output + max;
  state->role_graph().in([&output, end](int id) {
    if (output < end) *output++ = id;
  });
}

void OutRole(int max, SemparState *state, int64 *output) {
  int64 *end = output + max;
  state->role_graph().out([&output, end](int id) {
    if (output < end) *output++ = id;
  });
}

void LabeledRole(int max, SemparState *state, int64 *output) {
  int64 *end = output + max;
  state->role_graph().labeled([&output, end](int id) {
    if (output < end) *output++ = id;
  });
}

void UnlabeledRole(int max, SemparState *state, int64 *output) {
  int64 *end = output + max;
  state->role_graph().unlabeled([&output, end](int id) {
    if (output < end) *output++ = id;
  });
}

}  // namespace

void FixedFeatureExtractor::Init(
    syntaxnet::dragnn::ComponentSpec &spec, SharedResources *resources) {
  using std::placeholders::_1;
  using std::placeholders::_2;

  lexicon_ = &resources->lexicon;
  has_document_features_ = false;
  for (const auto &fixed : spec.fixed_feature()) {
    CHECK(fixed.has_size()) << fixed.DebugString();
    const string &fml = fixed.fml();
    size_t sep = fml.find(' ');
    string name = fml;
    string rest;
    if (sep != string::npos) {
      name = fml.substr(0, sep);
      rest = fml.substr(sep + 1);
    }
    max_num_ids_.emplace_back(fixed.size());

    if (name == "word") {
      CHECK(rest.empty()) << rest;
      has_document_features_ = true;
      functions_.push_back([](SemparState *state, int64 *output) {
        int c = state->current();
        if (c >= state->end() || c < state->begin()) return;
        *output = state->features()->word(c);
      });
    } else if (name == "prefix") {
      CHECK(rest.empty()) << rest;
      has_document_features_ = true;
      functions_.push_back([](SemparState *state, int64 *output) {
        int c = state->current();
        if (c >= state->end() || c < state->begin()) return;
        Affix *affix = state->features()->prefix(c);
        while (affix != nullptr) {
          *output++ = affix->id();
          affix = affix->shorter();
        }
      });
    } else if (name == "suffix") {
      CHECK(rest.empty()) << rest;
      has_document_features_ = true;
      functions_.push_back([](SemparState *state, int64 *output) {
        int c = state->current();
        if (c >= state->end() || c < state->begin()) return;
        Affix *affix = state->features()->suffix(c);
        while (affix != nullptr) {
          *output++ = affix->id();
          affix = affix->shorter();
        }
      });
    } else if (name == "capitalization") {
      has_document_features_ = true;
      functions_.push_back([](SemparState *state, int64 *output) {
        int c = state->current();
        if (c >= state->end() || c < state->begin()) return;
        *output = state->features()->capitalization(c);
      });
    } else if (name == "hyphen") {
      has_document_features_ = true;
      functions_.push_back([](SemparState *state, int64 *output) {
        int c = state->current();
        if (c >= state->end() || c < state->begin()) return;
        *output = state->features()->hyphen(c);
      });
    } else if (name == "punctuation") {
      has_document_features_ = true;
      functions_.push_back([](SemparState *state, int64 *output) {
        int c = state->current();
        if (c >= state->end() || c < state->begin()) return;
        *output = state->features()->punctuation(c);
      });
    } else if (name == "quote") {
      has_document_features_ = true;
      functions_.push_back([](SemparState *state, int64 *output) {
        int c = state->current();
        if (c >= state->end() || c < state->begin()) return;
        *output = state->features()->quote(c);
      });
    } else if (name == "digit") {
      has_document_features_ = true;
      functions_.push_back([](SemparState *state, int64 *output) {
        int c = state->current();
        if (c >= state->end() || c < state->begin()) return;
        *output = state->features()->digit(c);
      });
    } else if (name == "in-roles") {
      ParseFrameLimit(rest);
      functions_.push_back(std::bind(&InRole, fixed.size(), _1, _2));
    } else if (name == "out-roles") {
      ParseFrameLimit(rest);
      functions_.push_back(std::bind(&OutRole, fixed.size(), _1, _2));
    } else if (name == "labeled-roles") {
      ParseFrameLimit(rest);
      functions_.push_back(std::bind(&LabeledRole, fixed.size(), _1, _2));
    } else if (name == "unlabeled-roles") {
      ParseFrameLimit(rest);
      functions_.push_back(std::bind(&UnlabeledRole, fixed.size(), _1, _2));
    } else {
      LOG(FATAL) << "Unknown fixed feature: " << name;
    }
  }
}

void FixedFeatureExtractor::Preprocess(SemparState *state) {
  // Only precompute the lexical features if they are required.
  if (has_document_features_) {
    DocumentFeatures *f = new DocumentFeatures(lexicon_);
    f->Extract(*state->document());
    state->set_document_features(f);
  }
  state->set_role_frame_limit(role_frame_limit_);
}

void FixedFeatureExtractor::Extract(int channel,
                                    SemparState *state,
                                    int64 *output) const {
  functions_[channel](state, output);
}

void LinkFeatureExtractor::Init(
    syntaxnet::dragnn::ComponentSpec &spec, SharedResources *resources) {
  for (const auto &link : spec.linked_feature()) {
    CHECK(link.has_size()) << link.DebugString();
    const string &name = link.fml();
    int size = link.size();
    channel_size_.emplace_back(size);

    if (name == "focus") {
      CHECK_EQ(size, 1);
      functions_.push_back([](SemparState *state, int *output) {
        int c = state->current();
        if ((c >= state->begin()) && (c < state->end())) {
          *output = c - state->begin();
        }
      });
    } else if (name == "history") {
      history_limit_ = size;
      functions_.push_back([this](SemparState *state, int *output) {
        for (int i = 0; i < this->history_limit_; ++i) {
          *output++ = i;
        }
      });
    } else if (name == "frame-creation") {
      frame_creation_limit_ = size;
      functions_.push_back([this](SemparState *state, int *output) {
        CHECK(!state->shift_only());
        for (int i = 0; i < this->frame_creation_limit_; ++i) {
          if (i == state->parser_state()->AttentionSize()) break;
          *output++ = state->CreationStep(i);
        }
      });
    } else if (name == "frame-focus") {
      frame_focus_limit_ = size;
      functions_.push_back([this](SemparState *state, int *output) {
        CHECK(!state->shift_only());
        for (int i = 0; i < this->frame_focus_limit_; ++i) {
          if (i == state->parser_state()->AttentionSize()) break;
          *output++ = state->FocusStep(i);
        }
      });
    } else if (name == "frame-end") {
      frame_end_limit_ = size;
      functions_.push_back([this](SemparState *state, int *output) {
        CHECK(!state->shift_only());
        const auto *parser_state = state->parser_state();
        for (int i = 0; i < this->frame_end_limit_; ++i) {
          if (i == parser_state->AttentionSize()) break;
          int frame = parser_state->Attention(i);
          int end = parser_state->FrameEvokeEnd(frame);

          // 'end' is exclusive so end - 1 is the last token.
          // Also, ungrounded frames would have end = -1.
          *output++ = (end == -1) ? -1 : (end - 1 - parser_state->begin());
        }
      });
    } else {
      LOG(FATAL) << "Unknown link feature: " << name;
    }
  }
}

void LinkFeatureExtractor::Extract(int channel,
                                   SemparState *state,
                                   int *output) const {
  functions_[channel](state, output);
}

}  // namespace nlp
}  // namespace sling
