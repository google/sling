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

#include "nlp/parser/trainer/feature.h"

#include "frame/serialization.h"
#include "string/numbers.h"

REGISTER_COMPONENT_REGISTRY("sempar feature", sling::nlp::SemparFeature);

namespace sling {
namespace nlp {

using syntaxnet::dragnn::ComponentSpec;

namespace {

// Splits the given string on every occurrence of the given delimiter char.
std::vector<string> Split(const string &text, char delim) {
  std::vector<string> result;
  int token_start = 0;
  if (!text.empty()) {
    for (size_t i = 0; i < text.size() + 1; i++) {
      if (i == text.size() || text[i] == delim) {
        result.push_back(string(text.data() + token_start, i - token_start));
        token_start = i + 1;
      }
    }
  }
  return result;
}

}  // namespace

int SemparFeature::GetIntParam(const string &name, int default_value) const {
  const auto &it = params_.find(name);
  if (it == params_.end()) return default_value;
  int value;
  CHECK(safe_strto32(it->second, &value)) << it->second;
  return value;
}

bool SemparFeature::GetBoolParam(const string &name, bool default_value) const {
  const auto &it = params_.find(name);
  return (it == params_.end()) ? default_value : (it->second == "true");
}

float SemparFeature::GetFloatParam(
    const string &name, float default_value) const {
  const auto &it = params_.find(name);
  if (it == params_.end()) return default_value;
  float value;
  CHECK(safe_strtof(it->second, &value)) << it->second;
  return value;
}

const string &SemparFeature::GetParam(
    const string &name, const string &default_value) const {
  const auto &it = params_.find(name);
  return (it == params_.end()) ? default_value : it->second;
}

void SemparFeature::AddResourceToSpec(const string &name,
                                      const string &file,
                                      ComponentSpec *spec) {
  for (const auto &resource : spec->resource()) {
    if (resource.name() == name) {
      CHECK_EQ(resource.part_size(), 1) << resource.DebugString();
      CHECK_EQ(resource.part(0).file_pattern(), file) << resource.DebugString();
      return;
    }
  }
  auto *resource = spec->add_resource();
  resource->set_name(name);
  auto *part = resource->add_part();
  part->set_file_pattern(file);
}


string SemparFeature::GetResource(const syntaxnet::dragnn::ComponentSpec &spec,
                                  const string &name) {
  for (const auto &r : spec.resource()) {
    if (r.name() == name && r.part_size() > 0) {
      return r.part(0).file_pattern();
    }
  }
  return "";
}

SemparFeatureExtractor::~SemparFeatureExtractor() {
  for (auto &channel : channels_) {
    for (auto *feature : channel.features) {
      delete feature;
    }
  }
}

void SemparFeatureExtractor::AddChannel(const string &name,
                                        const string &fml,
                                        int embedding_dim) {
  CHECK_GT(embedding_dim, 0);
  CHECK(!name.empty());
  CHECK(!fml.empty());

  channels_.emplace_back();
  auto &channel = channels_.back();
  channel.name = name;
  channel.embedding_dim = embedding_dim;

  // Pared down FML parser. It doesn't support nested functions.
  std::vector<string> features = Split(fml, ' ');
  for (const string &feature : features) {
    if (feature.empty()) continue;
    size_t bracket = feature.find('(');
    string feature_name = feature;

    if (bracket != string::npos) {
      CHECK_EQ(feature.back(), ')') << feature;
      feature_name = feature.substr(0, bracket);
    }

    SemparFeature *f = SemparFeature::Create(feature_name);
    CHECK(f != nullptr) << feature_name;
    f->set_name(feature_name);
    f->set_fml(feature);
    channel.features.emplace_back(f);

    if (bracket != string::npos) {
      string inside = feature.substr(bracket + 1);
      inside.pop_back();  // remove closing bracket
      int arg;
      if (safe_strto32(inside, &arg)) {
        f->set_argument(arg);
      } else {
        std::vector<string> kv_pairs = Split(inside, ',');
        for (const string &kv : kv_pairs) {
          std::vector<string> key_value = Split(kv, '=');
          CHECK_EQ(key_value.size(), 2) << kv;
          f->SetParam(key_value[0], key_value[1]);
        }
      }
    }
  }
}

std::vector<std::pair<int, int>> SemparFeatureExtractor::Train(
    DocumentSource *corpus,
    const string &output_folder,
    SharedResources *resources,
    ComponentSpec *spec) {
  for (auto &channel : channels_) {
    for (auto *feature : channel.features) {
      feature->TrainInit(resources, *spec, output_folder);
    }
  }

  int count = 0;
  corpus->Rewind();
  while (true) {
    Store local(resources->global);
    Document *document = corpus->Next(&local);
    if (document == nullptr) break;

    count++;
    for (auto &channel : channels_) {
      for (auto *feature : channel.features) {
        feature->TrainProcess(*document);
      }
    }
    if (count % 10000 == 1) {
      LOG(INFO) << "SemparFeatureExtractor: " << count << " docs seen.";
    }
    delete document;
  }
  LOG(INFO) << "SemparFeatureExtractor: " << count << " docs seen.";

  std::vector<std::pair<int, int>> output;
  for (auto &channel : channels_) {
    for (auto *feature : channel.features) {
      int count = feature->TrainFinish(spec);
      LOG(INFO) << "Vocabulary size for " << channel.name
                << ":" << feature->name() << " = " << count;
      if (channel.vocabulary < count) channel.vocabulary = count;
    }
    LOG(INFO) << "Channel vocabulary size for " << channel.name << " = "
              << channel.vocabulary;
    output.emplace_back(channel.features.size(), channel.vocabulary);
  }

  return output;
}

void SemparFeatureExtractor::AddChannel(
    const syntaxnet::dragnn::FixedFeatureChannel &channel) {
  AddChannel(channel.name(), channel.fml(), channel.embedding_dim());
}

void SemparFeatureExtractor::AddChannel(
    const syntaxnet::dragnn::LinkedFeatureChannel &channel) {
  AddChannel(channel.name(), channel.fml(), channel.embedding_dim());
}

void SemparFeatureExtractor::Init(const ComponentSpec &spec,
                                  SharedResources *resources) {
  for (auto &channel : channels_) {
    for (auto *feature : channel.features) {
      feature->Init(spec, resources);
    }
  }
}

void SemparFeatureExtractor::RequestWorkspaces(WorkspaceRegistry *registry) {
  for (auto &channel : channels_) {
    for (auto *feature : channel.features) {
      feature->RequestWorkspaces(registry);
    }
  }
}

void SemparFeatureExtractor::Preprocess(SemparState *state) {
  for (auto &channel : channels_) {
    for (auto *feature : channel.features) {
      feature->Preprocess(state);
    }
  }
}

void SemparFeatureExtractor::Extract(SemparFeature::Args *args, int channel)
  const {
  CHECK_LT(channel, channels_.size());
  int index = 0;
  for (auto *feature : channels_[channel].features) {
    int old_size = args->output.size();
    feature->Extract(args);
    for (int i = old_size; i < args->output.size(); ++i) {
      auto &output = args->output[i];
      output.feature_index = index;
      if (args->debug) {
        output.debug = feature->FeatureToString(output.id);
      }
    }
    ++index;
  }
}

}  // namespace nlp
}  // namespace sling
