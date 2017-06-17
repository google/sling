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

#include <string>
#include <vector>

#include "base/flags.h"
#include "base/init.h"
#include "base/logging.h"
#include "base/macros.h"
#include "dragnn/core/interfaces/component.h"
#include "dragnn/protos/spec.pb.h"
#include "file/file.h"
#include "frame/object.h"
#include "frame/serialization.h"
#include "frame/store.h"
#include "nlp/document/document.h"
#include "nlp/parser/parser-action.h"
#include "nlp/parser/trainer/feature.h"
#include "nlp/parser/trainer/sempar-component.h"
#include "nlp/parser/trainer/shared-resources.h"
#include "string/strcat.h"
#include "tensorflow/core/platform/protobuf.h"

using sling::File;
using sling::FileDecoder;
using sling::Object;
using sling::Store;
using sling::StrCat;
using sling::nlp::Document;
using sling::nlp::ParserAction;
using sling::nlp::SemparComponent;
using sling::nlp::SemparFeature;
using sling::nlp::SharedResources;

using syntaxnet::dragnn::Component;
using syntaxnet::dragnn::ComponentSpec;
using syntaxnet::dragnn::InputBatchCache;
using syntaxnet::dragnn::MasterSpec;

using tensorflow::protobuf::TextFormat;

DEFINE_string(spec, "/tmp/sempar_out/master_spec", "Path to master spec.");
DEFINE_string(documents, "/tmp/foobar/doc.?", "Train documents file pattern.");

std::vector<int32> indices;
std::vector<int64> ids;
std::vector<float> weights;

int32 *AllocateIndices(int n) {
  indices.clear();
  indices.resize(n);
  return indices.data();
}

int64 *AllocateIds(int n) {
  ids.clear();
  ids.resize(n);
  return ids.data();
}

float *AllocateWeights(int n) {
  weights.clear();
  weights.resize(n);
  return weights.data();
}

int main(int argc, char **argv) {
  sling::InitProgram(&argc, &argv);

  CHECK(!FLAGS_spec.empty());
  CHECK(!FLAGS_documents.empty());

  LOG(INFO) << "Reading spec from " << FLAGS_spec;
  string contents;
  CHECK_OK(File::ReadContents(FLAGS_spec, &contents));

  MasterSpec spec;
  CHECK(TextFormat::ParseFromString(contents, &spec));

  std::vector<Component *> components;
  string global_store_path;
  string action_table_path;
  for (const auto &component_spec : spec.component()) {
    LOG(INFO) << "Making/initializing " << component_spec.name();
    auto *component =
        Component::Create(component_spec.backend().registered_name());
    component->InitializeComponent(component_spec);
    components.emplace_back(component);
    if (global_store_path.empty()) {
      global_store_path = SemparFeature::GetResource(component_spec, "commons");
    }
    if (action_table_path.empty()) {
      action_table_path =
          SemparFeature::GetResource(component_spec, "action-table");
    }
  }
  LOG(INFO) << "Made & initialized " << components.size() << " components";

  CHECK(!global_store_path.empty());
  CHECK(!action_table_path.empty());
  SharedResources resources;
  resources.LoadGlobalStore(global_store_path);
  resources.LoadActionTable(action_table_path);

  std::vector<string> document_files;
  CHECK_OK(File::Match(FLAGS_documents, &document_files));

  std::vector<string> kept_files;
  std::vector<string> kept_text;
  for (const string &file : document_files) {
    Store local(resources.global);
    FileDecoder decoder(&local, file);
    Object top = decoder.Decode();
    if (!top.invalid()) {
      kept_files.emplace_back(file);
      Document doc(top.AsFrame());
      kept_text.emplace_back(doc.PhraseText(0, doc.num_tokens()));
    }
  }
  LOG(INFO) << "Will process " << kept_files.size() << " docs out of "
            << document_files.size() << " input files";

  for (int i = 0; i < kept_files.size(); ++i) {
    LOG(INFO) << string(80, '*') << "\n";
    LOG(INFO) << "Processing doc: " << kept_text[i];
    string contents;
    CHECK_OK(File::ReadContents(kept_files[i], &contents));

    std::vector<string> input;
    input.push_back(contents);
    InputBatchCache input_data(input);

    int cidx = 0;
    for (Component *c : components) {
      SemparComponent *sc = static_cast<SemparComponent *>(c);
      CHECK(sc != nullptr);
      sc->InitializeData({} /* parent states */,
                         1 /* beam size */,
                         &input_data);
      CHECK(sc->IsReady());
      sc->InitializeTracing();

      const auto &cproto = spec.component(cidx);
      string name = StrCat(cproto.name(), " (shift_only=",
                           sc->shift_only(), ", left_to_right=",
                           sc->left_to_right(), ")");

      LOG(INFO) << "Component: " << name;
      while (!sc->IsTerminal()) {
        for (int channel = 0;
             channel < cproto.fixed_feature_size();
             ++channel) {
          int fixed_features = sc->GetFixedFeatures(AllocateIndices,
                                                    AllocateIds,
                                                    AllocateWeights,
                                                    channel);
          LOG(INFO) << fixed_features << " fixed features for channel "
                    << cproto.fixed_feature(channel).fml();
        }
        for (int channel = 0;
             channel < cproto.linked_feature_size();
             ++channel) {
          auto linked_features = sc->GetRawLinkFeatures(channel);
          LOG(INFO) << linked_features.size() << " linked features for channel "
                    << cproto.linked_feature(channel).fml();
          sc->AddTranslatedLinkFeaturesToTrace(linked_features, channel);
        }

        std::vector<std::vector<int>> oracles = c->GetOracleLabels();
        CHECK_EQ(oracles.size(), 1);
        CHECK_EQ(oracles[0].size(), 1);

        if (!sc->shift_only()) {
          const ParserAction &action = resources.table.Action(oracles[0][0]);
          LOG(INFO) << "Next gold : " << action.ToString(resources.global);
        } else {
          CHECK_EQ(oracles[0][0], 0);
          LOG(INFO) << "Next gold : SHIFT (shift-only system)";
        }
        c->AdvanceFromOracle();
      }
      const auto trace_protos = sc->GetTraceProtos();
      CHECK_EQ(trace_protos.size(), 1);
      CHECK_EQ(trace_protos[0].size(), 1);
      LOG(INFO) << "Trace proto: " << trace_protos[0][0].DebugString();
      cidx++;
    }
  }

  for (Component *c : components) delete c;

  return 0;
}
