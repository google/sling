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
#include "dragnn/core/compute_session.h"
#include "dragnn/core/compute_session_pool.h"
#include "dragnn/core/interfaces/component.h"
#include "dragnn/protos/spec.pb.h"
#include "file/file.h"
#include "nlp/document/document.h"
#include "nlp/document/document-source.h"
#include "nlp/parser/parser-action.h"
#include "nlp/parser/trainer/feature.h"
#include "nlp/parser/trainer/sempar-component.h"
#include "nlp/parser/trainer/shared-resources.h"
#include "string/strcat.h"
#include "tensorflow/core/platform/protobuf.h"

using sling::File;
using sling::StrCat;
using sling::nlp::Document;
using sling::nlp::DocumentSource;
using sling::nlp::ParserAction;
using sling::nlp::SemparComponent;
using sling::nlp::SemparFeature;
using sling::nlp::SharedResources;

using syntaxnet::dragnn::Component;
using syntaxnet::dragnn::ComponentSpec;
using syntaxnet::dragnn::ComputeSession;
using syntaxnet::dragnn::ComputeSessionPool;
using syntaxnet::dragnn::GridPoint;
using syntaxnet::dragnn::InputBatchCache;
using syntaxnet::dragnn::MasterSpec;

using tensorflow::protobuf::TextFormat;

DEFINE_string(spec, "/tmp/sempar_out/master_spec", "Path to master spec.");
DEFINE_string(documents, "/tmp/foobar/doc.?", "Train documents file pattern.");
DEFINE_int32(num_documents, 10, "Number of training documents to process.");

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
  CHECK(File::ReadContents(FLAGS_spec, &contents));

  MasterSpec spec;
  CHECK(TextFormat::ParseFromString(contents, &spec));

  GridPoint empty_grid_point;
  ComputeSessionPool pool(spec, empty_grid_point);
  auto session = pool.GetSession();
  session->SetTracing(true);

  string global_store_path;
  string action_table_path;
  for (const auto &component_spec : spec.component()) {
    LOG(INFO) << "Making/initializing " << component_spec.name();
    if (global_store_path.empty()) {
      global_store_path = SemparFeature::GetResource(component_spec, "commons");
    }
    if (action_table_path.empty()) {
      action_table_path =
          SemparFeature::GetResource(component_spec, "action-table");
    }
  }

  CHECK(!global_store_path.empty());
  CHECK(!action_table_path.empty());
  SharedResources resources;
  resources.LoadGlobalStore(global_store_path);
  resources.LoadActionTable(action_table_path);

  DocumentSource *corpus = DocumentSource::Create(FLAGS_documents);
  for (int i = 0; i < FLAGS_num_documents; ++i) {
    std::vector<string> input;
    string name;
    input.emplace_back();
    if (!corpus->NextSerialized(&name, &input.back())) break;

    LOG(INFO) << "Processing : " << name;
    session->SetInputData(input);
    for (int cidx = 0; cidx < spec.component_size(); ++cidx) {
      const string &name = spec.component(cidx).name();
      session->InitializeComponentData(name, 1 /* max beam size */);
      while (!session->IsTerminal(name)) {
        for (int c = 0; c < spec.component(cidx).linked_feature_size(); ++c) {
          session->GetTranslatedLinkFeatures(name, c);
        }
        for (int c = 0; c < spec.component(cidx).fixed_feature_size(); ++c) {
          session->GetInputFeatures(
              name, AllocateIndices, AllocateIds, AllocateWeights, c);
        }
        session->AdvanceFromOracle(name);
      }
    }

    const auto trace_protos = session->GetTraceProtos();
    CHECK_EQ(trace_protos.size(), 1);
    LOG(INFO) << "Trace proto: " << trace_protos[0].DebugString();
  }
  delete corpus;

  return 0;
}
