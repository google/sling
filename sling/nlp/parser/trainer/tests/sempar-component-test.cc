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

#include "dragnn/core/compute_session.h"
#include "dragnn/core/compute_session_pool.h"
#include "dragnn/core/interfaces/component.h"
#include "dragnn/protos/spec.pb.h"
#include "sling/base/flags.h"
#include "sling/base/init.h"
#include "sling/base/logging.h"
#include "sling/base/macros.h"
#include "sling/file/file.h"
#include "sling/nlp/document/document.h"
#include "sling/nlp/document/document-source.h"
#include "sling/nlp/parser/parser-action.h"
#include "sling/nlp/parser/trainer/sempar-component.h"
#include "sling/nlp/parser/trainer/shared-resources.h"
#include "sling/string/strcat.h"
#include "tensorflow/core/platform/protobuf.h"

using sling::File;
using sling::nlp::Document;
using sling::nlp::DocumentSource;
using sling::nlp::SharedResources;

using syntaxnet::dragnn::ComputeSession;
using syntaxnet::dragnn::ComputeSessionPool;
using syntaxnet::dragnn::GridPoint;
using syntaxnet::dragnn::MasterSpec;

using tensorflow::protobuf::TextFormat;

DEFINE_string(spec, "/tmp/sempar_out/master_spec", "Path to master spec.");
DEFINE_string(documents, "/tmp/foobar/doc.?", "Train documents file pattern.");
DEFINE_int32(num_documents, 10, "Number of training documents to process.");

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

  string global_store_path;
  string action_table_path;
  for (const auto &component_spec : spec.component()) {
    for (const auto &r : component_spec.resource()) {
      if (r.name() == "commons") {
        global_store_path = r.part(0).file_pattern();
      } else if (r.name() == "action-table") {
        action_table_path = r.part(0).file_pattern();
      }
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
    for (const auto &c: spec.component()) {
      int actions = 0;
      const string &name = c.name();
      session->InitializeComponentData(name, false);
      while (!session->IsTerminal(name)) {
        int batch_size = session->BatchSize(name);
        for (int i = 0; i < c.linked_feature_size(); ++i) {
          int size = batch_size * c.linked_feature(i).size();
          int *steps = new int[size];
          int *batch = new int[size];
          session->GetTranslatedLinkFeatures(name, i, size, steps, batch);
          delete steps;
          delete batch;
        }
        for (int i = 0; i < c.fixed_feature_size(); ++i) {
          int64 *output = new int64[batch_size * c.fixed_feature(i).size()];
          session->GetInputFeatures(name, i, output);
          delete output;
        }
        session->AdvanceFromOracle(name);
        actions++;
      }
      LOG(INFO) << "  " << name << " : " << actions << " actions";
    }
  }
  delete corpus;

  return 0;
}
