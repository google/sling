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

// Utility tool for generating an action table.
// Sample usage:
//   bazel-bin/nlp/parser/trainer/generate-action-table
//       --documents='/tmp/documents.*'
//       --commons=/tmp/common_store.encoded
//       --output_prefix='/tmp/out'
//
// This will create the table at /tmp/out.table, its summary at /tmp/out.summary
// and a list of unknown symbols at /tmp/out.unknown.

#include <string>
#include <vector>

#include "base/flags.h"
#include "base/init.h"
#include "base/logging.h"
#include "base/macros.h"
#include "file/file.h"
#include "frame/object.h"
#include "frame/serialization.h"
#include "frame/store.h"
#include "nlp/parser/trainer/action-table-generator.h"
#include "string/strcat.h"

using sling::File;
using sling::FileDecoder;
using sling::Object;
using sling::Store;
using sling::nlp::ActionTableGenerator;
using sling::nlp::Document;

DEFINE_string(documents, "", "File pattern of documents.");
DEFINE_string(commons, "", "Path to common store.");
DEFINE_string(output_prefix,
              "/tmp/out",
              "Output prefix for action table, summary etc.");

int main(int argc, char **argv) {
  sling::InitProgram(&argc, &argv);

  CHECK(!FLAGS_documents.empty()) << "No documents specified.";
  CHECK(!FLAGS_commons.empty()) << "No commons specified.";
  CHECK(!FLAGS_output_prefix.empty()) << "No output_prefix specified.";

  Store *global = new Store();
  sling::LoadStore(FLAGS_commons, global);

  ActionTableGenerator generator;
  generator.set_global_store(global);
  generator.set_coverage_percentile(99);
  generator.set_per_sentence(true);

  std::vector<string> files;
  CHECK_OK(File::Match(FLAGS_documents, &files));
  LOG(INFO) << "Processing " << files.size() << " documents..";
  int count = 0;
  for (const string &file : files) {
    Store local(global);
    FileDecoder decoder(&local, file);
    Object top = decoder.Decode();
    if (top.invalid()) continue;

    count++;
    Document document(top.AsFrame());
    generator.Add(document);
    if (count % 100 == 1) LOG(INFO) << count << " documents processed.";
  }
  LOG(INFO) << "Processed " << count << " documents.";

  string table_file = sling::StrCat(FLAGS_output_prefix, ".table");
  string summary_file = sling::StrCat(FLAGS_output_prefix, ".summary");
  string unknown_file = sling::StrCat(FLAGS_output_prefix, ".unknown");
  generator.Save(table_file, summary_file, unknown_file);

  LOG(INFO) << "Wrote action table to " << table_file
            << ", " << summary_file << ", " << unknown_file;
  delete global;

  return 0;
}
