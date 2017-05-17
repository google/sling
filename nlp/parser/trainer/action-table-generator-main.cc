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
//   bazel-bin/nlp/parser/trainer/action-table-generator-main \
//       '/tmp/documents.*' \
//       /tmp/common_store.encoded \
//       '/tmp/out.'
//
// This will create the table at /tmp/out.table, its summary at /tmp/out.summary
// and a list of unknown symbols at /tmp/out.unknown.

#include <string>
#include <vector>

#include "base/logging.h"
#include "base/macros.h"
#include "file/file.h"
#include "frame/object.h"
#include "frame/serialization.h"
#include "frame/store.h"
#include "nlp/parser/trainer/action-table-generator.h"
#include "string/strcat.h"

using sling::File;
using sling::Object;
using sling::Store;
using sling::nlp::ActionTableGenerator;
using sling::nlp::Document;

int main(int argc, char **argv) {
  File::Init();
  CHECK_GE(argc, 3)
      << "Usage: " << argv[0]
      << " <file pattern of documents> <common store> "
      << "[output prefix, e.g. /tmp/out.";

  string filepattern = argv[1];

  Store *global = new Store();
  sling::LoadStore(argv[2], global);

  ActionTableGenerator::Options options;
  options.per_sentence = true;
  options.coverage_percentile = 99;
  options.global = global;

  ActionTableGenerator generator(options);
  std::vector<string> files;
  CHECK_OK(File::Match(filepattern, &files));
  LOG(INFO) << "Processing " << files.size() << " documents..";
  int count = 0;
  for (const string &file : files) {
    Store local(global);
    string encoded_frame;
    CHECK_OK(File::ReadContents(file, &encoded_frame));
    Object top = sling::Decode(&local, encoded_frame);
    if (top.invalid()) continue;

    count++;
    Document document(top.AsFrame());
    generator.Add(document);
    if (count % 100 == 1) LOG(INFO) << count << " documents processed.";
  }
  LOG(INFO) << "Processed " << count << " documents.";

  string output_prefix = argv[3];
  if (output_prefix.empty()) output_prefix = "out";
  string table_file = sling::StrCat(output_prefix, ".table");
  string summary_file = sling::StrCat(output_prefix, ".summary");
  string unknown_file = sling::StrCat(output_prefix, ".unknown");
  generator.Save(table_file, summary_file, unknown_file);

  LOG(INFO) << "Wrote action table to " << table_file
            << ", " << summary_file << ", " << unknown_file;
  delete global;

  return 0;
}
