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

// Input arguments:
// - Path to a commons store.
// - File pattern of gold documents
// - File pattern of test documents
//
// Sample usage:
//   bazel-bin/nlp/parser/trainer/frame-evaluation-main
//       --gold_documents='/tmp/eval/gold.*'
//       --test_documents='/tmp/eval/gold.*'
//       --commons=/tmp/common_store.encoded

#include <iostream>
#include <string>
#include <vector>

#include "base/flags.h"
#include "base/init.h"
#include "base/logging.h"
#include "base/macros.h"
#include "nlp/parser/trainer/frame-evaluation.h"

DEFINE_string(gold_documents, "", "File pattern of gold documents.");
DEFINE_string(test_documents, "", "File pattern of test documents.");
DEFINE_string(commons, "", "Path to common store.");

using sling::nlp::FrameEvaluation;

int main(int argc, char **argv) {
  sling::InitProgram(&argc, &argv);

  CHECK(!FLAGS_gold_documents.empty()) << "No gold documents specified.";
  CHECK(!FLAGS_test_documents.empty()) << "No test documents specified.";
  CHECK(!FLAGS_commons.empty()) << "No commons specified.";

  std::vector<string> summary = FrameEvaluation::EvaluateAndSummarize(
      FLAGS_commons, FLAGS_gold_documents, FLAGS_test_documents);
  for (const string &line : summary) {
    std::cout << line << "\n";
  }

  return 0;
}
