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

// Utility to make and write a single test document.
// Also writes a tokens-only version of the document.
//
// Change main() to modify the test document being output.

#include <string>
#include <tuple>
#include <vector>

#include "base/flags.h"
#include "base/init.h"
#include "base/logging.h"
#include "base/macros.h"
#include "file/file.h"
#include "frame/object.h"
#include "frame/serialization.h"
#include "frame/store.h"
#include "nlp/document/document.h"
#include "nlp/document/token-breaks.h"
#include "nlp/parser/trainer/shared-resources.h"
#include "string/strcat.h"

using sling::File;
using sling::Frame;
using sling::Handle;
using sling::Store;
using sling::StrCat;
using sling::nlp::Document;
using sling::nlp::SharedResources;

DEFINE_string(commons, "/usr/local/google/home/grahul/sempar_ontonotes/commons",
              "Path to commons");
DEFINE_string(output, "",
              "Output file for writing the document. "
              "Tokens-only version is written to <FLAGS_output>-only-tokens.");

// Makes and returns a document using the supplied arguments:
// 'local': Store to create the document in.
// 'tokens': Ordered list of token strings.
// 'spans': [begin, end) token indices of spans.
// 'evoked': string ids of types of frames evoked by the spans above.
//           Should be the same length as 'spans'.
// 'links': (source frame index, target frame index, role id) tuples, one for
//          each link. Source/target frame indices should be into 'evoked'.
Document *MakeDocument(Store *local,
                       const std::vector<string> &tokens,
                       const std::vector<std::pair<int, int>> &spans,
                       const std::vector<string> &evoked,
                       const std::vector<std::tuple<int, int, string>> &links) {
  CHECK(local->globals() != nullptr);
  CHECK(local->globals()->frozen());
  CHECK_EQ(evoked.size(), spans.size());

  Document *doc = new Document(local);

  int offset = 0;
  for (const string &t : tokens) {
    doc->AddToken(
        offset, offset + t.size(), t,
        offset == 0 ? sling::nlp::NO_BREAK : sling::nlp::SPACE_BREAK);
    offset += t.size() + 1;
  }
  doc->Update();
  int index = 0;
  std::vector<Frame> frames;
  for (const auto &s : spans) {
    CHECK_LT(s.first, tokens.size()) << "Illegal span begin token";
    CHECK_LE(s.second, tokens.size()) << "Illegal span end token";
    sling::nlp::Span *span = doc->AddSpan(s.first, s.second);
    CHECK(span != nullptr) << "Can't add span " << s.first << ", " << s.second;

    Handle type = local->LookupExisting(evoked[index]);
    CHECK(!type.IsNil()) << "Unknown frame type: " << evoked[index];
    sling::Builder b(local);
    b.AddIsA(type);
    span->Evoke(b.Create());

    CHECK(!span->Evoked(type).IsNil()) << evoked[index];
    frames.emplace_back(span->Evoked(type));
    index++;
  }
  for (const auto &l : links) {
    CHECK_LT(std::get<0>(l), evoked.size()) << "Bad source frame index";
    CHECK_LT(std::get<1>(l), evoked.size()) << "Bad target frame index";
    Handle role = local->LookupExisting(std::get<2>(l));
    CHECK(!role.IsNil()) << "Unknown role: " << std::get<2>(l);
    frames[std::get<0>(l)].Add(role, frames[std::get<1>(l)]);
  }
  doc->Update();

  return doc;
}

Document *MakeDocument(Store *local, const std::vector<string> &tokens) {
  return MakeDocument(local, tokens, {}, {}, {});
}

void Write(const string &file, Document *doc) {
  string contents = sling::Encode(doc->top());
  CHECK(File::WriteContents(file, contents));
  LOG(INFO) << "Wrote to " << file << "\n" << sling::ToText(doc->top(), 2);
  delete doc;
}

int main(int argc, char **argv) {
  sling::InitProgram(&argc, &argv);

  CHECK(!FLAGS_commons.empty());
  CHECK(!FLAGS_output.empty());

  SharedResources resources;
  resources.LoadGlobalStore(FLAGS_commons);

  Store local(resources.global);
  std::vector<string> tokens = {"John", "lived", "in", "London", "."};
  Document *doc = MakeDocument(
      &local,
      tokens,
      {{0, 1}, {1, 2}, {3, 4}},
      {"/saft/person", "/pb/live-01", "/saft/location"},
      {
        std::make_tuple(1, 0, "/pb/arg0"),
        std::make_tuple(1, 2, "/pb/argm-loc")
      });
  Write(FLAGS_output, doc);

  Document *tokens_doc = MakeDocument(&local, tokens);
  Write(StrCat(FLAGS_output, "-only-tokens"), tokens_doc);

  return 0;
}
