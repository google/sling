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

#include "sling/nlp/parser/trainer/shared-resources.h"

#include "sling/base/macros.h"
#include "sling/file/file.h"
#include "sling/frame/serialization.h"

namespace sling {
namespace nlp {

void SharedResources::LoadActionTable(const string &file) {
  CHECK(global != nullptr);
  Store temp(global);
  sling::LoadStore(file, &temp);
  table.Init(&temp);
  roles.Init(table);
}

void SharedResources::LoadGlobalStore(const string &file) {
  delete global;
  global = new Store();
  sling::LoadStore(file, global);
  global->Freeze();
}

void SharedResources::Load(const syntaxnet::dragnn::ComponentSpec &spec) {
  bool commons = false;
  for (const auto &r : spec.resource()) {
    if (r.name() == "commons") {
      LoadGlobalStore(r.part(0).file_pattern());
      commons = true;
      break;
    }
  }

  for (const auto &r : spec.resource()) {
    string file = r.part(0).file_pattern();
    string contents;
    if (r.name() == "action-table") {
      CHECK(commons);
      LoadActionTable(file);
      break;
    } else if (r.name() == "word-vocab") {
      CHECK(File::ReadContents(file, &contents));
      lexicon.InitWords(contents.c_str(), contents.size());
      const auto &p = spec.transition_system().parameters();
      CHECK_NE(p.count("lexicon_normalize_digits"), 0) << spec.DebugString();
      CHECK_NE(p.count("lexicon_oov"), 0) << spec.DebugString();
      lexicon.set_normalize_digits(p.at("lexicon_normalize_digits") == "true");
      lexicon.set_oov(std::stoi(p.at("lexicon_oov")));
    } else if (r.name() == "prefix-table") {
      CHECK(File::ReadContents(file, &contents));
      lexicon.InitPrefixes(contents.c_str(), contents.size());
    } else if (r.name() == "suffix-table") {
      CHECK(File::ReadContents(file, &contents));
      lexicon.InitSuffixes(contents.c_str(), contents.size());
    }
  }
}

}  // namespace nlp
}  // namespace sling
