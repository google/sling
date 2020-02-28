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

#include "sling/base/flags.h"
#include "sling/base/init.h"
#include "sling/base/logging.h"
#include "sling/frame/serialization.h"
#include "sling/http/http-server.h"
#include "sling/nlp/kb/knowledge-service.h"

DEFINE_int32(port, 8080, "HTTP server port");
DEFINE_string(kb, "local/data/e/wiki/kb.sling", "Knowledge base");
DEFINE_string(names, "local/data/e/wiki/en/name-table.repo", "Name table");

using namespace sling;
using namespace sling::nlp;

int main(int argc, char *argv[]) {
  InitProgram(&argc, &argv);

  LOG(INFO) << "Loading knowledge base from " << FLAGS_kb;
  Store commons;
  LoadStore(FLAGS_kb, &commons);

  LOG(INFO) << "Start HTTP server on port " << FLAGS_port;
  HTTPServerOptions options;
  HTTPServer http(options, FLAGS_port);

  KnowledgeService kb;
  kb.Load(&commons, FLAGS_names);
  commons.Freeze();

  kb.Register(&http);
  http.Register("/", [](HTTPRequest *req, HTTPResponse *rsp) {
    rsp->TempRedirectTo("/kb");
  });

  CHECK(http.Start());

  LOG(INFO) << "HTTP server running";
  http.Wait();

  LOG(INFO) << "HTTP server done";
  return 0;
}

