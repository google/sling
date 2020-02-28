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
#include "sling/http/static-content.h"
#include "sling/http/web-service.h"
#include "sling/nlp/document/annotator.h"
#include "sling/nlp/document/document.h"
#include "sling/nlp/document/document-service.h"
#include "sling/nlp/kb/knowledge-service.h"

DEFINE_int32(port, 8080, "HTTP server port");
DEFINE_string(spec, "", "Document analyzer specification");
DEFINE_bool(kb, false, "Start knowledge base browser");
DEFINE_string(names, "local/data/e/wiki/en/name-table.repo", "Name table");

using namespace sling;
using namespace sling::nlp;

class Analyzer : public DocumentService {
 public:
  Analyzer(Store *commons, DocumentAnnotation *annotators)
    : DocumentService(commons), annotators_(annotators) {}

  // Register service.
  void Register(HTTPServer *http) {
    http->Register("/annotate", this, &Analyzer::HandleAnnotate);
    http->Register("/analyze", this, &Analyzer::HandleAnalyze);
    app_content_.Register(http);
    common_content_.Register(http);
  }

  // Annotate document and return analyzed document in DocView JSON format.
  void HandleAnnotate(HTTPRequest *request, HTTPResponse *response) {
    WebService ws(commons_, request, response);

    // Get input document.
    Document *document = GetInputDocument(&ws);
    if (document == nullptr) {
      response->SendError(400, "Bad Request", "document missing");
      return;
    }

    // Analyze document.
    annotators_->Annotate(document);

    // Return document in JSON format.
    Frame json = Convert(*document);
    ws.set_output(json);
    delete document;
  }

  // Annotate document and return analyzed document in requested format.
  void HandleAnalyze(HTTPRequest *request, HTTPResponse *response) {
    WebService ws(commons_, request, response);

    // Get input document.
    Document *document = GetInputDocument(&ws);
    if (document == nullptr) {
      response->SendError(400, "Bad Request", "document missing");
      return;
    }

    // Analyze document.
    annotators_->Annotate(document);

    // Return analyzed document.
    ws.set_output(document->top());
    delete document;
  }

 private:
  // Document analyzer.
  DocumentAnnotation *annotators_;

  // Static web content.
  StaticContent app_content_{"/doc", "sling/nlp/document/app"};
  StaticContent common_content_{"/common", "app"};
};

int main(int argc, char *argv[]) {
  InitProgram(&argc, &argv);

  // Load document annotation pipeline.
  Store commons;
  LOG(INFO) << "Loading analyzer";
  DocumentAnnotation annotators;
  annotators.Init(&commons, FLAGS_spec);

  // Initialize analyzer.
  Analyzer analyzer(&commons, &annotators);

  // Initialize knowledge base service.
  KnowledgeService kb;
  if (FLAGS_kb) kb.Load(&commons, FLAGS_names);

  commons.Freeze();

  LOG(INFO) << "Start HTTP server on port " << FLAGS_port;
  HTTPServerOptions httpopts;
  HTTPServer http(httpopts, FLAGS_port);

  analyzer.Register(&http);
  if (FLAGS_kb) kb.Register(&http);

  http.Register("/", [](HTTPRequest *req, HTTPResponse *rsp) {
    if (strcmp(req->path(), "/") == 0) {
      rsp->TempRedirectTo("/doc/analyzer.html");
    } else {
      rsp->SendError(404, "Not found", "file not found");
    }
  });

  http.Register("/favicon.ico", [](HTTPRequest *req, HTTPResponse *rsp) {
    rsp->RedirectTo("/common/image/appicon.ico");
  });

  CHECK(http.Start());

  LOG(INFO) << "HTTP server running";
  http.Wait();

  LOG(INFO) << "HTTP server done";
  return 0;
}
