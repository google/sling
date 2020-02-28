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
#include "sling/file/recordio.h"
#include "sling/frame/serialization.h"
#include "sling/http/http-server.h"
#include "sling/http/static-content.h"
#include "sling/http/web-service.h"
#include "sling/nlp/document/annotator.h"
#include "sling/nlp/document/document.h"
#include "sling/nlp/document/document-service.h"
#include "sling/nlp/kb/knowledge-service.h"
#include "sling/util/mutex.h"

DEFINE_int32(port, 8080, "HTTP server port");
DEFINE_string(commons, "", "Commons store");
DEFINE_bool(kb, false, "Start knowledge base browser");
DEFINE_string(names, "local/data/e/wiki/en/name-table.repo", "Name table");
DEFINE_string(spec, "", "Document analyzer specification");

using namespace sling;
using namespace sling::nlp;

class CorpusBrowser : public DocumentService {
 public:
  CorpusBrowser(Store *commons,
                RecordDatabase *db,
                DocumentAnnotation *annotators)
      : DocumentService(commons), db_(db), annotators_(annotators) {}

  // Register service.
  void Register(HTTPServer *http) {
    http->Register("/fetch", this, &CorpusBrowser::HandleFetch);
    http->Register("/forward", this, &CorpusBrowser::HandleForward);
    http->Register("/back", this, &CorpusBrowser::HandleBack);
    app_content_.Register(http);
    common_content_.Register(http);
  }

  void HandleFetch(HTTPRequest *request, HTTPResponse *response) {
    WebService ws(commons_, request, response);
    Text docid = ws.Get("docid");
    if (docid.empty()) {
      response->SendError(400, nullptr, "docid missing");
      return;
    }

    // Fetch document record from database.
    Record record;
    if (!FetchRecord(docid, &record)) {
      response->SendError(404, nullptr, "unknown document");
      return;
    }

    ProcessRecord(&record, &ws);
  }

  void HandleForward(HTTPRequest *request, HTTPResponse *response) {
    WebService ws(commons_, request, response);

    // Fetch next document from database.
    Record record;
    if (!FetchNext(&record)) {
      response->SendError(400, nullptr, "no more documents");
      return;
    }


    ProcessRecord(&record, &ws);
  }

  void HandleBack(HTTPRequest *request, HTTPResponse *response) {
    WebService ws(commons_, request, response);

    // Fetch previous document from database.
    Record record;
    if (!FetchBackward(&record)) {
      response->SendError(400, nullptr, "no more documents");
      return;
    }

    ProcessRecord(&record, &ws);
  }

  bool FetchRecord(Text key, Record *record) {
    MutexLock lock(&mu_);
    if (db_->Lookup(key.slice(), record)) {
      history_.emplace_back(db_->current_shard(), record->position);
      return true;
    } else {
      return false;
    }
  }

  bool FetchNext(Record *record) {
    MutexLock lock(&mu_);
    if (db_->Next(record)) {
      history_.emplace_back(db_->current_shard(), record->position);
      return true;
    } else {
      return false;
    }
  }

  bool FetchBackward(Record *record) {
    MutexLock lock(&mu_);
    if (history_.empty()) return false;
    history_.pop_back();
    if (history_.empty()) return false;
    int shard = history_.back().first;
    int64 position = history_.back().second;
    return db_->Read(shard, position, record);
  }

  void ProcessRecord(Record *record, WebService *ws) {
    // Create document.
    Store *store = ws->store();
    Frame top = Decode(store, record->value).AsFrame();
    Document document(top);

    // Analyze document.
    annotators_->Annotate(&document);

    // Return document in JSON format.
    Frame json = Convert(document);
    ws->set_output(json);
  }

 private:
  // Record database with documents.
  RecordDatabase *db_;

  // History of records read from database.
  std::vector<std::pair<int, int64>> history_;

  // Document analyzer.
  DocumentAnnotation *annotators_;

  // Static web content.
  StaticContent app_content_{"/doc", "sling/nlp/document/app"};
  StaticContent common_content_{"/common", "app"};

  // Mutex for accessing database.
  Mutex mu_;
};

int main(int argc, char *argv[]) {
  InitProgram(&argc, &argv);

  // Get corpus files.
  std::vector<string> files;
  for (int i = 1; i < argc; ++i) {
    File::Match(argv[i], &files);
  }
  CHECK(!files.empty()) << "No document database files";

  // Open record database.
  RecordFileOptions recopts;
  RecordDatabase db(files, recopts);

  // Load document annotation pipeline.
  Store commons;
  LOG(INFO) << "Loading analyzer";
  DocumentAnnotation annotators;
  annotators.Init(&commons, FLAGS_spec);

  // Initialize knowledge base service.
  KnowledgeService kb;
  if (FLAGS_kb) kb.Load(&commons, FLAGS_names);

  // Initialize corpus browser.
  CorpusBrowser browser(&commons, &db, &annotators);
  commons.Freeze();

  LOG(INFO) << "Start HTTP server on port " << FLAGS_port;
  HTTPServerOptions httpopts;
  HTTPServer http(httpopts, FLAGS_port);

  browser.Register(&http);
  if (FLAGS_kb) kb.Register(&http);

  http.Register("/", [](HTTPRequest *req, HTTPResponse *rsp) {
    if (strcmp(req->path(), "/") == 0) {
      rsp->TempRedirectTo("/doc/corpus.html");
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
