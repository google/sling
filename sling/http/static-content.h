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

#ifndef SLING_HTTP_STATIC_CONTENT_H_
#define SLING_HTTP_STATIC_CONTENT_H_

#include <string>

#include "sling/base/types.h"
#include "sling/http/http-server.h"

namespace sling {

// HTTP handler for serving static web content.
class StaticContent {
 public:
  // Initialize handler for serving files from a directory.
  StaticContent(const string &url, const string &path);

  // Register handler with HTTP server.
  void Register(HTTPServer *http);

  // Serve static web content from directory.
  void HandleFile(HTTPRequest *request, HTTPResponse *response);

 private:
  // URL path for static content.
  string url_;

  // Directory with static web content to be served.
  string dir_;
};

}  // namespace sling

#endif  // SLING_HTTP_STATIC_CONTENT_H_

