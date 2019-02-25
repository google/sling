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

#include "sling/http/static-content.h"

#include <string.h>
#include <time.h>
#include <string>

#include "sling/base/flags.h"
#include "sling/base/status.h"
#include "sling/file/file.h"
#include "sling/http/http-server.h"

// Use internal embedded file system for web content by default.
DEFINE_string(webdir, "/intern", "Base directory for serving web contents");
DEFINE_bool(webcache, true, "Enable caching of web content");

namespace sling {

// File extension to MIME type mapping.
struct MIMEMapping {
  const char *ext;
  const char *mime;
};

static const MIMEMapping mimetypes[] = {
  {"html", "text/html; charset=utf-8"},
  {"htm", "text/html; charset=utf-8"},
  {"xml", "text/xml; charset=utf-8"},
  {"jpeg", "image/jpeg"},
  {"jpg", "image/jpeg"},
  {"gif", "image/gif"},
  {"png", "image/png"},
  {"ico", "image/x-icon"},
  {"ttf", "font/ttf"},
  {"css", "text/css; charset=utf-8"},
  {"svg", "image/svg+xml; charset=utf-8"},
  {"js", "text/javascript; charset=utf-8"},
  {"zip", "application/zip"},
  {nullptr, nullptr},
};

// Find MIME type from extension.
static const char *GetMimeType(const char *ext) {
  if (ext == nullptr) return nullptr;
  for (const MIMEMapping *m = mimetypes; m->ext; ++m) {
    if (strcmp(ext, m->ext) == 0) return m->mime;
  }
  return nullptr;
}

// Get extension for file name.
static const char *GetExtension(const char *filename) {
  const char *ext = nullptr;
  for (const char *p = filename; *p; ++p) {
    if (*p == '/') {
      ext = nullptr;
    } else if (*p == '.') {
      ext = p + 1;
    }
  }
  return ext;
}

// Check if path is valid, especially that the path is not a relative path and
// does not contain any parent directory parts (..) that could escape the base
// directory.
static bool IsValidPath(const char *filename) {
  const char *p = filename;
  if (*p != '/' && *p != 0) return false;
  while (*p != 0) {
    if (p[0] == '.' && p[1] == '.') {
      if (p[2] == 0 || p[2] == '/') return false;
    }
    while (*p != 0 && *p != '/') p++;
    while (*p == '/') p++;
  }
  return true;
}

// Convert time to RFC date format.
static char *RFCTime(time_t t, char *buf) {
  struct tm tm;
  gmtime_r(&t, &tm);
  strftime(buf, 31, "%a, %d %b %Y %H:%M:%S GMT", &tm);
  return buf;
}

// Parse RFC date as time stamp.
static time_t ParseRFCTime(const char *timestr) {
  struct tm tm;
  if (strptime(timestr, "%a, %d %b %Y %H:%M:%S GMT", &tm) != nullptr) {
    return timegm(&tm);
  } else {
    return -1;
  }
}

StaticContent::StaticContent(const string &url, const string &path)
    : url_(url) {
  // Use configured directory for web content.
  dir_ = FLAGS_webdir;

  // Default to current directory.
  if (dir_.empty()) dir_ = ".";
  if (url_ == "/") url_ = "";

  // Add path for content.
  if (!path.empty() && path != "/") {
    dir_.push_back('/');
    dir_.append(path);
  }
  VLOG(3) << "Serve url " << url << " from " << dir_;
}

void StaticContent::Register(HTTPServer *http) {
  http->Register(url_, this, &StaticContent::HandleFile);
}

void StaticContent::HandleFile(HTTPRequest *request, HTTPResponse *response) {
  // Only GET and HEAD methods allowed.
  bool get_request = strcmp(request->method(), "GET") == 0;
  bool head_request = strcmp(request->method(), "HEAD") == 0;
  if (!get_request && !head_request) {
    response->SendError(405, "Method Not Allowed", nullptr);
    return;
  }

  // Get path.
  string path;
  if (!DecodeURLComponent(request->path(), &path)) {
    response->SendError(400, "Bad Request", nullptr);
    return;
  }

  // Check that path is valid.
  if (!IsValidPath(path.c_str())) {
    LOG(WARNING) << "Invalid request path: " << request->path();
    response->SendError(403, "Forbidden", nullptr);
    return;
  }

  // Remove trailing slash from file name.
  string filename = dir_ + path;
  VLOG(5) << "url: " << request->path() << " file: " << filename;
  bool trailing_slash = false;
  if (filename.back() == '/') {
    filename.pop_back();
    trailing_slash = true;
  }

  // Get file information.
  FileStat stat;
  Status st = File::Stat(filename, &stat);
  if (!st.ok()) {
    if (st.code() == EACCES) {
      response->SendError(403, "Forbidden", nullptr);
    } else if (st.code() == ENOENT) {
      response->SendError(404, "Not Found", nullptr);
    } else {
      string error = HTMLEscape(st.message());
      response->SendError(500, "Internal Server Error", error.c_str());
    }
    return;
  }

  // Redirect to index page for directory.
  if (stat.is_directory) {
    // Redirect to directory with slash if needed.
    if (!trailing_slash) {
      string dir = url_;
      dir.push_back('/');
      if (strlen(request->path()) > 0) {
        dir.append(request->path());
        dir.push_back('/');
      }
      response->RedirectTo(dir.c_str());
      return;
    }

    // Return index page for directory.
    filename.append("/index.html");
    st = File::Stat(filename, &stat);
    if (!st.ok() || stat.is_directory) {
      response->SendError(403, "Forbidden", "Directory browsing not allowed");
      return;
    }
  } else {
    // Regular files cannot have a trailing slash.
    if (trailing_slash) {
      response->SendError(404, "Not Found", nullptr);
      return;
    }
  }

  // Check if file has changed.
  const char *cached = request->Get("If-modified-since");
  const char *control = request->Get("Cache-Control");
  bool refresh = control != nullptr && strcmp(control, "maxage=0") == 0;
  if (!refresh && cached) {
    if (ParseRFCTime(cached) == stat.mtime) {
      response->set_status(304);
      response->SetContentLength(0);
      return;
    }
  }

  // Set content type from file extension.
  const char *mimetype = GetMimeType(GetExtension(filename.c_str()));
  if (mimetype != nullptr) {
    response->SetContentType(mimetype);
  }

  // Do not cache content if requested.
  if (!FLAGS_webcache) {
    response->Set("Cache-Control", "no-cache");
  } else {
    // Set file modified time.
    char datebuf[32];
    response->Set("Last-Modified", RFCTime(stat.mtime, datebuf));
  }

  // Do not return file content if only headers were requested.
  if (head_request) return;

  // Open requested file.
  File *file;
  st = File::Open(filename, "r", &file);
  if (!st.ok()) {
    if (st.code() == EACCES) {
      response->SendError(403, "Forbidden", nullptr);
    } else if (st.code() == ENOENT) {
      response->SendError(404, "Not Found", nullptr);
    } else {
      string error = HTMLEscape(st.message());
      response->SendError(500, "Internal Server Error", error.c_str());
    }
    return;
  }

  // Set content length to file size.
  response->SetContentLength(stat.size);

  // Return file contents.
  response->SendFile(file);
}

}  // namespace sling

