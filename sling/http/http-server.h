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

#ifndef SLING_HTTP_HTTP_SERVER_H_
#define SLING_HTTP_HTTP_SERVER_H_

#include <netinet/in.h>
#include <string.h>
#include <time.h>
#include <atomic>
#include <string>

#include "sling/base/status.h"
#include "sling/base/types.h"
#include "sling/file/file.h"
#include "sling/http/http-utils.h"
#include "sling/util/mutex.h"
#include "sling/util/thread.h"

using namespace std::placeholders;

namespace sling {

// Forward declarations.
class HTTPConnection;
class HTTPRequest;
class HTTPResponse;
class HTTPHandler;

// HTTP connection states.
enum HTTPState {
  HTTP_STATE_IDLE,
  HTTP_STATE_READ_HEADER,
  HTTP_STATE_READ_BODY,
  HTTP_STATE_PROCESSING,
  HTTP_STATE_WRITE_HEADER,
  HTTP_STATE_WRITE_BODY,
  HTTP_STATE_WRITE_FILE,
  HTTP_STATE_TERMINATE,
};

// HTTP header parsing states.
enum HTTPHeaderState {
  HDR_STATE_FIRSTWORD,
  HDR_STATE_FIRSTWS,
  HDR_STATE_SECONDWORD,
  HDR_STATE_SECONDWS,
  HDR_STATE_THIRDWORD,
  HDR_STATE_LINE,
  HDR_STATE_LF,
  HDR_STATE_CR,
  HDR_STATE_CRLF,
  HDR_STATE_CRLFCR,
  HDR_STATE_DONE,
  HDR_STATE_BOGUS,
};

// HTTP server configuration.
struct HTTPServerOptions {
  // Number of worker threads.
  int num_workers = 5;

  // Maximum number of worker threads.
  int max_workers = 200;

  // Number of events per worker poll.
  int max_events = 1;

  // Maximum idle time (in seconds) before connection is shut down.
  int max_idle = 600;

  // Initial buffer size.
  int initial_bufsiz = 1 << 10;

  // File data buffer size.
  int file_bufsiz = 1 << 16;

  // HTTP server name reported in Server: header.
  string server_name = "HTTPServer/1.0";
};

// HTTP server.
class HTTPServer {
 public:
  // HTTP handler.
  typedef std::function<void(HTTPRequest *, HTTPResponse *)> Handler;

  // Initialize HTTP server to listen on port.
  HTTPServer(const HTTPServerOptions &options, int port);
  ~HTTPServer();

  // Register handler for requests.
  void Register(const string &uri, const Handler &handler);

  // Register method for handling requests.
  template<class T> void Register(
      const string &uri, T *object,
      void (T::*method)(HTTPRequest *, HTTPResponse *)) {
    Register(uri, std::bind(method, object, _1, _2));
  }

  // Find handler for request.
  Handler FindHandler(HTTPRequest *request) const;

  // Start HTTP server listening on the port.
  Status Start();

  // Wait for shutdown.
  void Wait();

  // Shut down HTTP server.
  void Shutdown();

  // Configuration options.
  const HTTPServerOptions &options() const { return options_; }

 private:
  // HTTP context for serving requests under an URI.
  struct Context {
    Context(const string &u, const Handler &h) : uri(u), handler(h) {
      if (uri == "/") uri = "";
    }
    string uri;
    Handler handler;
  };

  // Worker handler.
  void Worker();

  // Accept new connection.
  void AcceptConnection();

  // Process I/O events for connection.
  void Process(HTTPConnection *conn, int events);

  // Add connection to server.
  void AddConnection(HTTPConnection *conn);

  // Remove connection from server.
  void RemoveConnection(HTTPConnection *conn);

  // Shut down idle connections.
  void ShutdownIdleConnections();

  // Handler for /helpz.
  void HelpHandler(HTTPRequest *req, HTTPResponse *rsp);

  // Handler for /connz.
  void ConnectionHandler(HTTPRequest *req, HTTPResponse *rsp);

  // Server configuration.
  HTTPServerOptions options_;

  // Port to listen on.
  int port_;

  // Socket for accepting new connections.
  int sock_;

  // File descriptor for epoll.
  int pollfd_;

  // Mutex for serializing access to server state.
  mutable Mutex mu_;

  // Registered HTTP handlers.
  std::vector<Context> contexts_;

  // List of active HTTP connections.
  HTTPConnection *connections_;

  // Worker threads.
  WorkerPool workers_;

  // Number of active worker threads.
  std::atomic<int> active_{0};

  // Number of idle worker threads.
  std::atomic<int> idle_{0};

  // Flag to determine if server is shutting down.
  bool stop_;
};

// HTTP connection.
class HTTPConnection {
 public:
  // Initialize new HTTP connection on socket.
  HTTPConnection(HTTPServer *server, int sock);
  ~HTTPConnection();

  // Process I/O for connection.
  Status Process();

  // Parse header. Returns true when header has been parsed.
  bool ParseHeader();

  // Dispatch request to handler.
  void Dispatch();

  // Reset connection to idle.
  void Reset();

  // Return HTTP request information.
  HTTPRequest *request() const { return request_; }

  // Return HTTP response information.
  HTTPResponse *response() const { return response_; }

  // Server for HTTP connection.
  HTTPServer *server() const { return server_; }

  // Append data to response.
  void AppendResponse(const char *data, int size);

  // Set file for streaming response. This will take ownership of the file.
  void SendFile(File *file) { file_ = file; }

  // Request and response body buffers.
  HTTPBuffer *request_buffer() { return &input_; }
  HTTPBuffer *response_buffer() { return &response_body_; }

  // Return connection state name.
  const char *State() const;

 private:
  // Receive data into buffer until it is full or all data that can be received
  // without blocking has been received.
  Status Recv(HTTPBuffer *buffer, bool *done);

  // Send data from buffer until all data has been sent or all the data that can
  // be sent without blocking has been sent.
  Status Send(HTTPBuffer *buffer, bool *done);

  // Shut down connection.
  void Shutdown();

  // HTTP server for connection.
  HTTPServer *server_;

  // Socket for connection.
  int sock_;

  // Last time event was received on connection.
  time_t last_;

  // HTTP connection list.
  HTTPConnection *next_;
  HTTPConnection *prev_;

  // Current HTTP request for connection.
  HTTPRequest *request_;

  // Current HTTP response for connection.
  HTTPResponse *response_;

  // HTTP input buffer.
  HTTPBuffer input_;

  // Buffers for request/response header/body.
  HTTPBuffer request_header_;
  HTTPBuffer response_header_;
  HTTPBuffer response_body_;

  // Request parsing state.
  HTTPState state_;
  HTTPHeaderState header_state_;

  // Mutex for serializing access to connection state.
  Mutex mu_;

  // Whether to keep connection after current request.
  bool keep_;

  // File for streaming response.
  File *file_ = nullptr;

  friend class HTTPServer;
};

// HTTP request.
class HTTPRequest {
 public:
  HTTPRequest(HTTPConnection *conn, HTTPBuffer *hdr);

  // Is this a valid HTTP request?
  bool valid() const { return valid_; }

  // Is this a HTTP/1.1 request?
  bool http11() const { return http11_; }

  // HTTP method.
  const char *method() const { return method_; }

  // HTTP URL path.
  const char *full_path() const { return full_path_; }
  const char *path() const { return path_; }
  void set_path(const char *path) { path_ = path; }

  // HTTP URL query.
  const char *query() const { return query_; }

  // HTTP URL fragment.
  const char *fragment() const { return fragment_; }

  // HTTP protocol.
  const char *protocol() const { return protocol_; }

  // HTTP content type.
  const char *content_type() const { return content_type_; }

  // HTTP content length or -1 if missing.
  int content_length() const { return content_length_; }

  // HTTP keep-alive flag.
  bool keep_alive() const { return keep_alive_; }

  // Get HTTP header.
  const char *Get(const char *name, const char *defval = nullptr) const;

  // HTTP request headers.
  const std::vector<HTTPHeader> &headers() const { return headers_; }

  // HTTP request body buffer.
  HTTPBuffer *buffer() { return conn_->request_buffer(); }

  // HTTP request body content.
  const char *content() const { return content_; }
  int content_size() const { return content_size_; }
  void set_content(const char *content, int size) {
   content_ = content;
   content_size_ = size;
  }

 private:
  // HTTP connect for request.
  HTTPConnection *conn_;

  // Is HTTP request valid?
  bool valid_ = false;

  // Is this a HTTP/1.1 request?
  bool http11_ = false;

  // HTTP method.
  const char *method_ = nullptr;

  // HTTP URI full path (including context).
  const char *full_path_ = nullptr;

  // HTTP URI path.
  const char *path_ = nullptr;

  // HTTP URI query.
  const char *query_ = nullptr;

  // HTTP URI fragment.
  const char *fragment_ = nullptr;

  // HTTP protocol.
  const char *protocol_ = nullptr;

  // Standard HTTP request headers.
  const char *content_type_ = nullptr;
  int content_length_ = -1;
  bool keep_alive_ = false;

  // HTTP request headers.
  std::vector<HTTPHeader> headers_;

  // HTTP request body.
  const char *content_ = nullptr;
  int content_size_ = 0;
};

// HTTP response.
class HTTPResponse {
 public:
  HTTPResponse(HTTPConnection *conn) : conn_(conn) {}
  ~HTTPResponse();

  // HTTP status code.
  int status() const { return status_; }
  void set_status(int status) { status_ = status; }

  // HTTP content type.
  const char *ContentType() const;
  void SetContentType(const char *type);

  // HTTP content length.
  int ContentLength() const;
  void SetContentLength(int length);

  // Get response header. Returns null if header is not set.
  const char *Get(const char *name, const char *defval = nullptr) const;

  // Set response header.
  void Set(const char *name, const char *value, bool overwrite = true);

  // HTTP response headers.
  const std::vector<HTTPHeader> &headers() const { return headers_; }

  // Append data to response.
  void Append(const char *data, int size) {
    conn_->AppendResponse(data, size);
  }
  void Append(const char *str) { if (str) Append(str, strlen(str)); }
  void Append(const string &str) { Append(str.data(), str.size()); }

  // Write HTTP header to buffer.
  void WriteHeader(HTTPBuffer *rsp);

  // Set file for streaming response. This will take ownership of the file.
  void SendFile(File *file) { conn_->SendFile(file); }

  // Return HTTP error message.
  void SendError(int status, const char *title, const char *msg);

  // Permanent redirect to another URL.
  void RedirectTo(const char *uri);

  // Temporary redirect to another URL.
  void TempRedirectTo(const char *uri);

  // HTTP response body buffer.
  HTTPBuffer *buffer() { return conn_->response_buffer(); }

 private:
  // HTTP connect for request.
  HTTPConnection *conn_;

  // HTTP status code for response.
  int status_ = 200;

  // HTTP response headers.
  std::vector<HTTPHeader> headers_;
};

}  // namespace sling

#endif  // SLING_HTTP_HTTP_SERVER_H_

