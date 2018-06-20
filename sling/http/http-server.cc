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

#include "sling/http/http-server.h"

#include <arpa/inet.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <stdlib.h>
#include <string.h>
#include <sys/epoll.h>
#include <sys/socket.h>
#include <unistd.h>
#include <time.h>

#include "sling/base/logging.h"
#include "sling/string/ctype.h"
#include "sling/string/numbers.h"

namespace sling {

namespace {

// Return system error.
Status Error(const char *context) {
  return Status(errno, context, strerror(errno));
}

// Returns value for ASCII hex digit.
int HexDigit(int c) {
  return (c <= '9') ? c - '0' : (c & 7) + 9;
}

// Convert time to RFC date format.
char *RFCTime(time_t t, char *buf) {
  struct tm tm;
  gmtime_r(&t, &tm);
  strftime(buf, 31, "%a, %d %b %Y %H:%M:%S GMT", &tm);
  return buf;
}

// Return text for HTTP status code.
const char *StatusText(int status) {
  switch (status) {
    case 200: return "OK";
    case 204: return "No Content";
    case 301: return "Moved Permanently";
    case 302: return "Moved";
    case 304: return "Not Modified";
    case 400: return "Bad Request";
    case 401: return "Not Authorized";
    case 403: return "Forbidden";
    case 404: return "Not Found";
    case 405: return "Method Not Allowed";
    case 500: return "Internal Server Error";
    case 501: return "Not Implemented";
    case 503: return "Service Unavailable";
    case 505: return "HTTP Version Not Supported";
  }

  return "Internal Error";
}

// Return 404 error.
void Handle404(HTTPRequest *request, HTTPResponse *response) {
  response->SetContentType("text/html");
  response->set_status(404);
  response->Append("<html><head>\n");
  response->Append("<title>404 Not Found</title>\n");
  response->Append("</head><body>\n");
  response->Append("<h1>Not Found</h1>\n");
  response->Append("<p>The requested URL ");
  response->Append(HTMLEscape(request->path()));
  response->Append(" was not found on this server.</p>\n");
  response->Append("</body></html>\n");
}

}  // namespace

// Decode URL component.
bool DecodeURLComponent(const char *url, int length, string *output) {
  const char *end = url + length;
  while (url < end) {
    char c = *url++;
    if (c == '%') {
      if (url + 2 >= end) return false;
      char x1 = *url++;
      if (!ascii_isxdigit(x1)) return false;
      char x2 = *url++;
      if (!ascii_isxdigit(x2)) return false;
      output->push_back((HexDigit(x1) << 4) + HexDigit(x2));
    } else {
      output->push_back(c);
    }
  }

  return true;
}

bool DecodeURLComponent(const char *url, string *output) {
  if (url == nullptr) return true;
  return DecodeURLComponent(url, strlen(url), output);
}

void HTTPBuffer::reset(int size) {
  if (size != capacity()) {
    if (size == 0) {
      free(floor);
      floor = ceil = start = end = nullptr;
    } else {
      floor = static_cast<char *>(realloc(floor, size));
      CHECK(floor != nullptr) << "Out of memory, " << size << " bytes";
      ceil = floor + size;
    }
  }
  start = end = floor;
}

string HTMLEscape(const char *text, int size) {
  string escaped;
  const char *p = text;
  const char *end = text + size;
  while (p < end) {
    char ch = *p++;
    switch (ch) {
      case '&':  escaped.append("&amp;"); break;
      case '<':  escaped.append("&lt;"); break;
      case '>':  escaped.append("&gt;"); break;
      case '"':  escaped.append("&quot;"); break;
      case '\'': escaped.append("&#39;");  break;
      default: escaped.push_back(ch);
    }
  }
  return escaped;
}

void HTTPBuffer::ensure(int minfree) {
  // Check if there is enough free space in buffer.
  if (ceil - end >= minfree) return;

  // Compute new size of buffer.
  int size = ceil - floor;
  int minsize = end + minfree - floor;
  while (size < minsize) {
    if (size == 0) {
      size = 1024;
    } else {
      size *= 2;
    }
  }

  // Expand buffer.
  char *p = static_cast<char *>(realloc(floor, size));
  CHECK(p != nullptr) << "Out of memory, " << size << " bytes";

  // Adjust pointers.
  start += p - floor;
  end += p - floor;
  floor = p;
  ceil = p + size;
}

void HTTPBuffer::clear() {
  free(floor);
  floor = ceil = start = end = nullptr;
}

char *HTTPBuffer::gets() {
  char *line = start;
  char *s = line;
  while (s < end) {
    switch (*s) {
      case '\n':
        if (s + 1 < end && (s[1] == ' ' || s[1] == '\t')) {
          // Replace HTTP header continuation with space.
          *s++ = ' ';
        } else {
          //  End of line found. Strip trailing whitespace.
          *s = 0;
          start = s + 1;
          while (s > line) {
            s--;
            if (*s != ' ' && *s != '\t') break;
            *s = 0;
          }
          return line;
        }
        break;

      case '\r':
      case '\t':
        // Replace whitespace with space.
        *s++ = ' ';
        break;

      default:
        s++;
    }
  }

  return nullptr;
}

void HTTPBuffer::append(const char *data, int size) {
  ensure(size);
  memcpy(end, data, size);
  end += size;
}

HTTPServer::HTTPServer(const HTTPServerOptions &options, int port)
    : options_(options), port_(port) {
  sock_ = -1;
  pollfd_ = -1;
  stop_ = false;
  connections_ = nullptr;

  // Register standard handlers.
  Register("/helpz", this, &HTTPServer::HelpHandler);
  Register("/connz", this, &HTTPServer::ConnectionHandler);
}

HTTPServer::~HTTPServer() {
  // Wait for workers to terminate.
  workers_.Join();

  // Close listening socket.
  if (sock_ != -1) close(sock_);

  // Close poll descriptor.
  if (pollfd_ != -1) close(pollfd_);

  // Delete connections.
  HTTPConnection *conn = connections_;
  while (conn != nullptr) {
    HTTPConnection *next = conn->next_;
    delete conn;
    conn = next;
  }
}

void HTTPServer::Register(const string &uri, const Handler &handler) {
  MutexLock lock(&mu_);
  contexts_.emplace_back(uri, handler);
}

HTTPServer::Handler HTTPServer::FindHandler(HTTPRequest *request) const {
  MutexLock lock(&mu_);

  // Find context with longest matching prefix.
  const char *path = request->path();
  int longest = -1;
  const Context *match = nullptr;
  for (const Context &c : contexts_) {
    int n = c.uri.size();
    const char *s = path + n;
    if (strncmp(c.uri.data(), path, n) == 0 && (*s == '/' || *s == 0)) {
      if (n > longest) {
        match = &c;
        longest = n;
      }
    }
  }

  if (longest >= 0) {
    // Remove matching URI prefix from path.
    request->set_path(path + longest);

    // Return handler.
    return match->handler;
  } else {
    // No match found. Return 404 handler.
    return &Handle404;
  }
}

Status HTTPServer::Start() {
  int rc;

  // Create poll file descriptor.
  pollfd_ = epoll_create(1);
  if (pollfd_ < 0) return Error("epoll_create");

  // Create listen socket.
  sock_ = socket(AF_INET, SOCK_STREAM, 0);
  if (sock_ < 0) return Error("socket");
  rc = fcntl(sock_, F_SETFL, O_NONBLOCK);
  if (rc < 0) return Error("fcntl");
  int on = 1;
  rc = setsockopt(sock_, SOL_SOCKET, SO_REUSEADDR, &on, sizeof(on));
  if (rc < 0) return Error("setsockopt");

  // Bind listen socket.
  struct sockaddr_in sin;
  sin.sin_addr.s_addr = htonl(INADDR_ANY);
  sin.sin_family = AF_INET;
  sin.sin_port = htons(port_);
  rc = bind(sock_, reinterpret_cast<struct sockaddr *>(&sin), sizeof(sin));
  if (rc < 0) return Error("bind");

  // Start listening on socket.
  rc = listen(sock_, SOMAXCONN);
  if (rc < 0) return Error("listen");

  // Add listening socket to poll descriptor.
  struct epoll_event ev;
  ev.events = EPOLLIN;
  ev.data.ptr = nullptr;
  rc = epoll_ctl(pollfd_, EPOLL_CTL_ADD, sock_, &ev);
  if (rc < 0) return Error("epoll_ctl");

  // Start workers.
  workers_.Start(options_.num_workers, [this](int index) { this->Worker(); });

  return Status::OK;
}

void HTTPServer::Worker() {
  // Allocate event structure.
  int max_events = options_.max_events;
  struct epoll_event *events = new epoll_event[max_events];

  // Keep processing events until server is shut down.
  while (!stop_) {
    // Get new events.
    int rc = epoll_wait(pollfd_, events, max_events, 2000);
    if (stop_) break;
    if (errno == EAGAIN) continue;
    if (rc < 0) {
      LOG(ERROR) << Error("epoll_wait");
      break;
    }
    if (rc == 0) {
      ShutdownIdleConnections();
      continue;
    }

    // Process events.
    for (int i = 0; i < rc; ++i) {
      struct epoll_event *ev = &events[i];
      auto *conn = reinterpret_cast<HTTPConnection *>(ev->data.ptr);
      if (conn == nullptr) {
        // New connection.
        AcceptConnection();
      } else {
        // Check if connection has been closed.
        if (ev->events & EPOLLHUP) {
          // Detach socket from poll descriptor.
          rc = epoll_ctl(pollfd_, EPOLL_CTL_DEL, conn->sock_, ev);
          if (rc < 0) {
            VLOG(2) << Error("epoll_ctl");
          } else {
            // Delete client connection.
            VLOG(3) << "Close HTTP connection";
            RemoveConnection(conn);
            delete conn;
          }
        } else {
          // Process connection data.
          VLOG(5) << "Process in state " << conn->state_;
          Status s = conn->Process(ev->events);
          if (!s.ok()) {
            LOG(ERROR) << "HTTP error: " << s;
            conn->state_ = HTTP_STATE_TERMINATE;
          }

          if (conn->state_ == HTTP_STATE_TERMINATE) {
            conn->Shutdown();
            VLOG(5) << "Shutdown HTTP connection";
          } else {
            // Update expected events.
            ev->events = 0;
            if (conn->AwaitsInput()) ev->events |= EPOLLIN;
            if (conn->HasOutput()) ev->events |= EPOLLOUT;
            rc = epoll_ctl(pollfd_, EPOLL_CTL_MOD, conn->sock_, ev);
            if (rc < 0) LOG(ERROR) << Error("epoll_ctl");
            conn->last_ = time(0);
            VLOG(5) << "Done processing in state " << conn->state_
                    << ", events " << ev->events;
          }
        }
      }
    }
  }

  // Free event structure.
  delete [] events;
}

void HTTPServer::Wait() {
  // Wait until all workers have terminated.
  workers_.Join();
}

void HTTPServer::Shutdown() {
  // Set stop flag to terminate worker threads.
  stop_ = true;
}

void HTTPServer::AcceptConnection() {
  int rc;

  // Accept new connection from listen socket.
  struct sockaddr_in addr;
  socklen_t len = sizeof(addr);
  int sock = accept(sock_, reinterpret_cast<struct sockaddr *>(&addr), &len);
  if (sock < 0) {
    if (errno != EAGAIN) LOG(WARNING) << Error("listen");
    return;
  }

  // Set non-blocking mode for socket.
  int flags = fcntl(sock, F_GETFL, 0);
  rc = fcntl(sock, F_SETFL, flags | O_NONBLOCK);
  if (rc < 0) LOG(WARNING) << Error("fcntl");

  // Create new connection.
  VLOG(3) << "New HTTP connection";
  HTTPConnection *conn = new HTTPConnection(this, sock);
  AddConnection(conn);

  // Add new connection to poll descriptor.
  struct epoll_event ev;
  ev.events = EPOLLIN | EPOLLOUT;
  ev.data.ptr = conn;
  rc = epoll_ctl(pollfd_, EPOLL_CTL_ADD, sock, &ev);
  if (rc < 0) LOG(WARNING) << Error("epoll_ctl");
}

void HTTPServer::AddConnection(HTTPConnection *conn) {
  MutexLock lock(&mu_);
  conn->next_ = connections_;
  conn->prev_ = nullptr;
  if (connections_ != nullptr) connections_->prev_ = conn;
  connections_ = conn;
}

void HTTPServer::RemoveConnection(HTTPConnection *conn) {
  MutexLock lock(&mu_);
  if (conn->prev_ != nullptr) conn->prev_->next_ = conn->next_;
  if (conn->next_ != nullptr) conn->next_->prev_ = conn->prev_;
  if (conn == connections_) connections_ = conn->next_;
  conn->next_ = conn->prev_ = nullptr;
}

void HTTPServer::ShutdownIdleConnections() {
  if (options_.max_idle <= 0) return;
  MutexLock lock(&mu_);
  time_t expire = time(0) - options_.max_idle;
  HTTPConnection *conn = connections_;
  while (conn != nullptr) {
    if (conn->last_ < expire) {
      conn->Shutdown();
      VLOG(5) << "Shut down idle connection";
    }
    conn = conn->next_;
  }
}

void HTTPServer::HelpHandler(HTTPRequest *req, HTTPResponse *rsp) {
  MutexLock lock(&mu_);
  rsp->SetContentType("text/html");
  rsp->set_status(404);
  rsp->Append("<html><head><title>helpz</title></head><body>\n");
  rsp->Append("Contexts:<ul>\n");
  for (const Context &c : contexts_) {
    if (c.uri.empty()) {
    rsp->Append("<li><a href=\"/\">/</a></li>\n");
    } else {
      rsp->Append("<li><a href=\"");
      rsp->Append(c.uri);
      rsp->Append("\">");
      rsp->Append(c.uri);
      rsp->Append("</a></li>\n");
    }
  }
  rsp->Append("</ul>\n");
  rsp->Append("</body></html>\n");
}

void HTTPServer::ConnectionHandler(HTTPRequest *req, HTTPResponse *rsp) {
  static const char *state_name[] = {
    "IDLE", "READ_HEADER", "READ_BODY", "PROCESSING",
    "WRITE_HEADER", "WRITE_BODY", "WRITE_FILE",
    "TERMINATE",
  };
  static const char *header_state_name[] = {
    "FIRSTWORD", "FIRSTWS", "SECONDWORD", "SECONDWS", "THIRDWORD",
    "LINE", "LF", "CR", "CRLF", "CRLFCR", "DONE", "BOGUS",
  };

  MutexLock lock(&mu_);
  rsp->SetContentType("text/html");
  rsp->set_status(404);
  rsp->Append("<html><head><title>connz</title></head><body>\n");
  rsp->Append("<table border=\"1\"><tr>\n");
  rsp->Append("<td>Socket</td>");
  rsp->Append("<td>Client address</td>");
  rsp->Append("<td>Socket status</td>");
  rsp->Append("<td>State</td>");
  rsp->Append("<td>Header state</td>");
  rsp->Append("<td>Keep</td>");
  rsp->Append("<td>Idle</td>");
  rsp->Append("</tr>\n");
  HTTPConnection *conn = connections_;
  time_t now = time(0);
  while (conn != nullptr) {
    rsp->Append("<tr>");

    // Socket.
    rsp->Append("<td>" + SimpleItoa(conn->sock_) + "</td>");

    // Client address.
    struct sockaddr_in peer;
    socklen_t plen = sizeof(peer);
    struct sockaddr *saddr = reinterpret_cast<sockaddr *>(&peer);
    if (getpeername(conn->sock_, saddr, &plen) == -1) {
      rsp->Append("<td>?</td>");
    } else {
      rsp->Append("<td>");
      rsp->Append(inet_ntoa(peer.sin_addr));
      rsp->Append(":");
      rsp->Append(SimpleItoa(ntohs(peer.sin_port)));
      rsp->Append("</td>");
    }

    // Socket state.
    int err = 0;
    socklen_t errlen = sizeof(err);
    int rc  = getsockopt(conn->sock_, SOL_SOCKET, SO_ERROR, &err, &errlen);
    const char *error = "OK";
    if (rc != 0) {
      error = strerror(rc);
    } else if (err != 0) {
      error = strerror(err);
    }
    rsp->Append("<td>");
    rsp->Append(error);
    rsp->Append("</td>");

    // Connection state.
    rsp->Append("<td>");
    rsp->Append(state_name[conn->state_]);
    rsp->Append("</td>");

    // Header parsing state.
    rsp->Append("<td>");
    rsp->Append(header_state_name[conn->header_state_]);
    rsp->Append("</td>");

    // Keep alive.
    rsp->Append(conn->keep_ ? "<td>Y</td>" : "<td>N</td>");

    // Idle time.
    rsp->Append("<td>" + SimpleItoa(now - conn->last_) + "</td>");

    rsp->Append("</tr>\n");
    conn = conn->next_;
  }
  rsp->Append("</table>\n");
  rsp->Append("</body></html>\n");
}

HTTPConnection::HTTPConnection(HTTPServer *server, int sock)
    : server_(server), sock_(sock) {
  next_ = prev_ = nullptr;
  request_ = nullptr;
  response_ = nullptr;

  state_ = HTTP_STATE_IDLE;
  header_state_ = HDR_STATE_FIRSTWORD;
  keep_ = false;
  last_ = time(0);
}

HTTPConnection::~HTTPConnection() {
  MutexLock lock(&mu_);

  // Close client connection.
  close(sock_);

  // Cleanup.
  if (file_ != nullptr) file_->Close();
  delete request_;
  delete response_;
}

Status HTTPConnection::Process(int events) {
  MutexLock lock(&mu_);
  bool done;
  switch (state_) {
    case HTTP_STATE_IDLE:
      // Allocate request header buffer.
      request_header_.reset(server_->options().initial_bufsiz);

      // Prepare for receiving HTTP header.
      state_ = HTTP_STATE_READ_HEADER;
      header_state_ = HDR_STATE_FIRSTWORD;
      keep_ = false;
      // Fall through

    case HTTP_STATE_READ_HEADER:
      if ((events & EPOLLIN) == 0) return Status::OK;
      while (1) {
        // Expand request header buffer.
        request_header_.ensure(1);

        // Receive more data.
        Status st = Recv(&request_header_, &done);
        if (!st.ok()) return st;
        if (done) return Status::OK;

        // Parse HTTP header.
        if (ParseHeader()) {
          break;
        } else if (header_state_ == HDR_STATE_BOGUS) {
          return Status(1, "Invalid HTTP header");
        }
      }

      // Create HTTP request from header.
      delete request_;
      request_ = new HTTPRequest(this, &request_header_);
      if (!request_->valid()) return Status(1, "Bad HTTP header");

      // Move unused portion of request header to request body.
      if (request_header_.remaining() > 0) {
        // Allocate response buffer.
        int n = server_->options().initial_bufsiz;
        if (n < request_header_.remaining()) n = request_header_.remaining();
        if (n < request_->content_length()) n = request_->content_length();
        request_body_.reset(n);
        request_body_.start = request_body_.floor;
        request_body_.end = request_body_.floor + n;
        memcpy(request_body_.start, request_header_.end, n);
        request_header_.end -= n;
      }
      state_ = HTTP_STATE_READ_BODY;
      // Fall through

    case HTTP_STATE_READ_BODY:
      // Check if any input data is ready.
      if ((events & EPOLLIN) == 0) return Status::OK;
      while (request_body_.size() < request_->content_length()) {
        // Receive more data.
        Status st = Recv(&request_body_, &done);
        if (!st.ok()) return st;
        if (done) return Status::OK;
      }
      state_ = HTTP_STATE_PROCESSING;
      // Fall through

    case HTTP_STATE_PROCESSING:
      // Set request body content.
      request_->set_content(request_body_.start, request_body_.size());

      // Dispatch request to handler.
      Dispatch();

      state_ = HTTP_STATE_WRITE_HEADER;
      // Fall through

    case HTTP_STATE_WRITE_HEADER:
      // Send HTTP response header.
      while (response_header_.size() > 0) {
        Status st = Send(&response_header_, &done);
        if (!st.ok()) return st;
        if (done) return Status::OK;
      }

      state_ = HTTP_STATE_WRITE_BODY;
      // Fall through

    case HTTP_STATE_WRITE_BODY:
      // Send HTTP response body.
      while (response_body_.size() > 0) {
        Status st = Send(&response_body_, &done);
        if (!st.ok()) return st;
        if (done) return Status::OK;
      }

      state_ = HTTP_STATE_WRITE_FILE;
      // Fall through

    case HTTP_STATE_WRITE_FILE:
      // Send file data.
      while (file_ != nullptr) {
        // Read next chunk from file.
        uint64 read;
        response_body_.reset(server_->options().file_bufsiz);
        Status st = file_->Read(response_body_.start,
                                response_body_.remaining(),
                                &read);
        response_body_.end = response_body_.start + read;

        if (!st.ok()) {
          // Error reading file.
          file_->Close();
          file_ = nullptr;
          return st;
        }

        if (read == 0) {
          // End of file.
          file_->Close();
          file_ = nullptr;
        } else {
          // Send next file chunk.
          while (response_body_.size() > 0) {
            Status st = Send(&response_body_, &done);
            if (!st.ok()) return st;
            if (done) return Status::OK;
          }
        }
      }

      // Check for persistent connection.
      if (keep_) {
        // Clear buffers.
        request_header_.clear();
        request_body_.clear();
        response_header_.clear();
        response_body_.clear();

        // Mark connection as idle.
        state_ = HTTP_STATE_IDLE;
        return Status::OK;
      }

      state_ = HTTP_STATE_TERMINATE;
      // Fall through

    case HTTP_STATE_TERMINATE:
      return Status::OK;

    default:
      return Status(1, "Invalid HTTP state");
  }
}

Status HTTPConnection::Recv(HTTPBuffer *buffer, bool *done) {
  *done = false;
  int rc = recv(sock_, buffer->end, buffer->remaining(), 0);
  if (rc <= 0) {
    *done = true;
    if (rc == 0) {
      // Connection closed.
      state_ = HTTP_STATE_TERMINATE;
      return Status::OK;
    } else if (errno == EAGAIN) {
      // No more data available for now.
      return Status::OK;
    } else {
      // Receive error.
      return Error("recv");
    }
  }
  buffer->end += rc;
  return Status::OK;
}

Status HTTPConnection::Send(HTTPBuffer *buffer, bool *done) {
  *done = false;
  int rc  = send(sock_, buffer->start, buffer->size(), MSG_NOSIGNAL);
  if (rc <= 0) {
    *done = true;
    if (rc == 0) {
      // Connection closed.
      state_ = HTTP_STATE_TERMINATE;
      return Status::OK;
    } else if (errno == EAGAIN) {
      // Output queue full.
      return Status::OK;
    } else {
      // Send error.
      return Error("send");
    }
  }
  buffer->start += rc;
  return Status::OK;
}

void HTTPConnection::Shutdown() {
  shutdown(sock_, SHUT_RDWR);
}

void HTTPConnection::Dispatch() {
  // Allocate response object.
  delete response_;
  response_ = new HTTPResponse(this);

  // Find handler for request.
  HTTPServer::Handler handler = server_->FindHandler(request_);

  // Dispatch request to handler.
  handler(request_, response_);

  // Add Date: and Server: headers.
  char datebuf[32];
  response_->Set("Server", server()->options().server_name.c_str(), false);
  response_->Set("Date", RFCTime(time(nullptr), datebuf), false);

  // Set content length.
  if (!response_body_.empty()) {
    response_->SetContentLength(response_body_.size());
  }

  // Check for persistent connection.
  if (request_->http11()) {
    keep_ = true;
  } else if (request_->keep_alive()) {
    keep_ = true;
    response_->Set("Connection", "keep-alive");
  }

  // Generate response header buffer.
  response_->WriteHeader(&response_header_);

  // The request and response objects are no longer needed.
  delete request_;
  delete response_;
  request_ = nullptr;
  response_ = nullptr;
}

bool HTTPConnection::ParseHeader() {
  while (request_header_.start < request_header_.end) {
    char c = *request_header_.start++;
    switch (header_state_) {
      case HDR_STATE_FIRSTWORD:
        switch (c) {
          case ' ':
          case '\t':
            header_state_ = HDR_STATE_FIRSTWS;
            break;

          case '\n':
          case '\r':
            header_state_ = HDR_STATE_BOGUS;
            return false;
        }
        break;

      case HDR_STATE_FIRSTWS:
        switch (c) {
          case ' ':
          case '\t':
            break;

          case '\n':
          case '\r':
            header_state_ = HDR_STATE_BOGUS;
            return false;

          default:
            header_state_ = HDR_STATE_SECONDWORD;
        }
        break;

      case HDR_STATE_SECONDWORD:
        switch (c) {
          case ' ':
          case '\t':
            header_state_ = HDR_STATE_SECONDWS;
            break;

          case '\n':
          case '\r':
            // The first line has only two words - an HTTP/0.9 request.
           return true;
        }
        break;

      case HDR_STATE_SECONDWS:
        switch (c) {
          case ' ':
          case '\t':
            break;

          case '\n':
          case '\r':
            header_state_ = HDR_STATE_BOGUS;
            return false;

          default:
            header_state_ = HDR_STATE_THIRDWORD;
        }
        break;

      case HDR_STATE_THIRDWORD:
        switch (c) {
          case ' ':
          case '\t':
            header_state_ = HDR_STATE_BOGUS;
            return false;

          case '\n':
            header_state_ = HDR_STATE_LF;
            break;

          case '\r':
            header_state_ = HDR_STATE_CR;
            break;
        }
        break;

      case HDR_STATE_LINE:
        switch (c) {
          case '\n':
            header_state_ = HDR_STATE_LF;
            break;

          case '\r':
            header_state_ = HDR_STATE_CR;
            break;
        }
        break;

      case HDR_STATE_LF:
        switch (c) {
          case '\n':
            // Two newlines in a row - a blank line - end of request.
           header_state_ = HDR_STATE_DONE;
           return true;

          case '\r':
            header_state_ = HDR_STATE_CR;
            break;

          default:
            header_state_ = HDR_STATE_LINE;
        }
        break;

      case HDR_STATE_CR:
        switch (c) {
          case '\n':
            header_state_ = HDR_STATE_CRLF;
            break;

          case '\r':
            // Two returns in a row - end of request.
            header_state_ = HDR_STATE_DONE;
            return true;

          default:
            header_state_ = HDR_STATE_LINE;
        }
        break;

      case HDR_STATE_CRLF:
        switch (c) {
          case '\n':
            // Two newlines in a row - end of request.
            header_state_ = HDR_STATE_DONE;
            return true;

          case '\r':
            header_state_ = HDR_STATE_CRLFCR;
            break;

          default:
            header_state_ = HDR_STATE_LINE;
        }
        break;

      case HDR_STATE_CRLFCR:
        switch (c) {
          case '\n':
          case '\r':
            // Two CRLFs or two CRs in a row - end of request.
            header_state_ = HDR_STATE_DONE;
            return true;

          default:
            header_state_ = HDR_STATE_LINE;
        }
        break;

      case HDR_STATE_BOGUS:
        return false;

      case HDR_STATE_DONE:
        return true;
    }
  }

  return false;
}

void HTTPConnection::AppendResponse(const char *data, int size) {
  if (response_body_.empty()) {
    int n = server_->options().initial_bufsiz;
    if (n < size) n = size;
    response_body_.reset(n);
  }
  response_body_.append(data, size);
}

HTTPRequest::HTTPRequest(HTTPConnection *conn, HTTPBuffer *hdr) : conn_(conn) {
  // Initialize request buffer.
  hdr->start = hdr->floor;

  // Get HTTP line.
  char *s = hdr->gets();
  if (!s) return;

  // Parse method.
  method_ = s;
  s = strchr(s, ' ');
  if (!s) return;
  *s++ = 0;

  // Parse URL path.
  if (*s) {
    path_ = s;

    // Parse URL query.
    char *q = strchr(s, '?');
    if (q) {
      *q++ = 0;
      query_ = q;
      s = q;
    }

    // Parse URL fragment.
    char *f = strchr(s, '#');
    if (f) {
      *f++ = 0;
      fragment_ = f;
      s = f;
    }
  }

  // Parse protocol version.
  if (*s) {
    char *p = strchr(s, ' ');
    if (p) {
      *p++ = 0;
      while (*p == ' ') p++;
      if (*p) {
        protocol_ = p;
      }
    }
  }

  if (protocol_ != nullptr && strcmp(protocol_, "HTTP/1.1") == 0) {
    http11_ = true;
    keep_alive_ = true;
  }

  VLOG(2) << "HTTP method: " << method_ << ", path: " << path_
          << ", query: " << query_ << ", protocol: " << protocol_;

  // Parse headers.
  char *l;
  while ((l = hdr->gets()) != nullptr) {
    // Split header line into key and value.
    if (!*l) continue;
    s = strchr(l, ':');
    if (!s) continue;
    *s++ = 0;
    while (*s == ' ') s++;
    if (!*s) continue;

    // Get standard HTTP headers.
    if (strcasecmp(l, "Content-Type") == 0) {
      content_type_ = s;
    } else if (strcasecmp(l, "Content-Length") == 0) {
      content_length_ = atoi(s);
    } else if (strcasecmp(l, "Connection") == 0) {
      keep_alive_ = strcasecmp(s, "keep-alive") == 0;
    }

    VLOG(3) << "HTTP request header: " << l << ": " << s;
    headers_.emplace_back(l, s);
  }

  // HTTP header successfully parsed.
  valid_ = true;
}

const char *HTTPRequest::Get(const char *name, const char *defval) const {
  for (const HTTPHeader &h : headers_) {
    if (strcasecmp(name, h.name) == 0) return h.value;
  }
  return defval;
}

HTTPResponse::~HTTPResponse() {
  for (HTTPHeader &h : headers_) {
    free(h.name);
    free(h.value);
  }
}

const char *HTTPResponse::ContentType() const {
  return Get("Content-Type");
}

void HTTPResponse::SetContentType(const char *type) {
  Set("Content-Type", type);
}

int HTTPResponse::ContentLength() const {
  const char *result = Get("Content-Length");
  if (result == nullptr) return -1;
  return atoi(result);
}

void HTTPResponse::SetContentLength(int length) {
  char number[16];
  FastInt32ToBufferLeft(length, number);
  Set("Content-Length", number);
}

const char *HTTPResponse::Get(const char *name, const char *defval) const {
  for (const HTTPHeader &h : headers_) {
    if (strcasecmp(name, h.name) == 0) return h.value;
  }
  return defval;
}

void HTTPResponse::Set(const char *name, const char *value, bool overwrite) {
  for (HTTPHeader &h : headers_) {
    if (strcasecmp(name, h.name) == 0) {
      if (overwrite) {
        free(h.value);
        h.value = strdup(value);
      }
      return;
    }
  }
  headers_.emplace_back(strdup(name), strdup(value));
}

void HTTPResponse::SendError(int status, const char *title, const char *msg) {
  if (title == nullptr) title = StatusText(status);

  SetContentType("text/html");
  set_status(status);

  Append("<html><head>\n");
  Append("<title>");
  if (title != nullptr) {
    Append(SimpleItoa(status));
    Append(" ");
    Append(title);
  } else {
    Append("Error ");
    Append(title);
  }
  Append("</title>\n");
  Append("</head><body>\n");
  if (msg != nullptr) {
    Append(msg);
  } else {
    Append("<p>Error ");
    Append(SimpleItoa(status));
    if (title != nullptr) {
      Append(": ");
      Append(title);
    }
    Append("</p>\n");
  }
  Append("</body></html>\n");
}

void HTTPResponse::RedirectTo(const char *uri) {
  string msg;
  string escaped_uri = HTMLEscape(uri);
  msg.append("<h1>Moved</h1>\n");
  msg.append("<p>This page has moved to <a href=\"");
  msg.append(escaped_uri);
  msg.append("\">");
  msg.append(escaped_uri);
  msg.append("</a>.</p>\n");

  Set("Location", uri);
  SendError(301, "Moved Permanently", msg.c_str());
}

void HTTPResponse::WriteHeader(HTTPBuffer *hdr) {
  // Output HTTP header line.
  if (conn_->request()->http11()) {
    hdr->append("HTTP/1.1");
  } else {
    hdr->append("HTTP/1.0");
  }
  hdr->append(" ");

  char statusnum[16];
  FastInt32ToBufferLeft(status_, statusnum);
  hdr->append(statusnum);
  hdr->append(" ");
  hdr->append(StatusText(status_));
  hdr->append("\r\n");

  VLOG(3) << "HTTP response: " << status_ << " " << StatusText(status_);

  // Output HTTP headers.
  for (const HTTPHeader &h : headers_) {
    hdr->append(h.name);
    hdr->append(": ");
    hdr->append(h.value);
    hdr->append("\r\n");
    VLOG(3) << "HTTP response header: " << h.name << ": " << h.value;
  }

  hdr->append("\r\n");
}

}  // namespace sling

