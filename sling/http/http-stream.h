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

#ifndef SLING_HTTP_HTTP_STREAM_H_
#define SLING_HTTP_HTTP_STREAM_H_

#include "sling/http/http-server.h"
#include "sling/stream/stream.h"

namespace sling {

// An InputStream for reading from a HTTP buffer.
class HTTPInputStream : public InputStream {
 public:
  HTTPInputStream(HTTPBuffer *buffer);

  // InputStream interface.
  bool Next(const void **data, int *size) override;
  void BackUp(int count) override;
  bool Skip(int count) override;
  int64 ByteCount() const override;

 private:
  HTTPBuffer *buffer_;
};

// An OutputStream backed by a HTTP buffer.
class HTTPOutputStream : public OutputStream {
 public:
  HTTPOutputStream(HTTPBuffer *buffer, int block_size = 8192);

  // OutputStream interface.
  bool Next(void **data, int *size) override;
  void BackUp(int count) override;
  int64 ByteCount() const override;

 private:
  HTTPBuffer *buffer_;
  int block_size_;
};

}  // namespace sling

#endif  // SLING_HTTP_HTTP_STREAM_H_

