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

#ifndef SLING_STREAM_BOUNDED_H_
#define SLING_STREAM_BOUNDED_H_

#include "sling/base/macros.h"
#include "sling/base/types.h"
#include "sling/stream/stream.h"

namespace sling {

// A bounded input stream that limits the size of the input to a particular
// size.
class BoundedInputStream : public InputStream {
 public:
  BoundedInputStream(InputStream *input, int64 limit);
  ~BoundedInputStream();

  // InputStream interface.
  bool Next(const void **data, int *size);
  void BackUp(int count);
  bool Skip(int count);
  int64 ByteCount() const;

 private:
  // Underlying input stream.
  InputStream *input_;

  // Number of bytes left to read. This can be negative if have overshoot the
  // limit of the stream.
  int64 left_;

  // Initial position of the underlying stream.
  int64 start_;

  DISALLOW_IMPLICIT_CONSTRUCTORS(BoundedInputStream);
};

}  // namespace sling

#endif  // SLING_STREAM_BOUNDED_H_

