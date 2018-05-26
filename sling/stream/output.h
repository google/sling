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

#ifndef SLING_STREAM_OUTPUT_H_
#define SLING_STREAM_OUTPUT_H_

#include "sling/base/macros.h"
#include "sling/base/types.h"
#include "sling/stream/stream.h"
#include "sling/string/text.h"
#include "sling/util/varint.h"

namespace sling {

// Output class for writing coded data to an output stream.
class Output {
 public:
  // Initializes output from a stream. This does not take ownership of the
  // underlying stream.
  explicit Output(OutputStream *stream);

  // Flush output to trim last buffer from the output stream.
  ~Output() { Flush(); }

  // Writes 'size' bytes to output.
  void Write(const char *data, int size);
  void Write(const unsigned char *data, int size) {
    Write(reinterpret_cast<const char *>(data), size);
  }

  // Writes string to output.
  void Write(Text str) { Write(str.data(), str.size()); }

  // Writes 32-bit varint to output.
  void WriteVarint32(uint32 value);

  // Writes 64-bit varint to output.
  void WriteVarint64(uint64 value);

  // Writes one character to output.
  void WriteChar(char ch) {
    if (current_ != limit_) {
      *current_++ = ch;
    } else {
      Write(&ch, 1);
    }
  }

  // Flushes output by trimming off the unused portion of the current output
  // buffer.
  void Flush();

  // Returns the output stream.
  OutputStream *stream() { return stream_; }

 private:
  // Gets new output buffer from stream.
  void Next();

  // Current output buffer.
  char *buffer_;

  // Current position in output buffer.
  char *current_;

  // End of output buffer
  char *limit_;

  // Underlying output stream where data is written to.
  OutputStream *stream_;

  DISALLOW_IMPLICIT_CONSTRUCTORS(Output);
};

}  // namespace sling

#endif  // SLING_STREAM_OUTPUT_H_

