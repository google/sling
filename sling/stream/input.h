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

#ifndef SLING_STREAM_INPUT_H_
#define SLING_STREAM_INPUT_H_

#include <string>

#include "sling/base/macros.h"
#include "sling/base/types.h"
#include "sling/stream/stream.h"
#include "sling/util/varint.h"

namespace sling {

// Input class for reading coded data from an input stream.
class Input {
 public:
  // Initializes input from a stream. This does not take ownership of the
  // underlying stream.
  explicit Input(InputStream *stream);
  ~Input();

  // Reads 'size' bytes from the input. Returns false if not all data could be
  // read, i.e. there is less data available from the input or an error
  // occurred.
  bool Read(char *data, int size);

  // Skips a number of bytes in the input.
  void Skip(int bytes);

  // Reads one byte from the input. This optimizes the case where we have data
  // buffered from the input stream.
  bool Next(char *ch) {
    if (!empty()) {
      *ch = *current_++;
      return true;
    } else {
      return Read(ch, 1);
    }
  }

  // Try reading 'size' bytes from the input. If the data is available in the
  // input buffer, a pointer to the data is returned. Otherwise the data must be
  // read using the Read() method.
  bool TryRead(int size, const char **data) {
    if (current_ + size < limit_) {
      *data = current_;
      current_ += size;
      return true;
    } else {
      return false;
    }
  }

  // Reads 'size' bytes from input and append them to the string.
  bool ReadString(int size, string *output);

  // Reads line from input into to the string. Return false on end of input.
  bool ReadLine(string *output);

  // Reads 32-bits varint from input. The fast case where we are sure enough
  // data is in the buffer is inlined. The loop will be unrolled by the
  // compiler, so this is faster than using the normal varint routines.
  bool ReadVarint32(uint32 *value) {
    if (limit_ - current_ >= Varint::kMax32) {
      uint32 result = 0;
      const char *ptr = current_;
      for (int i = 0; i < 64; i += 7) {
        uint32 byte = *ptr++;
        result |= (byte & 127) << i;
        if (byte < 128) {
          current_ = ptr;
          *value = result;
          return true;
        }
      }
      return false;
    } else {
      return ReadVarint32Fallback(value);
    }
  }

  // Reads 64-bits varint from input. The fast case where we are sure enough
  // data is in the buffer is inlined. The loop will be unrolled by the
  // compiler, so this is faster than using the normal varint routines.
  bool ReadVarint64(uint64 *value) {
    if (limit_ - current_ >= Varint::kMax64) {
      uint64 result = 0;
      const char *ptr = current_;
      for (int i = 0; i < 64; i += 7) {
        uint32 byte = *ptr++;
        result |= static_cast<uint64>(byte & 127) << i;
        if (byte < 128) {
          current_ = ptr;
          *value = result;
          return true;
        }
      }
      return false;
    } else {
      return ReadVarint64Fallback(value);
    }
  }

  // Peeks at the next input byte. Returns the next input byte without removing
  // it or returns -1 when there is no more input.
  int Peek();

  // Returns true when all input has been read.
  bool done() { return empty() && !Fill(); }

  // Returns the input stream.
  InputStream *stream() { return stream_; }

 private:
  // Returns true if input buffer is empty.
  bool empty() const { return current_ == limit_; }

  // Reads more data from input. If this returns true there will be more data in
  // the buffer. Returns false if the end of the input has been reached.
  bool Fill();

  // Fallback routines for varint decoding.
  bool ReadVarint32Fallback(uint32 *value);
  bool ReadVarint64Fallback(uint64 *value);

  // Current position in input buffer.
  const char *current_;

  // End of input buffer.
  const char *limit_;

  // Underlying input stream where data is read from.
  InputStream *stream_;

  // This flag is set when all input has been read from the input.
  bool done_;

  DISALLOW_IMPLICIT_CONSTRUCTORS(Input);
};

}  // namespace sling

#endif  // SLING_STREAM_INPUT_H_

