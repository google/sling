// Copyright 2008 Google Inc.  All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
//     * Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//     * Redistributions in binary form must reproduce the above
// copyright notice, this list of conditions and the following disclaimer
// in the documentation and/or other materials provided with the
// distribution.
//     * Neither the name of Google Inc. nor the names of its
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

// Based on ZeroCopy streams from Google Protocol Buffers.
//
// This file contains the InputStream and OutputStream interfaces, which
// represent abstract I/O streams.
//
// These interfaces are different from classic I/O streams in that they
// try to minimize the amount of data copying that needs to be done.
// To accomplish this, responsibility for allocating buffers is moved to
// the stream object, rather than being the responsibility of the caller.
// So, the stream can return a buffer which actually points directly into
// the final data structure where the bytes are to be stored, and the caller
// can interact directly with that buffer, eliminating an intermediate copy
// operation.
//
// As an example, consider the common case in which you are reading bytes
// from an array that is already in memory (or perhaps an mmap()ed file).
// With classic I/O streams, you would do something like:
//
//   char buffer[BUFFER_SIZE];
//   input->Read(buffer, BUFFER_SIZE);
//   DoSomething(buffer, BUFFER_SIZE);
//
// Then, the stream basically just calls memcpy() to copy the data from
// the array into your buffer.  With a zero-copy input stream, you would do
// this instead:
//
//   const void *buffer;
//   int size;
//   input->Next(&buffer, &size);
//   DoSomething(buffer, size);
//
// Here, no copy is performed.  The input stream returns a pointer directly
// into the backing array, and the caller ends up reading directly from it.

#ifndef SLING_STREAM_STREAM_H_
#define SLING_STREAM_STREAM_H_

#include <string>

#include "sling/base/macros.h"

namespace sling {

// Abstract input stream interface designed to minimize copying.
class InputStream {
 public:
  InputStream() {}
  virtual ~InputStream() = default;

  // Obtains a chunk of data from the stream.
  //
  // Preconditions:
  // * "size" and "data" are not null.
  //
  // Postconditions:
  // * If the returned value is false, there is no more data to return or
  //   an error occurred.  All errors are permanent.
  // * Otherwise, "size" points to the actual number of bytes read and "data"
  //   points to a pointer to a buffer containing these bytes.
  // * Ownership of this buffer remains with the stream, and the buffer
  //   remains valid only until some other method of the stream is called
  //   or the stream is destroyed.
  // * It is legal for the returned buffer to have zero size, as long
  //   as repeatedly calling Next() eventually yields a buffer with non-zero
  //   size.
  virtual bool Next(const void **data, int *size) = 0;

  // Backs up a number of bytes, so that the next call to Next() returns
  // data again that was already returned by the last call to Next().  This
  // is useful when writing procedures that are only supposed to read up
  // to a certain point in the input, then return.  If Next() returns a
  // buffer that goes beyond what you wanted to read, you can use BackUp()
  // to return to the point where you intended to finish.
  //
  // Preconditions:
  // * The last method called must have been Next().
  // * count must be less than or equal to the size of the last buffer
  //   returned by Next().
  //
  // Postconditions:
  // * The last "count" bytes of the last buffer returned by Next() will be
  //   pushed back into the stream.  Subsequent calls to Next() will return
  //   the same data again before producing new data.
  virtual void BackUp(int count) = 0;

  // Skips a number of bytes.  Returns false if the end of the stream is
  // reached or some input error occurred.  In the end-of-stream case, the
  // stream is advanced to the end of the stream (so ByteCount() will return
  // the total size of the stream).
  virtual bool Skip(int count) = 0;

  // Returns the total number of bytes read since this object was created.
  virtual int64 ByteCount() const = 0;

 private:
  DISALLOW_COPY_AND_ASSIGN(InputStream);
};

// Abstract output stream interface designed to minimize copying.
class OutputStream {
 public:
  OutputStream() {}
  virtual ~OutputStream() = default;

  // Obtains a buffer into which data can be written.  Any data written
  // into this buffer will eventually (maybe instantly, maybe later on)
  // be written to the output.
  //
  // Preconditions:
  // * "size" and "data" are not null.
  //
  // Postconditions:
  // * If the returned value is false, an error occurred.  All errors are
  //   permanent.
  // * Otherwise, "size" points to the actual number of bytes in the buffer
  //   and "data" points to the buffer.
  // * Ownership of this buffer remains with the stream, and the buffer
  //   remains valid only until some other method of the stream is called
  //   or the stream is destroyed.
  // * Any data which the caller stores in this buffer will eventually be
  //   written to the output (unless BackUp() is called).
  // * It is legal for the returned buffer to have zero size, as long
  //   as repeatedly calling Next() eventually yields a buffer with non-zero
  //   size.
  virtual bool Next(void **data, int *size) = 0;

  // Backs up a number of bytes, so that the end of the last buffer returned
  // by Next() is not actually written.  This is needed when you finish
  // writing all the data you want to write, but the last buffer was bigger
  // than you needed.  You don't want to write a bunch of garbage after the
  // end of your data, so you use BackUp() to back up.
  //
  // Preconditions:
  // * The last method called must have been Next().
  // * count must be less than or equal to the size of the last buffer
  //   returned by Next().
  // * The caller must not have written anything to the last "count" bytes
  //   of that buffer.
  //
  // Postconditions:
  // * The last "count" bytes of the last buffer returned by Next() will be
  //   ignored.
  virtual void BackUp(int count) = 0;

  // Returns the total number of bytes written since this object was created.
  virtual int64 ByteCount() const = 0;

 private:
  DISALLOW_COPY_AND_ASSIGN(OutputStream);
};

}  // namespace sling

#endif  // SLING_STREAM_STREAM_H_

