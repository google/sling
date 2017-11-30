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

#ifndef SLING_STREAM_FILE_INPUT_H_
#define SLING_STREAM_FILE_INPUT_H_

#include <string>
#include <vector>

#include "sling/base/macros.h"
#include "sling/base/types.h"
#include "sling/stream/input.h"

namespace sling {

// Input stream that runs a pipeline of input stream.
class InputPipeline : public InputStream {
 public:
  InputPipeline();
  ~InputPipeline();

  // Return the last steam in the pipeline.
  InputStream *last() const { return last_; }

  // Add input stream to pipeline. Takes ownership of the stream.
  void Add(InputStream *stream);

  // Implementation of InputStream interface.
  bool Next(const void **data, int *size) override;
  void BackUp(int count) override;
  bool Skip(int count) override;
  int64 ByteCount() const override;

 private:
  // Final input stream.
  InputStream *last_ = nullptr;

  // Input stream pipeline.
  std::vector<InputStream *> streams_;
};

// File input class that supports decompression of the input stream based on
// the file extension.
class FileInput : public Input {
 public:
  // Open file.
  explicit FileInput(const string &filename, int block_size = 1 << 20)
      : Input(Open(filename, block_size)) {}

  ~FileInput() { delete stream(); }

  // Open input file and add decompression for compressed input files.
  static InputStream *Open(const string &filename, int block_size = 1 << 20);

 private:
  DISALLOW_IMPLICIT_CONSTRUCTORS(FileInput);
};

}  // namespace sling

#endif  // SLING_STREAM_FILE_INPUT_H_

