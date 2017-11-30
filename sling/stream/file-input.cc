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

#include "sling/stream/file-input.h"

#include <string>
#include <vector>

#include "sling/stream/bzip2.h"
#include "sling/stream/file.h"
#include "sling/stream/gzip.h"

namespace sling {

InputPipeline::InputPipeline() {}

InputPipeline::~InputPipeline() {
  // Delete streams.
  for (int i = streams_.size() - 1; i >= 0; --i) {
    delete streams_[i];
  }
}

void InputPipeline::Add(InputStream *stream) {
  streams_.push_back(stream);
  last_ = stream;
}

bool InputPipeline::Next(const void **data, int *size) {
  return last_->Next(data, size);
}

void InputPipeline::BackUp(int count) {
  last_->BackUp(count);
}

bool InputPipeline::Skip(int count) {
  return last_->Skip(count);
}

int64 InputPipeline::ByteCount() const {
  return last_->ByteCount();
}

InputStream *FileInput::Open(const string &filename, int block_size) {
  // Open input file.
  InputStream *stream = new FileInputStream(filename, block_size);

  // Get file extension.
  int dot = filename.find_last_of('.');
  if (dot != -1) {
    string ext = filename.substr(dot);
    InputStream *decompressor = nullptr;
    if (ext == ".gz") {
      // Add GZIP decompressor.
      decompressor = new GZipDecompressor(stream, block_size);
    } else if (ext == ".bz2") {
      // Add BZIP2 decompressor.
      decompressor =  new BZip2Decompressor(stream, block_size);
    }

    // Create input pipeline for compressed files.
    if (decompressor != nullptr) {
      InputPipeline *pipeline = new InputPipeline();
      pipeline->Add(stream);
      pipeline->Add(decompressor);
      stream = pipeline;
    }
  }

  return stream;
}

}  // namespace sling

