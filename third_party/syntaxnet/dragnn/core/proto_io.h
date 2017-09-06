/* Copyright 2016 Google Inc. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef SYNTAXNET_DRAGNN_CORE_PROTO_IO_H_
#define SYNTAXNET_DRAGNN_CORE_PROTO_IO_H_

#include <string>

#include "base/types.h"
#include "tensorflow/core/lib/io/record_reader.h"
#include "tensorflow/core/lib/io/record_writer.h"

namespace syntaxnet {
namespace dragnn {

// A convenience wrapper to read protos with a RecordReader.
class ProtoRecordReader {
 public:
  explicit ProtoRecordReader(tensorflow::RandomAccessFile *file) {
    file_.reset(file);
    reader_.reset(new tensorflow::io::RecordReader(file_.get()));
  }

  explicit ProtoRecordReader(const string &filename) {
    TF_CHECK_OK(
        tensorflow::Env::Default()->NewRandomAccessFile(filename, &file_));
    reader_.reset(new tensorflow::io::RecordReader(file_.get()));
  }

  ~ProtoRecordReader() {
    reader_.reset();
  }

  template <typename T>
  tensorflow::Status Read(T *proto) {
    string buffer;
    tensorflow::Status status = reader_->ReadRecord(&offset_, &buffer);
    if (status.ok()) {
      CHECK(proto->ParseFromString(buffer));
      return tensorflow::Status::OK();
    } else {
      return status;
    }
  }

 private:
  uint64 offset_ = 0;
  std::unique_ptr<tensorflow::io::RecordReader> reader_;
  std::unique_ptr<tensorflow::RandomAccessFile> file_;
};

// A convenience wrapper to write protos with a RecordReader.
class ProtoRecordWriter {
 public:
  explicit ProtoRecordWriter(const string &filename) {
    TF_CHECK_OK(tensorflow::Env::Default()->NewWritableFile(filename, &file_));
    writer_.reset(new tensorflow::io::RecordWriter(file_.get()));
  }

  ~ProtoRecordWriter() {
    writer_.reset();
    file_.reset();
  }

  template <typename T>
  void Write(const T &proto) {
    TF_CHECK_OK(writer_->WriteRecord(proto.SerializeAsString()));
  }

 private:
  std::unique_ptr<tensorflow::io::RecordWriter> writer_;
  std::unique_ptr<tensorflow::WritableFile> file_;
};

}  // namespace dragnn
}  // namespace syntaxnet

#endif  // SYNTAXNET_DRAGNN_CORE_PROTO_IO_H_
