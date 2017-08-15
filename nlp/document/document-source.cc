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

#include "nlp/document/document-source.h"

#include <vector>

#include "base/logging.h"
#include "base/macros.h"
#include "file/file.h"
#include "frame/object.h"
#include "frame/serialization.h"
#include "util/zip-iterator.h"

namespace sling {
namespace nlp {

// Iterator implementation which assumes one encoded document per input file.
class EncodedDocumentSource : public DocumentSource {
 public:
  EncodedDocumentSource(const std::vector<string> &files) {
    files_ = files;
    index_ = 0;
  }

  bool NextSerialized(string *name, string *contents) override {
    if (index_ >= files_.size()) return false;
    *name = files_[index_];
    CHECK(File::ReadContents(files_[index_], contents));
    index_++;

    return true;
  }

  void Rewind() override {
    index_ = 0;
  }

 private:
  std::vector<string> files_;
  int index_;
};

// Iterator implementation for zip archives..
// Assumes that each encoded document is a separate file in the zip archive.
class ZipDocumentSource : public DocumentSource {
 public:
  ZipDocumentSource(const string &file) {
    file_ = file;
    iterator_ = new ZipIterator(file);
  }

  ~ZipDocumentSource() override {
    delete iterator_;
  }

  bool NextSerialized(string *name, string *contents) override {
    return iterator_->Next(name, contents);
  }

  void Rewind() override {
    delete iterator_;
    iterator_ = new ZipIterator(file_);
  }

 private:
  ZipIterator *iterator_ = nullptr;
  string file_;
};

Document *DocumentSource::Next(Store *store) {
  string name, contents;
  if (!NextSerialized(&name, &contents)) return nullptr;

  StringDecoder decoder(store, contents);
  return new Document(decoder.Decode().AsFrame());
}

DocumentSource *DocumentSource::Create(const string &file_pattern) {
  // TODO: Add more formats as needed.
  int size = file_pattern.size();
  if (size > 4 && file_pattern.substr(size - 4) == ".zip") {
    return new ZipDocumentSource(file_pattern);
  } else {
    std::vector<string> files;
    CHECK(File::Match(file_pattern, &files));
    return new EncodedDocumentSource(files);
  }

  return nullptr;
}

}  // namespace nlp
}  // namespace sling
