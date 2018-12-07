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

#include <iostream>
#include <string>

#include "sling/base/init.h"
#include "sling/base/flags.h"
#include "sling/base/logging.h"
#include "sling/base/types.h"
#include "sling/file/file.h"
#include "sling/file/recordio.h"
#include "sling/frame/serialization.h"
#include "sling/frame/store.h"
#include "sling/string/printf.h"
#include "sling/util/fingerprint.h"

DEFINE_bool(keys, false, "Only output keys");
DEFINE_bool(files, false, "Output file names");
DEFINE_bool(frames, false, "Record values as encoded frames");
DEFINE_string(key, "", "Only display records with matching key");
DEFINE_int32(indent, 2, "Indentation for structured data");
DEFINE_int32(limit, 0, "Maximum number of records to output");
DEFINE_bool(utf8, false, "Allow UTF8-encoded output");

using namespace sling;

int records_output = 0;

void DisplayFrame(const Slice &value) {
  Store store;
  Text encoded(value.data(), value.size());
  StringDecoder decoder(&store, encoded);

  StringPrinter printer(&store);
  printer.printer()->set_indent(FLAGS_indent);
  printer.printer()->set_shallow(false);
  printer.printer()->set_utf8(FLAGS_utf8);
  printer.Print(decoder.Decode());

  std::cout << printer.text();
}

void DisplayRaw(const Slice &value) {
  std::cout.write(value.data(), value.size());
}

void DisplayRecord(const Slice &key, const Slice &value) {
  // Display key.
  DisplayRaw(key);

  // Display value.
  if (!FLAGS_keys) {
    if (!key.empty()) std::cout << ": ";
    if (FLAGS_frames) {
      DisplayFrame(value);
    } else {
      DisplayRaw(value);
    }
  }

  std::cout << "\n";
  records_output++;
}

void DisplayFile(const string &filename) {
  if (FLAGS_files) std::cout << "File " << filename << ":\n";
  RecordReader reader(filename);
  while (!reader.Done()) {
    // Read next record.
    Record record;
    CHECK(reader.Read(&record));

    // Check for key match.
    if (!FLAGS_key.empty() && record.key != FLAGS_key) continue;

    // Display record.
    DisplayRecord(record.key, record.value);

    // Check record limit.
    if (FLAGS_limit > 0 && records_output >= FLAGS_limit) break;
  }
  CHECK(reader.Close());
}

int main(int argc, char *argv[]) {
  InitProgram(&argc, &argv);
  if (argc < 2) {
    std::cerr << argv[0] << " [OPTIONS] [FILE] ...\n";
    return 1;
  }

  std::vector<string> files;
  for (int i = 1; i < argc; ++i) {
    File::Match(argv[i], &files);
  }

  if (FLAGS_key.empty()) {
    for (const string &file : files) {
      DisplayFile(file);
      if (FLAGS_limit > 0 && records_output >= FLAGS_limit) break;
    }
  } else {
    uint64 fp = Fingerprint(FLAGS_key.data(), FLAGS_key.size());
    int shard = fp % files.size();
    DisplayFile(files[shard]);
  }

  return 0;
}

