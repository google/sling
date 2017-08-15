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

#include "base/init.h"
#include "base/logging.h"
#include "base/macros.h"
#include "stream/zipfile.h"
#include "util/zip-iterator.h"

// Compares ZipIterator and ZipFileReader.
int main(int argc, char **argv) {
  sling::InitProgram(&argc, &argv);

  sling::ZipIterator iter(argv[1]);
  int count = 0;
  string name, contents;
  while (iter.Next(&name, &contents)) count++;
  std::cout << "ZipIterator: " << count << " files in " << argv[1] << "\n";

  sling::ZipFileReader reader(argv[1]);
  std::cout << "ZipFileReader: " << reader.files().size()
            << " files in " << argv[1] << "\n";

  return 0;
}
