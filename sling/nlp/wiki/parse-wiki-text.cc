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

#include "sling/base/flags.h"
#include "sling/base/init.h"
#include "sling/base/logging.h"
#include "sling/base/types.h"
#include "sling/file/file.h"
#include "sling/nlp/wiki/wiki-parser.h"
#include "sling/nlp/wiki/wiki-extractor.h"

DEFINE_string(input, "test.txt", "input file with wiki text");

using namespace sling;
using namespace sling::nlp;

int main(int argc, char *argv[]) {
  InitProgram(&argc, &argv);

  string wikitext;
  CHECK(File::ReadContents(FLAGS_input, &wikitext));

  WikiParser parser(wikitext.c_str());
  parser.Parse();

  WikiExtractor extractor(parser);
  WikiTextSink sink;
  extractor.Extract(&sink);

  WikiPlainTextSink intro;
  extractor.ExtractIntro(&intro);

  std::cout << "<html>\n";
  std::cout << "<head>\n";
  std::cout << "<meta charset='utf-8'/>\n";
  std::cout << "</head>\n";
  std::cout << "<body>\n";
  std::cout <<  sink.text() << "\n";
  std::cout << "<h1>AST</h1>\n<pre>\n";
  if (!intro.text().empty()) {
    std::cout << "Intro: " << intro.text() << "<br><br>";
  }
  parser.PrintAST(0, 0);
  std::cout << "</pre>\n";
  std::cout << "</body></html>\n";

  return 0;
}

