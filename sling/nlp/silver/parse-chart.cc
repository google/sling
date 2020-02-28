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
#include <vector>

#include "sling/base/init.h"
#include "sling/base/logging.h"
#include "sling/base/types.h"
#include "sling/base/flags.h"
#include "sling/file/file.h"
#include "sling/file/recordio.h"
#include "sling/frame/object.h"
#include "sling/frame/serialization.h"
#include "sling/nlp/document/document.h"
#include "sling/nlp/document/document-tokenizer.h"
#include "sling/nlp/document/lex.h"
#include "sling/nlp/silver/mentions.h"
#include "sling/nlp/silver/chart.h"
#include "sling/nlp/silver/idf.h"

DEFINE_string(text, "", "Text to parse");
DEFINE_string(input, "", "File with text to parse");
DEFINE_string(item, "", "QID of item to parse");
DEFINE_string(lang, "en", "Language");
DEFINE_bool(resolve, false, "Resolve annotated entities");

using namespace sling;
using namespace sling::nlp;

int main(int argc, char *argv[]) {
  InitProgram(&argc, &argv);

  // Initialize annotator.
  Store commons;
  commons.LockGC();

  SpanAnnotator::Resources resources;
  resources.kb = "local/data/e/ner/kb.sling";
  resources.dictionary = "local/data/e/ner/" + FLAGS_lang + "/idf.repo";
  resources.language = FLAGS_lang;
  resources.resolve = FLAGS_resolve;

  string alias_file = "local/data/e/wiki/" + FLAGS_lang + "/phrase-table.repo";
  PhraseTable aliases;
  aliases.Load(&commons, alias_file);
  resources.aliases = &aliases;

  SpanAnnotator annotator;
  annotator.Init(&commons, resources);

  commons.Freeze();

  // Open document corpus.
  RecordFileOptions options;
  RecordDatabase db("local/data/e/wiki/" + FLAGS_lang + "/documents@10.rec",
                    options);

  // Initialize document.
  Store store(&commons);
  Frame frame(&store, Handle::nil());
  if (!FLAGS_item.empty()) {
    Record record;
    CHECK(db.Lookup(FLAGS_item, &record));
    frame = Decode(&store, record.value).AsFrame();
  }
  Document document(frame);
  if (frame.IsNil()) {
    string text;
    if (!FLAGS_text.empty()) {
      text = FLAGS_text;
    } else if (!FLAGS_input.empty()) {
      CHECK(File::ReadContents(FLAGS_input, &text));
    }

    DocumentTokenizer tokenizer;
    DocumentLexer lexer(&tokenizer);
    CHECK(lexer.Lex(&document, text));
  }

  // Create unannotated output document.
  Document output(document);
  output.ClearAnnotations();

  // Annotate document.
  annotator.Annotate(document, &output);

  // Output annotated document.
  std::cout << ToLex(output) << "\n";

  return 0;
}

