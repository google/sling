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

// Utility tool for using a trained parser. It loads a parser from a Myelin flow
// file, and runs it in one of the following modes.
// A. If --text is set to some text, it runs the model over that text, outputs
//    the frames inferred from the text along with the processing speed.
//    The output frames are printed in textual form, whose indentation is
//    controlled by --indent.
// B. If --benchmark is true, then it runs the parser over the corpus
//    specified via --corpus, and reports the processing speed.
// C. If --evaluate is true, then it takes gold documents via --corpus, runs
//    the parser over them, and reports frame evaluation numbers.
//
// For B and C, --maxdocs can be used to limit the processing to the specified
// number of documents.

#include <iostream>
#include <string>
#include <vector>

#include "sling/base/clock.h"
#include "sling/base/init.h"
#include "sling/base/logging.h"
#include "sling/base/types.h"
#include "sling/base/flags.h"
#include "sling/file/recordio.h"
#include "sling/frame/object.h"
#include "sling/frame/serialization.h"
#include "sling/myelin/compiler.h"
#include "sling/myelin/profile.h"
#include "sling/nlp/document/document.h"
#include "sling/nlp/document/document-corpus.h"
#include "sling/nlp/document/document-tokenizer.h"
#include "sling/nlp/document/lex.h"
#include "sling/nlp/parser/frame-evaluation.h"
#include "sling/nlp/parser/parser.h"
#include "sling/string/printf.h"

DEFINE_string(parser, "", "Input file with flow model");
DEFINE_string(text, "", "Text to parse");
DEFINE_string(output, "", "Output filename");
DEFINE_int32(indent, 2, "Indentation for SLING output");
DEFINE_string(corpus, "", "Input corpus");
DEFINE_bool(parse, false, "Parse input corpus");
DEFINE_bool(benchmark, false, "Benchmark parser");
DEFINE_bool(lex, false, "Output documents in LEX format");
DEFINE_bool(evaluate, false, "Evaluate parser");
DEFINE_int32(maxdocs, -1, "Maximum number of documents to process");

using namespace sling;
using namespace sling::nlp;

Document *RemoveAnnotations(Document *document) {
  Store *store = document->store();
  Handle h_mention = store->Lookup("mention");
  Handle h_theme = store->Lookup("theme");
  Builder b(store);
  for (const Slot &s : document->top()) {
    if (s.name != Handle::id() &&
        s.name != h_mention &&
        s.name != h_theme) {
      b.Add(s.name, s.value);
    }
  }
  return new Document(b.Create());
}

// Parallel corpus for evaluating parser on golden corpus.
class ParserEvaulationCorpus : public ParallelCorpus {
 public:
  ParserEvaulationCorpus(Store *commons, const Parser *parser,
                         const string &eval_corpus_filename)
      : commons_(commons),
        parser_(parser),
        corpus_(commons, eval_corpus_filename) {}

  bool Next(Store **store, Document **golden, Document **predicted) override {
    // Stop if we have reached the maximum number of documents.
    num_documents_++;
    if (FLAGS_maxdocs != -1 && num_documents_ >= FLAGS_maxdocs) return false;

    // Create a local store for both golden and parsed document.
    Store *locals = new Store(commons_);

    // Read next document from corpus.
    Document *document = corpus_.Next(locals);
    if (document == nullptr) {
      delete locals;
      return false;
    }

    // Create new document and remove annotations.
    Document *parsed = RemoveAnnotations(document);

    // Parse the document.
    parser_->Parse(parsed);
    parsed->Update();

    // Return golden and predicted documents.
    *store = locals;
    *golden = document;
    *predicted = parsed;

    return true;
  }

 private:
  Store *commons_;           // commons store
  const Parser *parser_;     // parser being evaluated
  DocumentCorpus corpus_;    // evaluation corpus with golden annotations
  int num_documents_ = 0;    // number of documents processed
};

int main(int argc, char *argv[]) {
  InitProgram(&argc, &argv);

  // Load parser.
  LOG(INFO) << "Load parser from " << FLAGS_parser;
  Clock clock;
  clock.start();
  Store commons;
  Parser parser;
  parser.Load(&commons, FLAGS_parser);
  commons.Freeze();
  clock.stop();
  LOG(INFO) << clock.ms() << " ms loading parser";

  // Parse input text.
  if (!FLAGS_text.empty()) {
    // Create document tokenizer.
    DocumentTokenizer tokenizer;

    // Create document
    Store store(&commons);
    Document document(&store);

    // Parse sentence.
    tokenizer.Tokenize(&document, FLAGS_text);
    clock.start();
    parser.Parse(&document);
    document.Update();
    clock.stop();

    if (FLAGS_lex) {
      std::cout << ToLex(document) << "\n";
    } else {
      std::cout << ToText(document.top(), FLAGS_indent) << "\n";
    }
    LOG(INFO) << document.num_tokens() / clock.secs() << " tokens/sec";
  }

  // Parse input corpus.
  if (FLAGS_parse) {
    CHECK(!FLAGS_corpus.empty());
    LOG(INFO) << "Parse " << FLAGS_corpus;
    DocumentCorpus corpus(&commons, FLAGS_corpus);
    int num_documents = 0;
    RecordWriter *writer = nullptr;
    if (!FLAGS_output.empty()) {
      writer = new RecordWriter(FLAGS_output);
    }
    for (;;) {
      if (FLAGS_maxdocs != -1 && num_documents >= FLAGS_maxdocs) break;

      Store store(&commons);
      Document *document = corpus.Next(&store);

      if (document == nullptr) break;
      num_documents++;

      document->ClearAnnotations();
      parser.Parse(document);
      document->Update();
      if (writer != nullptr) {
        writer->Write(std::to_string(num_documents), Encode(document->top()));
      } else if (FLAGS_lex) {
        std::cout << ToLex(*document) << "\n";
      } else {
        std::cout << ToText(document->top(), FLAGS_indent) << "\n";
      }

      delete document;
    }
  }

  // Benchmark parser on corpus.
  if (FLAGS_benchmark) {
    CHECK(!FLAGS_corpus.empty());
    LOG(INFO) << "Benchmarking parser on " << FLAGS_corpus;
    DocumentCorpus corpus(&commons, FLAGS_corpus);
    int num_documents = 0;
    int num_tokens = 0;
    clock.start();
    for (;;) {
      if (FLAGS_maxdocs != -1 && num_documents >= FLAGS_maxdocs) break;

      Store store(&commons);
      Document *document = corpus.Next(&store);
      if (document == nullptr) break;

      num_documents++;
      num_tokens += document->num_tokens();
      if (num_documents % 100 == 0) {
        std::cout << num_documents << " documents\r";
        std::cout.flush();
      }
      parser.Parse(document);

      delete document;
    }
    clock.stop();
    LOG(INFO) << num_documents << " documents, "
              << num_tokens << " tokens, "
              << num_tokens / clock.secs() << " tokens/sec";
  }

  // Evaluate parser on gold corpus.
  if (FLAGS_evaluate) {
    CHECK(!FLAGS_corpus.empty());
    LOG(INFO) << "Evaluating parser on " << FLAGS_corpus;
    ParserEvaulationCorpus corpus(&commons, &parser, FLAGS_corpus);
    FrameEvaluation::Output eval;
    FrameEvaluation::Evaluate(&corpus, &eval);
    FrameEvaluation::Scores scores;
    eval.GetScores(&scores);
    for (const auto &score : scores) {
      std::cout << score.first << "=" << score.second << "\n";
    }
  }

  // Output profile report.
  LogProfile(parser.network());

  return 0;
}

