#include <functional>
#include <iostream>
#include <string>

#include "sling/base/init.h"
#include "sling/base/logging.h"
#include "sling/base/types.h"
#include "sling/base/flags.h"
#include "sling/file/recordio.h"
#include "sling/frame/store.h"
#include "sling/frame/serialization.h"
#include "sling/nlp/document/document.h"
#include "sling/nlp/document/document-source.h"
#include "sling/nlp/parser/parser-action.h"
#include "sling/nlp/parser/trainer/transition-generator.h"

DEFINE_string(corpus, "", "Input corpus");

using namespace sling::nlp;

int main(int argc, char *argv[]) {
  sling::InitProgram(&argc, &argv);

  DocumentSource *corpus = DocumentSource::Create(FLAGS_corpus);
  sling::Store commons;
  commons.Freeze();
  for (int i = 0;; ++i) {
    sling::Store store(&commons);
    Document *document = corpus->Next(&store);
    if (document == nullptr) break;

    Generate(*document, [&](const ParserAction &action) {
      std::cout << "Doc " << i << " " << action.ToString(&store) << "\n";
    });
    delete document;
  }

  return 0;
}

