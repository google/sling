#include <string>

#include "base/flags.h"
#include "base/init.h"
#include "base/logging.h"
#include "base/macros.h"
#include "frame/store.h"
#include "nlp/document/document.h"
#include "nlp/document/document-source.h"
#include "nlp/parser/trainer/shared-resources.h"

DEFINE_string(documents, "", "File pattern of documents.");
DEFINE_string(commons, "", "Path to common store.");

using sling::Store;
using sling::nlp::SharedResources;
using sling::nlp::Document;
using sling::nlp::DocumentSource;

int main(int argc, char **argv) {
  sling::InitProgram(&argc, &argv);

  SharedResources resources;
  resources.LoadGlobalStore(FLAGS_commons);

  DocumentSource *corpus = DocumentSource::Create(FLAGS_documents);
  int count = 0;
  string query = "gqui from $SEM/google3docs/dev-full.gold.rio "
                 "proto nlp_saft.Document where '";
  while (true) {
    Store store(resources.global);
    Document *document = corpus->Next(&store);
    if (document == nullptr) break;

    string text = document->GetText();
    if (text.empty()) text = document->PhraseText(0, document->num_tokens());
    LOG(INFO) << "Doc " << count << ": " << text;
    string finaltext;
    for (char c : text) {
      if (c == '\'' or c == '"') finaltext.push_back('\\');
      finaltext.push_back(c);
    }
    if (finaltext.size() > 30) finaltext = finaltext.substr(0, 30) + "%";
    query += " joinstrings(token[*].word, \" \") like \"" + finaltext + "\" or ";
    count++;
    delete document;
  }
  query.pop_back();
  query.pop_back();
  query.pop_back();
  query += "' select 'joinstrings(token[*].word, \" \")'";
  LOG(INFO) << query;
  delete corpus;

  return 0;
}


