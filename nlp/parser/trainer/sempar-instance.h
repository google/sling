#ifndef NLP_PARSER_TRAINER_SEMPAR_SENTENCE_H_
#define NLP_PARSER_TRAINER_SEMPAR_SENTENCE_H_

#include "frame/store.h"
#include "nlp/document/document.h"
#include "syntaxnet/workspace.h"

namespace sling {
namespace nlp {

// Container for holding a document, its local store, and workspaces for
// feature extraction.
struct SemparInstance {
  ~SemparInstance() {
    delete document;
    delete store;
    delete workspaces;
  }

  // Serialization of the document. Cleared once the document is decoded.
  string encoded;

  // The document itself, its local store, and workspace set.
  Document *document = nullptr;
  Store *store = nullptr;
  syntaxnet::WorkspaceSet *workspaces = nullptr;
};

}  // namespace nlp
}  // namespace sling

#endif  // NLP_PARSER_TRAINER_SEMPAR_SENTENCE_H_
