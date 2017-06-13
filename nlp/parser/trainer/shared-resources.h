#ifndef NLP_PARSER_TRAINER_SHARED_RESOURCES_H_
#define NLP_PARSER_TRAINER_SHARED_RESOURCES_H_

#include "frame/store.h"
#include "nlp/parser/action-table.h"

namespace sling {
namespace nlp {

// Container for resources that are typically shared (e.g. across features).
// TODO: This can be extended more generally by adding a WorkspaceSet.
struct SharedResources {
  ActionTable table;
  Store *global = nullptr;  // owned

  ~SharedResources() { delete global; }

  // Loads global store from 'file'.
  void LoadGlobalStore(const string &file);

  // Loads action table from 'file'.
  void LoadActionTable(const string &file);
};

}  // namespace nlp
}  // namespace sling

#endif // NLP_PARSER_TRAINER_SHARED_RESOURCES_H_
