#include "nlp/parser/trainer/shared-resources.h"

#include "base/macros.h"
#include "frame/serialization.h"

namespace sling {
namespace nlp {

void SharedResources::LoadActionTable(const string &file) {
  CHECK(global != nullptr);
  Store temp(global);
  sling::LoadStore(file, &temp);
  table.Init(&temp);
  table.set_action_checks(false);
}

void SharedResources::LoadGlobalStore(const string &file) {
  delete global;
  global = new Store();
  sling::LoadStore(file, global);
  global->Freeze();
}

}  // namespace nlp
}  // namespace sling
