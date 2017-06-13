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
}

void SharedResources::LoadGlobalStore(const string &file) {
  delete global;
  global = new Store();
  sling::LoadStore(file, global);
}

}  // namespace nlp
}  // namespace sling
