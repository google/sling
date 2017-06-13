#include "nlp/parser/trainer/document-batch.h"

#include "frame/object.h"
#include "frame/serialization.h"
#include "nlp/document/document.h"
#include "syntaxnet/workspace.h"

namespace sling {
namespace nlp {

void DocumentBatch::SetData(const std::vector<string> &data) {
  items_.resize(data.size());
  for (int i = 0; i < data.size(); ++i) {
    auto &item = items_[i];
    item.encoded = data[i];
    item.document = nullptr;
    item.workspace = new syntaxnet::WorkspaceSet();
  }
}

const std::vector<string> DocumentBatch::GetSerializedData() const {
  std::vector<string> output;
  output.resize(size());
  for (int i = 0; i < size(); ++i) {
    CHECK(items_[i].document != nullptr);
    output[i] = Encode(items_[i].document->top());
  }
  return output;
}

void DocumentBatch::Decode(Store *global) {
  for (int i = 0; i < size(); ++i) {
    items_[i].store = new Store(global);
    StringDecoder decoder(items_[i].store, items_[i].encoded);
    Object top = decoder.Decode();
    CHECK(!top.invalid());
    items_[i].document = new Document(top.AsFrame());
    items_[i].encoded.clear();
  }
}

}  // namespace nlp
}
