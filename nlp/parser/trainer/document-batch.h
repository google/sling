#ifndef NLP_PARSER_TRAINER_DOCUMENT_BATCH_H_
#define NLP_PARSER_TRAINER_DOCUMENT_BATCH_H_

#include <string>
#include <utility>
#include <vector>

#include "dragnn/core/interfaces/input_batch.h"
#include "frame/store.h"
#include "nlp/parser/trainer/sempar-instance.h"

namespace sling {
namespace nlp {

// InputBatch implementation for SLING document batches.
class DocumentBatch : public syntaxnet::dragnn::InputBatch {
 public:
  // Translates from a vector of serialized Document frames.
  void SetData(const std::vector<string> &data) override;

  // Translates to a vector of serialized Document frames.
  const std::vector<string> GetSerializedData() const override;

  // Returns the size of the batch.
  int size() const { return items_.size(); }
  SemparInstance *item(int i) { return &items_[i]; }

  // Decodes the documents in the batch. 'global' is used to construct the
  // local stores.
  void Decode(Store *global);

 private:
  // Document batch.
  std::vector<SemparInstance> items_;
};

}  // namespace nlp
}  // namespace sling

#endif  // NLP_PARSER_TRAINER_DOCUMENT_BATCH_H_
