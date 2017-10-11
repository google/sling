/* Copyright 2016 Google Inc. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/


#include <string>
#include <unordered_map>
#include <vector>

#include "base/types.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "util/embeddings.h"

using tensorflow::OpKernel;
using tensorflow::OpKernelConstruction;
using tensorflow::OpKernelContext;
using tensorflow::Tensor;
using tensorflow::DT_FLOAT;
using tensorflow::TensorShape;
using tensorflow::errors::InvalidArgument;
using tensorflow::error::OUT_OF_RANGE;
using tensorflow::DEVICE_CPU;

namespace syntaxnet {
namespace dragnn {

class WordEmbeddingInitializer : public OpKernel {
 public:
  explicit WordEmbeddingInitializer(OpKernelConstruction *context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("vocabulary", &vocabulary_path_));
    OP_REQUIRES_OK(context, context->GetAttr("vectors", &vectors_path_));

    // Convert the seeds into a single 64-bit seed.  NB: seed=0,seed2=0 converts
    // into seed_=0, which causes Eigen PRNGs to seed non-deterministically.
    int seed, seed2;
    OP_REQUIRES_OK(context, context->GetAttr("seed", &seed));
    OP_REQUIRES_OK(context, context->GetAttr("seed2", &seed2));
    seed_ = static_cast<uint64>(seed) | static_cast<uint64>(seed2) << 32;

    // Sets up number and type of inputs and outputs.
    OP_REQUIRES_OK(context, context->MatchSignature({}, {DT_FLOAT}));
  }

  void Compute(OpKernelContext *context) override {
    std::unordered_map<string, int64> vocab;
    OP_REQUIRES_OK(context, LoadVocabulary(&vocab));

    // Load the embedding vectors into a matrix.  Since the |embedding_matrix|
    // output cannot be allocated until the embedding dimension is known, delay
    // allocation until the first iteration of the loop.
    sling::EmbeddingReader reader(vectors_path_);
    Tensor *embedding_matrix;
    OP_REQUIRES_OK(context,
                   InitRandomEmbeddingMatrix(vocab.size(), reader.dim(),
                                             context, &embedding_matrix));
    int64 count = 0;
    while (reader.Next()) {
      if (vocab.find(reader.word()) != vocab.end()) {
        SetNormalizedRow(reader.embedding(), vocab[reader.word()],
                         embedding_matrix);
        ++count;
      }
    }
    LOG(INFO) << "Initialized with " << count << " pre-trained embedding "
              << "vectors out of a vocabulary of " << vocab.size();
  }

 private:
  // Loads the |vocabulary| from the |vocabulary_path_| file.
  // The file is assumed to list one word per line, including <UNKNOWN>.
  // The zero-based line number is taken as the id of the corresponding word.
  tensorflow::Status LoadVocabulary(
      std::unordered_map<string, int64> *vocabulary) const {
    vocabulary->clear();
    string text;
    TF_RETURN_IF_ERROR(
        ReadFileToString(tensorflow::Env::Default(), vocabulary_path_, &text));

    // Chomp a trailing newline, if any, to avoid producing a spurious empty
    // term at the end of the vocabulary file.
    if (!text.empty() && text.back() == '\n') text.pop_back();
    for (const string &line : tensorflow::str_util::Split(text, "\n")) {
      if (vocabulary->find(line) != vocabulary->end()) {
        return InvalidArgument("Vocabulary file at ", vocabulary_path_,
                               " contains multiple instances of term: ", line);
      }

      const int64 index = vocabulary->size();
      (*vocabulary)[line] = index;
    }

    return tensorflow::Status::OK();
  }

  // Allocates the |embedding_matrix| based on the vocabulary size and embedding
  // dimension and initializes it to random values, or returns non-OK on error.
  tensorflow::Status InitRandomEmbeddingMatrix(
      int words, int dims, OpKernelContext *context,
      Tensor **embedding_matrix) const {
    const int rows = words;
    const int columns = dims;
    TF_RETURN_IF_ERROR(context->allocate_output(0, TensorShape({rows, columns}),
                                                embedding_matrix));
    auto matrix = (*embedding_matrix)->matrix<float>();
    Eigen::internal::NormalRandomGenerator<float> prng(seed_);
    matrix = matrix.random(prng) * (1.0f / sqrtf(columns));
    return tensorflow::Status::OK();
  }

  // Sets embedding_matrix[row] to a normalized version of the given vector.
  void SetNormalizedRow(const std::vector<float> &vector, const int row,
                        Tensor *embedding_matrix) {
    float norm = 0.0f;
    for (int col = 0; col < vector.size(); ++col) {
      float val = vector[col];
      norm += val * val;
    }
    norm = sqrt(norm);
    for (int col = 0; col < vector.size(); ++col) {
      embedding_matrix->matrix<float>()(row, col) = vector[col] / norm;
    }
  }

  // Path to the vocabulary file.
  string vocabulary_path_;

  // Seed for random initialization.
  uint64 seed_ = 0;

  // Path to recordio with word embedding vectors.
  string vectors_path_;
};

REGISTER_KERNEL_BUILDER(Name("WordEmbeddingInitializer").Device(DEVICE_CPU),
                        WordEmbeddingInitializer);

}  // namespace dragnn
}  // namespace syntaxnet
