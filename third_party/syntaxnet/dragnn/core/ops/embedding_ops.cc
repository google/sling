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

#include "tensorflow/core/framework/op.h"

namespace syntaxnet {
namespace dragnn {

REGISTER_OP("WordEmbeddingInitializer")
    .Output("word_embeddings: float")
    .Attr("vectors: string")
    .Attr("vocabulary: string = ''")
    .Attr("seed: int = 0")
    .Attr("seed2: int = 0")
    .Doc(R"doc(
Reads word embeddings from an sstable of dist_belief.TokenEmbedding protos for
every word specified in a text vocabulary file.
Embedding vectors that are not found in the input embeddings are
initialized randomly from a normal distribution with zero mean and
std dev = 1.0 / sqrt(embedding_size).

word_embeddings: a tensor containing word embeddings from the specified table.
vectors: path to TF record file of word embedding vectors.
vocabulary: path to vocabulary file, which contains one unique word per line, in
  order.
seed: If either `seed` or `seed2` are set to be non-zero, the random number
  generator is seeded by the given seed.  Otherwise, it is seeded by a random
  seed.
seed2: A second seed to avoid seed collision.
)doc");

}  // namespace dragnn
}  // namespace syntaxnet

