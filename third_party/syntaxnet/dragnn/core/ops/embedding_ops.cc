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
    .Attr("task_context: string = ''")
    .Attr("vocabulary: string = ''")
    .Attr("cache_vectors_locally: bool = true")
    .Attr("num_special_embeddings: int = 3")
    .Attr("embedding_init: float = 1.0")
    .Attr("seed: int = 0")
    .Attr("seed2: int = 0")
    .Doc(R"doc(
Reads word embeddings from an sstable of dist_belief.TokenEmbedding protos for
every word specified in a text vocabulary file.

word_embeddings: a tensor containing word embeddings from the specified table.
vectors: path to TF record file of word embedding vectors.
task_context: file path at which to read the task context, for its "word-map"
  input.  Exactly one of `task_context` or `vocabulary` must be specified.
vocabulary: path to vocabulary file, which contains one unique word per line, in
  order.  Exactly one of `task_context` or `vocabulary` must be specified.
cache_vectors_locally: Whether to cache the vectors file to a local temp file
  before parsing it.  This greatly reduces initialization time when the vectors
  are stored remotely, but requires that "/tmp" has sufficient space.
num_special_embeddings: Number of special embeddings to allocate, in addition to
  those allocated for real words.
embedding_init: embedding vectors that are not found in the input sstable are
  initialized randomly from a normal distribution with zero mean and
  std dev = embedding_init / sqrt(embedding_size).
seed: If either `seed` or `seed2` are set to be non-zero, the random number
  generator is seeded by the given seed.  Otherwise, it is seeded by a random
  seed.
seed2: A second seed to avoid seed collision.
)doc");

}  // namespace dragnn
}  // namespace syntaxnet

