// Copyright 2017 Google Inc. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef SLING_UTIL_EMBEDDINGS_H_
#define SLING_UTIL_EMBEDDINGS_H_

#include <string>
#include <vector>

#include "sling/base/types.h"
#include "sling/stream/file.h"
#include "sling/stream/input.h"
#include "sling/stream/output.h"

namespace sling {

// Read embeddings in Mikolov format, see https://github.com/tmikolov/word2vec.
class EmbeddingReader {
 public:
  // Initialize embedding reader.
  EmbeddingReader(const string &filename);

  // Number of words in embedding file.
  int num_words() const { return num_words_; }

  // Embedding dimension.
  int dim() const { return dim_; }

  // Get next embedding from file.
  bool Next();

  // Current word.
  const string &word() const { return word_; }

  // Current embedding.
  const std::vector<float> &embedding() { return embedding_; }

  // Normalize embedding vectors to unit length.
  bool normalize() const { return normalize_; }
  void set_normalize(bool normalize) { normalize_ = normalize; }

 private:
  // Read next word from input.
  void NextWord(string *output);

  // Input stream.
  FileInputStream stream_;
  Input input_;

  // Number of words.
  int num_words_;

  // Current  word number.
  int current_word_;

  // Embedding dimension.
  int dim_;

  // Current word.
  string word_;

  // Current embedding vector.
  std::vector<float> embedding_;

  // Normalize embedding vectors to unit length.
  bool normalize_ = false;
};

// Write embeddings in Mikolov format.
class EmbeddingWriter {
 public:
  // Initialize embedding writer.
  EmbeddingWriter(const string &filename, int num_words, int dim);

  // Write next embedding.
  void Write(const string &word, const std::vector<float> &embedding);

  // Flush and close output.
  bool Close();

 private:
  // Output stream.
  FileOutputStream stream_;
  Output output_;
};

}  // namespace sling

#endif  // SLING_UTIL_EMBEDDINGS_H_

