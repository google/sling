// Copyright 2017 Google Inc.
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

#include "sling/nlp/document/lexical-encoder.h"

#include "sling/myelin/builder.h"
#include "sling/myelin/gradient.h"
#include "sling/util/embeddings.h"

using namespace sling::myelin;

namespace sling {
namespace nlp {

void LexicalFeatures::LoadLexicon(Flow *flow) {
  // Load word vocabulary.
  Flow::Blob *vocabulary = flow->DataBlock("lexicon");
  CHECK(vocabulary != nullptr);
  int delimiter = vocabulary->GetAttr("delimiter", '\n');
  Vocabulary::BufferIterator it(vocabulary->data, vocabulary->size, delimiter);
  lexicon_.InitWords(&it);
  bool normalize = vocabulary->GetAttr("normalize_digits", false);
  int oov = vocabulary->GetAttr("oov", -1);
  lexicon_.set_normalize_digits(normalize);
  lexicon_.set_oov(oov);

  // Load affix tables.
  Flow::Blob *prefix_table = flow->DataBlock("prefixes");
  if (prefix_table != nullptr) {
    lexicon_.InitPrefixes(prefix_table->data, prefix_table->size);
  }
  Flow::Blob *suffix_table = flow->DataBlock("suffixes");
  if (suffix_table != nullptr) {
    lexicon_.InitSuffixes(suffix_table->data, suffix_table->size);
  }
}

void LexicalFeatures::SaveLexicon(myelin::Flow *flow) const {
  // Save word vocabulary.
  Flow::Blob *vocabulary = flow->AddBlob("lexicon", "dict");
  vocabulary->SetAttr("delimiter", 10);
  vocabulary->SetAttr("oov", lexicon_.oov());
  vocabulary->SetAttr("normalize_digits", lexicon_.normalize_digits());
  string buffer;
  lexicon_.WriteVocabulary(&buffer);
  vocabulary->data = flow->AllocateMemory(buffer);
  vocabulary->size = buffer.size();

  // Save prefixes.
  if (lexicon_.prefixes().size() > 0) {
    buffer.clear();
    lexicon_.WritePrefixes(&buffer);
    Flow::Blob *blob = flow->AddBlob("prefixes", "affix");
    blob->data = flow->AllocateMemory(buffer);
    blob->size = buffer.size();
  }

  // Save suffixes.
  if (lexicon_.suffixes().size() > 0) {
    buffer.clear();
    lexicon_.WriteSuffixes(&buffer);
    Flow::Blob *blob = flow->AddBlob("suffixes", "affix");
    blob->data = flow->AllocateMemory(buffer);
    blob->size = buffer.size();
  }
}

void LexicalFeatures::InitializeLexicon(Vocabulary::Iterator *words,
                                        const LexiconSpec &spec) {
  // Build dictionary.
  std::unordered_map<string, int> dictionary;
  words->Reset();
  Text word;
  int count;
  int unknown = 0;
  while (words->Next(&word, &count)) {
    if (count < spec.threshold) {
      unknown += count;
    } else if (spec.normalize_digits) {
      string normalized = word.str();
      for (char &c : normalized) {
        if (c >= '0' && c <= '9') c = '9';
      }
      dictionary[normalized] += count;
    } else {
      dictionary[word.str()] += count;
    }
  }

  // Build word list.
  std::vector<std::pair<string, int>> word_list;
  word_list.emplace_back("<UNKNOWN>", unknown);
  for (const auto &it : dictionary) {
    word_list.emplace_back(it.first, it.second);
  }

  // Build lexicon.
  Vocabulary::VectorMapIterator it(word_list);
  lexicon_.InitWords(&it);
  lexicon_.set_oov(0);
  lexicon_.set_normalize_digits(spec.normalize_digits);

  // Build affix tables.
  prefix_size_ = spec.max_prefix;
  suffix_size_ = spec.max_suffix;
  lexicon_.BuildPrefixes(spec.max_prefix);
  lexicon_.BuildSuffixes(spec.max_suffix);
}

LexicalFeatures::Variables LexicalFeatures::Build(Flow *flow,
                                                  const Library &library,
                                                  const Spec &spec,
                                                  bool learn) {
  // Build function for feature extraction and mapping.
  FlowBuilder tf(flow, name_);
  std::vector<Flow::Variable *> features;
  Flow::Variable *word_embeddings = nullptr;
  if (spec.word_dim > 0) {
    int num_words = lexicon_.size();
    if (spec.train_word_embeddings) {
      word_embeddings = tf.Parameter("word_embeddings", DT_FLOAT,
                                     {num_words, spec.word_dim});
    } else {
      word_embeddings =
          tf.Name(tf.Const(nullptr, DT_FLOAT, {num_words, spec.word_dim}),
                  "word_embeddings");
    }
    auto *f = tf.Placeholder("word", DT_INT32, {1, 1});
    auto *gather = tf.Gather(word_embeddings, f);
    features.push_back(gather);
    if (!spec.train_word_embeddings) {
      gather->producer->set(Flow::Operation::NOGRADIENT);
    }
  }
  if (spec.prefix_dim > 0) {
    auto *f = tf.Feature("prefix", lexicon_.prefixes().size(), prefix_size_,
                         spec.prefix_dim);
    features.push_back(f);
  }
  if (spec.suffix_dim > 0) {
    auto *f = tf.Feature("suffix", lexicon_.suffixes().size(), suffix_size_,
                         spec.suffix_dim);
    features.push_back(f);
  }
  if (spec.hyphen_dim > 0) {
    auto *f = tf.Feature("hyphen", DocumentFeatures::HYPHEN_CARDINALITY, 1,
                         spec.hyphen_dim);
    features.push_back(f);
  }
  if (spec.caps_dim > 0) {
    auto *f = tf.Feature("caps", DocumentFeatures::CAPITALIZATION_CARDINALITY,
                         1, spec.caps_dim);
    features.push_back(f);
  }
  if (spec.punct_dim > 0) {
    auto *f = tf.Feature("punct", DocumentFeatures::PUNCTUATION_CARDINALITY,
                         1, spec.punct_dim);
    features.push_back(f);
  }
  if (spec.quote_dim > 0) {
    auto *f = tf.Feature("quote", DocumentFeatures::QUOTE_CARDINALITY,
                         1, spec.quote_dim);
    features.push_back(f);
  }
  if (spec.digit_dim > 0) {
    auto *f = tf.Feature("digit", DocumentFeatures::DIGIT_CARDINALITY,
                         1, spec.digit_dim);
    features.push_back(f);
  }

  // Concatenate feature embeddings.
  Variables vars;
  vars.fv = tf.Name(tf.Concat(features), "feature_vector");
  vars.fv->set_out()->set_ref();

  // Build gradient function for feature extractor.
  if (learn) {
    Gradient(flow, tf.func(), library);
    vars.dfv = flow->Var("gradients/" + name_ + "/d_feature_vector");
  } else {
    vars.dfv = nullptr;
  }

  // Initialize word embeddings.
  if (learn && word_embeddings != nullptr) {
    if (spec.train_word_embeddings) {
      // Word embeddings will be loaded after network has been built.
      pretrained_embeddings_ = spec.word_embeddings;
    } else if (!spec.word_embeddings.empty()) {
      // Load static pre-trained word embeddings.
      LoadWordEmbeddings(word_embeddings, spec.word_embeddings);
    }
  }

  return vars;
}

void LexicalFeatures::Initialize(const Network &net) {
  // Get tensors.
  features_ = net.GetCell(name_);
  word_feature_ = net.LookupParameter(name_ + "/word");
  prefix_feature_ = net.LookupParameter(name_ + "/prefix");
  suffix_feature_ = net.LookupParameter(name_ + "/suffix");
  hyphen_feature_ = net.LookupParameter(name_ + "/hyphen");
  caps_feature_ = net.LookupParameter(name_ + "/capitalization");
  punct_feature_ = net.LookupParameter(name_ + "/punctuation");
  quote_feature_ = net.LookupParameter(name_ + "/quote");
  digit_feature_ = net.LookupParameter(name_ + "/digit");
  feature_vector_ = net.GetParameter(name_ + "/feature_vector");
  word_embeddings_ = net.GetParameter(name_ + "/word_embeddings");

  // Optionally initialize gradient cell.
  gfeatures_ = net.LookupCell("gradients/" + name_);
  if (gfeatures_ != nullptr) {
    const string &gname = gfeatures_->name();
    d_feature_vector_ = net.GetParameter(gname + "/d_feature_vector");
    primal_ = net.GetParameter(gname + "/primal");
  }

  // Get feature sizes.
  if (prefix_feature_ != nullptr) prefix_size_ = prefix_feature_->elements();
  if (suffix_feature_ != nullptr) suffix_size_ = suffix_feature_->elements();

  // Load pre-trained word embeddings.
  if (!pretrained_embeddings_.empty()) {
    InitWordEmbeddings(pretrained_embeddings_);
  }
}

int LexicalFeatures::LoadWordEmbeddings(Flow::Variable *matrix,
                                        const string &filename) {
  // Read word embeddings.
  EmbeddingReader reader(filename);
  reader.set_normalize(true);

  // Check that embedding matrix matches embeddings and vocabulary.
  CHECK_EQ(matrix->rank(), 2);
  CHECK_EQ(matrix->type, DT_FLOAT);
  CHECK_EQ(matrix->dim(0), lexicon_.size());
  CHECK_EQ(matrix->dim(1), reader.dim());
  CHECK(matrix->data != nullptr);

  // Initialize matrix with pre-trained word embeddings.
  int rowsize = reader.dim() * sizeof(float);
  int found = 0;
  while (reader.Next()) {
    // Check if word is in vocabulary
    int row = lexicon_.LookupWord(reader.word(), nullptr);
    if (row == lexicon_.oov()) continue;

    // Copy embedding to matrix.
    void *f = matrix->data + row * rowsize;
    memcpy(f, reader.embedding().data(), rowsize);
    found++;
  }

  return found;
};

int LexicalFeatures::InitWordEmbeddings(const string &filename) {
  // Read word embeddings.
  EmbeddingReader reader(filename);
  reader.set_normalize(true);

  // Check that embedding matrix matches embeddings and vocabulary.
  CHECK(word_embeddings_ != nullptr);
  CHECK_EQ(word_embeddings_->rank(), 2);
  CHECK_EQ(word_embeddings_->type(), DT_FLOAT);
  CHECK_EQ(word_embeddings_->dim(0), lexicon_.size());
  CHECK_EQ(word_embeddings_->dim(1), reader.dim());
  CHECK(word_embeddings_->data() != nullptr);

  // Initialize matrix with pre-trained word embeddings.
  int rowsize = reader.dim() * sizeof(float);
  int found = 0;
  while (reader.Next()) {
    // Check if word is in vocabulary
    int row = lexicon_.LookupWord(reader.word(), nullptr);
    if (row == lexicon_.oov()) continue;

    // Copy embedding to matrix.
    void *f = word_embeddings_->data() + word_embeddings_->offset(row);
    memcpy(f, reader.embedding().data(), rowsize);
    found++;
  }
  return found;
};

void LexicalFeatureExtractor::Compute(const DocumentFeatures &features,
                                      int index, float *fv) {
  // Extract word feature.
  if (lex_.word_feature_) {
    *data_.Get<int>(lex_.word_feature_) = features.word(index);
  }

  // Extract prefix feature.
  if (lex_.prefix_feature_) {
    Affix *affix = features.prefix(index);
    int *a = data_.Get<int>(lex_.prefix_feature_);
    for (int n = 0; n < lex_.prefix_size_; ++n) {
      if (affix != nullptr) {
        *a++ = affix->id();
        affix = affix->shorter();
      } else {
        *a++ = -1;
      }
    }
  }

  // Extract suffix feature.
  if (lex_.suffix_feature_) {
    Affix *affix = features.suffix(index);
    int *a = data_.Get<int>(lex_.suffix_feature_);
    for (int n = 0; n < lex_.suffix_size_; ++n) {
      if (affix != nullptr) {
        *a++ = affix->id();
        affix = affix->shorter();
      } else {
        *a++ = -1;
      }
    }
  }

  // Extract hyphen feature.
  if (lex_.hyphen_feature_) {
    *data_.Get<int>(lex_.hyphen_feature_) = features.hyphen(index);
  }

  // Extract capitalization feature.
  if (lex_.caps_feature_) {
    *data_.Get<int>(lex_.caps_feature_) = features.capitalization(index);
  }

  // Extract punctuation feature.
  if (lex_.punct_feature_) {
    *data_.Get<int>(lex_.punct_feature_) = features.punctuation(index);
  }

  // Extract quote feature.
  if (lex_.quote_feature_) {
    *data_.Get<int>(lex_.quote_feature_) = features.quote(index);
  }

  // Extract digit feature.
  if (lex_.digit_feature_) {
    *data_.Get<int>(lex_.digit_feature_) = features.digit(index);
  }

  // Set reference to output feature vector.
  data_.SetReference(lex_.feature_vector_, fv);

  // Map features through embeddings.
  data_.Compute();
}

void LexicalFeatureExtractor::Extract(const Document &document,
                                      int begin, int end, Channel *fv) {
  // Extract lexical features from document.
  DocumentFeatures features(&lex_.lexicon_);
  features.Extract(document, begin, end);

  // Compute feature vectors.
  int length = end - begin;
  fv->resize(length);
  for (int i = 0; i < length; ++i) {
    float *f = reinterpret_cast<float *>(fv->at(i));
    Compute(features, i, f);
  }
}

Channel *LexicalFeatureLearner::Extract(const Document &document,
                                        int begin, int end) {
  // Clear previous feature vectors.
  for (auto *e : extractors_) delete e;
  extractors_.clear();

  // Extract lexical features from document.
  DocumentFeatures features(&lex_.lexicon_);
  features.Extract(document, begin, end);

  // Compute feature vector for all tokens in range.
  int length = end - begin;
  fv_.resize(length);
  for (int i = 0; i < length; ++i) {
    auto *e = new LexicalFeatureExtractor(lex_);
    extractors_.push_back(e);
    float *f = reinterpret_cast<float *>(fv_.at(i));
    e->Compute(features, i, f);
  }
  return &fv_;
}

void LexicalFeatureLearner::Backpropagate(Channel *dfv) {
  CHECK_EQ(dfv->size(), fv_.size());
  for (int i = 0; i < fv_.size(); ++i) {
    gradient_.Set(lex_.d_feature_vector_, dfv, i);
    gradient_.Set(lex_.primal_, extractors_[i]->data());
    gradient_.Compute();
  }
}

BiLSTM::Outputs LexicalEncoder::Build(Flow *flow,
                                      const Library &library,
                                      const LexicalFeatures::Spec &spec,
                                      Vocabulary::Iterator *words,
                                      int dim, bool learn) {
  if (words != nullptr) {
    lex_.InitializeLexicon(words, spec.lexicon);
  }
  auto lexvars = lex_.Build(flow, library, spec, learn);
  return bilstm_.Build(flow, library, dim, lexvars.fv, lexvars.dfv);
}

void LexicalEncoder::Initialize(const myelin::Network &net) {
  lex_.Initialize(net);
  bilstm_.Initialize(net);
}

myelin::BiChannel LexicalEncoderInstance::Compute(const Document &document,
                                                  int begin, int end) {
  // Extract feature and map through feature embeddings.
  features_.Extract(document, begin, end, &fv_);

  // Compute hidden states of LSTMs.
  return bilstm_.Compute(&fv_);
}

myelin::BiChannel LexicalEncoderLearner::Compute(const Document &document,
                                                 int begin, int end) {
  // Extract feature and map through feature embeddings.
  myelin::Channel *fv = features_.Extract(document, begin, end);

  // Compute hidden states of LSTMs.
  return bilstm_.Compute(fv);
}

void LexicalEncoderLearner::Backpropagate() {
  // Backpropagate hidden state gradients through LSTMs.
  myelin::Channel *dfv = bilstm_.Backpropagate();

  // Backpropagate feature vector gradients to feature embeddings.
  features_.Backpropagate(dfv);
}

}  // namespace nlp
}  // namespace sling

