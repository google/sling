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
// =============================================================================

#include "nlp/parser/trainer/feature.h"

#include "file/file.h"
#include "stream/file-input.h"
#include "string/strcat.h"
#include "syntaxnet/affix.h"
#include "syntaxnet/char_properties.h"
#include "syntaxnet/proto_io.h"
#include "syntaxnet/workspace.h"
#include "util/utf8/unicodetext.h"
#include "util/utf8/unilib_utf8_utils.h"

namespace sling {
namespace nlp {

using syntaxnet::AffixTable;
using syntaxnet::ProtoRecordReader;
using syntaxnet::ProtoRecordWriter;
using syntaxnet::VectorIntWorkspace;
using syntaxnet::dragnn::ComponentSpec;

class PrecomputedFeature : public SemparFeature {
 public:
   void RequestWorkspaces(syntaxnet::WorkspaceRegistry *registry) override {
    workspace_id_ = registry->Request<VectorIntWorkspace>(name());
  }

  void Preprocess(SemparState *state) override {
    auto *workspaces = state->instance()->workspaces;
    if (workspaces->Has<VectorIntWorkspace>(workspace_id_)) return;

    int size = state->num_tokens();
    VectorIntWorkspace *workspace = new VectorIntWorkspace(size);
    for (int i = 0; i < size; ++i) {
      const string &s = state->document()->token(i).text();
      workspace->set_element(i, Get(i, s));
    }
    workspaces->Set<VectorIntWorkspace>(workspace_id_, workspace);
  }

  void Extract(Args *args) override {
    int index = args->state->current() + argument();
    if (index < 0 || index >= args->state->end()) {
      return;
    }
    int64 id = args->workspaces()->Get<VectorIntWorkspace>(
        workspace_id_).element(index);
    args->output_ids.emplace_back(id);
  }

 protected:
  virtual int64 Get(int index, const string &word) = 0;

  // Workspace index.
  int workspace_id_ = -1;
};

// Feature that returns the id of the current word (offset via argument()).
class WordFeature : public PrecomputedFeature {
 public:
  void TrainInit(SharedResources *resources, const string &output_folder) {
    // Add an unknown word to the dictionary for representing OOV words.
    dictionary_file_ = StrCat(output_folder, "/", DictionaryName());
    Add(kUnknown);
  }

  void TrainProcess(const Document &document) override {
    for (int t = 0; t < document.num_tokens(); ++t) {
      const auto &token = document.token(t);
      string word = token.text();
      syntaxnet::utils::NormalizeDigits(&word);
      if (!word.empty() && !HasSpaces(word)) Add(word);
    }
  }

  int TrainFinish(ComponentSpec *spec) override {
    // Write dictionary to file.
    string contents;
    for (const string &w : id_to_word_) {
      StrAppend(&contents, !contents.empty() ? "\n" : "", w);
    }
    CHECK_OK(File::WriteContents(dictionary_file_, contents));

    // Add path to the dictionary to the spec.
    AddResourceToSpec(DictionaryName(), dictionary_file_, spec);

    return id_to_word_.size();
  }

  void Init(const ComponentSpec &spec, SharedResources *resources) override {
    string file = GetResource(spec, DictionaryName());
    CHECK(!file.empty()) << spec.DebugString();
    FileInput input(file);
    string word;
    while (input.ReadLine(&word)) {
      Add(word);
      if (word == kUnknown) oov_ = id_to_word_.size() - 1;
    }
  }

  string FeatureToString(int64 id) const override {
    return id_to_word_.at(id);
  }

 protected:
  virtual string DictionaryName() const {
    return "word-vocab";
  }

  virtual void Add(const string &word) {
    const auto &it = words_.find(word);
    if (it == words_.end()) {
      int64 id = words_.size();
      words_[word] = id;
      id_to_word_.emplace_back(word);
      CHECK_EQ(id_to_word_.size(), 1 + id);
    }
  }

  virtual int64 Get(int index, const string &word) {
    string s = word;
    syntaxnet::utils::NormalizeDigits(&s);
    const auto &it = words_.find(s);
    return it == words_.end() ? oov_ : it->second;
  }

  bool HasSpaces(const string &word) {
    for (char c : word) {
      if (c == ' ') return true;
    }
    return false;
  }

  // Unknown word.
  static constexpr char kUnknown[] = "<UNKNOWN>";

  // Id of the unknown word.
  int64 oov_ = 0;

  // Path of dictionary under construction.
  string dictionary_file_;

 private:
  // Word -> Id.
  std::unordered_map<string, int64> words_;

  // Id -> Word.
  std::vector<string> id_to_word_;
};

constexpr char WordFeature::kUnknown[];

REGISTER_SEMPAR_FEATURE("word", WordFeature);

class PrefixFeature : public WordFeature {
 public:
  ~PrefixFeature() override {
    delete affixes_;
  }

  void TrainInit(SharedResources *resources, const string &output_folder) {
    dictionary_file_ = StrCat(output_folder, "/", DictionaryName());
    length_ = GetIntParam("length", 3);
    affixes_ = new AffixTable(AffixType(), length_);
  }

  int TrainFinish(ComponentSpec *spec) override {
    syntaxnet::ProtoRecordWriter writer(dictionary_file_);
    affixes_->Write(&writer);

    // Add path to the dictionary to the spec.
    AddResourceToSpec(DictionaryName(), dictionary_file_, spec);

    return affixes_->size() + 1;  // +1 for OOV
  }

  void Init(const ComponentSpec &spec, SharedResources *resources) override {
    string filename = GetResource(spec, DictionaryName());
    CHECK(!filename.empty()) << spec.DebugString();

    length_ = GetIntParam("length", 3);
    affixes_ = new AffixTable(AffixType(), length_);
    ProtoRecordReader reader(filename);
    affixes_->Read(&reader);
    oov_ = affixes_->size();
  }

  string FeatureToString(int64 id) const override {
    return (id == oov_) ? kUnknown : affixes_->AffixForm(id);
  }

 protected:
  virtual AffixTable::Type AffixType() const {
    return AffixTable::PREFIX;
  }

  string DictionaryName() const override {
    return "prefix-table";
  }

  void Add(const string &word) override {
    affixes_->AddAffixesForWord(word.c_str(), word.size());
  }

  int64 Get(int index, const string &word) override {
    UnicodeText text;
    text.PointToUTF8(word.c_str(), word.size());
    if (length_ > text.size()) return oov_;

    UnicodeText::const_iterator start, end;
    start = end = text.begin();
    for (int i = 0; i < length_; ++i) ++end;
    string affix(start.utf8_data(), end.utf8_data() - start.utf8_data());
    int affix_id = affixes_->AffixId(affix);
    return affix_id == -1 ? oov_ : affix_id;
  }

 protected:
  AffixTable *affixes_ = nullptr;
  int length_ = 0;
  static constexpr char kUnknown[] = "<UNKNOWN_AFFIX>";
};

constexpr char PrefixFeature::kUnknown[];

REGISTER_SEMPAR_FEATURE("prefix", PrefixFeature);

class SuffixFeature : public PrefixFeature {
 protected:
  AffixTable::Type AffixType() const override {
    return AffixTable::SUFFIX;
  }

  string DictionaryName() const override {
    return "suffix-table";
  }

  int64 Get(int index, const string &word) override {
    UnicodeText text;
    text.PointToUTF8(word.c_str(), word.size());
    if (length_ > text.size()) return oov_;

    UnicodeText::const_iterator start, end;
    start = end = text.end();
    for (int i = 0; i < length_; ++i) --start;
    string affix(start.utf8_data(), end.utf8_data() - start.utf8_data());
    int affix_id = affixes_->AffixId(affix);
    return affix_id == -1 ? oov_ : affix_id;
  }
};

REGISTER_SEMPAR_FEATURE("suffix", SuffixFeature);

class HyphenFeature : public PrecomputedFeature {
 public:
  // Enumeration of values.
  enum Category {
    NO_HYPHEN = 0,
    HAS_HYPHEN = 1,
    CARDINALITY = 2,
  };

  // Returns the final domain size of the feature.
  int TrainFinish(syntaxnet::dragnn::ComponentSpec *spec) override {
    return CARDINALITY;
  }

  string FeatureToString(int64 id) const override {
    if (id == NO_HYPHEN) return "NO_HYPHEN";
    if (id == HAS_HYPHEN) return "HAS_HYPHEN";
    return "<INVALID_HYPHEN>";
  }

 protected:
  int64 Get(int index, const string &word) override {
    return (word.find('-') != string::npos ? HAS_HYPHEN : NO_HYPHEN);
  }
};

REGISTER_SEMPAR_FEATURE("hyphen", HyphenFeature);

// Feature that categorizes the capitalization of the word. If the option
// utf8=true is specified, lowercase and uppercase checks are done with UTF8
// compliant functions.
class CapitalizationFeature : public PrecomputedFeature {
 public:
  enum Category {
    LOWERCASE = 0,                     // normal word
    UPPERCASE = 1,                     // all-caps
    CAPITALIZED = 2,                   // has one cap and one non-cap
    CAPITALIZED_SENTENCE_INITIAL = 3,  // same as above but sentence-initial
    NON_ALPHABETIC = 4,                // contains no alphabetic characters
    CARDINALITY = 5,
  };

  // Returns the final domain size of the feature.
  int TrainFinish(syntaxnet::dragnn::ComponentSpec *spec) override {
    return CARDINALITY;
  }

  // Returns a string representation of the enum value.
  string FeatureToString(int64 id) const override {
    Category category = static_cast<Category>(id);
    switch (category) {
      case LOWERCASE: return "LOWERCASE";
      case UPPERCASE: return "UPPERCASE";
      case CAPITALIZED: return "CAPITALIZED";
      case CAPITALIZED_SENTENCE_INITIAL: return "CAPITALIZED_SENTENCE_INITIAL";
      case NON_ALPHABETIC: return "NON_ALPHABETIC";
      default: return "<INVALID_CAPITALIZATION>";
    }
  }

 protected:
  int64 Get(int index, const string &word) override {
    bool has_upper = false;
    bool has_lower = false;
    const char *str = word.c_str();
    for (int i = 0; i < word.length(); ++i) {
      char c = str[i];
      has_upper = (has_upper || (c >= 'A' && c <= 'Z'));
      has_lower = (has_lower || (c >= 'a' && c <= 'z'));
    }

    // Compute simple values.
    if (!has_upper && has_lower) return LOWERCASE;
    if (has_upper && !has_lower) return UPPERCASE;
    if (!has_upper && !has_lower) return NON_ALPHABETIC;

    // Else has_upper && has_lower; a normal capitalized word.  Check the break
    // level to determine whether the capitalized word is sentence-initial.
    return (index == 0) ? CAPITALIZED_SENTENCE_INITIAL : CAPITALIZED;
  }
};

REGISTER_SEMPAR_FEATURE("capitalization", CapitalizationFeature);

namespace {

int UTF8FirstLetterNumBytes(const char *utf8_str) {
  if (*utf8_str == '\0') return 0;
  return UniLib::OneCharLen(utf8_str);
}

}  // namespace

// A feature for computing whether the focus token contains any punctuation
// for ternary features.
class PunctuationAmountFeature : public WordFeature {
 public:
  // Enumeration of values.
  enum Category {
    NO_PUNCTUATION = 0,
    SOME_PUNCTUATION = 1,
    ALL_PUNCTUATION = 2,
    CARDINALITY = 3,
  };

  // Returns the final domain size of the feature.
  int TrainFinish(syntaxnet::dragnn::ComponentSpec *spec) override {
    return CARDINALITY;
  }

  string FeatureToString(int64 id) const override {
    Category category = static_cast<Category>(id);
    switch (category) {
      case NO_PUNCTUATION: return "NO_PUNCTUATION";
      case SOME_PUNCTUATION: return "SOME_PUNCTUATION";
      case ALL_PUNCTUATION: return "ALL_PUNCTUATION";
      default: return "<INVALID_PUNCTUATION>";
    }
  }

 protected:
  int64 Get(int index, const string &word) override {
    bool has_punctuation = false;
    bool all_punctuation = true;

    const char *start = word.c_str();
    const char *end = word.c_str() + word.size();
    while (start < end) {
      int char_length = UTF8FirstLetterNumBytes(start);
      bool char_is_punct =
          syntaxnet::is_punctuation_or_symbol(start, char_length);
      all_punctuation &= char_is_punct;
      has_punctuation |= char_is_punct;
      if (!all_punctuation && has_punctuation) return SOME_PUNCTUATION;
      start += char_length;
    }
    if (!all_punctuation) return NO_PUNCTUATION;
    return ALL_PUNCTUATION;
  }
};

REGISTER_SEMPAR_FEATURE("punctuation", PunctuationAmountFeature);

// A feature for a feature that returns whether the word is an open or
// close quotation mark, based on its relative position to other quotation marks
// in the sentence.
class QuoteFeature : public WordFeature {
 public:
  // Enumeration of values.
  enum Category {
    NO_QUOTE = 0,
    OPEN_QUOTE = 1,
    CLOSE_QUOTE = 2,
    UNKNOWN_QUOTE = 3,
    CARDINALITY = 4,
  };

  // Returns the final domain size of the feature.
  int TrainFinish(syntaxnet::dragnn::ComponentSpec *spec) override {
    return CARDINALITY;
  }

  string FeatureToString(int64 id) const override {
    Category category = static_cast<Category>(id);
    switch (category) {
      case NO_QUOTE: return "NO_QUOTE";
      case OPEN_QUOTE: return "OPEN_QUOTE";
      case CLOSE_QUOTE: return "CLOSE_QUOTE";
      case UNKNOWN_QUOTE: return "UNKNOWN_QUOTE";
      default: return "<INVALID_QUOTE>";
    }
  }

  // Override preprocess to compute open and close quotes from prior context of
  // the sentence.
  void Preprocess(SemparState *state) override {
    auto *workspaces = state->instance()->workspaces;
    if (workspaces->Has<VectorIntWorkspace>(workspace_id_)) return;

    // For double quote ", it is unknown whether they are open or closed without
    // looking at the prior tokens in the sentence.  in_quote is true iff an odd
    // number of " marks have been seen so far in the sentence (similar to the
    // behavior of some tokenizers).
    int size = state->num_tokens();
    VectorIntWorkspace *workspace = new VectorIntWorkspace(size);
    bool in_quote = false;
    for (int i = 0; i < size; ++i) {
      const string &s = state->document()->token(i).text();
      int64 id = Get(i, s);
      if (id == UNKNOWN_QUOTE) {
        // Update based on in_quote and flip in_quote.
        id = in_quote ? CLOSE_QUOTE : OPEN_QUOTE;
        in_quote = !in_quote;
      }
      workspace->set_element(i, id);
    }
    workspaces->Set<VectorIntWorkspace>(workspace_id_, workspace);
  }

 protected:
  int64 Get(int index, const string &word) override {
    // Penn Treebank open and close quotes are multi-character.
    if (word == "``") return OPEN_QUOTE;
    if (word == "''") return CLOSE_QUOTE;
    if (word.length() == 1) {
      int char_len = UTF8FirstLetterNumBytes(word.c_str());
      bool is_open = syntaxnet::is_open_quote(word.c_str(), char_len);
      bool is_close = syntaxnet::is_close_quote(word.c_str(), char_len);
      if (is_open && !is_close) return OPEN_QUOTE;
      if (is_close && !is_open) return CLOSE_QUOTE;
      if (is_open && is_close) return UNKNOWN_QUOTE;
    }
    return NO_QUOTE;
  }
};

REGISTER_SEMPAR_FEATURE("quote", QuoteFeature);

// Feature that computes whether a word has digits or not.
class DigitFeature : public WordFeature {
 public:
  // Enumeration of values.
  enum Category {
    NO_DIGIT = 0,
    SOME_DIGIT = 1,
    ALL_DIGIT = 2,
    CARDINALITY = 3,
  };

  // Returns the final domain size of the feature.
  int TrainFinish(syntaxnet::dragnn::ComponentSpec *spec) override {
    return CARDINALITY;
  }

  string FeatureToString(int64 id) const override {
    Category category = static_cast<Category>(id);
    switch (category) {
      case NO_DIGIT: return "NO_DIGIT";
      case SOME_DIGIT: return "SOME_DIGIT";
      case ALL_DIGIT: return "ALL_DIGIT";
      default: return "<INVALID_DIGIT>";
    }
  }

 protected:
  int64 Get(int index, const string &word) override {
    bool has_digit = isdigit(word[0]);
    bool all_digit = has_digit;
    for (size_t i = 1; i < word.length(); ++i) {
      bool char_is_digit = isdigit(word[i]);
      all_digit = all_digit && char_is_digit;
      has_digit = has_digit || char_is_digit;
      if (!all_digit && has_digit) return SOME_DIGIT;
    }
    if (!all_digit) return NO_DIGIT;
    return ALL_DIGIT;
  }
};

REGISTER_SEMPAR_FEATURE("digit", DigitFeature);

}  // namespace nlp
}  // namespace sling
