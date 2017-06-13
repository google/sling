#include "nlp/parser/trainer/feature.h"

#include "file/file.h"
#include "stream/file-input.h"
#include "syntaxnet/workspace.h"

namespace sling {
namespace nlp {

using syntaxnet::VectorIntWorkspace;
using syntaxnet::dragnn::ComponentSpec;

// Feature that returns the id of the current word (offset via argument()).
class WordFeature : public SemparFeature {
 public:
  void TrainInit(SharedResources *resources, const string &output_folder) {
    // Add an unknown word to the dictionary for representing OOV words.
    dictionary_file_ = StrCat(output_folder, "/word-vocab");
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
    AddResourceToSpec("word-vocab", dictionary_file_, spec);

    return id_to_word_.size();
  }

  void Init(const ComponentSpec &spec, SharedResources *resources) override {
    string file = GetResource(spec, "word-vocab");
    CHECK(!file.empty()) << spec.DebugString();
    FileInput input(file);
    string word;
    while (input.ReadLine(&word)) {
      Add(word);
      if (word == kUnknown) oov_ = id_to_word_.size() - 1;
    }
  }

  void RequestWorkspaces(syntaxnet::WorkspaceRegistry *registry) override {
    workspace_id_ = registry->Request<VectorIntWorkspace>("word");
  }

  void Preprocess(SemparState *state) override {
    auto *workspaces = state->instance()->workspaces;
    if (workspaces->Has<VectorIntWorkspace>(workspace_id_)) return;

    int base = state->parser_state()->begin();
    int size = state->parser_state()->end() - base;
    VectorIntWorkspace *workspace = new VectorIntWorkspace(size);
    for (int i = 0; i < size; ++i) {
      workspace->set_element(i, Get(state->document()->token(i + base).text()));
    }
    workspaces->Set<VectorIntWorkspace>(workspace_id_, workspace);
  }

  void Extract(Args *args) override {
    int index = args->parser_state()->current() + argument();
    if (index < args->parser_state()->begin() ||
        index >= args->parser_state()->end()) {
      return;
    }
    int64 id = args->workspaces()->Get<VectorIntWorkspace>(
        workspace_id_).element(index - args->parser_state()->begin());
    args->output_ids.emplace_back(id);
  }

  const string &FeatureToString(int64 id) const override {
    return id_to_word_.at(id);
  }


 private:
  void Add(const string &word) {
    const auto &it = words_.find(word);
    if (it == words_.end()) {
      int64 id = words_.size();
      words_[word] = id;
      id_to_word_.emplace_back(word);
      CHECK_EQ(id_to_word_.size(), 1 + id);
    }
  }

  int64 Get(const string &word) {
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

  // Path of dictionary under construction.
  string dictionary_file_;

  // Word -> Id.
  std::unordered_map<string, int64> words_;

  // Id -> Word.
  std::vector<string> id_to_word_;

  // Id of the unknown word.
  int64 oov_ = 0;

  // Workspace index.
  int workspace_id_ = -1;
};

constexpr char WordFeature::kUnknown[];

REGISTER_SEMPAR_FEATURE("word", WordFeature);

}  // namespace nlp
}  // namespace sling
