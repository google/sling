#ifndef NLP_PARSER_TRAINER_FEATURE_H_
#define NLP_PARSER_TRAINER_FEATURE_H_

#include <string>
#include <unordered_map>
#include <vector>

#include "base/registry.h"
#include "dragnn/protos/spec.pb.h"
#include "nlp/parser/trainer/shared-resources.h"
#include "nlp/parser/trainer/transition-state.h"

namespace sling {
namespace nlp {

class SemparFeature : public RegisterableClass<SemparFeature> {
 public:
  struct Args {
    SemparState *state = nullptr;  // not owned
    bool debug = false;
    std::vector<int64> output_ids;
    std::vector<string> output_strings;

    syntaxnet::WorkspaceSet *workspaces() {
      return state->instance()->workspaces;
    }

    ParserState *parser_state() {
      return state->parser_state();
    }
  };

  // Accessors/mutators used while constructing the feature function.
  void SetParam(const string &name, const string &val) { params_[name] = val; }

  int GetIntParam(const string &name, int default_value) const;
  bool GetBoolParam(const string &name, bool default_value) const;
  float GetFloatParam(const string &name, float default_value) const;
  const string &GetParam(const string &name, const string &default_value) const;

  int argument() const { return arg_; }
  void set_argument(int a) { arg_ = a; }

  const string &name() const { return name_; }
  void set_name(const string &n) { name_ = n; }
  const string &fml() const { return fml_; }
  void set_fml(const string &f) { fml_ = f; }

  // Methods used to generate resources required by the feature function.
  // Only called during training.
  virtual void TrainInit(
      SharedResources *resources, const string &output_folder) {}
  virtual void TrainProcess(const Document &doc) {}
  virtual int TrainFinish(syntaxnet::dragnn::ComponentSpec *spec) = 0;

  // Methods used during feature extraction.
  virtual void Init(const syntaxnet::dragnn::ComponentSpec &spec,
                    SharedResources *resources) {}
  virtual void RequestWorkspaces(syntaxnet::WorkspaceRegistry *registry) {}
  virtual void Preprocess(SemparState *state) {}
  virtual void Extract(Args *args) = 0;
  virtual const string &FeatureToString(int64 id) const = 0;

 protected:
  static string GetResource(const syntaxnet::dragnn::ComponentSpec &spec,
                            const string &name);

  static void AddResourceToSpec(const string &name,
                                const string &file,
                                syntaxnet::dragnn::ComponentSpec *spec);

 private:
  string name_;
  string fml_;
  int arg_ = 0;
  std::unordered_map<string, string> params_;
};

#define REGISTER_SEMPAR_FEATURE(name, component) \
    REGISTER_CLASS_COMPONENT(SemparFeature, name, component);

class SemparFeatureExtractor {
 public:
  void AddChannel(const string &name,
                  const string &fml,
                  int embedding_dim);
  void AddChannel(const syntaxnet::dragnn::FixedFeatureChannel &channel);
  void AddChannel(const syntaxnet::dragnn::LinkedFeatureChannel &channel);

  void Train(const std::vector<string> &train_files,
             const string &output_folder,
             bool fill_vocabulary_size,
             SharedResources *resources,
             syntaxnet::dragnn::ComponentSpec *spec);

  void Init(
      const syntaxnet::dragnn::ComponentSpec &spec, SharedResources *resources);
  void RequestWorkspaces(syntaxnet::WorkspaceRegistry *registry);
  void Preprocess(SemparState *state);
  void Extract(SemparFeature::Args *args);

 private:
  struct Channel {
    string name;
    int embedding_dim = -1;
    int vocabulary = -1;
    std::vector<SemparFeature *> features;
  };

  std::vector<Channel> channels_;
};

}  // namespace nlp
}  // namespace sling

#endif // NLP_PARSER_TRAINER_FEATURE_H_
