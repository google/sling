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

#ifndef NLP_PARSER_PARSER_H_
#define NLP_PARSER_PARSER_H_

#include <limits>
#include <string>
#include <unordered_map>
#include <vector>

#include "base/logging.h"
#include "base/types.h"
#include "file/file.h"
#include "frame/store.h"
#include "myelin/compute.h"
#include "myelin/flow.h"
#include "myelin/profile.h"
#include "nlp/document/document.h"
#include "nlp/document/features.h"
#include "nlp/document/lexicon.h"
#include "nlp/parser/action-table.h"
#include "nlp/parser/parser-state.h"
#include "nlp/parser/roles.h"

namespace sling {
namespace nlp {

class ParserInstance;

// Frame semantics parser model.
class Parser {
 public:
  // Profile summary for each cell.
  struct Profile {
    Profile(Parser *parser)
      : lr(parser->lr_.cell), rl(parser->rl_.cell), ff(parser->ff_.cell) {}

    myelin::ProfileSummary lr;                // profile summary for LR LSTM
    myelin::ProfileSummary rl;                // profile summary for RL LSTM
    myelin::ProfileSummary ff;                // profile summary for FF
  };

  ~Parser() { delete profile_; }

  // Load and initialize parser model.
  void Load(Store *store, const string &filename);

  // Parse document.
  void Parse(Document *document) const;

  // Enable profiling. Must be called before Load().
  void EnableProfiling() {
    network_.options().profiling = true;
    network_.options().external_profiler = true;
  }

  // Return profile summary for parser.
  Profile *profile() const { return profile_; }

 private:
  // LSTM cell.
  struct LSTM {
    // Cell.
    myelin::Cell *cell;                       // LSTM cell
    bool reverse;                             // LSTM direction
    myelin::Tensor *profile;                  // LSTM profiling block

    // Connectors.
    myelin::Connector *control;               // LSTM control layer
    myelin::Connector *hidden;                // LSTM hidden layer

    // Features.
    myelin::Tensor *word_feature;             // word feature
    myelin::Tensor *prefix_feature;           // prefix feature
    myelin::Tensor *suffix_feature;           // suffix feature
    myelin::Tensor *hyphen_feature;           // hyphenation feature
    myelin::Tensor *caps_feature;             // capitalization feature
    myelin::Tensor *punct_feature;            // punctuation feature
    myelin::Tensor *quote_feature;            // quote feature
    myelin::Tensor *digit_feature;            // digit feature

    // Links.
    myelin::Tensor *c_in;                     // link to LSTM control input
    myelin::Tensor *c_out;                    // link to LSTM control output
    myelin::Tensor *h_in;                     // link to LSTM hidden input
    myelin::Tensor *h_out;                    // link to LSTM hidden output
  };

  // Feed-forward cell.
  struct FF {
    myelin::Cell *cell;                       // feed-forward cell
    myelin::Connector *step;                  // FF step hidden activations
    myelin::Tensor *profile;                  // FF profiling block

    // Features.
    myelin::Tensor *lr_focus_feature;         // LR LSTM input focus feature
    myelin::Tensor *rl_focus_feature;         // RL LSTM input focus feature

    myelin::Tensor *lr_attention_feature;     // LR LSTM frame attention feature
    myelin::Tensor *rl_attention_feature;     // LR LSTM frame attention feature

    myelin::Tensor *frame_create_feature;     // FF frame create feature
    myelin::Tensor *frame_focus_feature;      // FF frame focus feature

    myelin::Tensor *history_feature;          // history feature

    myelin::Tensor *out_roles_feature;        // out roles feature
    myelin::Tensor *in_roles_feature;         // in roles feature
    myelin::Tensor *unlabeled_roles_feature;  // unlabeled roles feature
    myelin::Tensor *labeled_roles_feature;    // labeled roles feature

    int attention_depth = 0;                  // number of attention features
    int history_size = 0;                     // number of history features
    int out_roles_size = 0;                   // max number of out roles
    int in_roles_size = 0;                    // max number of in roles
    int labeled_roles_size = 0;               // max number of unlabeled roles
    int unlabeled_roles_size = 0;             // max number of labeled roles

    // Links.
    myelin::Tensor *lr_lstm;                  // link to LR LSTM hidden layer
    myelin::Tensor *rl_lstm;                  // link to RL LSTM hidden layer
    myelin::Tensor *steps;                    // link to FF step hidden layer
    myelin::Tensor *hidden;                   // link to FF hidden layer output
    myelin::Tensor *output;                   // link to FF logit layer output
  };

  // Initialize LSTM cell.
  void InitLSTM(const string &name, LSTM *lstm, bool reverse);

  // Initialize FF cell.
  void InitFF(const string &name, FF *ff);

  // Lookup cells, connectors, and parameters.
  myelin::Cell *GetCell(const string &name);
  myelin::Connector *GetConnector(const string &name);
  myelin::Tensor *GetParam(const string &name, bool optional = false);

  // Parser network.
  myelin::Library library_;
  myelin::Network network_;

  // Cells.
  LSTM lr_;                                   // left-to-right LSTM cell
  LSTM rl_;                                   // right-to-left LSTM cell
  FF ff_;                                     // feed-forward cell

  // Profile summary.
  Profile *profile_ = nullptr;

  // Number of output actions.
  int num_actions_;

  // Lexicon.
  Lexicon lexicon_;

  // Global store for parser.
  Store *store_ = nullptr;

  // Parser action table.
  ActionTable actions_;

  // Maximum attention index considered (exclusive).
  int frame_limit_ = 5;

  // Set of roles considered.
  RoleSet roles_;

  // Symbols.
  Names names_;
  Name n_document_tokens_{names_, "/s/document/tokens"};
  Name n_token_text_{names_, "/s/token/text"};
  Name n_token_break_{names_, "/s/token/break"};

  friend class ParserInstance;
};

// Parser state for running an instance of the parser on a document.
class ParserInstance {
 public:
  ParserInstance(const Parser *parser, Document *document, int begin, int end);

  // Attach connectors for LR LSTM.
  void AttachLR(int input, int output);

  // Attach connectors for RL LSTM.
  void AttachRL(int input, int output);

  // Attach connectors for FF.
  void AttachFF(int output);

  // Extract features for LSTM.
  void ExtractFeaturesLSTM(int token,
                           const DocumentFeatures &features,
                           const Parser::LSTM &lstm,
                           myelin::Instance *data);

  // Extract features for FF.
  void ExtractFeaturesFF(int step);

 private:
  // Get feature vector for FF.
  int *GetFF(myelin::Tensor *type) {
    return type ? ff_.Get<int>(type) : nullptr;
  }

  // Parser model.
  const Parser *parser_;

  // Parser transition state.
  ParserState state_;

  // Instances for network computations.
  myelin::Instance lr_;
  myelin::Instance rl_;
  myelin::Instance ff_;

  // Channels for connectors.
  myelin::Channel lr_c_;
  myelin::Channel lr_h_;
  myelin::Channel rl_c_;
  myelin::Channel rl_h_;
  myelin::Channel ff_step_;

  // Frame creation and focus steps.
  std::vector<int> create_step_;
  std::vector<int> focus_step_;

  friend class Parser;
};

}  // namespace nlp
}  // namespace sling

#endif  // NLP_PARSER_PARSER_H_

