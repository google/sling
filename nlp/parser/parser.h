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
#include "myelin/dictionary.h"
#include "myelin/flow.h"
#include "nlp/document/document.h"
#include "nlp/parser/action-table.h"
#include "nlp/parser/parser-state.h"

namespace sling {
namespace nlp {

class ParserInstance;

// Frame semantics parser model.
class Parser {
 public:
  // Load and initialize parser model.
  void Load(Store *store, const string &filename);

  // Parse document.
  void Parse(Document *document) const;

 private:
  // Lookup cells, connectors, and parameters.
  myelin::Cell *GetCell(const string &name);
  myelin::Connector *GetConnector(const string &name);
  myelin::Tensor *GetParam(const string &name);

  // Look up word in vocabulary. Return OOV for unknown words.
  int LookupWord(const string &word) const;

  // Parser network.
  myelin::Library library_;
  myelin::Network network_;

  // Parser cells.
  myelin::Cell *lr_;                         // left-to-right LSTM cell
  myelin::Cell *rl_;                         // right-to-left LSTM cell
  myelin::Cell *ff_;                         // feed-forward cell

  // Connectors.
  myelin::Connector *lr_control_;            // left-to-right LSTM control layer
  myelin::Connector *lr_hidden_;             // left-to-right LSTM hidden layer
  myelin::Connector *rl_control_;            // right-to-left LSTM control layer
  myelin::Connector *rl_hidden_;             // right-to-left LSTM hidden layer
  myelin::Connector *ff_step_;               // FF step hidden activations

  // Left-to-right LSTM network parameters and links.
  myelin::Tensor *lr_feature_words_;         // word feature
  myelin::Tensor *lr_c_in_;                  // link to LSTM control input
  myelin::Tensor *lr_c_out_;                 // link to LSTM control output
  myelin::Tensor *lr_h_in_;                  // link to LSTM hidden input
  myelin::Tensor *lr_h_out_;                 // link to LSTM hidden output

  // Right-to-left LSTM network parameters and links.
  myelin::Tensor *rl_feature_words_;         // word feature
  myelin::Tensor *rl_c_in_;                  // link to LSTM control input
  myelin::Tensor *rl_c_out_;                 // link to LSTM control output
  myelin::Tensor *rl_h_in_;                  // link to LSTM hidden input
  myelin::Tensor *rl_h_out_;                 // link to LSTM hidden output

  // Feed-forward network parameters and links.
  myelin::Tensor *ff_feature_lr_focus_;      // LR LSTM input focus feature
  myelin::Tensor *ff_feature_rl_focus_;      // RL LSTM input focus feature

  myelin::Tensor *ff_feature_lr_attention_;  // LR LSTM frame attention feature
  myelin::Tensor *ff_feature_rl_attention_;  // LR LSTM frame attention feature

  myelin::Tensor *ff_feature_frame_create_;  // FF frame create feature
  myelin::Tensor *ff_feature_frame_focus_;   // FF frame focus feature

  myelin::Tensor *ff_feature_history_;       // history feature
  myelin::Tensor *ff_feature_roles_;         // roles feature

  myelin::Tensor *ff_lr_lstm_;               // link to LR LSTM hidden layer
  myelin::Tensor *ff_rl_lstm_;               // link to RL LSTM hidden layer
  myelin::Tensor *ff_steps_;                 // link to FF step hidden layer
  myelin::Tensor *ff_hidden_;                // link to FF hidden layer output
  myelin::Tensor *ff_output_;                // link to FF logit layer output

  // Number of attention features.
  int attention_depth_;

  // Number of history features.
  int history_size_;

  // Maximum number of role features.
  int max_roles_;

  // Number of output actions.
  int num_actions_;

  // Lexicon.
  myelin::Dictionary lexicon_;
  bool normalize_digits_ = false;
  int oov_ = -1;

  // Global store for parser.
  Store *store_ = nullptr;

  // Parser action table.
  ActionTable actions_;

  // Maximum attention index considered (exclusive).
  int frame_limit_ = 5;

  // Starting offset for (source, role) features.
  int outlink_offset_;

  // Starting offset for (role, target) features.
  int inlink_offset_;

  // Starting offset for (source, target) features.
  int unlabeled_link_offset_;

  // Starting offset for (source, role, target) features.
  int labeled_link_offset_;

  // Set of roles considered.
  HandleMap<int> roles_;

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

  // Extract features for LR LSTM.
  void ExtractFeaturesLR(int current);

  // Extract features for RL LSTM.
  void ExtractFeaturesRL(int current);

  // Extract features for FF.
  void ExtractFeaturesFF(int step);

 private:
  // Parser model.
  const Parser *parser;

  // Parser transition state.
  ParserState state;

  // Instances for network computations.
  myelin::Instance lr;
  myelin::Instance rl;
  myelin::Instance ff;

  // Channels for connectors.
  myelin::Channel lr_c;
  myelin::Channel lr_h;
  myelin::Channel rl_c;
  myelin::Channel rl_h;
  myelin::Channel ff_step;

  // Word ids.
  std::vector<int> words;

  // Frame creation and focus steps.
  std::vector<int> create_step;
  std::vector<int> focus_step;

  friend class Parser;
};

}  // namespace nlp
}  // namespace sling

#endif  // NLP_PARSER_PARSER_H_

