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

#ifndef SLING_NLP_PARSER_PARSER_H_
#define SLING_NLP_PARSER_PARSER_H_

#include <limits>
#include <string>
#include <unordered_map>
#include <vector>

#include "sling/base/logging.h"
#include "sling/base/types.h"
#include "sling/file/file.h"
#include "sling/frame/store.h"
#include "sling/myelin/compute.h"
#include "sling/myelin/flow.h"
#include "sling/nlp/document/document.h"
#include "sling/nlp/document/lexical-encoder.h"
#include "sling/nlp/parser/action-table.h"
#include "sling/nlp/parser/parser-state.h"
#include "sling/nlp/parser/roles.h"

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

  // Enable profiling. Must be called before Load().
  void EnableProfiling() {
    network_.options().profiling = true;
    network_.options().global_profiler = true;
  }

  // Enable fast fallback. Must be called before Load().
  void EnableFastFallback() { fast_fallback_ = true; }

  // Run parser on GPU if available. Must be called before Load().
  void EnableGPU();

  // Neural network for parser.
  const myelin::Network &network() const { return network_; }

  // Return the lexical encoder.
  const LexicalEncoder &encoder() const { return encoder_; }

 private:
  // Feed-forward cell.
  struct FF {
    myelin::Cell *cell;                       // feed-forward cell

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
    myelin::Tensor *prediction;               // link to FF argmax
  };

  // Initialize FF cell.
  void InitFF(const string &name, FF *ff);

  // Lookup cells and parameters.
  myelin::Cell *GetCell(const string &name);
  myelin::Tensor *GetParam(const string &name, bool optional = false);

  // Parser network.
  myelin::Library library_;
  myelin::Network network_;

  // Lexical encoder.
  LexicalEncoder encoder_;

  // Feed-forward cell.
  FF ff_;

  // Number of output actions.
  int num_actions_;

  // Global store for parser.
  Store *store_ = nullptr;

  // Parser action table.
  ActionTable actions_;

  // Maximum attention index considered (exclusive).
  int frame_limit_ = 5;

  // Set of roles considered.
  RoleSet roles_;

  // Fast fallback using argmax.
  bool fast_fallback_ = false;

  // Run parser on GPU.
  bool use_gpu_ = false;

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

  // Attach channel for FF.
  void AttachFF(int output, const myelin::BiChannel &bilstm);

  // Extract features for FF.
  void ExtractFeaturesFF(int step);

 private:
  // Get feature vector for FF.
  int *GetFF(myelin::Tensor *type) {
    return type ? ff_.Get<int>(type) : nullptr;
  }

  // Parser model.
  const Parser *parser_;

  // Instance for lexical encoding computation.
  LexicalEncoderInstance encoder_;

  // Parser transition state.
  ParserState state_;

  // Instance for network computations.
  myelin::Instance ff_;

  // Channels.
  myelin::Channel ff_step_;

  // Frame creation and focus steps.
  std::vector<int> create_step_;
  std::vector<int> focus_step_;

  friend class Parser;
};

}  // namespace nlp
}  // namespace sling

#endif  // SLING_NLP_PARSER_PARSER_H_

