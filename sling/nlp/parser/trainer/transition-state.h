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

#ifndef SLING_NLP_PARSER_TRAINER_TRANSITION_STATE_H_
#define SLING_NLP_PARSER_TRAINER_TRANSITION_STATE_H_

#include <memory>
#include <string>
#include <vector>

#include "dragnn/core/interfaces/transition_state.h"
#include "sling/frame/store.h"
#include "sling/nlp/document/document.h"
#include "sling/nlp/document/features.h"
#include "sling/nlp/parser/action-table.h"
#include "sling/nlp/parser/parser-action.h"
#include "sling/nlp/parser/parser-state.h"
#include "sling/nlp/parser/roles.h"
#include "sling/nlp/parser/trainer/sempar-instance.h"
#include "sling/nlp/parser/trainer/shared-resources.h"
#include "sling/nlp/parser/trainer/transition-generator.h"
#include "sling/nlp/parser/trainer/transition-system-type.h"
#include "sling/string/strcat.h"

namespace sling {
namespace nlp {

// TransitionState implementation for Sempar.
// It supports two transition systems:
// - The normal mode using the full set of actions using a ParserState.
// - A shift-only mode where the only action allowed is shift. In this
//   mode, it further supports left-to-right and right-to-left sub-modes.
class SemparState : public syntaxnet::dragnn::TransitionState {
 public:
  SemparState(SemparInstance *instance,
              const SharedResources &resources,
              TransitionSystemType type,
              bool shift_only_left_to_right);

  ~SemparState();

  // Get the score associated with this transition state.
  const float GetScore() const override;

  // Set the score associated with this transition state.
  void SetScore(const float score) override;

  // Depicts this state as a debug string.
  string DebugString() const;

  // Returns whether the state can performs more actions.
  bool IsFinal() const;

  // Returns whether the state can perform action 'action_index'.
  bool Allowed(int action_index) const;

  // Performs action 'action_index'.
  void PerformAction(int action_index);

  // Returns the next gold action in the gold sequence (or -1 if done).
  // Calls to NextGoldAction() should be interleaved with PerformAction().
  int NextGoldAction();

  // Accessors.
  ParserState *parser_state() { return parser_state_; }
  SemparInstance *instance() { return instance_; }
  const SemparInstance *instance() const { return instance_; }
  Document *document() { return instance_->document; }
  const Document *document() const { return instance_->document; }
  Store *store() { return instance_->store; }
  int num_tokens() const { return instance_->document->num_tokens(); }

  string current_token_text() const {
    int c = current();
    if (c < 0 || c >= num_tokens()) {
      return StrCat("<", c, ">");
    }
    return document()->token(c).text();
  }

  // Returns the size of the actions history.
  int HistorySize() const { return history_.size(); }

  // Returns an action from the history, where offset = 0 corresponds to the
  // latest action.
  int History(int offset) const {
    return history_[history_.size() - 1 - offset];
  }

  // Accessors.
  const ActionTable *action_table() const { return &resources_->table; }

  Store *store() const {
    return parser_state_ == nullptr ? nullptr : parser_state_->store();
  }

  const TransitionGenerator *gold_transition_generator() const {
    return gold_transition_generator_;
  }
  void set_gold_transition_generator(TransitionGenerator *g) {
    gold_transition_generator_ = g;
  }

  const ParserState *parser_state() const { return parser_state_; }

  // Returns the index of the step which created/focused the frame at
  // position 'index' in the attention buffer.
  int CreationStep(int index) const {
    if (index < 0 || index >= parser_state_->AttentionSize()) return -1;
    return step_info_.CreationStep(parser_state_->Attention(index));
  }

  int FocusStep(int index) const {
    if (index < 0 || index >= parser_state_->AttentionSize()) return -1;
    return step_info_.FocusStep(parser_state_->Attention(index));
  }

  // Whether the state is for a shift-only instance.
  bool shift_only() const { return system_type_ == SHIFT_ONLY; }

  // Returns the number of steps taken by the state so far.
  int NumSteps() const {
    return shift_only() ? shift_only_state_.steps_taken : step_info_.NumSteps();
  }

  // Current position (works for both SHIFT_ONLY and SEMPAR cases).
  int current() const {
    return shift_only() ? shift_only_state_.current() :
        parser_state()->current();
  }

  // Begin/End position.
  int begin() const {
    return shift_only() ? shift_only_state_.begin : parser_state()->begin();
  }

  int end() const {
    return shift_only() ? shift_only_state_.end : parser_state()->end();
  }

  // Sets the lexical feature vector.
  void set_document_features(DocumentFeatures *f) {
    delete features_;
    features_ = f;
  }

  // Returns the lexical features.
  DocumentFeatures *features() const { return features_; }

  // Returns the role graph.
  const RoleGraph &role_graph() const { return role_graph_; }

  // Sets frame limit for computing the role graph.
  void set_role_frame_limit(int l) { role_frame_limit_ = l; }

 private:
  // Holds frame -> step information, i.e. at which step was a frame created or
  // brought to focus.
  struct StepInformation {
    // Number of steps (i.e. actions) taken so far.
    int steps = 0;

    // Number of steps since the last shift action.
    int steps_since_shift = 0;

    // Absolute frame index -> Step at which the frame was created.
    std::vector<int> creation_step;

    // Absolute frame index -> Most recent step at which the frame was focused.
    std::vector<int> focus_step;

    // Accessor.
    int NumSteps() const { return steps; }
    int NumStepsSinceShift() const { return steps_since_shift; }

    // Updates the step information using 'action' that resulted in 'state'.
    void Update(const ParserAction &action, const ParserState &state);

    // Returns the creation/focus step for the frame at absolute index 'index'.
    int CreationStep(int index) const { return creation_step[index]; }
    int FocusStep(int index) const { return focus_step[index]; }
  };

  // State information for shift-only cases.
  struct ShiftOnlyState {
    int begin = 0;   // beginning token index
    int end = 0;     // ending token index (exclusive)
    int steps_taken = 0;
    bool left_to_right = true;

    int current() const {
      return left_to_right ? (begin + steps_taken) : (end - 1 - steps_taken);
    }

    int size() const { return end - begin; }
  };

  // Computes the set of allowed actions for the current ParserState.
  void ComputeAllowed();

  // Shared resources. Not owned.
  const SharedResources *resources_ = nullptr;

  // Type of transition system.
  TransitionSystemType system_type_;

  // State information in case system_type_ is SHIFT_ONLY.
  ShiftOnlyState shift_only_state_;

  // Underlying ParserState if system_type is SEMPAR. Owned.
  ParserState *parser_state_ = nullptr;

  // Instance that is being examined with this state. Not owned.
  SemparInstance *instance_ = nullptr;

  // Bitmap of allowed actions for 'parser_state_'.
  std::vector<bool> allowed_;

  // Step information.
  StepInformation step_info_;

  // Lexical features. Owned.
  DocumentFeatures *features_ = nullptr;

  // The current score of this state.
  float score_;

  // History of previous actions.
  std::vector<int> history_;

  // Maximum size of the history.
  static const int kMaxHistory = 10;

  // Gold transition generator. Not owned. Only used during training.
  const TransitionGenerator *gold_transition_generator_ = nullptr;

  // Gold sequence for the token range. Only populated during training.
  TransitionSequence gold_sequence_;

  // Index of the next gold action to be output. Only used during training.
  int next_gold_index_ = 0;

  // Whether the state is being used to report gold transitions (i.e. training).
  mutable bool gold_transitions_required_ = false;

  // Role graph.
  RoleGraph role_graph_;

  // Role frame limit.
  int role_frame_limit_ = 0;
};

}  // namespace nlp
}  // namespace sling

#endif  // SLING_NLP_PARSER_TRAINER_TRANSITION_STATE_H_
