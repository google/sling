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

#ifndef NLP_PARSER_TRAINER_TRANSITION_STATE_H_
#define NLP_PARSER_TRAINER_TRANSITION_STATE_H_

#include <memory>
#include <string>
#include <vector>

#include "dragnn/core/interfaces/cloneable_transition_state.h"
#include "dragnn/core/interfaces/transition_state.h"
#include "dragnn/protos/trace.pb.h"
#include "frame/store.h"
#include "nlp/document/document.h"
#include "nlp/parser/action-table.h"
#include "nlp/parser/parser-action.h"
#include "nlp/parser/parser-state.h"
#include "nlp/parser/trainer/gold-transition-generator.h"
#include "nlp/parser/trainer/sempar-instance.h"
#include "nlp/parser/trainer/shared-resources.h"
#include "nlp/parser/trainer/transition-system-type.h"

namespace sling {
namespace nlp {

// TransitionState implementation for Sempar.
// It supports two transition systems:
// - The normal mode using the full set of actions using a ParserState.
// - A shift-only mode where the only action allowed is shift. In this
//   mode, it further supports left-to-right and right-to-left sub-modes.
class SemparState
    : public syntaxnet::dragnn::CloneableTransitionState<SemparState> {
 public:
  SemparState(SemparInstance *instance,
              const SharedResources &resources,
              TransitionSystemType type,
              bool shift_only_left_to_right);

  SemparState(const SemparState *other);
  ~SemparState();

  // Initializes this TransitionState from a previous TransitionState.
  void Init(const TransitionState &parent) override;

  // Produces a new state with the same backing data as this state.
  std::unique_ptr<SemparState> Clone() const override;

  // Return the beam index of the state passed into the initializer of this
  // TransitionState.
  const int ParentBeamIndex() const override;

  // Get the current beam index for this state.
  const int GetBeamIndex() const override;

  // Set the current beam index for this state.
  void SetBeamIndex(const int index) override;

  // Get the score associated with this transition state.
  const float GetScore() const override;

  // Set the score associated with this transition state.
  void SetScore(const float score) override;

  // Depicts this state as an HTML-language string.
  string HTMLRepresentation() const override;

  // **** END INHERITED INTERFACE ****

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

  syntaxnet::dragnn::ComponentTrace *mutable_trace() {
    CHECK(trace_ != nullptr) << "Trace is not initialized";
    return trace_;
  }
  void set_trace(syntaxnet::dragnn::ComponentTrace *trace) {
    delete trace_;
    trace_ = trace;
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

  const GoldTransitionGenerator *gold_transition_generator() const {
    return gold_transition_generator_;
  }
  void set_gold_transition_generator(GoldTransitionGenerator *g) {
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
    if (!shift_only()) return parser_state()->current();
    return shift_only_state_.left_to_right ? shift_only_state_.steps_taken :
        (shift_only_state_.size - 1 - shift_only_state_.steps_taken);
  }

  // End position (works for both SHIFT_ONLY and SEMPAR cases).
  int end() const {
    return shift_only() ? shift_only_state_.size : parser_state()->end();
  }

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
    int steps_taken = 0;
    int size = 0;
    bool left_to_right = true;
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

  // The current score of this state.
  float score_;

  // The current beam index of this state.
  int current_beam_index_;

  // The parent beam index for this state.
  int parent_beam_index_;

  // History of previous actions.
  std::vector<int> history_;

  // Maximum size of the history.
  static const int kMaxHistory = 10;

  // Trace of the history to produce this state.
  syntaxnet::dragnn::ComponentTrace *trace_ = nullptr;

  // Gold transition generator. Not owned. Only used during training.
  const GoldTransitionGenerator *gold_transition_generator_ = nullptr;

  // Gold sequence for the token range. Only populated during training.
  GoldTransitionSequence gold_sequence_;

  // Index of the next gold action to be output. Only used during training.
  int next_gold_index_ = 0;

  // Whether the state is being used to report gold transitions (i.e. training).
  mutable bool gold_transitions_required_ = false;
};

}  // namespace nlp
}  // namespace sling

#endif  // NLP_PARSER_TRAINER_TRANSITION_STATE_H_
