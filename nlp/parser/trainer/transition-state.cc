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

#include "nlp/parser/trainer/transition-state.h"

#include "base/logging.h"
#include "string/strcat.h"

namespace sling {
namespace nlp {

SemparState::SemparState(SemparInstance *instance,
                         const SharedResources &resources,
                         TransitionSystemType type,
                         bool shift_only_left_to_right) {
  instance_ = instance;
  resources_ = &resources;
  system_type_ = type;
  CHECK(!action_table()->action_checks())
      << "Fingerprint-based checks not currently supported in "
      << "SemparTransitiontate.";
  score_ = 0;

  if (!shift_only()) {
    allowed_.assign(action_table()->NumActions(), false);
    parser_state_ = new ParserState(instance->store, 0, num_tokens());
    ComputeAllowed();
  } else {
    shift_only_state_.begin = 0;
    shift_only_state_.end = num_tokens();
    shift_only_state_.steps_taken = 0;
    shift_only_state_.left_to_right = shift_only_left_to_right;
  }
}

SemparState::~SemparState() {
  delete parser_state_;
  delete features_;
}

const float SemparState::GetScore() const { return score_; }

void SemparState::SetScore(const float score) { score_ = score; }

string SemparState::DebugString() const {
  if (shift_only()) {
    int size = document()->num_tokens();
    return StrCat("steps_taken=", shift_only_state_.steps_taken,
                  ",current_token:",
                  size <= current() ? " <EOS>" :
                  (current() < 0 ? "<BOS>" :
                  document()->token(current()).text()));
  } else {
    return parser_state_->DebugString();
  }
}

int SemparState::NextGoldAction() {
  if (IsFinal()) return -1;
  if (shift_only()) return 0;  // only one action

  if (gold_sequence_.actions().empty()) {
    CHECK(gold_transition_generator_ != nullptr);
    gold_transition_generator_->Generate(
        *instance_->document,
        parser_state_->begin(),
        parser_state_->end(),
        &gold_sequence_,
        nullptr  /* report */);
    next_gold_index_ = 0;
  }

  const ParserAction &action = gold_sequence_.action(next_gold_index_);
  int index = action_table()->Index(action);
  LOG_IF(FATAL, index == -1) << action.ToString(store());
  gold_transitions_required_ = true;

  return index;
}

bool SemparState::IsFinal() const {
  return shift_only() ?
      (shift_only_state_.steps_taken >= shift_only_state_.size()) :
      parser_state_->done();
}

bool SemparState::Allowed(int action) const {
  if (IsFinal()) return false;
  return shift_only() ? (action == 0) : allowed_.at(action);
}

void SemparState::PerformAction(int action_index) {
  if (shift_only()) {
    CHECK_EQ(action_index, 0);
    ++shift_only_state_.steps_taken;
    return;
  }

  const ParserAction &action = action_table()->Action(action_index);

  if (gold_transitions_required_) {
    // If we are truly in training mode, then only gold actions are applicable.
    const ParserAction &expected = gold_sequence_.action(next_gold_index_);
    if (action != expected) {
      string debug = "Given gold action != expected gold action.";
      StrAppend(&debug, "\nParser State: ", parser_state_->DebugString());
      StrAppend(&debug, "\nExpected : ", expected.ToString(store()));
      StrAppend(&debug, "\nGot : ", action.ToString(store()));
      LOG(FATAL) << debug;
    }

    // Since the action table only allows a large percentile of all actions,
    // it is possible that the gold action is not allowed as per the table.
    // If so, explicitly whitelist the action.
    if (!allowed_[action_index]) {
      LOG_FIRST_N(WARNING, 50) << "Forcibly enabling disallowed gold action: "
          << action.ToString(store());
      allowed_[action_index] = true;
    }
    next_gold_index_++;
  }

  CHECK(allowed_[action_index]) << "Action not allowed for document:"
      << instance_->document->GetText() << ", action: "
      << action.ToString(store()) << " at state:\n"
      << parser_state_->DebugString();

  parser_state_->Apply(action);

  // Update history.
  history_.emplace_back(action_index);
  if (history_.size() > kMaxHistory) {
    history_.erase(history_.begin(), history_.begin() + 1);
  }

  // Update step information.
  step_info_.Update(action, *parser_state_);

  // Compute the set of allowed actions for the resulting state.
  ComputeAllowed();

  // Update role graph.
  if ((role_frame_limit_ > 0) && (action.type != ParserAction::SHIFT)) {
    role_graph_.Compute(*parser_state(), role_frame_limit_, resources_->roles);
  }
}

void SemparState::ComputeAllowed() {
  CHECK(!shift_only());

  // Disable all actions by default.
  allowed_.assign(allowed_.size(), false);

  // If we are at the end, then STOP is the only allowed action.
  if (IsFinal()) {
    allowed_[action_table()->StopIndex()] = true;
    return;
  }

  // If we have taken too many actions at this token, then just advance.
  // We use a small padding on the action limit to allow for variations not
  // seen in the training corpus.
  int max_wait_till_shift = 4 + action_table()->max_actions_per_token();
  if (step_info_.NumStepsSinceShift() > max_wait_till_shift) {
    allowed_[action_table()->ShiftIndex()] = true;
    return;
  }

  // Compute the rest of the allowed actions as per the action table.
  action_table()->Allowed(*parser_state_, {} /* fingerprints */, &allowed_);
}

void SemparState::StepInformation::Update(const ParserAction &action,
                                          const ParserState &state) {
  // Note that except for SHIFT and STOP, all actions set the focus.
  bool focus_set =
      (action.type != ParserAction::SHIFT) &&
      (action.type != ParserAction::STOP);
  if (focus_set && state.AttentionSize() > 0) {
    int focus = state.Attention(0);
    if (creation_step.size() < focus + 1) {
      creation_step.resize(focus + 1);
      creation_step[focus] = steps;
    }
    if (focus_step.size() < focus + 1) focus_step.resize(focus + 1);
    focus_step[focus] = steps;
  }
  steps++;
  steps_since_shift = (action.type == ParserAction::SHIFT) ? 0 :
    (steps_since_shift + 1);
}

}  // namespace nlp
}  // namespace sling
