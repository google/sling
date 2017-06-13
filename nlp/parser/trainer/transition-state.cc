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

#include "nlp/parser/trainer/transition-state.h"

#include "base/logging.h"
#include "string/strcat.h"

namespace sling {
namespace nlp {

SemparState::SemparState(SemparInstance *instance,
                         const SharedResources &resources,
                         TransitionSystemType type) {
  instance_ = instance;
  resources_ = &resources;
  system_type_ = type;
  CHECK(!action_table()->action_checks())
      << "Fingerprint-based checks not currently supported in "
      << "SemparTransitiontate.";
  score_ = 0;
  current_beam_index_ = -1;
  parent_beam_index_ = 0;

  if (!shift_only()) {
    allowed_.assign(action_table()->NumActions(), false);
    parser_state_ = new ParserState(instance->store, 0, num_tokens());
    ComputeAllowed();
  } else {
    shift_only_state_.current = 0;
    shift_only_state_.size = num_tokens();
  }
}

SemparState::SemparState(const SemparState *other) {
  if (other->parser_state_ != nullptr) {
    parser_state_ = new ParserState(*other->parser_state_);
  }
  instance_ = other->instance_;
  resources_ = other->resources_;
  system_type_ = other->system_type_;
  shift_only_state_ = other->shift_only_state_;
  allowed_ = other->allowed_;
  step_info_ = other->step_info_;
  score_ = other->score_;
  history_ = other->history_;
  gold_transition_generator_ = other->gold_transition_generator_;
  current_beam_index_ = other->current_beam_index_;
  parent_beam_index_ = other->parent_beam_index_;
  next_gold_index_ = other->next_gold_index_;

  // Copy trace if it exists.
  if (other->trace_ != nullptr) {
    trace_ = new syntaxnet::dragnn::ComponentTrace(*other->trace_);
  }
}

SemparState::~SemparState() {
  delete parser_state_;
  delete trace_;
}

void SemparState::Init(const TransitionState &parent) {
  score_ = parent.GetScore();
  parent_beam_index_ = parent.GetBeamIndex();
}

std::unique_ptr<SemparState> SemparState::Clone() const {
  std::unique_ptr<SemparState> copy(new SemparState(this));
  return copy;
}

const int SemparState::ParentBeamIndex() const {
  return parent_beam_index_;
}

const int SemparState::GetBeamIndex() const {
  return current_beam_index_;
}

void SemparState::SetBeamIndex(const int index) {
  current_beam_index_ = index;
}

const float SemparState::GetScore() const { return score_; }

void SemparState::SetScore(const float score) { score_ = score; }

string SemparState::HTMLRepresentation() const {
  return shift_only() ? StrCat("current=", shift_only_state_.current) :
      parser_state_->DebugString();
}

int SemparState::NextGoldAction() {
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
  return shift_only() ? (shift_only_current() == shift_only_state_.size) :
      parser_state_->done();
}

bool SemparState::Allowed(int action) const {
  if (IsFinal()) return false;
  return shift_only() ? (action == 0) : allowed_.at(action);
}

void SemparState::PerformAction(int action_index) {
  if (shift_only()) {
    CHECK_EQ(action_index, 0);
    ++shift_only_state_.current;
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

  CHECK(parser_state_->Apply(action))
      << action.ToString(store()) << " at state:\n"
      << parser_state_->DebugString();

  // Update history.
  history_.emplace_back(action_index);
  if (history_.size() > kMaxHistory) {
    history_.erase(history_.begin(), history_.begin() + 1);
  }

  // Update step information.
  step_info_.Update(action, *parser_state_);

  // Compute the set of allowed actions for the resulting state.
  ComputeAllowed();
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
