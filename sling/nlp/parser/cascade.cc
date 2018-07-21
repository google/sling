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

#include "sling/nlp/parser/cascade.h"

#include "sling/frame/serialization.h"

REGISTER_COMPONENT_REGISTRY("delegate runtime", sling::nlp::Delegate);

namespace sling {
namespace nlp {

// Delegate that assumes a softmax output.
class SoftmaxDelegate : public Delegate {
 public:
  void Initialize(const Cascade *cascade, const Frame &spec) override {
    input_ = cell_->GetParameter(cell_->name() + "/input");
    output_ = cell_->GetParameter(cell_->name() + "/output");

    // Read delegate's action table.
    Store *store = spec.store();
    Array actions(store, spec.GetHandle("actions"));
    CHECK(actions.valid()) << name();

    Handle n_type = store->Lookup("/table/action/type");
    Handle n_delegate = store->Lookup("/table/action/delegate");
    Handle n_length = store->Lookup("/table/action/length");
    Handle n_source = store->Lookup("/table/action/source");
    Handle n_target = store->Lookup("/table/action/target");
    Handle n_label = store->Lookup("/table/action/label");
    Handle n_role = store->Lookup("/table/action/role");
    for (int i = 0; i < actions.length(); ++i) {
      ParserAction action;
      Frame frame(store, actions.get(i));
      action.type = static_cast<ParserAction::Type>(frame.GetInt(n_type));
      if (frame.Has(n_delegate)) action.delegate = frame.GetInt(n_delegate);
      if (frame.Has(n_length)) action.length = frame.GetInt(n_length);
      if (frame.Has(n_source)) action.source = frame.GetInt(n_source);
      if (frame.Has(n_target)) action.target = frame.GetInt(n_target);
      if (frame.Has(n_label)) action.label = frame.GetHandle(n_label);
      if (frame.Has(n_role)) action.role = frame.GetHandle(n_role);
      actions_.push_back(action);
    }
  }

  void Compute(
    myelin::Instance *instance, ParserAction *action) const override {
    int best_index = *instance->Get<int>(output_);

    // NOTE: A more general and slightly more expensive approach would be
    // to call another virtual method here:
    //   Overlay(actions_[best_index], action);
    // Right now we overwrite the under-construction action with the output.
    *action = actions_[best_index];
  }

 private:
  // Location of the delegate output (argmax of the softmax layer).
  myelin::Tensor *output_ = nullptr;

  // Action table for the delegate.
  std::vector<ParserAction> actions_;
};

REGISTER_DELEGATE_RUNTIME("SoftmaxDelegate", SoftmaxDelegate);

Cascade::Cascade() {
  shift_.type = ParserAction::SHIFT;
  stop_.type = ParserAction::STOP;
}

Cascade::~Cascade() {
  for (auto *d : delegates_) delete d;
}

void Cascade::Initialize(const myelin::Network &network, const Frame &spec) {
  Store *store = spec.store();
  Array delegates(store, spec.GetHandle("delegates"));
  CHECK(delegates.valid());

  // Create delegates from the spec.
  //
  // For each delegate, the spec contains (among possibly other things):
  // - Name of the the Myelin cell that implements it.
  // - The name of the runtime (i.e. subclass of 'Delegate') used to run it.
  // - The textual name (e.g. ShiftOrNot) of the delegate.
  std::vector<Frame> delegate_specs;
  for (int i = 0; i < delegates.length(); ++i) {
    Frame frame(store, delegates.get(i));
    string runtime = frame.GetText("runtime").str();

    Delegate *d = Delegate::Create(runtime);
    d->set_cell(network.GetCell(frame.GetText("cell").str()));
    d->set_name(frame.GetText("name").str());
    d->set_runtime(runtime);

    delegates_.push_back(d);
    delegate_specs.push_back(frame);
  }

  // Initialize delegates. Delegates can choose to access other delegates in
  // the cascade at this point.
  int i = 0;
  for (auto *d : delegates_) {
    d->Initialize(this, delegate_specs[i++]);
  }
}

void Cascade::FallbackAction(
    const ParserState *state, ParserAction *action) const {
  *action = (state->current() < state->end()) ? shift_ : stop_;
}

void DelegateInstance::Compute(
    myelin::Channel *activations, int step, ParserAction *output) {
  instance_.Clear();
  instance_.Set(delegate_->input(), activations, step);
  instance_.Compute();
  delegate_->Compute(&instance_, output);
}

CascadeInstance::CascadeInstance(const Cascade *cascade)
    : cascade_(cascade) {
  for (auto *d : cascade->delegates_) {
    instances_.push_back(new DelegateInstance(d));
  }
}

CascadeInstance::~CascadeInstance() {
  for (auto *i : instances_) delete i;
}

void CascadeInstance::Compute(myelin::Channel *activations,
                              int step,
                              ParserState *state,
                              ParserAction *output) {
  int current = 0;
  const ActionTable *actions = cascade_->actions_;
  while (true) {
    // Execute the current delegate's instance.
    instances_[current]->Compute(activations, step, output);

    // If there is a cascade down the chain then follow it.
    // To avoid potential infinite loops, cascades to delegates
    // up in the chain are disallowed.
    bool is_cascade = output->type == ParserAction::CASCADE;
    if (is_cascade && (output->delegate > current)) {
      current = output->delegate;
      continue;
    }

    // If we have an applicable action then we are done with the cascade.
    if (!is_cascade &&
        !actions->Beyond(actions->Index(*output)) &&
        state->CanApply(*output)) {
      return;
    }

    // Return a fallback action.
    cascade_->FallbackAction(state, output);
    return;
  }
}


}  // namespace nlp
}  // namespace sling
