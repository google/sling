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

#ifndef SLING_NLP_PARSER_CASCADE_H_
#define SLING_NLP_PARSER_CASCADE_H_

#include <string>
#include <vector>

#include "sling/base/registry.h"
#include "sling/frame/object.h"
#include "sling/frame/store.h"
#include "sling/myelin/compute.h"
#include "sling/myelin/flow.h"
#include "sling/nlp/parser/action-table.h"
#include "sling/nlp/parser/parser-action.h"
#include "sling/nlp/parser/parser-state.h"

namespace sling {
namespace nlp {

class Cascade;
class CascadeInstance;

// Delegate runtime implementation.
class Delegate : public Component<Delegate> {
 public:
  virtual ~Delegate() {}

  // Initializes the delegate, which is a part of 'cascade', and whose
  // specification is in 'spec'. The delegate implementation is
  // already available in 'cell_'.
  virtual void Initialize(const Cascade *cascade, const Frame &spec) = 0;

  // Modifies 'action' with the result of the already computed 'instance'.
  virtual void Compute(
    myelin::Instance *instance, ParserAction *action) const = 0;

  // Returns the location of the delegate input.
  virtual myelin::Tensor *input() const { return input_; }

  // Cell accessors.
  myelin::Cell *cell() const { return cell_; }
  void set_cell(myelin::Cell *cell) { cell_ = cell; }

  // Other accessors.
  const string &name() const { return name_; }
  const string &runtime() const { return runtime_; }
  void set_name(const string &n) { name_ = n; }
  void set_runtime(const string &r) { runtime_ = r; }

 protected:
  // Input to the delegate.
  myelin::Tensor *input_ = nullptr;

  // Delegate cell.
  myelin::Cell *cell_ = nullptr;

  // Name and runtime.
  string name_;
  string runtime_;
};

#define REGISTER_DELEGATE_RUNTIME(type, component) \
    REGISTER_COMPONENT_TYPE(sling::nlp::Delegate, type, component)

// Cascade model.
class Cascade {
 public:
  Cascade();
  ~Cascade();

  // Initializes the cascade by reading its specification from 'spec'
  // and implementation from 'network'.
  void Initialize(const myelin::Network &network, const Frame &spec);

  // Delegate accessors.
  Delegate *delegate(int i) const { return delegates_[i]; }
  int size() const { return delegates_.size(); }

  // Action table accessors.
  const ActionTable *actions() const { return actions_; }
  void set_actions(const ActionTable *t) { actions_ = t; }

  // Sets 'action' to the fallback action for 'state'.
  void FallbackAction(const ParserState *state, ParserAction *action) const;

 private:
  friend class CascadeInstance;

  // List of delegates.
  std::vector<Delegate *> delegates_;

  // Action table.
  const ActionTable *actions_ = nullptr;

  // Fallback actions.
  ParserAction shift_;
  ParserAction stop_;
};

// Instance for running a single delegate.
class DelegateInstance {
 public:
  DelegateInstance(const Delegate *d) : delegate_(d), instance_(d->cell()) {}

  // Runs the delegate with the specified input activation, and populates
  // 'output' with the resulting action.
  void Compute(myelin::Channel *activations, int step, ParserAction *output);

 private:
  // Delegate. Not owned.
  const Delegate *delegate_ = nullptr;

  // Underlying Myelin instance.
  myelin::Instance instance_ = nullptr;
};

// Runs an instance of a cascade on a ParserState.
class CascadeInstance {
 public:
  CascadeInstance(const Cascade *cascade);
  ~CascadeInstance();

  // Outputs in 'output' the result of running the whole cascade on 'state'.
  // The activation at index 'step' is used as input to all the delegates.
  void Compute(myelin::Channel *activations,
               int step,
               ParserState *state,
               ParserAction *output);

 private:
  const Cascade *const cascade_ = nullptr;     // cascade; not owned
  std::vector<DelegateInstance *> instances_;  // delegate-specific instances
};

}  // namespace nlp
}  // namespace sling

#endif  // SLING_NLP_PARSER_CASCADE_H_
