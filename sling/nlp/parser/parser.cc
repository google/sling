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

#include <functional>

#include "sling/nlp/parser/parser.h"

#include "sling/frame/serialization.h"
#include "sling/myelin/profile.h"
#include "sling/myelin/kernel/dragnn.h"
#include "sling/nlp/document/document.h"
#include "sling/nlp/document/features.h"
#include "sling/nlp/document/lexicon.h"
#include "sling/nlp/parser/action-table.h"

using namespace std::placeholders;

namespace sling {
namespace nlp {

void Parser::Load(Store *store, const string &model) {
  // Load and analyze parser flow file.
  myelin::Flow flow;
  CHECK(flow.Load(model));

  // FIXME(ringgaard): Patch feature cell output.
  flow.Var("features/feature_vector")->set_in();

  // Register DRAGNN kernel to support legacy parser models.
  RegisterDragnnLibrary(compiler_.library());

  // Compile parser flow.
  compiler_.Compile(&flow, &network_);

  // Initialize lexical encoder.
  encoder_.Initialize(network_);
  encoder_.LoadLexicon(&flow);

  // Load commons store from parser model.
  myelin::Flow::Blob *commons = flow.DataBlock("commons");
  CHECK(commons != nullptr);
  StringDecoder decoder(store, commons->data, commons->size);
  decoder.DecodeAll();

  // Read the cascade specification and implementation from the flow.
  Frame cascade_spec(store, "/cascade");
  CHECK(cascade_spec.valid());
  cascade_.Initialize(network_, cascade_spec);

  // Initialize action table.
  store_ = store;
  ActionTable actions;
  actions.Init(store);
  roles_.Init(actions.list());

  // Initialize decoder feature model.
  myelin::Flow::Blob *spec = flow.DataBlock("spec");
  decoder_ = network_.GetCell("ff_trunk");
  feature_model_.Init(decoder_, spec, &roles_, actions.frame_limit());
}

void Parser::Parse(Document *document) const {
  // Parse each sentence of the document.
  for (SentenceIterator s(document); s.more(); s.next()) {
    // Set up trace if feature tracing is enabled.
    Trace *trace = trace_ ? new Trace(s.begin(), s.end()) : nullptr;

    // Run the lexical encoder for sentence.
    LexicalEncoderInstance encoder(encoder_);
    if (trace) {
      encoder.set_trace(std::bind(&Trace::AddLSTM, trace, _1, _2, _3));
    }
    auto bilstm = encoder.Compute(*document, s.begin(), s.end());

    // Initialize decoder.
    ParserState state(document, s.begin(), s.end());
    ParserFeatureExtractor features(&feature_model_, &state);
    myelin::Instance decoder(decoder_);
    myelin::Channel activations(feature_model_.hidden());
    CascadeInstance cascade(&cascade_);

    // Run decoder to predict transitions.
    for (;;) {
      // Allocate space for next step.
      activations.push();

      // Attach instance to recurrent layers.
      decoder.Clear();
      features.Attach(bilstm, &activations, &decoder);

      // Extract features.
      features.Extract(&decoder);
      if (trace) features.TraceFeatures(&decoder, trace);

      // Compute decoder activations.
      decoder.Compute();

      // Run the cascade.
      ParserAction action;
      cascade.Compute(&activations, &state, &action, trace);

      // Apply action to parser state.
      state.Apply(action);

      // Check if we are done.
      if (action.type == ParserAction::STOP) break;
    }

    // Write feature trace to document.
    if (trace) {
      trace->Write(document);
      delete trace;
    }
  }
}

}  // namespace nlp
}  // namespace sling

