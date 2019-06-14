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

#include "sling/nlp/parser/trace.h"

namespace sling {
namespace nlp {

void Trace::Step::Add(int *ptr, int num, const string &name) {
  // Nothing to do if the feature is missing.
  if (ptr == nullptr || num == 0) return;

  auto &features = ff_features[name];
  for (int i = 0; i < num; ++i) {
    // Note: -2 signals end of feature indices.
    if (ptr[i] != -2) features.push_back(ptr[i]);
  }
}

void Trace::Action(const ParserAction &action) {
  steps.back().actions.emplace_back(action, action);
}

void Trace::Fallback(const ParserAction &fallback) {
  steps.back().actions.back().second = fallback;
}

void Trace::AddLSTM(int token, const string &name, int val) {
  if (lstm_features.size() <= token) lstm_features.resize(token + 1);
  size_t index = name.rfind('/');
  if (index != string::npos) {
    string shortname = name.substr(index + 1);
    lstm_features[token][shortname].emplace_back(val);
  } else {
    lstm_features[token][name].emplace_back(val);
  }
}

void Trace::Write(Document *document) const {
  Store *store = document->store();
  Builder builder(store);
  builder.Add("begin", begin);
  builder.Add("end", end);

  // Write encoder features.
  Array lstm_array(store, lstm_features.size());
  for (int t = 0; t < lstm_features.size(); ++t) {
    Builder lstm(store);
    lstm.Add("/trace/token", document->token(t).word());
    lstm.Add("/trace/index", t);
    for (const auto &kv : lstm_features[t]) {
      Array values(store, kv.second.size());
      for (int v = 0; v < kv.second.size(); ++v) {
        values.set(v,  Handle::Integer(kv.second[v]));
      }
      lstm.Add("/trace/" + kv.first, values);
    }
    lstm_array.set(t, lstm.Create().handle());
  }
  builder.Add("/trace/lstm_features", lstm_array);

  // Write steps.
  Array steps_array(store, steps.size());
  for (int i = 0; i < steps.size(); ++i) {
    Builder step(store);
    int current = steps[i].current;
    step.Add("/trace/current", current);
    step.Add("/trace/index", i);
    string word = "<EOS>";
    if (current < document->num_tokens()) {
      word = document->token(current).word();
    }
    step.Add("/trace/current_word", word);

    // Write decoder features.
    Array ff(store, steps[i].ff_features.size());
    int ff_count = 0;
    for (const auto &kv : steps[i].ff_features) {
      Builder feature(store);
      feature.Add("/trace/feature", kv.first);
      Array values(store, kv.second.size());
      for (int v = 0; v < kv.second.size(); ++v) {
        values.set(v,  Handle::Integer(kv.second[v]));
      }
      feature.Add("/trace/values", values);
      ff.set(ff_count++, feature.Create().handle());
    }
    step.Add("/trace/ff_features", ff);

    // Write (predicted, final) actions.
    Array actions(store, steps[i].actions.size());
    for (int a = 0; a < steps[i].actions.size(); ++a) {
      Frame predicted = steps[i].actions[a].first.AsFrame(store, "/trace/");
      Frame applied = steps[i].actions[a].second.AsFrame(store, "/trace/");
      Builder action(store);
      action.Add("/trace/predicted", predicted);
      action.Add("/trace/final", applied);
      actions.set(a, action.Create().handle());
    }
    step.Add("/trace/actions", actions);
    steps_array.set(i, step.Create().handle());
  }
  builder.Add("/trace/steps", steps_array);

  document->AddExtra(store->Lookup("trace"), builder.Create().handle());
}

}  // namespace nlp
}  // namespace sling
