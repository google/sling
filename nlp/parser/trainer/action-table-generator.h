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

#ifndef NLP_PARSER_TRAINER_ACTION_TABLE_GENERATOR_H_
#define NLP_PARSER_TRAINER_ACTION_TABLE_GENERATOR_H_

#include <string>
#include <unordered_map>

#include "frame/store.h"
#include "nlp/document/document.h"
#include "nlp/parser/action-table.h"
#include "nlp/parser/trainer/gold-transition-generator.h"

namespace sling {
namespace nlp {

// Generates an ActionTable from supplied list of documents.
class ActionTableGenerator {
 public:
  struct Options {
    // Global store. Not owned.
    Store *global = nullptr;

    // Whether or not to generate per sentence gold sequences.
    bool per_sentence = true;

    // Coverage percentile used to prune actions.
    int coverage_percentile = 100;
  };

  ActionTableGenerator(const Options &options);

  // Adds golden actions from 'document' to the action table.
  void Add(const Document &document);

  // Saves the action table, action table summary, and unknown symbols
  // to their respective files.
  void Save(const string &table_file,
            const string &summary_file,
            const string &unknown_symbols_file) const;

 private:
  // Retrieves unknown symbols from 'document' to be logged later.
  void GetUnknownSymbols(const Document &document);

  // Dumps unknown symbols as a text store.
  void OutputUnknownSymbols(const string &filename) const;

  // Generates and uses the gold sub-sequence for [begin, end).
  void Process(const Document &document, int start, int end);

  // Generation options.
  Options options_;

  // Gold sequence generator.
  GoldTransitionGenerator generator_;

  // Unknown (i.e. non-global) symbol -> raw frequency.
  std::unordered_map<string, int> unknown_;

  // Under construction table.
  ActionTable table_;
};

}  // namespace nlp
}  // namespace sling

#endif  // NLP_PARSER_TRAINER_ACTION_TABLE_GENERATOR_H_
