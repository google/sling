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

#ifndef SLING_NLP_PARSER_ACTION_TABLE_H_
#define SLING_NLP_PARSER_ACTION_TABLE_H_

#include <unordered_map>
#include <vector>

#include "sling/base/types.h"
#include "sling/frame/object.h"
#include "sling/frame/store.h"
#include "sling/nlp/parser/parser-action.h"
#include "sling/nlp/parser/parser-state.h"
#include "sling/util/table-writer.h"

namespace sling {
namespace nlp {

// Stores the following information from a training corpus:
// - Assignment of 0-based indices to ParserActions seen in the training data.
// - Coverage-based bounds on the source/target arguments of the actions.
//
// Usage:
// - While running over a training corpus, the caller needs to call
//    Add(action) to register the action in the table.
// - This table can be queried via the Allowed() method, which returns the
//   indices of ParserActions possible at the given state.
class ActionTable {
 public:
  // Adds 'action' to the table.
  void Add(const ParserAction &action);

  // Accessor/mutator for max_actions_per_token_. Note that this field needs
  // to be set explicitly and is not automatically computed.
  int max_actions_per_token() const { return max_actions_per_token_; }
  void set_max_actions_per_token(int m) {
    if (max_actions_per_token_ < m) max_actions_per_token_ = m;
  }

  // Sets the indices of only the allowed actions for 'state' to true.
  void Allowed(const ParserState &state, std::vector<bool> *allowed) const;

  // Checks if actions is beyond bounds.
  bool Beyond(int index) const { return beyond_bounds_[index]; }

  // Returns the integer index of 'action'.
  int Index(const ParserAction &action) const {
    const auto &it = index_.find(action);
    return (it == index_.end()) ? -1 : it->second.first;
  }

  // Shortcuts for accessing the indices for STOP and SHIFT.
  int StopIndex() const { return stop_index_; }
  int ShiftIndex() const { return shift_index_; }

  // Returns the number of possible actions.
  int NumActions() const { return actions_.size(); }

  // Returns the ParserAction for integer index 'i'.
  const ParserAction &Action(int i) const { return actions_[i]; }

  // Returns the maximum span length.
  int max_span_length() const { return max_span_length_; }

  // Accessors for various indices.
  int max_refer_target() const { return max_refer_target_; }
  int max_connect_source() const { return max_connect_source_; }
  int max_connect_target() const { return max_connect_target_; }
  int max_assign_source() const { return max_assign_source_; }
  int max_embed_target() const { return max_embed_target_; }
  int max_elaborate_source() const { return max_elaborate_source_; }
  int frame_limit() const { return frame_limit_; }

  // Saves the action table to 'file'. 'global' is optionally used to lookup
  // global symbols used as action arguments. 'percentile' is used to compute
  // the maximum indices and span length for various actions.
  void Save(const Store *global, int percentile, const string &file) const;

  // Saves the action table using 100 percentile indices/lengths.
  void Save(const Store *global, const string &file) const {
    Save(global, 100, file);
  }

  // Returns the serialization of the table as per 'percentile'.
  string Serialize(const Store *global, int percentile) const;

  // Outputs summary to 'file'
  void OutputSummary(const string &file) const;

  // Outputs summary to 'writer'
  void OutputSummary(TableWriter *writer) const;

  // Initialize the action table from 'store'.
  void Init(Store *store);

 private:
  // Represents a histogram where the bins are small integers.
  class Histogram {
   public:
    explicit Histogram(const string &xaxis) : xaxis_(xaxis) {}

    // Adds a count of 'count' to 'bin'.
    void Add(int bin, int count);
    void Add(int bin) { Add(bin, 1); }

    // Returns the maximum bin.
    int MaxBin() const { return counts_.size() - 1; }

    // Returns the index of the smallest bin whose cumulative count equals or
    // exceeds the percentile 'p'.
    int PercentileBin(int p) const;

    // Outputs histogram to 'writer'.
    void ToTable(TableWriter *writer) const;

   private:
    // X-axis title.
    string xaxis_;

    // Bin -> Count. Assumes small bin values.
    std::vector<int64> counts_;

    // Total count across bins.
    int total_ = 0;
  };

  // Mapping from ParserAction -> (0-based index, raw count).
  std::unordered_map<ParserAction, std::pair<int, int>, ParserActionHash>
      index_;

  // Index -> ParserAction with that index.
  std::vector<ParserAction> actions_;

  // ith entry -> whether ith action violates the max-span-length or
  // max-source-index or max-target-index bounds.
  std::vector<bool> beyond_bounds_;

  // Histograms of integer arguments for various actions. Only populated during
  // a pass over the training data.
  Histogram refer_target_{"Refer Target Histogram"};
  Histogram embed_target_{"Embed Target Histogram"};
  Histogram elaborate_source_{"Elaborate Source Histogram"};
  Histogram connect_source_{"Connect Source Histogram"};
  Histogram connect_target_{"Connect Target Histogram"};
  Histogram assign_source_{"Assign Source Histogram"};
  Histogram overall_index_{"Overall Frame Index Histogram"};
  Histogram span_length_{"Span Length Histogram"};

  // Max permitted values for various action arguments during inference.
  // These might be the maximum bins of the histograms above or might be set
  // according to some percentile-based heuristics. These are loaded from the
  // serialized store file.
  int max_refer_target_;
  int max_embed_target_;
  int max_elaborate_source_;
  int max_connect_source_;
  int max_connect_target_;
  int max_assign_source_;
  int frame_limit_ = 5;

  // Maximum of all the above indices.
  int max_index_ = 0;

  // Maximum length of a span involved in EVOKE or REFER actions.
  int max_span_length_ = 0;

  // Maximum number of actions taken per token.
  int max_actions_per_token_ = -1;

  // Indices of STOP and SHIFT.
  int stop_index_ = 0;
  int shift_index_ = 0;
};

}  // namespace nlp
}  // namespace sling

#endif  // SLING_NLP_PARSER_ACTION_TABLE_H_
