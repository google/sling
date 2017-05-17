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

#ifndef NLP_PARSER_ACTION_TABLE_H_
#define NLP_PARSER_ACTION_TABLE_H_

#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "base/types.h"
#include "frame/object.h"
#include "frame/store.h"
#include "nlp/parser/parser-action.h"
#include "nlp/parser/parser-state.h"
#include "util/table-writer.h"

namespace sling {
namespace nlp {

// Stores the following information from a training corpus:
// - Assignment of 0-based indices to ParserActions seen in the training data.
// - For each training data phrase that evokes frame(s), a mapping to the
//   corresponding EVOKE/REFER actions.
// - For each type involved in the other types of actions, a mapping to the
//   other arguments of the action. For example if type t participates in
//   CONNECT(source frame type = t, target frame type = t', role = r), then
//   the mapping for t would include (t', r). This is used to figure out what
//   possible actions a frame of type t can participate in.
//
// Usage:
// - While running over a training corpus, the caller needs to call
//    Add(state, fingerprints, action) to register the action in the table.
// - This table can be queried via the Allowed() method, which returns the
//   indices of ParserActions possible at the given state.
class ActionTable {
 public:
  // Adds to the table 'action' that was taken at 'state'. If 'action' is an
  // EVOKE or REFER action, 'fp' should be the fingerprint of the span.
  void Add(const ParserState &state, const ParserAction &action, uint64 fp = 0);

  // Accessors/mutators for whether checks would be done on actions.
  bool action_checks() const { return action_checks_; }
  void set_action_checks(bool b) { action_checks_ = b; }

  // Accessor/mutator for max_actions_per_token_. Note that this field needs
  // to be set explicitly and is not automatically computed.
  int max_actions_per_token() const { return max_actions_per_token_; }
  void set_max_actions_per_token(int m) {
    if (max_actions_per_token_ < m) max_actions_per_token_ = m;
  }

  // Returns the indices of actions allowed for 'state'. 'fingerprints' has the
  // list of fingerprints of possible spans that start at the current location,
  // where the ith fingerprint corresponds to a span of length i.
  // 'allowed' should already be big enough to hold all the actions, and be
  // all false.
  void Allowed(const ParserState &state,
               const std::vector<uint64> &fingerprints,
               std::vector<bool> *allowed) const;

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
  // A pair of handles.
  typedef std::pair<Handle, Handle> HandlePair;

  // Hasher for a pair of handles.
  struct HandlePairHash {
    size_t operator()(const HandlePair &p) const {
      HandleHash h;
      return h(p.first) ^ h(p.second);
    }
  };

  // Shorthand for a hashset keyed by a pair of handles.
  typedef std::unordered_set<HandlePair, HandlePairHash> HandlePairSet;

  // Given a phrase fingerprint, stores constraints on allowed actions that
  // depend on the fingerprint.
  struct FingerprintConstraint {
    // Indices of all EVOKE actions allowed on this fingerprint.
    std::unordered_set<int> evoke;

    // All REFER actions allowed on this fingerprint. These refer actions also
    // store the type of the frame that they refer to.
    std::unordered_set<ParserAction, ParserActionHash> refer;
  };

  // Given a frame type t, stores constraints on allowed actions dependent on t.
  struct TypeConstraint {
    // (Role, Value) pairs for ASSIGN actions for frames of type t.
    HandlePairSet assign;

    // (Source type, role) for EMBED actions whose target frame type is t.
    HandlePairSet embed;

    // (Target type, role) for ELABORATE actions whose source frame type is t.
    HandlePairSet elaborate;

    // Target type -> Roles for CONNECT actions whose source frame type is t.
    HandleMap<HandleSet> connect;
  };

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
    std::vector<int> counts_;

    // Total count across bins.
    int total_ = 0;
  };

  // Reads a set of handle pairs from 'slot' in 'frame'. The slot value is
  // an array of nested frames, with each nested frame containing two slots
  // 'subslot1' and 'subslot2'.
  void Load(const Frame &frame,
            Handle slot,
            Handle subslot1,
            Handle subslot2,
            HandlePairSet *pairs);

  // Saves a set of handle pairs as an array of nested frames with slots
  // 'subslot{1,2}'. The array resides as the value of 'slot' in 'builder'.
  void Save(const HandlePairSet &pairs,
            Handle slot,
            Handle subslot1,
            Handle subslot2,
            Builder *builder) const;

  // Phrase fingerprint -> Allowed REFER/EVOKE actions.
  std::unordered_map<uint64, FingerprintConstraint> fingerprint_;

  // Frame type -> Information about various actions associated with frame type.
  HandleMap<TypeConstraint> type_;

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

  // Maximum of all the above indices.
  int max_index_ = 0;

  // Maximum length of a span involved in EVOKE or REFER actions.
  int max_span_length_ = 0;

  // Maximum number of actions taken per token.
  int max_actions_per_token_ = -1;

  // Indices of STOP and SHIFT.
  int stop_index_ = 0;
  int shift_index_ = 0;

  // Whether or not to use fingerprint/typechecks to compute allowed actions.
  bool action_checks_ = true;
};

}  // namespace nlp
}  // namespace sling

#endif  // NLP_PARSER_ACTION_TABLE_H_
