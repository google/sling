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

#ifndef NLP_PARSER_TRAINER_TRANSITION_GENERATOR_H_
#define NLP_PARSER_TRAINER_TRANSITION_GENERATOR_H_

#include <string>
#include <vector>

#include "frame/object.h"
#include "frame/store.h"
#include "nlp/document/document.h"
#include "nlp/parser/parser-action.h"

namespace sling {
namespace nlp {

// A transition sequence is just an ordered list of actions.
class TransitionSequence {
 public:
  // Appends 'action' to the sequence.
  void Add(const ParserAction &action) { actions_.emplace_back(action); }

  // Accessor.
  const std::vector<ParserAction> &actions() const { return actions_; }
  const ParserAction &action(int i) const { return actions_[i]; }

  // Returns the index of the first action for 'token'. Note that the last
  // action for a token is always SHIFT.
  int FirstAction(int token) const;

  // Returns the actions as human-readable strings.
  std::vector<string> AsStrings(Store *store) const;

  // Clears the sequence.
  void Clear() { actions_.clear(); }

 private:
  std::vector<ParserAction> actions_;
};

// Generates transition sequences from document frames.
//
// Implements the following protocol:
// - EVOKE: If frames F1 and F2 are evoked by spans S1 and S2 where S1 starts
//   before S2, then F1 will be evoked before F2.
// - EVOKE: In the above case, if S1 and S2 have the same start token, then F1
//   will be evoked first iff S1 is longer than S2.
// - ASSIGN: Once a frame is created, any ASSIGN actions for it will be
//   performed then and there.
// - CONNECT: Once a frame is created, any links between it and any existing
//   frame in any direction will be evoked via CONNECT actions then and there.
// - EMBED: Once a frame is created, any links TO it from a yet to be created
//   thematic frame will be materialized via EMBED actions then and there.
// - ELABORATE: Once a frame is created, ELABORATE will be called for all links
//   FROM it to yet-to-be-built thematic frames.
// - REFER: This action is output if a span evokes a previously evoked frame.
//   It is output when the span's start token is reached.
//
// Note that EVOKE, EMBED, and ELABORATE can trigger more actions.
class TransitionGenerator {
 public:
  // End result of generating the transition sequence.
  struct Report {
    // Maximum attention index used by CONNECT/EMBED/ELABORATE/ASSIGN/REFER.
    int max_attention_index = -1;

    // Number of frames for which there was no EVOKE/EMBED/ELABORATE/REFER.
    int frames_not_output = 0;

    // Number of edges for which there was no CONNECT/EMBED/ASSIGN/ELABORATE.
    int edges_not_output = 0;

    // Debug string representing the frames/edges not output.
    string not_output_debug;
  };

  // Looks up necessary global symbols in 'store'. Should be called only once.
  void Init(Store *store) { names_.Bind(store); }

  // Generates in 'sequence' the transition sequence for the interpretation
  // in 'document'. Generation report is output in 'report'.
  void Generate(const Document &document,
                TransitionSequence *sequence,
                Report *report = nullptr) const;

  // Generates in 'sequence' the transition sequence for the interpretation
  // in 'document' but limited to the range [begin, end). Any frame reachable
  // from a span in [begin, end) is processed during sequence generation.
  // Generation report is output in 'report'.
  //
  // IMPORTANT NOTE:
  // This method is best used when the subgraph of frames reachable from the
  // spans in [begin, end) doesn't link to any other frames in 'document'.
  // Otherwise all the other frames will be pulled into the sequence via
  // EMBED or ELABORATE, since they are not evoked by any span in [begin, end).
  void Generate(const Document &document,
                int begin,
                int end,
                TransitionSequence *sequence,
                Report *report = nullptr) const;

 private:
  // Information about an incoming/outgoing link from a frame. By default, the
  // frame is assumed to be the source.
  struct Edge {
    Edge(bool i, Handle r, Handle n) {
      incoming = i;
      role = r;
      neighbor = n;
      used = false;
    }

    // If the edge is incoming, i.e. the frame is the target and not the source.
    bool incoming = false;

    // The role of the edge.
    Handle role = Handle::nil();

    // The other endpoint of the edge.
    Handle neighbor = Handle::nil();

    // Whether the edge has been used to generate an action.
    bool used = false;

    // Index of the reverse edge, if any, in the neighbor's FrameInfo.
    int inverse = -1;
  };

  // Holds book-keeping info about a frame in the document.
  struct FrameInfo {
    // Handle to the frame.
    Handle handle;

    // First type of the frame.
    Handle first_type = Handle::nil();

    // Whether the frame has been output via an action or not.
    bool output = false;

    // Incoming and outgoing edges of the frame.
    std::vector<Edge> edges;
  };

  // Represents an action using FrameInfo structs of frames as arguments,
  // instead of attention indices.
  struct Action {
    // Type of the action.
    ParserAction::Type type;

    // The frame being created in EVOKE/EMBED/ELABORATE, or the source
    // frame in CONNECT/ASSIGN, or the referred frame in REFER.
    FrameInfo *frame = nullptr;

    // Role argument in CONNECT, ASSIGN, EMBED, and ELABORATE.
    Handle role = Handle::nil();

    // Target frame in CONNECT/EMBED, or existing frame argument in ELABORATE.
    FrameInfo *other_frame = nullptr;

    // Target argument in ASSIGN. This doesn't necessarily have to be a frame.
    Handle value = Handle::nil();

    // Length argument in EVOKE/REFER.
    int length = 0;

    // Constructors.
    explicit Action(ParserAction::Type t) : type(t) {}
    Action(ParserAction::Type t, FrameInfo *f) : type(t), frame(f) {}
    Action(ParserAction::Type t, FrameInfo *f, int l) : type(t), frame(f) {
      length = l;
    }
  };

  // Two-way mapping from frame handle to its attention index. An index of zero
  // maps to the center of attention.
  class AttentionIndex {
   public:
    // Adds 'handle' to the mapping and makes it the center of attention.
    void Add(Handle handle);

    // Makes 'handle' the center of attention.
    void MakeCenter(Handle handle);

    // Returns the attention index of 'handle', or -1 if unknown.
    int Index(Handle handle) const {
      const auto &it = frame_to_index_.find(handle);
      return it == frame_to_index_.end() ? -1 : it->second;
    }

    // Updates the index as per the consequence of 'action'.
    void Update(const Action &action);

    // Translates 'action'' into a ParserAction using attention index arguments.
    ParserAction Translate(const Document &document, const Action &action);

    // Returns the maximum attention index used so far in any Translate call.
    int MaxIndex() const { return max_index_; }

   private:
    // Updates the maximum attention index.
    void SetMaxIndex(int i) { if (i > max_index_) max_index_ = i; }

    HandleMap<int> frame_to_index_;  // frame handle -> attention index
    std::vector<Handle> index_to_frame_;  // attention index -> frame handle
    int max_index_ = -1;
  };

  // Initializes the FrameInfo struct for 'frame' if it is not already so
  // as per 'initialized'. Recursively calls itself on linked frames.
  void InitInfo(const Frame &frame,
                HandleMap<FrameInfo *> *frame_info,
                HandleSet *initialized) const;

  // Lazily bound names.
  Names names_;
  Name n_name_{names_, "name"};
  Name n_thing_{names_, "/s/thing"};
  Name n_evokes_{names_, "/s/phrase/evokes"};
};

}  // namespace nlp
}  // namespace sling

#endif  // NLP_PARSER_TRAINER_TRANSITION_GENERATOR_H_
