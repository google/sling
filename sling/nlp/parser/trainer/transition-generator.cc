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

#include "sling/nlp/parser/trainer/transition-generator.h"

#include <algorithm>
#include <unordered_set>

#include "sling/base/macros.h"
#include "sling/string/strcat.h"

namespace sling {
namespace nlp {

std::vector<string> TransitionSequence::AsStrings(Store *store) const {
  std::vector<string> debug;
  for (const ParserAction &action : actions()) {
    debug.emplace_back(action.ToString(store));
  }

  return debug;
}

int TransitionSequence::FirstAction(int token) const {
  // The desired action is immediately after the SHIFT action for the
  // previous token. Since there is exactly one SHIFT per token, we simply
  // count (token - 1) SHIFTs.
  DCHECK_GT(actions_.size(), 0);
  if (token == 0) return 0;

  int skips = 0;
  for (int i = 0; i < actions_.size(); ++i) {
    if (actions_[i].type == ParserAction::SHIFT) ++skips;
    if (skips == token) return i + 1;
  }

  return actions_.size();  // should not reach here
}

void TransitionGenerator::AttentionIndex::Add(Handle handle) {
  const auto &it = frame_to_index_.find(handle);
  if (it == frame_to_index_.end()) {
    index_to_frame_.push_back(handle);
    frame_to_index_[handle] = index_to_frame_.size() - 1;
  }
  MakeCenter(handle);
}

void TransitionGenerator::AttentionIndex::MakeCenter(Handle handle) {
  int current_index = Index(handle);
  if (current_index == 0) return;  // already the center

  for (int i = current_index; i >= 0; --i) {
    index_to_frame_[i] = (i == 0) ? handle : index_to_frame_[i - 1];
    frame_to_index_[index_to_frame_[i]] = i;
  }
}

void TransitionGenerator::AttentionIndex::Update(const Action &action) {
  switch (action.type) {
    case ParserAction::EVOKE: {
      Add(action.frame->handle);
      break;
    }
    case ParserAction::REFER: {
      MakeCenter(action.frame->handle);
      break;
    }
    case ParserAction::EMBED: {
      Add(action.frame->handle);
      break;
    }
    case ParserAction::ELABORATE: {
      Add(action.frame->handle);
      break;
    }
    case ParserAction::CONNECT: {
      MakeCenter(action.frame->handle);
      break;
    }
    case ParserAction::ASSIGN: {
      MakeCenter(action.frame->handle);
      break;
    }
    case ParserAction::SHIFT:
    case ParserAction::STOP:
      break;
    default:
      LOG(FATAL) << "Unknown action";
      break;
  }
}

void TransitionGenerator::InitInfo(const Frame &frame,
                                   HandleMap<FrameInfo *> *frame_info,
                                   HandleSet *initialized) const {
  Handle handle = frame.handle();
  if (initialized->find(handle) != initialized->end()) return;
  initialized->insert(handle);

  FrameInfo *&info = (*frame_info)[handle];
  if (info == nullptr) info = new FrameInfo();
  info->handle = handle;
  std::vector<Frame> pending;  // frames whose FrameInfo needs to be initialized
  std::unordered_set<uint64> seen;  // processed slots
  for (const Slot &slot : frame) {
    if (slot.name.IsId() || slot.name == n_name_.handle()) continue;

    // Ignore local slot names. We are only interested in global roles.
    if (slot.name.IsLocalRef()) continue;
    if (slot.name.IsIsA() && !slot.value.IsGlobalRef()) continue;

    // Ignore duplicates.
    uint64 key = slot.name.raw();
    key = (key << 32) | slot.value.raw();
    if (seen.find(key) != seen.end()) continue;
    seen.insert(key);

    if (slot.name.IsIsA() && info->first_type.IsNil()) {
      info->first_type = slot.value;
    } else {
      info->edges.emplace_back(false /* incoming */, slot.name, slot.value);

      // Check for self-loops.
      if (frame.handle() == slot.value) {
        info->edges.back().inverse = info->edges.size() - 1;
        continue;
      }

      // Add incoming edge to the linked frame's FrameInfo only if the linked
      // frame is local. We don't want to output EVOKE/EMBED for global frames.
      Object target(frame.store(), slot.value);
      if (target.IsFrame() && target.IsLocal()) {
        FrameInfo *&neighbor_info = (*frame_info)[slot.value];
        if (neighbor_info == nullptr) neighbor_info = new FrameInfo();

        auto &edges = neighbor_info->edges;
        edges.emplace_back(true /* incoming */, slot.name, handle);
        edges.back().inverse = info->edges.size() - 1;
        info->edges.back().inverse = edges.size() - 1;
        pending.emplace_back(target.AsFrame());
      }
    }
  }

  // Set a default type for the frame.
  if (info->first_type.IsNil()) info->first_type = n_thing_.handle();

  // Recursively process all local frames that this frame points to.
  for (const Frame &f : pending) InitInfo(f, frame_info, initialized);
}

ParserAction TransitionGenerator::AttentionIndex::Translate(
    const Document &document,
    const Action &action) {
  ParserAction output;
  output.type = action.type;
  switch (action.type) {
    case ParserAction::EVOKE: {
      CHECK_NE(action.length, 0);
      output.length = action.length;
      output.label = action.frame->first_type;
      break;
    }
    case ParserAction::REFER: {
      CHECK_NE(action.length, 0);
      output.length = action.length;
      output.target = Index(action.frame->handle);
      SetMaxIndex(output.target);
      break;
    }
    case ParserAction::EMBED: {
      output.label = action.frame->first_type;
      output.role = action.role;
      output.target = Index(action.other_frame->handle);
      SetMaxIndex(output.target);
      break;
    }
    case ParserAction::ELABORATE: {
      output.label = action.frame->first_type;
      output.role = action.role;
      output.source = Index(action.other_frame->handle);
      SetMaxIndex(output.source);
      break;
    }
    case ParserAction::CONNECT: {
      output.source = Index(action.frame->handle);
      output.target = Index(action.other_frame->handle);
      output.role = action.role;
      SetMaxIndex(output.source);
      SetMaxIndex(output.target);
      break;
    }
    case ParserAction::ASSIGN: {
      output.source = Index(action.frame->handle);
      output.role = action.role;
      output.label = action.value;
      SetMaxIndex(output.source);
      break;
    }
    case ParserAction::SHIFT:
    case ParserAction::STOP:
    default:
      break;
  }

  return output;
}

void TransitionGenerator::Generate(
    const Document &document,
    TransitionSequence *sequence,
    TransitionGenerator::Report *report) const {
  Generate(document, 0, document.num_tokens(), sequence, report);
}

void TransitionGenerator::Generate(
    const Document &document,
    int begin,
    int end,
    TransitionSequence *sequence,
    TransitionGenerator::Report *report) const {
  Store *store = document.store();

  // Sort the spans by start token, and then in decreasing order of length.
  // This ordering is used to EVOKE/REFER the corresponding evoked frames.
  std::vector<Span *> spans;
  for (int i = 0; i < document.num_spans(); ++i) {
    Span *span = document.span(i);
    if (span->begin() >= begin && span->end() <= end) spans.emplace_back(span);
  }
  std::sort(spans.begin(), spans.end(), [](Span *s, Span *t) {
    return (s->begin() < t->begin()) ||
        ((s->begin() == t->begin()) && (s->end() > t->end()));
  });

  // Frame handle -> FrameInfo.
  HandleMap<FrameInfo *> frame_info;

  // Initialize FrameInfo of all evoked frames.
  HandleSet initialized;
  std::vector<Span *> spans_with_frames;
  for (Span *span : spans) {
    for (const Slot &slot : span->mention()) {
      if (slot.name == n_evokes_.handle()) {
        if (spans_with_frames.empty() || spans_with_frames.back() != span) {
          spans_with_frames.emplace_back(span);
        }
        InitInfo(Frame(store, slot.value), &frame_info, &initialized);
      }
    }
  }

  // Initialize FrameInfo of all thematic frames.
  for (Handle theme : document.themes()) {
    Object object(store, theme);
    if (object.IsFrame()) {
      InitInfo(object.AsFrame(), &frame_info, &initialized);
    }
  }

  // Insert SHIFT actions between consecutive EVOKE/REFER actions. Other
  // actions triggered by EVOKE will be added to the list later.
  int previous_start = begin;
  std::vector<Action> stack;
  HandleSet evoked;
  for (Span *span : spans_with_frames) {
    // Issue SHIFT actions from previous span till this span. This includes
    // skipping all tokens for the previous span.
    for (int i = previous_start; i < span->begin(); ++i) {
      stack.emplace_back(ParserAction::SHIFT);
    }
    previous_start = span->begin();

    // Issue an EVOKE/REFER action for every evoked frame.
    for (const Slot &slot : span->mention()) {
      if (slot.name == n_evokes_.handle()) {
        FrameInfo *f = frame_info[slot.value];
        if (evoked.count(slot.value) == 0) {
          stack.emplace_back(ParserAction::EVOKE, f, span->length());
          evoked.insert(slot.value);
        } else {
          stack.emplace_back(ParserAction::REFER, f, span->length());
        }
      }
    }
  }

  // Add SHIFTs for tokens after the last frame, and the final STOP action.
  for (int i = previous_start; i < end; ++i) {
    stack.emplace_back(ParserAction::SHIFT);
  }
  stack.emplace_back(ParserAction::STOP);

  // Reverse 'stack' and treat it as a stack.
  // This is for simplicity -- e.g. we can pop an EVOKE action from the stack,
  // and push triggered actions like CONNECT, ASSIGN etc. onto the stack.
  // Note that reversing + treating as stack continues to preserve span-ordering
  // of evoked frames.
  std::reverse(stack.begin(), stack.end());

  AttentionIndex attention;
  while (!stack.empty()) {
    Action action = stack.back();
    stack.pop_back();

    // Translate the action using real attention indices and output it.
    sequence->Add(attention.Translate(document, action));

    // Update the attention index by performing the action.
    attention.Update(action);

    // New frame creation can trigger more actions. Add them to the stack.
    if (action.type == ParserAction::EVOKE ||
        action.type == ParserAction::EMBED ||
        action.type == ParserAction::ELABORATE) {
      action.frame->output = true;

      // Process CONNECT actions.
      for (auto &edge : action.frame->edges) {
        if (edge.used) continue;
        const auto &it = frame_info.find(edge.neighbor);
        if (it == frame_info.end()) continue;

        FrameInfo *neighbor_info = it->second;
        if (neighbor_info->output) {
          // Neighbor has already been output earlier. Output a CONNECT action.
          stack.emplace_back(ParserAction::CONNECT);
          Action &latest = stack.back();
          latest.frame = edge.incoming ? neighbor_info : action.frame,
          latest.role = edge.role;
          latest.other_frame = edge.incoming ? action.frame : neighbor_info;
          edge.used = true;
          CHECK_NE(edge.inverse, -1);
          neighbor_info->edges[edge.inverse].used = true;
        }
      }

      // Process EMBED actions.
      for (auto &edge : action.frame->edges) {
        if (edge.used) continue;
        const auto &it = frame_info.find(edge.neighbor);
        if (it == frame_info.end()) continue;

        // EMBED a frame if it links to this frame, hasn't been output earlier,
        // and if it is not evoked by any span.
        FrameInfo *neighbor_info = it->second;
        if (!neighbor_info->output && edge.incoming) {
          const auto &range = document.EvokingSpans(neighbor_info->handle);
          if (range.begin() == range.end()) {
            stack.emplace_back(ParserAction::EMBED);
            Action &latest = stack.back();
            latest.frame = neighbor_info;
            latest.role = edge.role;
            latest.other_frame = action.frame;
            edge.used = true;
            neighbor_info->edges[edge.inverse].used = true;
          }
        }
      }

      // Process ELABORATE actions.
      for (auto &edge : action.frame->edges) {
        if (edge.used) continue;
        const auto &it = frame_info.find(edge.neighbor);
        if (it == frame_info.end()) continue;

        // ELABORATE this frame by creating new frames and linking to them.
        // Links must be TO the new frames, and the new frames must not be
        // evoked by spans.
        FrameInfo *neighbor_info = it->second;
        const auto &range = document.EvokingSpans(neighbor_info->handle);
        if (range.begin() == range.end() &&
            !neighbor_info->output &&
            !edge.incoming) {
          stack.emplace_back(ParserAction::ELABORATE, neighbor_info);
          Action &latest = stack.back();
          latest.role = edge.role;
          latest.other_frame = action.frame;
          edge.used = true;
          neighbor_info->edges[edge.inverse].used = true;
        }
      }

      // Process ASSIGN actions.
      // Note: Since we push ASSIGN actions on the stack last, they are the
      // first one to be popped out. This ensures that ASSIGN's index argument
      // will always be zero.
      for (auto &edge : action.frame->edges) {
        if (edge.used) continue;

        Object neighbor(store, edge.neighbor);
        if (!neighbor.IsLocal()) {
          // Treat global values as constants & output ASSIGN actions for them.
          DCHECK(!edge.incoming);
          stack.emplace_back(ParserAction::ASSIGN, action.frame);
          Action &latest = stack.back();
          latest.role = edge.role;
          latest.value = edge.neighbor;
          edge.used = true;
        }
      }
    }
  }

  // Output the report.
  if (report != nullptr) {
    report->max_attention_index = attention.MaxIndex();
    for (const auto &kv : frame_info) {
      Handle frame_handle = kv.first;
      const FrameInfo &info = *kv.second;
      bool frame_output = info.output;
      if (!frame_output) {
        // Frame was never pushed into the attention buffer.
        Frame frame(store, frame_handle);
        report->frames_not_output++;
        StrAppend(&report->not_output_debug, "Frame:", frame.Id(), "\n");
      }

      // Check for any unprocessed edges.
      for (const auto &edge : info.edges) {
        if (!edge.incoming && !edge.used) {
          report->edges_not_output++;
          if (frame_output) {
            StrAppend(&report->not_output_debug,
                      Object(store, frame_handle).DebugString(), " ->(",
                      Object(store, edge.role).DebugString(), ")-> ",
                      Object(store, edge.neighbor).DebugString(), "\n");
          }
        }
      }
    }
  }

  // Cleanup.
  for (auto &kv : frame_info) delete kv.second;
}

}  // namespace nlp
}  // namespace sling
