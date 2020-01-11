// Copyright 2019 Google Inc.
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

#include "sling/nlp/parser/transition-generator.h"

#include <algorithm>
#include <deque>
#include <vector>

#include "sling/frame/object.h"
#include "sling/frame/store.h"

namespace sling {
namespace nlp {
namespace {

// Map from a frame to information pertinent to transition generation,
// represented as a graph keyed by individual frame nodes.
class FrameGraph {
 public:
  // Represents one incoming/outgoing edge of a frame node.
  struct Edge {
    Edge(Handle r, bool in, Handle n) : role(r), neighbor(n), incoming(in) {}

    // Edge label.
    Handle role;

    // Other end of the edge.
    Handle neighbor;

    // Whether the edge is coming into this frame.
    bool incoming;

    // Pointer to the reverse edge. Not owned.
    Edge *inverse = nullptr;

    // Whether the edge has been used to output action(s).
    bool accounted = false;
  };

  // Information about a single frame.
  struct Node {
    ~Node() {
      for (auto *edge : edges) delete edge;
    }

    Handle handle = Handle::nil();  // frame handle
    Handle type = Handle::nil();    // type of the frame
    bool evoked = false;            // whether the frame is evoked by text
    bool accounted = false;         // whether the node has been used
    std::vector<Edge *> edges;      // in/out edges of the frame (owned)
  };

  ~FrameGraph() {
    for (const auto &n : nodes_) delete n.second;
  }

  // Retrieves the node for the frame 'handle'.
  Node *node(Handle handle) const {
    auto it = nodes_.find(handle);
    return it != nodes_.end() ? it->second : nullptr;
  }

  // Gets or creates a node for 'handle'.
  Node *Get(Handle handle) {
    auto it = nodes_.find(handle);
    if (it != nodes_.end()) return it->second;

    Node *e = new Node();
    e->handle = handle;
    nodes_[handle] = e;
    return e;
  }

 private:
  HandleMap<Node *> nodes_;
};

// Represents an action modulo attention indices.
struct Action {
  Action(ParserAction::Type type)
      : core(type), frame(nullptr), neighbor(nullptr) {}
  Action(ParserAction::Type type, FrameGraph::Node *f)
      : core(type), frame(f), neighbor(nullptr) {}

  ParserAction core;                     // core action
  FrameGraph::Node *frame = nullptr;     // frame responsible for the action
  FrameGraph::Node *neighbor = nullptr;  // neighboring frame (if applicable)
};

// Map from token -> mentions starting/ending/both at it.
struct TokenToMentions {
  void Start(const Span *span) {
    if (!starting.empty()) {
      QCHECK_GE(starting.back()->end(), span->end());
    }
    starting.emplace_back(span);
  }

  void End(const Span *span) {
    if (!ending.empty()) {
      QCHECK_LE(ending.front()->begin(), span->begin());
    }
    ending.insert(ending.begin(), span);
  }

  void Singleton(const Span *span) {
    singletons.emplace_back(span);
  }

  bool empty() const {
    return starting.empty() && ending.empty() && singletons.empty();
  }

  std::vector<const Span *> starting;
  std::vector<const Span *> ending;
  std::vector<const Span *> singletons;
};

int Index(const Handles &attention, Handle item) {
  for (int i = 0; i < attention.size(); ++i) {
    if (attention[attention.size() - 1 - i] == item) return i;
  }
  return -1;
}

// Translates 'action' into an attention-resolved version.
ParserAction Translate(const Handles &attention, const Action &action) {
  ParserAction output(action.core.type);
  if (action.core.length != 0) output.length = action.core.length;
  if (!action.core.role.IsNil()) output.role = action.core.role;

  switch (output.type) {
    case ParserAction::EVOKE:
      output.label = action.frame->type;
      break;
    case ParserAction::REFER:
      output.target = Index(attention, action.frame->handle);
      break;
    case ParserAction::CONNECT:
      output.source = Index(attention, action.frame->handle);
      output.target = Index(attention, action.neighbor->handle);
      break;
    case ParserAction::ASSIGN:
      output.source = Index(attention, action.frame->handle);
      output.label = action.core.label;
      break;
    default:
    break;
  }

  return output;
}

// Updates 'attention' as a result of execution 'action'.
void Update(const Action &action, Handles *attention) {
  auto type = action.core.type;
  if (type == ParserAction::EVOKE) {
    attention->emplace_back(action.frame->handle);
  } else if (type == ParserAction::REFER ||
             type == ParserAction::ASSIGN ||
             type == ParserAction::CONNECT) {
    QCHECK_GT(attention->size(), 0);
    auto handle = action.frame->handle;
    if (handle != attention->back()) {
      int i = attention->size() - 1;
      while (i >= 0 && (*attention)[i] != handle) --i;
      QCHECK_GT(i, -1);
      attention->erase(attention->begin() + i);
      attention->emplace_back(handle);
    }
  }
}

void InitNode(
    const Frame &frame, FrameGraph *frame_graph, HandleSet *initialized) {
  Handle handle = frame.handle();
  if (initialized->find(handle) != initialized->end()) return;
  initialized->insert(handle);

  auto *node = frame_graph->Get(handle);
  for (const Slot &slot : frame) {
    if (slot.name.IsId()) continue;

    Frame value(frame.store(), slot.value);
    if (!value.IsFrame()) continue;

    if (slot.name.IsIsA() && node->type.IsNil()) {
      node->type = slot.value;
    } else {
      auto *edge = new FrameGraph::Edge(slot.name, false, slot.value);
      node->edges.emplace_back(edge);
      if (slot.value == handle) {
        edge->inverse = edge;
        continue;
      }
      if (value.IsAnonymous()) {
        auto *neighbor = frame_graph->Get(slot.value);
        auto *reverse = new FrameGraph::Edge(slot.name, true, handle);
        neighbor->edges.emplace_back(reverse);
        reverse->inverse = edge;
        edge->inverse = reverse;
        InitNode(value, frame_graph, initialized);
      }
    }
  }
}

void InitializeGeneration(
    const Document &document,
    int begin, int end,
    FrameGraph *frame_graph,
    std::vector<TokenToMentions> *token_to_mentions) {
  HandleSet initialized;
  token_to_mentions->resize(end - begin);

  std::vector<Span *> spans = document.spans();
  std::sort(spans.begin(), spans.end(), [](Span *a, Span *b) {
    if (a->begin() == b->begin()) {
      return a->length() > b->length();
    } else {
      return a->begin() < b->begin();
    }
  });

  Handles evoked(document.store());
  for (const Span *span : spans) {
    if (span->deleted() || span->begin() < begin || span->end() > end) continue;

    evoked.clear();
    span->AllEvoked(&evoked);
    for (Handle handle : evoked) {
      Frame frame(document.store(), handle);
      InitNode(frame, frame_graph, &initialized);
      frame_graph->node(handle)->evoked = true;
    }

    int index = span->begin() - begin;
    if (span->length() == 1) {
      (*token_to_mentions)[index].Singleton(span);
    } else {
      (*token_to_mentions)[index].Start(span);
      (*token_to_mentions)[span->end() - 1 - begin].End(span);
    }
  }

  for (Handle theme : document.themes()) {
    Frame frame(document.store(), theme);
    InitNode(frame, frame_graph, &initialized);
  }
}

void CollectSpanActions(Store *store,
                        const std::vector<TokenToMentions> &token_to_mentions,
                        const FrameGraph &frame_graph,
                        std::deque<Action> *actions) {
  HandleSet marked;
  HandleSet evoked;

  Handles all_evoked(store);
  for (int i = 0; i < token_to_mentions.size(); ++i) {
    const TokenToMentions &t2m = token_to_mentions[i];

    // Evoke singletons.
    for (const Span *span : t2m.singletons) {
      span->AllEvoked(&all_evoked);
      for (Handle h : all_evoked) {
        if (marked.find(h) != marked.end()) {
          QCHECK(evoked.find(h) != evoked.end());
        }
        if (evoked.find(h) != evoked.end()) {
          auto refer = Action(ParserAction::REFER, frame_graph.node(h));
          refer.core.length = span->length();   // = 1
          actions->emplace_back(refer);
        } else {
          QCHECK_EQ(span->length(), 1);
          auto evoke = Action(ParserAction::EVOKE, frame_graph.node(h));
          evoke.core.length = span->length();   // = 1
          evoke.core.label = frame_graph.node(h)->type;
          actions->emplace_back(evoke);
          marked.emplace(h);
          evoked.emplace(h);
        }
      }
    }

    for (const Span *span : t2m.ending) {
      QCHECK_GT(span->length(), 1);
      span->AllEvoked(&all_evoked);
      for (Handle h : all_evoked) {
        QCHECK(marked.find(h) != marked.end());
        if (evoked.find(h) != evoked.end()) continue;

        QCHECK_GT(span->length(), 1);
        auto evoke = Action(ParserAction::EVOKE, frame_graph.node(h));
        evoke.core.label = frame_graph.node(h)->type;
        actions->emplace_back(evoke);
        evoked.emplace(h);
      }
    }

    for (const Span *span : t2m.starting) {
      QCHECK_GT(span->length(), 1);
      span->AllEvoked(&all_evoked);
      for (Handle h : all_evoked) {
        if (marked.find(h) != marked.end()) {
          QCHECK(evoked.find(h) != evoked.end());
        }
        if (evoked.find(h) != evoked.end()) {
          auto refer = Action(ParserAction::REFER, frame_graph.node(h));
          refer.core.length = span->length();
          actions->emplace_back(refer);
        } else {
          auto mark = Action(ParserAction::MARK, frame_graph.node(h));
          actions->emplace_back(mark);
          marked.emplace(h);
        }
      }
    }

    actions->emplace_back(ParserAction::SHIFT, nullptr);
  }
}

void OutputActions(Store *store,
                   const FrameGraph &frame_graph,
                   std::deque<Action> *actions,
                   std::function<void(const ParserAction &)> callback) {
  Handles attention(store);
  while (!actions->empty()) {
    Action action = actions->front();
    ParserAction output = Translate(attention, action);
    if (callback) callback(output);
    Update(action, &attention);
    actions->pop_front();

    ParserAction::Type type = output.type;
    if (type == ParserAction::EVOKE) {
      action.frame->accounted = true;

      // CONNECT.
      for (auto *edge : action.frame->edges) {
        if (edge->accounted) continue;

        FrameGraph::Node *neighbor = frame_graph.node(edge->neighbor);
        if (neighbor == nullptr || !neighbor->accounted) continue;

        Action connect(ParserAction::CONNECT);
        connect.core.role = edge->role;
        connect.frame = edge->incoming ? neighbor : action.frame;
        connect.neighbor = edge->incoming ? action.frame : neighbor;
        edge->accounted = true;
        edge->inverse->accounted = true;
        actions->push_front(connect);
      }

      // ASSIGN.
      for (auto *edge : action.frame->edges) {
        if (edge->accounted || edge->incoming) continue;
        if (store->IsAnonymous(edge->neighbor)) continue;
        Action assign(ParserAction::ASSIGN, action.frame);
        assign.core.role = edge->role;
        assign.core.label = edge->neighbor;
        edge->accounted = true;
        actions->push_front(assign);
      }
    }
  }
}

}  // namespace

void Generate(const Document &document,
              int begin, int end,
              std::function<void(const ParserAction &)> callback) {
  FrameGraph frame_graph;
  std::vector<TokenToMentions> token_to_mentions;

  // Collect information about all frames.
  Store *store = document.store();
  InitializeGeneration(document, begin, end, &frame_graph, &token_to_mentions);

  // Collect span-related actions, i.e. SHIFT, MARK, EVOKE, REFER, STOP.
  // Collected in a deque for efficiently popping them off the front.
  std::deque<Action> actions;
  CollectSpanActions(store, token_to_mentions, frame_graph, &actions);

  // Output all actions, including any CONNECT, ASSIGN etc.
  OutputActions(store, frame_graph, &actions, callback);
}

void Generate(const Document &document,
              std::function<void(const ParserAction &)> callback) {
  Generate(document, 0, document.length(), callback);
}

}  // namespace nlp
}  // namespace sling
