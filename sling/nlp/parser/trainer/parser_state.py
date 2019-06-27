# Copyright 2017 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http:#www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sling

from action import Action

# ParserState maintains the under-construction frame graph for a given text.
class ParserState:
  # Represents a span in the text.
  class Span:
    def __init__(self, start, length):
      self.start = start
      self.end = start + length
      self.evoked = []   # frame(s) evoked by the span


  # Represents the beginning token of a span.
  class Mark:
    def __init__(self, token, step):
      self.token = token
      self.step = step


  # Represents a frame.
  class Frame:
    def __init__(self, t):
      self.type = t

      # Frame slots.
      self.edges = []

      # Boundaries of the first span (if any) that evoked the frame,
      self.start = -1
      self.end = -1

      # All span(s), if any, that evoked the frame.
      self.spans = []

      # Steps that focused / created the frame.
      self.focus = 0
      self.creation = 0


  def __init__(self, document, spec):
    self.document = document
    self.spec = spec
    self.begin = 0
    self.end = len(document.tokens)
    self.current = 0                 # current input position
    self.frames = []                 # frames added so far
    self.spans = {}                  # spans added so far
    self.steps = 0                   # no. of steps taken so far
    self.actions = []                # actual steps taken so far
    self.graph = []                  # edges in the frame graph
    self.done = False                # if the graph is complete
    self.attention = []              # frames in attention buffer
    self.marks = []                  # marked (i.e. open) spans
    self.embed = []                  # current embedded frames
    self.elaborate = []              # current elaborated frames
    self.max_mark_nesting = 5        # max number of open marks

    # Token -> Spans over it.
    self.token_to_spans = [[] for _ in range(len(document.tokens))]


  # Returns a string representation of the parser state.
  def __repr__(self):
    s = "Curr:" + str(self.current) + " in [" + str(self.begin) + \
        ", " + str(self.end) + ")" + " " + str(len(self.frames)) + " frames"
    for index, f in enumerate(self.attention):
      if index == 10: break
      s += "\n   - Attn " + str(index) + ":" + str(f.type) + \
           " Creation:" + str(f.creation) + \
           ", Focus:" + str(f.focus) + ", #Edges:" + str(len(f.edges)) + \
           " (" + str(len(f.spans)) + " spans) "
      if len(f.spans) > 0:
        for span in f.spans:
          words = self.document.tokens[span.start].word
          if span.end > span.start + 1:
            words += ".." + self.document.tokens[span.end - 1].word
          s += words + " = [" + str(span.start) + ", " + str(span.end) + ") "
    return s


  # Computes the role graph.
  def compute_role_graph(self):
    # No computation required if none of the actions have roles.
    if len(self.spec.actions.roles) == 0: return

    del self.graph
    self.graph = []
    limit = min(self.spec.frame_limit, len(self.attention))
    for i in range(limit):
      frame = self.attention[i]
      for role, value in frame.edges:
        role_id = self.spec.actions.role_indices.get(role, None)
        if role_id is not None:
          target = -1
          if isinstance(value, ParserState.Frame):
            target = self.index(value)
            if target == -1 or target >= self.spec.frame_limit: continue
          self.graph.append((i, role_id, target))


  # Returns the step at which index-th attention frame was created.
  def creation_step(self, index):
    if index >= len(self.attention) or index < 0: return -1
    return self.attention[index].creation


  # Returns the most recent step at which index-th attention frame was focused.
  def focus_step(self, index):
    if index >= len(self.attention) or index < 0: return -1
    return self.attention[index].focus


  # Returns whether [start, end) crosses an existing span.
  def _crosses(self, start, end):
    for token in range(start, end):
      for s in self.token_to_spans[token]:
        if (s.start - start) * (s.end - end) > 0:
          return True
    return False


  # Check fails if there is any crossing span in the parser state.
  def check_spans(self):
    spans = list(self.spans.keys())
    spans.sort(key=lambda s: (s[0], s[0] - s[1]))
    cover = [None] * len(self.document.tokens)
    for s in spans:
      c = cover[s[0]]
      for i in range(s[0], s[1]):
        assert c == cover[i], (c, cover[i], spans, self.actions)
        cover[i] = s


  # Returns whether 'action_index' is allowed in the current state.
  def is_allowed(self, action_index):
    if self.done: return False

    actions = self.spec.actions
    if action_index == actions.stop(): return self.current == self.end
    if action_index == actions.shift(): return self.current < self.end
    if action_index == actions.mark():
      return self.current < self.end and len(self.marks) < self.max_mark_nesting

    action = actions.table[action_index]
    if action.type == Action.REFER:
      end = self.current + action.length
      if end > self.end or \
        action.target >= self.attention_size() or \
        self._crosses(self.current, end):
          return False

      existing = self._get_span(self.current, end)
      if existing is not None:
        target = self.attention[action.target]
        for f in existing.evoked:
          if f is target: return False
      return True

    if action.type == Action.EVOKE:
      if action.length is None:
        if len(self.marks) == 0 or self.marks[-1].token == self.current \
          or self.current == self.end:
          return False
        return not self._crosses(self.marks[-1].token, self.current + 1)
      else:
        end = self.current + action.length
        if end > self.end or self._crosses(self.current, end):
          return False
        existing = self._get_span(self.current, end)
        if existing is not None:
          for f in existing.evoked:
            if f.type == action.label: return False
        return True
    elif action.type == Action.CONNECT:
      s = self.attention_size()
      if action.source >= s or action.target >= s: return False
      source = self.attention[action.source]
      target = self.attention[action.target]
      for role, value in source.edges:
        if role == action.role and value is target: return False
      return True
    elif action.type == Action.EMBED:
      if action.target >= self.attention_size(): return False
      target = self.attention[action.target]
      for t, role, value in self.embed:
        if t == action.label and role == action.role and value is target:
          return False
      return True
    elif action.type == Action.ELABORATE:
      if action.source >= self.attention_size(): return False
      source = self.attention[action.source]
      for t, role, value in self.elaborate:
        if t == action.label and role == action.role and value is source:
          return False
      return True
    elif action.type == Action.ASSIGN:
      if action.source >= self.attention_size(): return False
      source = self.attention[action.source]
      for role, value in source.edges:
        if role == action.role and value == action.label: return False
      return True
    else:
      raise ValueError("Unknown action : ", action)


  # Returns the attention index of 'frame'.
  def index(self, frame):
    for i in range(len(self.attention)):
      if self.attention[i] is frame:
        return i
    return -1


  # Returns the frame at attention index 'index'.
  def frame(self, index):
    return self.attention[index]


  # Returns the size of the attention buffer.
  def attention_size(self):
    return len(self.attention)


  # Returns the role graph.
  def role_graph(self):
    return self.graph


  # Returns the end token (inclusive) of the span, if any, that evoked/referred
  # the frame at attention index 'index'.
  def frame_end_inclusive(self, index):
    if index >= len(self.attention) or index < 0:
      return -1
    elif self.attention[index].end == -1:
      return -1
    else:
      return self.attention[index].end - 1


  # Advances the state using 'action'.
  def advance(self, action):
    self.actions.append(action)
    if action.type == Action.STOP:
      self.done = True
    elif action.type == Action.SHIFT:
      self.current += 1
      del self.embed[:]
      del self.elaborate[:]
    elif action.type == Action.MARK:
      self.marks.append(ParserState.Mark(self.current, len(self.actions) - 1))
    elif action.type == Action.EVOKE:
      begin = self.current
      end = self.current + 1
      if action.length is None:
        begin = self.marks.pop().token
      else:
        assert action.length > 0
        end = self.current + action.length
      s = self._make_span(begin, end)
      f = self._make_frame(action.label)
      f.start = begin
      f.end = end
      f.spans.append(s)
      s.evoked.append(f)
      self.frames.append(f)
      self._add_to_attention(f)
    elif action.type == Action.REFER:
      f = self.attention[action.target]
      f.focus = self.steps
      s = self._make_span(self.current, self.current + action.length)
      s.evoked.append(f)
      f.spans.append(s)
      self._refocus_attention(action.target)
    elif action.type == Action.CONNECT:
      f = self.attention[action.source]
      f.edges.append((action.role, self.attention[action.target]))
      f.focus = self.steps
      self._refocus_attention(action.source)
    elif action.type == Action.EMBED:
      target = self.attention[action.target]
      f = self._make_frame(action.label)
      f.edges.append((action.role, target))
      self.frames.append(f)
      self._add_to_attention(f)
      self.embed.append((action.label, action.role, target))
    elif action.type == Action.ELABORATE:
      source = self.attention[action.source]
      f = self._make_frame(action.label)
      source.edges.append((action.role, f))
      self.frames.append(f)
      self._add_to_attention(f)
      self.elaborate.append((action.label, action.role, source))
    elif action.type == Action.ASSIGN:
      source = self.attention[action.source]
      source.focus = self.steps
      source.edges.append((action.role, action.label))
      self._refocus_attention(action.source)
    else:
      raise ValueError("Unknown action type: ", action.type)

    self.steps += 1
    if action.type != Action.SHIFT and action.type != Action.STOP:
      # Recompute the role graph because we modified the attention buffer.
      if self.spec: self.compute_role_graph()


  # Write the frame graph to 'document', which should be a SLING Document.
  def write(self, document=None):
    if document is None:
      document = self.document

    store = document.frame.store()
    document.remove_annotations()
    frames = {}

    for f in self.frames:
      frame = store.frame({})
      if f.type is not None:
        frame["isa"] = f.type
      frames[f] = frame
      if len(f.spans) == 0:
        document.add_theme(frame)

    for f in self.frames:
      frame = frames[f]
      for role, value in f.edges:
        if isinstance(value, ParserState.Frame):
          # Add slot whose value is a reference to another frame.
          assert value in frames, str(value.__dict__)
          frame.append(role, frames[value])
        else:
          # Add slot whose value is a reference to a global frame (cf. ASSIGN).
          assert type(value) == sling.Frame, "%r" % value
          frame.append(role, value)

    for _, s in self.spans.items():
      # Note: mention.frame is the actual mention frame.
      mention = document.add_mention(s.start, s.end)
      for f in s.evoked:
        assert f in frames
        mention.frame.append("evokes", frames[f])

    document.update()


  # Returns the string representation of the framing.
  def data(self, **kwargs):
    return self.document.frame.data(**kwargs)


  # Returns the encoded representation of the framing.
  def encoded(self):
    return self.data(binary=True, shallow=True)


  # Returns the textual representation of the framing.
  def textual(self):
    return self.data(binary=False, pretty=True, shallow=True)


  # Adds frame 'f' to the attention buffer.
  def _add_to_attention(self, f):
    f.focus = self.steps
    self.attention.insert(0, f)


  # Makes the frame at attention index 'index' the center of attention.
  def _refocus_attention(self, index):
    f = self.attention[index]
    f.focus = self.steps
    if index > 0: self.attention.insert(0, self.attention.pop(index))


  # Gets and existing [begin, end) span or None.
  def _get_span(self, begin, end):
    key = (begin, end)
    return self.spans.get(key, None)


  # Creates and returns a [begin, end) span.
  def _make_span(self, begin, end):
    # See if an existing span can be returned.
    key = (begin, end)
    existing = self.spans.get(key, None)
    if existing is not None: return existing
    s = ParserState.Span(begin, end - begin)
    self.spans[key] = s
    for i in range(begin, end):
      self.token_to_spans[i].append(s)
    return s


  # Creates and returns a frame of type 't'.
  def _make_frame(self, t):
    f = ParserState.Frame(t)
    f.creation = self.steps
    f.focus = self.steps
    return f

