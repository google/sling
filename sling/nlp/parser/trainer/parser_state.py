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
    self.current = 0                               # current input position
    self.frames = []                               # frames added so far
    self.spans = []                                # spans added so far
    self.steps = 0                                 # no. of steps taken so far
    self.actions = []                              # actual steps taken so far
    self.graph = []                                # edges in the frame graph
    self.allowed = [False] * spec.actions.size()   # allowed actions bitmap
    self.done = False                              # if the graph is complete
    self.attention = []                            # frames in attention buffer
    self.nesting = []                              # current nesting information
    self.embed = []                                # current embedded frames
    self.elaborate = []                            # current elaborated frames


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
          words = self.document.tokens[span.start].text
          if span.end > span.start + 1:
            words += ".." + self.document.tokens[span.end - 1].text
          s += words + " = [" + str(span.start) + ", " + str(span.end) + ") "
    return s


  # Computes the role graph.
  def compute_role_graph(self):
    # No computation required if none of the actions have roles.
    if len(self.spec.actions.roles) == 0: return

    del self.graph
    self.graph = []
    limit = min(self.spec.frame_limit, len(self.attention))
    for i in xrange(limit):
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


  # Returns whether 'action_index' is allowed in the current state.
  def is_allowed(self, action_index):
    if self.done: return False

    actions = self.spec.actions
    if self.current == self.end: return action_index == actions.stop()
    if action_index == actions.stop(): return False
    if action_index == actions.shift(): return True
    action = actions.table[action_index]

    if action.type == Action.EVOKE or action.type == Action.REFER:
      if self.current + action.length > self.end: return False
      if action.type == Action.REFER and action.target >= self.attention_size():
        return False

      # No existing spans, so everything is allowed.
      if len(self.nesting) == 0: return True

      # Proposed span can't be longer than the most nested span.
      outer = self.nesting[-1]
      assert outer.start <= self.current
      assert outer.end > self.current
      gap = outer.end - self.current
      if gap != action.length: return gap > action.length

      if outer.start < self.current:
        return True
      else:
        if action.type == Action.EVOKE:
          for f in outer.evoked:
            if f.type == action.label: return False
          return True
        else:
          target = self.attention[action.target]
          for f in outer.evoked:
            if f is target: return False
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
    for i in xrange(len(self.attention)):
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
    else:
      return self.attention[index].end - 1


  # Advances the state using 'action'.
  def advance(self, action):
    self.actions.append(action)
    if action.type == Action.STOP:
      self.done = True
    elif action.type == Action.SHIFT:
      self.current += 1
      while len(self.nesting) > 0:
        if self.nesting[-1].end <= self.current:
          self.nesting.pop()
        else:
          break
      del self.embed[:]
      del self.elaborate[:]
    elif action.type == Action.EVOKE:
      s = self._make_span(action.length)
      f = self._make_frame(action.label)
      f.start = self.current
      f.end = self.current + action.length
      f.spans.append(s)
      s.evoked.append(f)
      self.frames.append(f)
      self._add_to_attention(f)
    elif action.type == Action.REFER:
      f = self.attention[action.target]
      f.focus = self.steps
      s = self._make_span(action.length)
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
      self._add_to_attention(f)
      self.embed.append((action.label, action.role, target))
    elif action.type == Action.ELABORATE:
      source = self.attention[action.source]
      f = self._make_frame(action.label)
      source.edges.append((action.role, f))
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
      self.compute_role_graph()


  # Write the frame graph to 'document', which should be a SLING Document.
  def write(self, document=None):
    if document is None:
      document = self.document

    store = document.frame.store()
    document.remove_annotations()
    frames = {}

    for f in self.frames:
      frame = store.frame({"isa": f.type})
      frames[f] = frame
      if len(f.spans) == 0:
        document.add_theme(frame)

    for f in self.frames:
      frame = frames[f]
      for role, value in f.edges:
        if isinstance(value, ParserState.Frame):
          # Add slot whose value is a reference to another frame.
          assert value in frames
          frame.append(role, frames[value])
        else:
          # Add slot whose value is a reference to a global frame (cf. ASSIGN).
          assert type(value) == sling.Frame, "%r" % value
          frame.append(role, value)

    for s in self.spans:
      # Note: mention.frame is the actual mention frame.
      mention = document.add_mention(s.start, s.end)
      for f in s.evoked:
        assert f in frames
        mention.frame.append("/s/phrase/evokes", frames[f])

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


  # Creates and returns a span of length 'length'.
  def _make_span(self, length):
    # See if an existing span can be returned.
    if len(self.nesting) > 0:
      last = self.nesting[-1]
      if last.start == self.current and last.end == self.current + length:
        return last
    s = ParserState.Span(self.current, length)
    self.spans.append(s)
    self.nesting.append(s)
    return s


  # Creates and returns a frame of type 't'.
  def _make_frame(self, t):
    f = ParserState.Frame(t)
    f.creation = self.steps
    f.focus = self.steps
    return f

