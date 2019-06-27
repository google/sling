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

# Outputs a list of transitions that represent a given document's frame graph.
class TransitionGenerator:
  # Bookkeeping for one frame.
  class FrameInfo:
    def __init__(self, handle):
      self.handle = handle
      self.type = None
      self.edges = []
      self.from_mention = False

      # Whether this frame has been evoked.
      self.output = False


  # Labeled edge between two frames. Each edge is used to issue a CONNECT,
  # EMBED, ELABORATE, or ASSIGN action.
  class Edge:
    def __init__(self, incoming=None, role=None, value=None):
      self.incoming = incoming
      self.role = role
      self.neighbor = value
      self.inverse = None
      self.used = False


  # Simple action that will be translated into an Action object later on.
  class SimpleAction:
    def __init__(self, t=None):
      self.action = Action(t)
      self.info = None
      self.other_info = None


  def __init__(self, commons):
    self.commons = commons
    self._id = commons["id"]
    self._isa = commons["isa"]


  # Returns whether 'handle' refers to another frame.
  def is_ref(self, handle):
    return type(handle) == sling.Frame


  # Creates a FrameInfo object for 'frame' and recursively for all frames
  # pointed to by it.
  def _init_info(self, frame, frame_info, initialized):
    if frame in initialized:
      return
    initialized[frame] = True

    info = frame_info.get(frame, None)
    if info is None:
      info = TransitionGenerator.FrameInfo(frame)
      frame_info[frame] = info

    pending = []
    for role, value in frame:
      if not self.is_ref(value) or role == self._id: continue
      if role == self._isa and value.islocal(): continue

      if role == self._isa and info.type is None:
        info.type = value
      else:
        edge = TransitionGenerator.Edge(incoming=False, role=role, value=value)
        info.edges.append(edge)
        if value == frame:
          edge.inverse = edge
          continue

        if value.islocal():
          nb_info = frame_info.get(value, None)
          if nb_info is None:
            nb_info = TransitionGenerator.FrameInfo(value)
            frame_info[value] = nb_info
          nb_edge = TransitionGenerator.Edge(
              incoming=True, role=role, value=frame)
          nb_info.edges.append(nb_edge)
          nb_edge.inverse = edge
          edge.inverse = nb_edge
          pending.append(value)

    # Initialize bookkeeping for all frames pointed to by this frame.
    for p in pending:
      self._init_info(p, frame_info, initialized)


  # Translates 'simple' action to an Action using indices from 'attention'.
  def _translate(self, attention, simple):
    action = Action(t=simple.action.type)
    if simple.action.length is not None:
      action.length = simple.action.length
    if simple.action.role is not None:
      action.role = simple.action.role

    if action.type == Action.EVOKE:
       action.label = simple.info.type
    elif action.type == Action.REFER:
      action.target = attention.index(simple.info.handle)
    elif action.type == Action.EMBED:
      action.label = simple.info.type
      action.target = attention.index(simple.other_info.handle)
    elif action.type == Action.ELABORATE:
      action.label = simple.info.type
      action.source = attention.index(simple.other_info.handle)
    elif action.type == Action.CONNECT:
      action.source = attention.index(simple.info.handle)
      action.target = attention.index(simple.other_info.handle)
    elif action.type == Action.ASSIGN:
      action.source = attention.index(simple.info.handle)
      action.label = simple.action.label

    return action


  # Updates frame indices in 'attention' as a result of the action 'simple'.
  def _update(self, attention, simple):
    t = simple.action.type
    if t in [Action.EVOKE, Action.EMBED, Action.ELABORATE]:
      # Insert a new frame at the center of attention.
      attention.insert(0, simple.info.handle)
    elif t in [Action.REFER, Action.ASSIGN, Action.CONNECT]:
      # Promote an existing frame to the center of attention.
      attention.remove(simple.info.handle)
      attention.insert(0, simple.info.handle)


  # Builds and returns a simple action of type 'type'.
  def _simple_action(self, type=None):
    return TransitionGenerator.SimpleAction(type)


  # Stores mentions starting or ending or both at a given token.
  class TokenToMentions:
    def __init__(self):
      self.starting = []
      self.ending = []
      self.singletons = []

    # Record 'mention' at starting at this token.
    def start(self, mention):
      if len(self.starting) > 0:
        # Check that the mention respects nesting.
        assert self.starting[-1].end >= mention.end
      self.starting.append(mention)

    # Record 'mention' as ending at this token.
    def end(self, mention):
      if len(self.ending) > 0:
        # Check that the mention respects nesting.
        assert self.ending[0].begin <= mention.begin
      self.ending.insert(0, mention)  # most-nested is at the front

    # Record 'mention' as starting and ending at this token.
    def singleton(self, mention):
      self.singletons.append(mention)

    # Returns if there are no mentions starting/ending at this token.
    def empty(self):
      return len(self.starting) + len(self.ending) + len(self.singletons) == 0

    # Returns a string representation of the object.
    def __repr__(self):
      return "Starting:" + str(self.starting) + ", Ending:" + \
        str(self.ending) + ", Singletons:" + str(self.singletons)


  # Generates transition sequence for 'document' which should be an instance of
  # AnnotatedDocument.
  def generate(self, document):
    frame_info = {}    # frame -> whether it is evoked from a span
    initialized = {}   # frame -> whether the frame's book-keeping is done

    # Initialize book-keeping for all evoked frames.
    for m in document.mentions:
      for evoked in m.evokes():
        self._init_info(evoked, frame_info, initialized)
        frame_info[evoked].from_mention = True

    # Initialize book-keeping for all thematic frames.
    for theme in document.themes:
      self._init_info(theme, frame_info, initialized)

    # Record start/end boundaries of all mentions.
    token_to_mentions = []
    for _ in range(len(document.tokens)):
      token_to_mentions.append(TransitionGenerator.TokenToMentions())

    for m in document.mentions:
      if m.length == 1:
        token_to_mentions[m.begin].singleton(m)
      else:
        token_to_mentions[m.begin].start(m)
        token_to_mentions[m.end - 1].end(m)

    # Single token mentions are handled via EVOKE(length=1), and others
    # are handled via MARK at the beginning token and EVOKE(length=None)
    # at the end token.

    simple_actions = []
    marked = {}   # frames for which we have output a MARK
    evoked = {}   # frames for which we have output an EVOKE
    for index in range(len(document.tokens)):
      t2m = token_to_mentions[index]

      # First evoke/refer the singleton mentions.
      for singleton in t2m.singletons:
        for frame in singleton.evokes():
          # If the frame is already evoked, refer to it.
          if frame in marked:
            assert frame in evoked, "Referring to marked but not evoked frame"
          if frame in evoked:
            refer = self._simple_action(Action.REFER)
            refer.info = frame_info[frame]
            refer.action.length = singleton.length  # should be 1
            simple_actions.append(refer)
            continue

          # Otherwise evoke a new frame.
          evoke = self._simple_action(Action.EVOKE)
          evoke.action.length = singleton.length  # should be 1
          evoke.info = frame_info[frame]
          evoke.action.label = evoke.info.type
          simple_actions.append(evoke)
          marked[frame] = True
          evoked[frame] = True

      # Output EVOKE for any frames whose spans end here.
      for mention in t2m.ending:
        assert mention.length > 1, mention.length  # singletons already handled
        for frame in mention.evokes():
          assert frame in marked   # frame should be already MARKed
          if frame in evoked:
            # Already handled via REFER at mention.begin.
            continue

          evoke = self._simple_action(Action.EVOKE)
          evoke.info = frame_info[frame]
          evoke.action.label = evoke.info.type
          simple_actions.append(evoke)
          evoked[frame] = True

      # Output MARK for any frames whose spans begin here.
      for mention in t2m.starting:
        assert mention.length > 1, mention.length  # singletons already handled
        for frame in mention.evokes():
          # Check if this is a fresh frame or a refer.
          if frame in marked:
            assert frame in evoked, "Referring to marked but not evoked frame"
          if frame in evoked:
            refer = self._simple_action(Action.REFER)
            refer.info = frame_info[frame]
            refer.action.length = mention.length
            simple_actions.append(refer)
            continue

          mark = self._simple_action(Action.MARK)
          mark.info = frame_info[frame]
          simple_actions.append(mark)
          marked[frame] = True

      # Move to the next token.
      simple_actions.append(self._simple_action(Action.SHIFT))
    simple_actions.append(self._simple_action(Action.STOP))

    # Recursively output more actions (e.g. CONNECT, EMBED, ELABORATE, ASSIGN)
    # from the current set of EVOKE/REFER actions. Then translate each
    # action using final attention indices. This is done in reverse order
    # for convenience.
    simple_actions.reverse()
    actions = []
    attention = []
    while len(simple_actions) > 0:
      simple_action = simple_actions.pop()
      actions.append(self._translate(attention, simple_action))
      self._update(attention, simple_action)
      t = simple_action.action.type
      if t == Action.EVOKE or t == Action.EMBED or t == Action.ELABORATE:
        simple_action.info.output = True

        # CONNECT actions triggered by the newly output frame.
        for e in simple_action.info.edges:
          if e.used: continue

          nb = frame_info.get(e.neighbor, None)
          if nb is not None and nb.output:
            connect = self._simple_action(Action.CONNECT)
            connect.action.role = e.role
            connect.info = nb if e.incoming else simple_action.info
            connect.other_info = simple_action.info if e.incoming else nb
            simple_actions.append(connect)
            e.used = True
            e.inverse.used = True

        # EMBED actions triggered by the newly output frame.
        for e in simple_action.info.edges:
          if e.used or not e.incoming: continue

          nb = frame_info.get(e.neighbor, None)
          if nb is not None and not nb.output and not nb.from_mention:
            embed = self._simple_action(Action.EMBED)
            embed.action.role = e.role
            embed.info = nb
            embed.other_info = simple_action.info
            simple_actions.append(embed)
            e.used = True
            e.inverse.used = True

        # ELABORATE actions triggered by the newly output frame.
        for e in simple_action.info.edges:
          if e.used or e.incoming: continue

          nb = frame_info.get(e.neighbor, None)
          if nb is not None and not nb.output and not nb.from_mention:
            elaborate = self._simple_action(Action.ELABORATE)
            elaborate.action.role = e.role
            elaborate.info = nb
            elaborate.other_info = simple_action.info
            simple_actions.append(elaborate)
            e.used = True
            e.inverse.used = True

        # ASSIGN actions triggered by the newly output frame.
        for e in simple_action.info.edges:
          if not e.used and not e.neighbor.islocal() and not e.incoming:
            assign = self._simple_action(Action.ASSIGN)
            assign.info = simple_action.info
            assign.action.role = e.role
            assign.action.label = e.neighbor
            simple_actions.append(assign)
            e.used = True

    return actions

