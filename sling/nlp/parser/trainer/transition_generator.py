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
      self.mention = None
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
    self._thing = commons["/s/thing"]  # fallback type
    assert self._thing.isglobal()


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

    # Assign a fallback type.
    if info.type is None:
      info.type = self._thing

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
    if t == Action.EVOKE or t == Action.EMBED or t == Action.ELABORATE:
      # Insert a new frame at the center of attention.
      attention.insert(0, simple.info.handle)
    elif t == Action.REFER or t == Action.ASSIGN or t == Action.CONNECT:
      # Promote an existing frame to the center of attention.
      attention.remove(simple.info.handle)
      attention.insert(0, simple.info.handle)


  # Builds and returns a simple action of type 'type'.
  def _simple_action(self, type=None):
    return TransitionGenerator.SimpleAction(type)


  # Generates transition sequence for 'document' which should be an instance of
  # AnnotatedDocument.
  def generate(self, document):
    frame_info = {}
    initialized = {}

    # Initialize book-keeping for all evoked frames.
    for m in document.mentions:
      for evoked in m.evokes():
        self._init_info(evoked, frame_info, initialized)
        frame_info[evoked].mention = m

    # Initialize book-keeping for all thematic frames.
    for theme in document.themes:
      self._init_info(theme, frame_info, initialized)

    simple_actions = []
    start = 0
    evoked = {}
    for m in document.mentions:
      # Insert SHIFT actions between evoked frames.
      for i in xrange(start, m.begin):
        simple_actions.append(self._simple_action(Action.SHIFT))
      start = m.begin

      for frame in m.evokes():
        simple_action = self._simple_action()
        simple_action.action.length = m.length
        simple_action.info = frame_info[frame]

        # See if we are evoking a new or an existing frame.
        # Output an appropriate EVOKE/REFER action respectively.
        if frame not in evoked:
          simple_action.action.type = Action.EVOKE
          evoked[frame] = True
        else:
          simple_action.action.type = Action.REFER
        simple_actions.append(simple_action)

    # Output SHIFT actions after the last evoked frame.
    for index in xrange(start, document.size()):
      simple_actions.append(self._simple_action(Action.SHIFT))

    # Generate the final STOP action.
    simple_actions.append(self._simple_action(Action.STOP))

    # Recursively generate more actions (e.g. CONNECT, EMBED, ELABORATE, ASSIGN)
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
          if nb is not None and not nb.output and nb.mention is None:
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
          if nb is not None and not nb.output and nb.mention is None:
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

