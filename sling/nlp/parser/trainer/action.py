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

# Represents a single transition.
class Action:
  SHIFT = 0              # shift to the next token
  STOP = 1               # stop processing
  EVOKE = 2              # make a span and evoke a frame
  REFER = 3              # make a span and re-evoke an existing frame
  CONNECT = 4            # link two frames
  ASSIGN = 5             # assign a constant slot to a frame 
  EMBED = 6              # create a new frame and link it to an existing frame
  ELABORATE = 7          # create a new frame and link an existing frame to it
  CASCADE = 8            # cascade to another delegate
  MARK = 9               # mark the current token as the beginning of a span
  NUM_ACTIONS = 10

  def __init__(self, t=None):
    # Action type.
    self.type = None

    # Length for REFER, and EVOKE. Note that while EVOKE with length > 1 is
    # supported, multi-token spans are generated via MARK and EVOKE pairs, and
    # singleton spans are generated via EVOKE(length = 1) (i.e. without MARK).
    self.length = None

    # Source frame for CONNECT, ELABORATE, ASSIGN.
    self.source = None

    # Target frame for CONNECT, EMBED, REFER.
    self.target = None

    # Role for CONNECT, EMBED, ELABORATE, ASSIGN.
    self.role = None

    # Type/constant label for EVOKE, ELABORATE, EMBED, ASSIGN.
    self.label = None

    # Delegate index for CASCADE. Note that CASCADE actions can also use
    # other fields in the action.
    self.delegate = None

    if t is not None:
      assert t < Action.NUM_ACTIONS
      assert t >= 0
      self.type = t


  # Converts the action to a tuple.
  def _totuple(self):
    return (self.type, self.length, self.source, self.target,
            self.role, self.label, self.delegate)


  # Methods to enable hashing of actions.
  def __hash__(self):
    return hash(self._totuple())


  def __eq__(self, other):
    return self._totuple() == other._totuple()


  # Returns the string representation of the action.
  def __repr__(self):
    names = {
      Action.SHIFT: "SHIFT",
      Action.STOP: "STOP",
      Action.EVOKE: "EVOKE",
      Action.REFER: "REFER",
      Action.CONNECT: "CONNECT",
      Action.ASSIGN: "ASSIGN",
      Action.EMBED: "EMBED",
      Action.ELABORATE: "ELABORATE",
      Action.CASCADE: "CASCADE",
      Action.MARK: "MARK"
    }

    s = names[self.type]
    for k, v in sorted(self.__dict__.items()):
      if v is not None and k != "type":
        if s != "":
          s = s + ", "
        s = s + k + ": " + str(v)
    return "(" + s + ")"

  # Returns frame representation of the action.
  def as_frame(self, store, slot_prefix="/table/action/"):
    frame = store.frame({})
    for s in self.__dict__.keys():
      val = getattr(self, s)
      if val is not None:
        if isinstance(val, sling.Frame):
          assert val.id is not None
          val = store[val.id]  # ensure we use a store-owned version
        frame[slot_prefix + s] = val
    return frame

  # Reads the action from 'frame'.
  def from_frame(self, frame, slot_prefix="/table/action/"):
    for s in self.__dict__.keys():
      name = slot_prefix + s
      setattr(self, s, frame[name])

  # Returns whether the action is a cascade.
  def is_cascade(self):
    return self.type == Action.CASCADE

  def is_shift(self):
    return self.type == Action.SHIFT

  def is_stop(self):
    return self.type == Action.STOP

  def is_mark(self):
    return self.type == Action.MARK
