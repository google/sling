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

# Action table.
class Actions:
  def __init__(self):
    # List of actions in id order.
    self.table = []

    # Action id -> Raw count.
    self.counts = []

    # Action id -> Disallowed or not.
    self.disallowed = []

    # Action -> id.
    self.indices = {}

    # Role id -> role and the inverse mapping.
    self.roles = []
    self.role_indices = {}

    # Various maximum indices/lengths.
    self.max_span_length = None
    self.max_connect_source = None
    self.max_connect_target = None
    self.max_refer_target = None
    self.max_embed_target = None
    self.max_elaborate_source = None
    self.frame_limit = None

    # Short-cuts for ids of SHIFT, STOP, and MARK actions.
    self.shift_index = None
    self.stop_index = None
    self.mark_index = None


  # Accessors.
  def stop(self):
    return self.stop_index


  def shift(self):
    return self.shift_index


  def mark(self):
    return self.mark_index


  def size(self):
    return len(self.table)


  def action(self, index):
    return self.table[index]


  def index(self, action):
    return self.indices.get(action, None)


  # Adds 'action' to the table.
  def add(self, action, count=1):
    index = self.indices.get(action, len(self.table))
    if index == len(self.table):
      self.indices[action] = index
      self.table.append(action)
      self.counts.append(0)
      if action.role is not None and action.role not in self.role_indices:
        role_index = len(self.roles)
        self.role_indices[action.role] = role_index
        self.roles.append(action.role)
    self.counts[index] = self.counts[index] + count

    if action.type == Action.SHIFT:
      self.shift_index = index
    if action.type == Action.STOP:
      self.stop_index = index
    if action.type == Action.MARK:
      self.mark_index = index


  # Returns the value of 'action_field' that covers 'percentile' amount
  # of actions of the 'action_type' as per raw counts.
  # Also sets the actions that are beyond this value as disallowed.
  def _maxvalue(self, action_type, action_field, percentile):
    # Compute the count histogram as per 'action_field'.
    values = []
    for action, count in zip(self.table, self.counts):
      if action_type is None or action.type == action_type:
        value = action.__dict__[action_field]
        if value is not None:
          values.extend([value] * count)

    # Degenerate case.
    if len(values) == 0: return 0

    values.sort()

    # Compute the value that covers 'percentile' of the histogram.
    index = int(len(values) * (1.0 * percentile) / 100)
    cutoff = values[index]

    # Mark actions that are beyond this cutoff as disallowed.
    for index, action in enumerate(self.table):
      if action_type is None or action.type == action_type:
        value = action.__dict__[action_field]
        disallowed = value is not None and value > cutoff
        self.disallowed[index] = self.disallowed[index] | disallowed

    return cutoff


  # Compute percentile-based cutoffs for various indices/length fields,
  # and mark actions beyond cutoffs as disallowed.
  def prune(self, percentile):
    p = percentile
    if p < 1 and type(p) is float: p = int(p * 100)

    self.disallowed = [False] * self.size()
    self.max_span_length = self._maxvalue(None, "length", p)
    self.max_connect_source = self._maxvalue(Action.CONNECT, "source", p)
    self.max_connect_target = self._maxvalue(Action.CONNECT, "target", p)
    self.max_assign_source = self._maxvalue(Action.ASSIGN, "source", p)
    self.max_refer_target = self._maxvalue(Action.REFER, "target", p)
    self.max_embed_target = self._maxvalue(Action.EMBED, "target", p)
    self.max_elaborate_source = self._maxvalue(Action.ELABORATE, "source", p)


  # Encodes and returns the action table as a SLING frame.
  def encoded(self, store):
    table = store.frame({"id": "/table"})

    table["/table/max_span_length"] = self.max_span_length
    table["/table/max_connect_source"] = self.max_connect_source
    table["/table/max_connect_target"] = self.max_connect_target
    table["/table/max_assign_source"] = self.max_assign_source
    table["/table/max_refer_target"] = self.max_refer_target
    table["/table/max_embed_target"] = self.max_embed_target
    table["/table/max_elaborate_source"] = self.max_elaborate_source
    table["/table/frame_limit"] = self.frame_limit

    actions_array = store.array(self.size())
    for index, action in enumerate(self.table):
      f = action.as_frame(store)
      f["/table/action/count"] = self.counts[index]
      f["/table/action/disallowed"] = self.disallowed[index]
      actions_array[index] = f
    table["/table/actions"] = actions_array

    table["/table/symbols"] = ["/table/action/" + i for i in [
        "type", "length", "source", "target", "role", \
        "label", "delegate", "count", "disallowed"]]
    return table

  # Decodes the action table afresh from 'frame'.
  def decode(self, frame):
    assert len(self.table) == 0, "Action table not empty!"
    assert frame.id == "/table"
    self.max_span_length = frame["/table/max_span_length"]
    self.max_connect_source = frame["/table/max_connect_source"]
    self.max_connect_target = frame["/table/max_connect_target"]
    self.max_assign_source = frame["/table/max_assign_source"]
    self.max_refer_target = frame["/table/max_refer_target"]
    self.max_embed_target = frame["/table/max_embed_target"]
    self.max_elaborate_source = frame["/table/max_elaborate_source"]
    self.frame_limit = frame["/table/frame_limit"]

    actions = frame["/table/actions"]
    self.disallowed = []
    for i in range(len(actions)):
      action = Action()
      action.from_frame(actions[i])
      self.add(action, count=actions[i]["/table/action/count"])
      self.disallowed.append(actions[i]["/table/action/disallowed"])


  # Returns a textual representation of the action table.
  def __str__(self):
    s = ["Action %d = %s" % (i, a) for i, a in enumerate(self.table)]
    return "\n".join(s)

