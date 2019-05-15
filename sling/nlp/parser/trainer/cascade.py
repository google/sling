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

import inspect
import sling

from action import Action
from action_table import Actions

"""A cascade is a mechanism to output a parser action in a staged manner.
Each stage of the cascade is handled by a Delegate, which either computes
a piece of the eventual action, or delegates the computation to another
delegate, or both.

This mechanism allows each delegate to focus on a small part of the action,
thus allowing scalability (e.g. smaller softmax layer per delegate), as well
as generality (since custom delegates can be added easily).

Cascade model architecture is assumed to be the following:

                                     --> Delegate 0
                                     |
FF Input -> FF hidden layer output ----> Delegate 1
                                     |
                                     --> Delegate 2
                                     ...

That is, the output of the feed forward network's hidden layer is input for
all the delegates."""


"""Cascade delegate interface."""
class Delegate(object):
  def __init__(self):
    self.model = None
    self.lossfn = None
    self.actions = None

  """Sets the loss function for the delegate."""
  def set_loss(self, lossfn):
    self.lossfn = lossfn

  """Sets the implementation-specific model function for the delegate."""
  def set_model(self, model):
    self.model = model

  """Delegate interface."""

  """Prepares the delegate. Called before anything else.
  'cascade' is the cascade that this delegate is a part of, and 'actions'
  is the global action table for the parser."""
  def build(self, cascade, actions):
    pass

  """Translates 'action' into a series of delegate-specific actions and
  appends them to 'output'."""
  def translate(self, action, output):
    pass

  """Computes the delegate loss, where 'ff_activation' is the FF unit's
  activation, and 'gold' is the gold transition for the delegate."""
  def loss(self, state, ff_activation, gold):
    pass

  """Predicts and returns the delegate's output action based on the input
  FF unit's hidden layer, and the output action (if any) of the previous
  delegate."""
  def predict(self, state, previous_action, ff_activation):
    pass

  """Saves the delegate specification in the SLING frame 'frame'. This will
  eventually become a part of the flow file."""
  def as_frame(self, frame):
    pass


"""Delegate that uses a softmax layer to decide what action to output."""
class SoftmaxDelegate(Delegate):
  def __init__(self):
    super(SoftmaxDelegate, self).__init__()
    self.softmax_size = None

  """Returns the size of the softmax layer."""
  def size(self):
    return self.softmax_size

  """Returns the integer index of 'action' in the softmax layer."""
  def index(self, action):
    pass

  """Returns the action corresponding to 'index' in the softmax layer."""
  def action(index, previous_action=None):
    pass

  """Computes the delegate loss, where 'ff_activation' is the FF unit's
  activation, and 'gold' is the gold transition for the delegate."""
  def loss(self, state, ff_activation, gold):
    logits = self.model(ff_activation, train=True)
    gold_index = self.index(gold)
    assert gold_index >= 0, gold
    return self.lossfn(logits, gold_index)

  """Predicts and returns the delegate's output action based on the input
  FF unit's hidden layer, and the output action (if any) of the previous
  delegate."""
  def predict(self, state, previous_action, ff_activation):
    best_index = self.model(ff_activation, train=False)
    return self.action(best_index, previous_action)

  """Saves the delegate specification in the SLING frame 'frame'."""
  def as_frame(self, frame):
    """Specify the runtime implementation of this delegate."""
    frame["runtime"] = "SoftmaxDelegate"

    """Save the action table for this delegate in the frame."""
    actions = frame.store().array(self.size())
    for i in range(self.size()):
      action = self.action(i, previous_action=None)
      actions[i] = action.as_frame(frame.store())
    frame["actions"] = actions


"""A few Delegate implementations."""

"""FlatDelegate covers all actions in its softmax layer and doesn't delegate
anything."""
class FlatDelegate(SoftmaxDelegate):
  def build(self, cascade, actions):
    self.actions = actions
    self.softmax_size = actions.size()

  def translate(self, action, output):
    output.append(action)
      
  def index(self, action):
    return self.actions.index(action)
    
  def action(self, index, previous_action):
    return self.actions.action(index)


"""Delegate that decides whether to SHIFT or not. Not-SHIFT decisions are
delegated to the next delegate."""
class ShiftOrNotDelegate(SoftmaxDelegate):
  def build(self, cascade, actions):
    self.softmax_size = 2
    self.shift = Action(Action.SHIFT)
    self.not_shift = Action(Action.CASCADE)
    assert self is cascade.delegates[0]  # Should be the top delegate

    # Assume we will always cascade to the next delegate.
    assert cascade.size() > 1
    self.not_shift.delegate = 1

  def translate(self, action, output):
    if action.type == Action.SHIFT:
      output.append(self.shift)
    else:
      output.append(self.not_shift)
      
  def index(self, action):
    return 0 if action.type == Action.SHIFT else 1
    
  def action(self, index, previous_action):
    return self.shift if index == 0 else self.not_shift


"""Delegate that decides only among non-SHIFT actions."""
class NotShiftDelegate(SoftmaxDelegate):
  def build(self, cascade, actions):
    self.shift = actions.shift()
    self.actions = actions
    self.softmax_size = self.actions.size() - 1  # except SHIFT

  def translate(self, action, output):
    output.append(action)

  def index(self, action):
    i = self.actions.index(action)
    if i > self.shift: i -= 1
    return i

  def action(self, index, previous_action):
    if index >= self.shift: index += 1
    return self.actions.action(index)


"""Delegate that decides whether to SHIFT or MARK or neither. Non SHIFT/MARK
decisions are delegated to the next delegate."""
class ShiftMarkDelegate(SoftmaxDelegate):
  def build(self, cascade, actions):
    self.softmax_size = 3
    self.shift = Action(Action.SHIFT)
    self.mark = Action(Action.MARK)
    self.neither = Action(Action.CASCADE)
    assert self is cascade.delegates[0]  # Should be the top delegate

    # Assume we will always cascade to the next delegate.
    assert cascade.size() > 1
    self.neither.delegate = 1

  def translate(self, action, output):
    if action.type == Action.SHIFT:
      output.append(self.shift)
    elif action.type == Action.MARK:
      output.append(self.mark)
    else:
      output.append(self.neither)
      
  def index(self, action):
    if action.type == Action.SHIFT: return 0
    if action.type == Action.MARK: return 1
    return 2
    
  def action(self, index, previous_action):
    if index == 0: return self.shift
    if index == 1: return self.mark
    return self.neither


"""Delegate that decides only among non-SHIFT/MARK actions."""
class NotShiftOrMarkDelegate(SoftmaxDelegate):
  def build(self, cascade, actions):
    self.actions = actions
    mark = actions.mark()
    if mark is not None:
      self.first = min(actions.shift(), mark)
      self.second = max(actions.shift(), mark)
      self.softmax_size = self.actions.size() - 2  # except SHIFT, MARK
    else:
      self.first = actions.shift()
      self.second = None
      self.softmax_size = self.actions.size() - 1  # except SHIFT

  def translate(self, action, output):
    output.append(action)

  def index(self, action):
    i = self.actions.index(action)
    if i < self.first: return i
    if self.second is not None:
      if i < self.second: return i - 1
      return i - 2
    else:
      return i - 1

  def action(self, index, previous_action):
    if index >= self.first:
      index += 1
    if self.second is not None and index >= self.second - 1:
      index += 1
    return self.actions.action(index)


"""Returns whether 'action' EVOKEs a PropBank frame."""
def is_pbevoke(action):
  return action.type == Action.EVOKE and action.label.id.startswith("/pb/")


"""Delegate that decides among non-SHIFT/MARK and non-PropBank EVOKE actions.
PropBank EVOKE actions are delegated further."""
class ExceptPropbankEvokeDelegate(SoftmaxDelegate):
  def build(self, cascade, actions):
    """Build table of actions handled by the delegate."""
    self.table = Actions()
    for action in actions.table:
      if action.type != Action.SHIFT and \
        action.type != Action.MARK and not is_pbevoke(action):
        self.table.add(action)
    self.softmax_size = self.table.size() + 1  # +1 for CASCADE action
    self.pb_index = self.table.size()          # last action is CASCADE

    # Assume we will delegate PropBank EVOKES to PropbankEvokeDelegate.
    self.pb_action = Action(Action.CASCADE)
    self.pb_action.delegate = cascade.index_of("PropbankEvokeDelegate")

  def translate(self, action, output):
    if is_pbevoke(action):
      output.append(self.pb_action)
    else:
      output.append(action)

  def index(self, action):
    if action.is_cascade() and action.delegate == self.pb_action.delegate:
      return self.pb_index
    return self.table.index(action)

  def action(self, index, previous_action):
    if index == self.pb_index:
      return self.pb_action
    return self.table.action(index)


"""Handles only PropBank EVOKE actions."""
class PropbankEvokeDelegate(SoftmaxDelegate):
  def build(self, cascade, actions):
    self.table = Actions()
    for action in actions.table:
      if is_pbevoke(action): self.table.add(action)
    self.softmax_size = self.table.size()

  def translate(self, action, output):
    output.append(action)

  def index(self, action):
    return self.table.index(action)

  def action(self, index, previous_action):
    return self.table.action(index)


"""Cascade interface."""
class Cascade(object):
  def __init__(self, actions):
    self.actions = actions
    self.delegates = []

  """Adds a delegate to the cascade."""
  def add(self, delegate):
    self.delegates.append(delegate)

  """Initializes the cascade from a list of delegate classes."""
  def initialize(self, delegate_classes):
    for delegate in delegate_classes:
      self.add(delegate())
    for delegate in self.delegates:
      delegate.build(self, self.actions)

  """Returns the index of 'delegate'. 'delegate' could be a class, class name
  or an instance of Delegate."""
  def index_of(self, delegate):
    if inspect.isclass(delegate):
      for index, d in enumerate(self.delegates):
        if d.__class__ is delegate: return index
      raise ValueError("Can't find delegate %r" % delegate.__name__)
    elif type(delegate) is str:
      for index, d in enumerate(self.delegates):
        if d.__class__.__name__ == delegate: return index
      raise ValueError("Can't find delegate %r" % delegate)
    elif isinstance(delegate, Delegate):
      for index, d in enumerate(self.delegates):
        if d is delegate: return index
      raise ValueError("Can't find delegate %r" % delegate.__class__.__name__)
    raise ValueError("Can't handle delegate", delegate)

  """Returns the number of delegates in the cascade."""
  def size(self):
    return len(self.delegates)

  """Translates a sequence of actions into cascade-specific actions."""
  def translate(self, sequence):
    output = []
    for action in sequence:
      delegate_index = 0
      while True:
        """Get the current delegate's actions."""
        self.delegates[delegate_index].translate(action, output)

        """Move to the next delegate if needed, else stop."""
        if output[-1].is_cascade():
          delegate_index = output[-1].delegate
        else:
          break
    return output

  """Returns the loss for the specified delegate."""
  def loss(self, delegate_index, state, ff_activation, gold):
    return self.delegates[delegate_index].loss(state, ff_activation, gold)

  """Returns the predicted action for the specified delegate."""
  def predict(self, delegate_index, state, previous_action, ff_activation):
    return self.delegates[delegate_index].predict(
      state, previous_action, ff_activation)

  """Returns a string visualization of the cascade."""
  def __repr__(self):
    s = self.__class__.__name__ + "(" + str(len(self.delegates)) + " delegates)"
    for d in self.delegates:
      s += "\n  " + d.__class__.__name__
      if isinstance(d, SoftmaxDelegate):
        s += " (softmax size=" + str(d.size()) + ")"
    return s

  """Saves the cascade specificaton to a frame in 'store'."""
  def as_frame(self, store, delegate_cell_prefix):
    name = self.__module__ + "." + self.__class__.__name__
    frame = store.frame({"id": "/cascade", "name": name})
    delegates = store.array(self.size())
    for index, delegate in enumerate(self.delegates):
      d = store.frame({"name": delegate.__class__.__name__, "index": index})
      d["cell"] = delegate_cell_prefix + str(index)
      delegate.as_frame(d)
      delegates[index] = d
    frame["delegates"] = delegates
    return frame


"""Cascade implementations."""

"""FlatCascade only has one delegate which covers all actions."""
class FlatCascade(Cascade):
  def __init__(self, actions):
    super(FlatCascade, self).__init__(actions)
    self.initialize([FlatDelegate])

"""Cascade that decides on SHIFT vs all other actions."""
class ShiftCascade(Cascade):
  def __init__(self, actions):
    super(ShiftCascade, self).__init__(actions)
    self.initialize([ShiftOrNotDelegate, NotShiftDelegate])

"""Cascade that decides on SHIFT/MARK vs all other actions."""
class ShiftMarkCascade(Cascade):
  def __init__(self, actions):
    super(ShiftMarkCascade, self).__init__(actions)
    self.initialize([ShiftMarkDelegate, NotShiftOrMarkDelegate])

"""Cascade that decides via separate delegates:
a) Whether to SHIFT/MARK or not,
b) If not, whether to output a non-PropBank EVOKE action or not,
c) If not, which PropBank EVOKE action to output."""
class ShiftPropbankEvokeCascade(Cascade):
  def __init__(self, actions):
    super(ShiftPropbankEvokeCascade, self).__init__(actions)
    self.initialize(
      [ShiftMarkDelegate, ExceptPropbankEvokeDelegate, PropbankEvokeDelegate])


"""Prints softmax computation cost estimates of a bunch of cascades."""
def print_cost_estimates(commons_path, corpora_path):
  from corpora import Corpora

  train = Corpora(corpora_path, commons_path, gold=True)
  actions = Actions()
  for document in train:
    for action in document.gold:
      actions.add(action)

  train.rewind()

  cascades = [cascade_class(actions) for cascade_class in \
    [FlatCascade, ShiftCascade, ShiftMarkCascade, ShiftPropbankEvokeCascade]]
  costs = [0] * len(cascades)
  counts = [[0] * cascade.size() for cascade in cascades]
  for document in train:
    gold = document.gold
    for index, cascade in enumerate(cascades):
      cascade_gold_sequence = cascade.translate(gold)
      delegate = 0
      cost = 0
      for cascade_gold in cascade_gold_sequence:
        cost += cascade.delegates[delegate].size()
        counts[index][delegate] += 1
        if cascade_gold.is_cascade():
          delegate = cascade_gold.delegate
        else:
          delegate = 0
      costs[index] += cost
  for c, cost, cascade in zip(counts, costs, cascades):
    print("\n", cascade.__class__.__name__, "cost =", cost, "\n", \
      "Delegate invocations:", c, "\n", cascade)

if __name__ == '__main__':
  import sling.flags as flags
  flags.define('--commons',
               help='Commons store',
               default='',
               type=str)
  flags.define('--input',
               help='Input corpora',
               default='',
               type=str)
  flags.parse()
  print_cost_estimates(flags.arg.commons, flags.arg.input)

