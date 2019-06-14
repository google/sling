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


# Stores the full trace of running the SLING parser on a document.
# A trace is a list of steps, and a step is a cascade of one or more actions.
# A trace is saved as a slot in the document frame.
#
# This file can also be used to compare two recordios with traces:
#
# python sling/nlp/parser/trainer/trace.py \
#   --base=<base recordio file with tracing information> \
#   --expt=<expt recordio file with tracing information> \
#   --commons=<path to commons>
#   [--diff=/path/where/sample/diff/will/be/stored.txt]
#
# This will compare base and expt document pairs, and verify equality in the
# following order: lstm features, ff features, predicted & final actions.
# At the first disparity, it will throw a ValueError with a diagnostic message
# and print the two documents in text format to the file specified by '--diff'.


# Tracing information.
class Trace:
  # Represents a (predicted, final) parser action pair.
  class Action:
    def __init__(self, predicted, final, score=None):
      self.predicted = predicted
      self.final = final
      self.score = score

    # Returns the actions as frames.
    def as_frames(self, store):
      predicted = self.predicted.as_frame(store, slot_prefix='/trace/')
      predicted['/trace/_str'] = str(self.predicted)
      if self.score is not None:
        predicted['/trace/score'] = score
      final = predicted
      if not self.final is self.predicted:
        final = self.final.as_frame(store, slot_prefix='/trace/')
        final['/trace/_str'] = str(self.final)
      return (predicted, final)

  # Represents a cascade of actions for one decoder step.
  class Step:
    def __init__(self, state, ff_features):
      self.current = state.current
      self.ff_features = ff_features
      self.actions = []

    # Adds a (predicted, final) action pair to the step.
    def add_action(self, predicted, final, score=None):
      self.actions.append(Trace.Action(predicted, final, score))


  def __init__(self, spec, state, lstm_features):
    self.spec = spec
    self.lstm_features = lstm_features
    self.document = state.document
    self.begin = state.begin
    self.end = state.end

    # List of steps.
    self.steps = []

  # Adds a fresh decoder step with 'ff_features' as decoder features.
  def start_step(self, state, ff_features):
    self.steps.append(Trace.Step(state, ff_features))

  # Adds an action to the latest decoder step.
  def action(self, predicted, final, score=None):
    assert len(self.steps) > 0
    self.steps[-1].add_action(predicted, final, score)

  # Writes encoder features to 'trace'.
  def write_lstm_features(self, trace):
    assert len(self.lstm_features) == len(self.spec.lstm_features)
    store = self.document.store
    tokens = self.document.tokens
    frames = []
    for i in range(len(tokens)):
      frames.append(store.frame(\
        {"/trace/index": i, "/trace/token": tokens[i].word}))

    for f, vals in zip(self.spec.lstm_features, self.lstm_features):
      assert len(vals.indices) == len(tokens)
      for token, values in enumerate(vals.indices):
        if type(values) is int:
          values = [values]
        assert type(values) is list
        frames[token]["/trace/" + f.name] = values

    frames_array = store.array(len(frames))
    for i, frame in enumerate(frames):
      frames_array[i] = frame
    trace["/trace/lstm_features"] = frames_array

  # Writes step tracing information to 'trace'.
  def write_steps(self, trace):
    store = self.document.store
    steps = store.array(len(self.steps))
    trace["/trace/steps"] = steps
    for i, step in enumerate(self.steps):
      word = "<EOS>"
      if step.current < len(self.document.tokens):
        word = self.document.tokens[step.current].word

      # Decoder features.
      ff_features = []
      for f, indices in step.ff_features:
        # Convert 'None' link feature values to -1.
        indices = [-1 if index is None else index for index in indices]
        ff_features.append(\
          store.frame({"/trace/feature": f.name, "/trace/values": indices}))

      # Actions in the step.
      actions = store.array(len(step.actions))
      for idx, action in enumerate(step.actions):
        (predicted, final) = action.as_frames(store)
        actions[idx] = store.frame(\
          {"/trace/predicted": predicted, "/trace/final": final})

      frame = store.frame({
        "/trace/index": i,
        "/trace/current": step.current,
        "/trace/current_word": word,
        "/trace/ff_features" : ff_features,
        "/trace/actions" : actions
      })
      steps[i] = frame

  # Writes the trace to the underlying document.
  def write(self):
    trace = self.document.store.frame({"begin": self.begin, "end": self.end})
    self.write_lstm_features(trace)
    self.write_steps(trace)
    self.document.frame["trace"] = trace


if __name__ == "__main__":
  # Compares traces of two aligned recordio files.

  import os
  import sling
  import sling.flags as flags

  # Utility to check assertions and throw an error if the assertion is
  # violated.
  class Checker:
    # Initializes the checker with the document index, the base and expt
    # documents, and the filename where the first discrepancy is stored.
    def __init__(self, index, base_doc, expt_doc, diff_file=None):
      self.index = index
      self.base_doc = base_doc
      self.expt_doc = expt_doc
      self.diff_file = diff_file

      # Sanity check: the two documents should have the same tokens.
      if len(base_doc.tokens) != len(expt_doc.tokens):
        self.error('Differing number of tokens at document %d' % index)
      for i in range(len(base_doc.tokens)):
        self.check_eq(base_doc.tokens[i].word, expt_doc.tokens[i].word, \
          'token %d word' % i)
        self.check_eq(base_doc.tokens[i].brk, expt_doc.tokens[i].brk, \
          "token %d brk" % i)

    # Throws an error with 'message', and writes the document pair to the
    # pre-specified file.
    def error(self, message):
      if self.diff_file is not None:
        with open(self.diff_file, 'w') as f:
          f.write("Document Index:" + str(self.index) + "\n")
          f.write("Base document\n")
          f.write(self.base_doc.frame.data(pretty=True))
          f.write('\n\n')
          f.write("Expt document\n")
          f.write(self.expt_doc.frame.data(pretty=True))
          f.write('\n\n')
          f.write(message)
          print("One pair of differing docs written to", self.diff_file)
      raise ValueError(message)

    # Checks that lhs == rhs.
    def check_eq(self, lhs, rhs, message):
      if lhs != rhs:
        # Augment the message with the document index and the two values.
        message = ("Document %d " % self.index) + message + ": %s vs %s"
        self.error(message % (str(lhs), str(rhs)))

    # Checks that the two frames are equal (modulo slot re-ordering), and
    # ignoring roles in 'ignored_slots', which should be a list of names
    # of roles that should be ignored.
    def frame_eq(self, lhs, rhs, message, ignored_slots=None):
      if ignored_slots is None: ignored_slots = []

      lhs_set = set()
      for key, value in lhs:
        if key.id not in ignored_slots:
          lhs_set.add((key.id, value))
      rhs_set = set()
      for key, value in rhs:
        if key.id not in ignored_slots:
          rhs_set.add((key.id, value))

      diff = lhs_set.symmetric_difference(rhs_set)
      if len(diff) > 0:
        # Augment the message and report error.
        message = ("Document %d " % self.index) + message
        message += ", %s vs %s" % (lhs.data(), rhs.data())
        message += ", symmetric difference = " + str(diff)
        self.error(message)

  # Compares two recordios for equality of tracing, stopping at the first error.
  def compare(arg):
    base_reader = sling.RecordReader(arg.base)
    expt_reader = sling.RecordReader(arg.expt)

    commons = sling.Store()
    commons.load(arg.commons)
    schema = sling.DocumentSchema(commons)
    commons.freeze()

    store = sling.Store(commons)
    index = -1
    for (_, base_val), (_, expt_val) in zip(base_reader, expt_reader):
      index += 1
      base_doc = sling.Document(frame=store.parse(base_val), schema=schema)
      expt_doc = sling.Document(frame=store.parse(expt_val), schema=schema)

      # Basic checks.
      base = base_doc.frame["trace"]
      expt = expt_doc.frame["trace"]
      if base is None and expt_doc is not None:
        checker.error('No trace in base document at index %d' % index)
      elif base is not None and expt_doc is None:
        checker.error('No trace in expt document at index %d' % index)
      if base is None:
        continue

      # Traces should be over the same token range.
      checker = Checker(index, base_doc, expt_doc, arg.diff)
      checker.check_eq(base["begin"], expt["begin"], "Trace Begin")
      checker.check_eq(base["end"], expt["end"], "Trace End")

      # Check LSTM features.
      base_lstm = base["/trace/lstm_features"]
      expt_lstm = expt["/trace/lstm_features"]
      checker.check_eq(len(base_lstm), len(expt_lstm), "LSTM Features Length")
      for i in range(len(base_lstm)):
        checker.frame_eq(base_lstm[i], expt_lstm[i], \
          "LSTM features for token %d (%s)" % (i, base_doc.tokens[i].word))

      if arg.skip_steps:
        continue

      # Check steps.
      base_steps = base["/trace/steps"]
      expt_steps = expt["/trace/steps"]
      min_steps = min(len(base_steps), len(expt_steps))
      for i in range(min_steps):
        message = "Step %d's current token index" % i
        checker.check_eq(base_steps[i]["/trace/current"], \
          expt_steps[i]["/trace/current"], message)

        # Check FF features for the step.
        base_ff = base_steps[i]["/trace/ff_features"]
        expt_ff = expt_steps[i]["/trace/ff_features"]
        checker.check_eq(len(base_ff), len(expt_ff), \
          "# of FF features for step %d" % i)

        base_dict = {f["/trace/feature"] : f["/trace/values"] for f in base_ff}
        expt_dict = {f["/trace/feature"] : f["/trace/values"] for f in expt_ff}
        for k, v in base_dict.items():
          checker.check_eq(k in expt_dict, True, \
            "Step %d: FF feature %s not in expt" % (i, k))
          checker.check_eq(v, expt_dict[k], \
            "Step %d: FF feature %s has a different value in expt" % (i, k))
        for k, v in expt_dict.items():
          checker.check_eq(k in base_dict, True, \
            "Step %d: FF feature %s not in base" % (i, k))

        # Check action(s) in the step.
        base_actions = base_steps[i]["/trace/actions"]
        expt_actions = expt_steps[i]["/trace/actions"]
        for idx in range(min(len(base_actions), len(expt_actions))):
          checker.frame_eq(base_actions[idx]["/trace/predicted"], \
            expt_actions[idx]["/trace/predicted"],
            "Step %d, predicted action %d" % (i, idx),
            ["/trace/_str"])
          checker.frame_eq(base_actions[idx]["/trace/final"], \
            expt_actions[idx]["/trace/final"],
            "Step %d, final action %d" % (i, idx),
            ["/trace/_str"])

        # There should be the same number of actions in the step.
        checker.check_eq(len(base_actions), len(expt_actions), \
          "Step %d: # of actions" % i)

      # There should be the same number of steps.
      checker.check_eq(len(base_steps), len(expt_steps), "# of Steps")

    base_reader.close()
    expt_reader.close()


  flags.define('--base',
               help='Base recordio',
               default="",
               type=str,
               metavar='FILE')
  flags.define('--expt',
               help='Expt recordio',
               default="",
               type=str,
               metavar='FILE')
  flags.define('--commons',
               help='Commons',
               default="",
               type=str,
               metavar='FILE')
  flags.define('--diff',
               help='File where sample diff (if any) will be written',
               default="/tmp/diff.txt",
               type=str,
               metavar='FILE')
  flags.define('--skip_steps',
               help='Whether to skip diffing steps (only comparing features)',
               default=False,
               action='store_true')
  flags.parse()
  assert os.path.exists(flags.arg.base)
  assert os.path.exists(flags.arg.expt)
  assert os.path.exists(flags.arg.commons)

  compare(flags.arg)
