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

import cascade
import os
import pickle
import sling
import struct
import tempfile
import unicodedata

from action_table import Actions
from corpora import Corpora
from lexicon import Lexicon
from parser_state import ParserState


# Stores raw feature indices.
class Feature:
  def __init__(self):
    self.indices = []       # list of lists of indices
    self.has_empty = False  # whether some of the lists in 'indices' are empty
    self.has_multi = False  # whether some of the lists have >1 values


  # Adds 'index' to 'indices'. 'index' could be an int or a list of ints.
  def add(self, index):
    if type(index) is int:
      self.indices.append(index)
    else:
      assert type(index) is list
      if len(index) == 1:
        self.indices.append(index[0])
      else:
        self.indices.append(index)
        if len(index) == 0: self.has_empty = True
        if len(index) > 1: self.has_multi = True


  def __repr__(self):
    return str(self.indices)


# Specification for a single link or fixed feature.
class FeatureSpec:
  def __init__(self, name, dim, vocab=None, activation=None, num=1):
    self.name = name
    self.dim = dim                     # embedding dimensionality
    self.vocab_size = vocab            # vocabulary size (fixed features only)
    self.activation_size = activation  # activation size (link features only)
    self.num = num                     # no. of links / fixed feature ids


# Training specification.
class Spec:
  # Fallback feature values.
  NO_HYPHEN = 0
  HAS_HYPHEN = 1
  HYPHEN_CARDINALITY = 2

  LOWERCASE = 0
  UPPERCASE = 1
  CAPITALIZED = 2
  INITIAL = 3
  NON_ALPHABETIC = 4
  CAPITALIZATION_CARDINALITY = 5

  NO_PUNCTUATION = 0
  SOME_PUNCTUATION = 1
  ALL_PUNCTUATION = 2
  PUNCTUATION_CARDINALITY = 3

  NO_QUOTE = 0
  OPEN_QUOTE = 1
  CLOSE_QUOTE = 2
  UNKNOWN_QUOTE = 3
  QUOTE_CARDINALITY = 4

  NO_DIGIT = 0
  SOME_DIGIT = 1
  ALL_DIGIT = 2
  DIGIT_CARDINALITY = 3

  def __init__(self):
    # Lexicon generation settings.
    self.words_normalize_digits = True
    self.suffixes_max_length = 3

    # Action table percentile.
    self.actions_percentile = 99

    # Network dimensionalities.
    self.lstm_hidden_dim = 256
    self.ff_hidden_dim = 128

    # Fixed feature dimensionalities.
    self.affix_features = True
    self.shape_features = True
    self.words_dim = 32
    self.suffixes_dim = 16
    self.fallback_dim = 8  # dimensionality of each fallback feature
    self.roles_dim = 16

    # History feature size.
    self.history_limit = 5

    # Frame limit for other link features.
    self.frame_limit = 5

    # Link feature dimensionalities.
    self.link_dim_lstm = 32
    self.link_dim_ff = 64

    # Bins for distance from the current token to the topmost marked token.
    self.distance_bins = [0, 1, 2, 3, 6, 10, 15, 20]

    # Resources.
    self.commons = None
    self.actions = None
    self.words = None
    self.suffix = None

    # To be determined.
    self.num_actions = None
    self.lstm_features = []
    self.ff_fixed_features = []
    self.ff_link_features = []
    self.cascade = None


  # Builds an action table from 'corpora'.
  def _build_action_table(self, corpora):
    corpora.rewind()
    corpora.set_gold(True)
    self.actions = Actions()
    self.actions.frame_limit = self.frame_limit
    for document in corpora:
      assert document.size() == 0 or len(document.gold) > 0
      for action in document.gold:
        self.actions.add(action)

    self.actions.prune(self.actions_percentile)

    # Save the actions table in commons.
    actions_frame = self.actions.encoded(self.commons)

    # Re-read the actions table from the commons, so all frames come
    # from the commons store.
    self.actions = Actions()
    self.actions.decode(actions_frame)

    self.num_actions = self.actions.size()
    print(self.num_actions, "gold actions")
    allowed = self.num_actions - sum(self.actions.disallowed)
    print("num allowed actions:", allowed)
    print(len(self.actions.roles), "unique roles in action table")


  # Writes parts of the spec to a flow blob.
  def to_flow(self, fl):
    blob = fl.blob("spec")

    # Separately write some fields for the convenience of the Myelin runtime.
    blob.add_attr("frame_limit", self.frame_limit)
    bins = [str(d) for d in self.distance_bins]
    blob.add_attr("mark_distance_bins", ' '.join(bins))

    # Temporarily remove fields that can't or don't need to be pickled.
    fields_to_ignore = [ 'commons', 'actions', 'words', 'suffix', 'cascade']
    cache = {}
    for k, v in self.__dict__.items():
      if k in fields_to_ignore:
        cache[k] = v

    for k in fields_to_ignore:
      delattr(self, k)
    blob.data = pickle.dumps(self.__dict__)

    # Resurrect deleted fields.
    for k, v in cache.items():
      setattr(self, k, v)


  # Reads spec from a flow.
  def from_flow(self, fl):
    blob = fl.blob("spec")
    temp_dict = pickle.loads(blob.data)
    self.__dict__.update(temp_dict)

    # Read non-pickled fields.
    # Read common store.
    self.commons = sling.Store()
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    filename = temp_file.name
    with open(filename, "wb") as f:
      f.write(fl.blob("commons").data)
    temp_file.close()
    self.commons.load(filename)
    _ = sling.DocumentSchema(self.commons)
    self.commons.freeze()
    os.unlink(filename)

    # Read action table from the commons.
    self.actions = Actions()
    self.actions.decode(self.commons["/table"])

    # Read cascade specification. This is done by calling eval()
    # on the class constructor. The classname is stored in the cascade frame.
    frame = self.commons["/cascade"]
    self.cascade = eval(frame["name"])(self.actions)
    print(self.cascade)

    # Read word lexicon.
    blob = fl.blob("lexicon")
    self.words = Lexicon(self.words_normalize_digits)

    # Delimiter is stored as the string form of the character.
    # e.g. \n is stored as the string "10".
    delimiter_str = chr(int(blob.get_attr("delimiter")))
    vocab_str = blob.data.tobytes().decode()
    self.words.read(vocab_str, delimiter_str)
    print(self.words.size(), "words read from flow's lexicon")

    # Read suffix table.
    self.suffix = Lexicon(self.words_normalize_digits, oov_item=None)
    data = fl.blob("suffixes").data
    def read_int(mview):
      output = 0
      shift_bits = 0
      index = 0
      while index < len(mview):
        part = mview[index]
        index += 1
        output |= (part & 127) << shift_bits
        shift_bits += 7
        if part & 128 == 0:
          break
      return output, mview[index:]

    affix_type, data = read_int(data)                # affix type
    assert affix_type == 1

    max_length, data = read_int(data)                # max length
    assert max_length == self.suffixes_max_length

    num, data = read_int(data)                       # num affixes
    for _ in range(num):
      num_bytes, data = read_int(data)
      word = data[0:num_bytes].tobytes().decode()
      self.suffix.add(word)
      data = data[num_bytes:]
      num_chars, data = read_int(data)
      if num_chars > 0:
        shorter_index, data = read_int(data)
    print(self.suffix.size(), "suffixes read from flow's affix table")


  # Returns suffix(es) of 'word'.
  def get_suffixes(self, word):
    assert type(word) is str
    output = []
    end = min(self.suffixes_max_length, len(word))
    for start in range(end, 0, -1):
      output.append(word[-start:])
    output.append("")  # empty suffix

    return output


  # Dumps suffixes in the AffixTable format (cf. sling/nlp/document/affix.cc).
  def write_suffix_table(self, buf=None):
    if buf is None: buf = bytearray()

    # Writes 'num' in varint encoding to 'b'.
    def writeint(num, b):
      while True:
        part = num & 127
        num = num >> 7
        if num > 0:
          b.append(part | 128)
        else:
          b.append(part)
          break


    writeint(1, buf)  # 1 = AffixTable::SUFFIX
    writeint(self.suffixes_max_length, buf)
    writeint(self.suffix.size(), buf)
    for i in range(self.suffix.size()):
      v = self.suffix.value(i)

      assert type(v) is str, type(v)
      v_bytes = v.encode('utf-8')

      writeint(len(v_bytes), buf)      # number of bytes
      for x in v_bytes: buf.append(x)  # the bytes themselves
      writeint(len(v), buf)            # number of code points
      if len(v) > 0:
        shorter = v[1:]
        shorter_idx = self.suffix.index(shorter)
        assert shorter_idx is not None, (shorter, v, v_bytes)
        writeint(shorter_idx, buf)     # id of the shorter suffix
    return bytes(buf)


  # Adds LSTM feature to the specification.
  def add_lstm_fixed(self, name, dim, vocab, num=1):
    self.lstm_features.append(
        FeatureSpec(name, dim=dim, vocab=vocab, num=num))

  # Adds fixed feature to the specification.
  def add_ff_fixed(self, name, dim, vocab, num):
    self.ff_fixed_features.append(
        FeatureSpec(name, dim=dim, vocab=vocab, num=num))


  # Adds recurrent link feature from the FF unit to the specification.
  def add_ff_link(self, name, num):
    self.ff_link_features.append(
        FeatureSpec(name, dim=self.link_dim_ff, \
        activation=self.ff_hidden_dim, num=num))


  # Adds link feature from the LSTMs to the FF unit to the specification.
  def add_lstm_link(self, name, num):
    self.ff_link_features.append(
        FeatureSpec(name, dim=self.link_dim_lstm, \
        activation=self.lstm_hidden_dim, num=num))


  # Specifies all fixed and link features.
  def _specify_features(self):
    # LSTM features.
    self.add_lstm_fixed("word", self.words_dim, self.words.size())
    if self.affix_features:
      self.add_lstm_fixed(
          "suffix", self.suffixes_dim, self.suffix.size(), \
          self.suffixes_max_length + 1)  # +1 to account for the empty affix
    if self.shape_features:
      self.add_lstm_fixed(
          "capitalization", self.fallback_dim, Spec.CAPITALIZATION_CARDINALITY)
      self.add_lstm_fixed("hyphen", self.fallback_dim, Spec.HYPHEN_CARDINALITY)
      self.add_lstm_fixed(
          "punctuation", self.fallback_dim, Spec.PUNCTUATION_CARDINALITY)
      self.add_lstm_fixed("quote", self.fallback_dim, Spec.QUOTE_CARDINALITY)
      self.add_lstm_fixed("digit", self.fallback_dim, Spec.DIGIT_CARDINALITY)

    self.lstm_input_dim = sum([f.dim for f in self.lstm_features])
    print("LSTM input dim", self.lstm_input_dim)
    assert self.lstm_input_dim > 0

    # Feed forward features.
    num_roles = len(self.actions.roles)
    fl = self.frame_limit
    if num_roles > 0:
      num = 32
      dim = self.roles_dim
      self.add_ff_fixed("in-roles", dim, num_roles * fl, num)
      self.add_ff_fixed("out-roles", dim, num_roles * fl, num)
      self.add_ff_fixed("labeled-roles", dim, num_roles * fl * fl, num)
      self.add_ff_fixed("unlabeled-roles", dim, fl * fl, num)

    # Distance to the top of the mark stack.
    self.add_ff_fixed("mark-distance", 32, len(self.distance_bins) + 1, 1)

    # Link features.
    self.add_ff_link("frame-creation-steps", fl)
    self.add_ff_link("frame-focus-steps", fl)
    self.add_lstm_link("frame-end-lr", fl)
    self.add_lstm_link("frame-end-rl", fl)
    self.add_ff_link("history", self.history_limit)
    self.add_lstm_link("lr", 1)
    self.add_lstm_link("rl", 1)

    # Link features that look at the stack of marked tokens.
    mark_depth = 1   # 1 = use only the top of the stack
    self.add_lstm_link("mark-lr", mark_depth)
    self.add_lstm_link("mark-rl", mark_depth)
    self.add_ff_link("mark-step", mark_depth)

    self.ff_input_dim = sum([f.dim for f in self.ff_fixed_features])
    self.ff_input_dim += sum(
        [f.dim * f.num for f in self.ff_link_features])
    print("FF_input_dim", self.ff_input_dim)
    assert self.ff_input_dim > 0


  # Builds the spec using the specified corpora.
  def build(self, commons_path, corpora_path):
    # Prepare lexical dictionaries.
    self.words = Lexicon(self.words_normalize_digits)
    self.suffix = Lexicon(self.words_normalize_digits, oov_item=None)

    # Initialize training corpus.
    corpora = Corpora(corpora_path, commons_path)

    # Collect word and affix lexicons.
    for document in corpora:
      for token in document.tokens:
        word = token.word
        self.words.add(word)
        for s in self.get_suffixes(word):
          self.suffix.add(s)
    print("Words:", self.words.size(), "items in lexicon, including OOV")
    print("Suffix:", self.suffix.size(), "items in lexicon")

    # Load common store, but not freeze it yet. We will add the action table
    # and cascade specification to it.
    self.commons = sling.Store()
    self.commons.load(commons_path)
    schema = sling.DocumentSchema(self.commons)

    # Prepare action table and cascade.
    self._build_action_table(corpora)
    self.cascade = cascade.ShiftMarkCascade(self.actions)
    print(self.cascade)

    # Save cascade specification in commons.
    _ = self.cascade.as_frame(self.commons, delegate_cell_prefix="delegate")

    # Freeze the common store.
    self.commons.freeze()

    # Add feature specs.
    self._specify_features()


  # Loads embeddings for words in the lexicon.
  def load_word_embeddings(self, embeddings_file):
    word_embeddings = [None] * self.words.size()
    f = open(embeddings_file, 'rb')

    # Read header.
    header = f.readline().strip()
    size = int(header.split()[0])
    dim = int(header.split()[1])
    assert dim == self.words_dim, "%r vs %r" % (dim, self.words_dim)

    # Read vectors for known words.
    count = 0
    fmt = "f" * dim
    vector_size = 4 * dim  # 4 being sizeof(float)
    oov = self.words.oov_index
    for _ in range(size):
      word = bytearray()
      while True:
        ch = f.read(1)
        if ch[0] == 32: break
        word.append(ch[0])

      vector = list(struct.unpack(fmt, f.read(vector_size)))
      ch = f.read(1)
      assert ch[0] == 10, "%r" % ch     # end of line expected

      word = word.decode()
      index = self.words.index(word)
      if index != oov and word_embeddings[index] is None:
        word_embeddings[index] = vector
        count += 1

    f.close()
    word_embedding_indices =\
        [i for i, v in enumerate(word_embeddings) if v is not None]
    word_embeddings = [v for v in word_embeddings if v is not None]

    print("Loaded", count, "pre-trained embeddings from file with", size, \
        "vectors. Vectors for remaining", (self.words.size() - count), \
        "words will be randomly initialized.")
    return word_embeddings, word_embedding_indices


  # Returns raw indices of LSTM features for all tokens in 'document'.
  def raw_lstm_features(self, document):
    output = []
    categories = []
    for token in document.tokens:
      categories.append([unicodedata.category(ch) for ch in token.word])

    for f in self.lstm_features:
      features = Feature()
      output.append(features)
      if f.name == "word":
        for token in document.tokens:
          features.add(self.words.index(token.word))
      elif f.name == "suffix":
        for token in document.tokens:
          suffixes = self.get_suffixes(token.word)
          ids = [self.suffix.index(s) for s in suffixes]
          ids = [i for i in ids if i is not None]  # ignore unknown suffixes
          features.add(ids)
      elif f.name == "hyphen":
        for index, token in enumerate(document.tokens):
          hyphen = any(c == 'Pd' for c in categories[index])
          features.add(Spec.HAS_HYPHEN if hyphen else Spec.NO_HYPHEN)
      elif f.name == "capitalization":
        for index, token in enumerate(document.tokens):
          has_upper = any(c == 'Lu' for c in categories[index])
          has_lower = any(c == 'Ll' for c in categories[index])

          value = Spec.CAPITALIZED
          if not has_upper and has_lower:
            value = Spec.LOWERCASE
          elif has_upper and not has_lower:
            value = Spec.UPPERCASE
          elif not has_upper and not has_lower:
            value = Spec.NON_ALPHABETIC
          elif index == 0 or token.brk >= 3:  # 3 = SENTENCE_BREAK
            value = Spec.INITIAL
          features.add(value)
      elif f.name == "punctuation":
        for index in range(len(document.tokens)):
          all_punct = all(c[0] == 'P' for c in categories[index])
          some_punct = any(c[0] == 'P' for c in categories[index])

          if all_punct:
            features.add(Spec.ALL_PUNCTUATION)
          elif some_punct:
            features.add(Spec.SOME_PUNCTUATION)
          else:
            features.add(Spec.NO_PUNCTUATION)

      elif f.name == "digit":
        for index in range(len(document.tokens)):
          all_digit = all(c == 'Nd' for c in categories[index])
          some_digit = any(c == 'Nd' for c in categories[index])

          if all_digit:
            features.add(Spec.ALL_DIGIT)
          elif some_digit:
            features.add(Spec.SOME_DIGIT)
          else:
            features.add(Spec.NO_DIGIT)

      elif f.name == "quote":
        in_quote = False
        for index, token in enumerate(document.tokens):
          value = Spec.NO_QUOTE
          for cat, ch in zip(categories[index], token.word):
            if cat == 'Pi':
              value = Spec.OPEN_QUOTE
            elif cat == 'Pf':
              value = Spec.CLOSE_QUOTE
            elif cat == 'Po' and (ch == '\'' or ch == '"'):
              value = Spec.UNKNOWN_QUOTE
            elif cat == 'Sk' and ch == '`':
              value = Spec.UNKNOWN_QUOTE
          if value != Spec.NO_QUOTE:
            token = document.tokens[index]
            if token.word == "``":
              value = Spec.OPEN_QUOTE
            elif token.word == "''":
              value = Spec.CLOSE_QUOTE
            if value == Spec.UNKNOWN_QUOTE:
              value = Spec.CLOSE_QUOTE if in_quote else Spec.OPEN_QUOTE
              in_quote = not in_quote
          features.add(value)
      else:
        raise ValueError("LSTM feature '", f.name, "' not implemented")
    return output


  # Returns the index of the bin corresponding to the distance of the topmost
  # marked token from the current token.
  def _mark_distance(self, t1, t2):
    d = t2 - t1
    for i, x in enumerate(self.distance_bins):
      if d <= x: return i
    return len(self.distance_bins)


  # Returns raw indices of all fixed FF features for 'state'.
  def raw_ff_fixed_features(self, feature_spec, state):
    role_graph = state.role_graph()
    num_roles = len(self.actions.roles)
    fl = self.frame_limit
    raw_features = []
    if feature_spec.name == "in-roles":
      for e in role_graph:
        if e[2] is not None and e[2] < fl and e[2] >= 0:
          raw_features.append(e[2] * num_roles + e[1])
    elif feature_spec.name == "out-roles":
      for e in role_graph:
        raw_features.append(e[0] * num_roles + e[1])
    elif feature_spec.name == "unlabeled-roles":
      for e in role_graph:
        if e[2] is not None and e[2] < fl and e[2] >= 0:
          raw_features.append(e[2] * fl + e[0])
    elif feature_spec.name == "labeled-roles":
      for e in role_graph:
        if e[2] is not None and e[2] < fl and e[2] >= 0:
          raw_features.append(e[0] * fl * num_roles + e[2] * num_roles + e[1])
    elif feature_spec.name == "mark-distance":
      if len(state.marks) > 0:
        d = self._mark_distance(state.marks[-1].token, state.current)
        raw_features.append(d)
    else:
      raise ValueError("FF feature '", feature_spec.name, "' not implemented")

    return raw_features


  # Returns link features for the FF unit.
  def translated_ff_link_features(self, feature_spec, state):
    name = feature_spec.name
    num = feature_spec.num

    output = []
    if name == "history":
      for i in range(num):
        output.append(None if i >= state.steps else state.steps - i - 1)
    elif name in ["lr", "rl"]:
      index = None
      if state.current < state.end:
        index = state.current - state.begin
      output.append(index)
    elif name in ["frame-end-lr", "frame-end-rl"]:
      for i in range(num):
        index = None
        end = state.frame_end_inclusive(i)
        if end != -1:
          index = end - state.begin
        output.append(index)
    elif name == "frame-creation-steps":
      for i in range(num):
        step = state.creation_step(i)
        output.append(None if step == -1 else step)
    elif name == "frame-focus-steps":
      for i in range(num):
        step = state.focus_step(i)
        output.append(None if step == -1 else step)
    elif name in ["mark-lr", "mark-rl"]:
      for i in range(num):
        index = None
        if len(state.marks) > i:
          index = state.marks[-1 - i].token - state.begin
        output.append(index)
    elif name == "mark-step":
      for i in range(num):
        index = None
        if len(state.marks) > i:
          index = state.marks[-1 - i].step
        output.append(index)
    else:
      raise ValueError("Link feature not implemented:" + name)

    return output


  # Debugging methods.

  # Returns feature strings for FF feature indices provided in 'indices'.
  # All indices are assumed to belong to a single feature whose spec is in
  # 'feature_spec'.
  def ff_fixed_feature_strings(self, feature_spec, indices):
    limit = self.frame_limit
    roles = self.actions.roles
    nr = len(roles)

    strings = []
    if feature_spec.name == "out-roles":
      strings = [str(i / nr) + "->" + str(roles[i % nr]) for i in indices]
    elif feature_spec.name == "in-roles":
      strings = [str(roles[i % nr]) + "->" + str(i / nr) for i in indices]
    elif feature_spec.name == "unlabeled-roles":
      strings = [str(i / limit) + "->" + str(i % limit) for i in indices]
    elif feature_spec.name == "labeled-roles":
      t = limit * nr
      for i in indices:
        value = str(i / t) + "->" + str(roles[(i % t) % nr])
        value += "->" + str((i % t) / nr)
        strings.append(value)
    else:
      raise ValueError(feature_spec.name + " not implemented")
    return str(strings)


  # Traces the seqeuence of gold actions in 'document'.
  def oracle_trace(self, document):
    assert len(document.gold) > 0, "No gold actions"
    state = ParserState(document, self)
    for gold in document.gold:
      print("Taking gold action", gold)
      print("On state:", state)

      gold_index = self.actions.indices.get(gold, None)
      assert gold_index is not None, "Unknown gold action: %r" % gold
      assert state.is_allowed(gold_index), "Disallowed gold action: %r" % gold
      state.advance(gold)

    print("Final state after", len(document.gold), "actions:", state)
