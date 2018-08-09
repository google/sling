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
import sling
import struct
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
    self.num = num                     # no. of links / no. of fixed feature ids


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

  def __init__(self, small=False):
    self.small = small

    # Lexicon generation settings.
    self.words_normalize_digits = True
    self.suffixes_max_length = 3

    # Action table percentile.
    self.actions_percentile = 99

    if small:
      # Network dimensionalities.
      self.lstm_hidden_dim = 6
      self.ff_hidden_dim = 12

      # Fixed feature dimensionalities.
      self.oov_features = False
      self.words_dim = 4
      self.suffixes_dim = 2
      self.fallback_dim = 2  # dimensionality of each fallback feature
      self.roles_dim = 2

      # History feature size.
      self.history_limit = 2

      # Frame limit for other link features.
      self.frame_limit = 2

      # Link feature dimensionalities.
      self.link_dim_lstm = 8
      self.link_dim_non_lstm = 10
    else:
      # Network dimensionalities.
      self.lstm_hidden_dim = 256
      self.ff_hidden_dim = 128

      # Fixed feature dimensionalities.
      self.oov_features = True
      self.words_dim = 32
      self.suffixes_dim = 16
      self.fallback_dim = 8  # dimensionality of each fallback feature
      self.roles_dim = 16

      # History feature size.
      self.history_limit = 4

      # Frame limit for other link features.
      self.frame_limit = 5

      # Link feature dimensionalities.
      self.link_dim_lstm = 32
      self.link_dim_non_lstm = 64

    # Resources.
    self.commons = None
    self.commons_path = None
    self.actions = None
    self.words = None
    self.suffix = None
    self.word_embeddings = None
    self.word_embedding_indices = None

    # To be determined.
    self.num_actions = None
    self.lstm_features = []
    self.ff_fixed_features = []
    self.ff_link_features = []
    self.cascade = None


  # Builds an action table from 'corpora'.
  def _build_action_table(self, corpora):
    corpora.rewind()
    self.actions = Actions()
    self.actions.frame_limit = self.frame_limit
    for document in corpora:
      assert document.size() == 0 or len(document.gold) > 0
      for action in document.gold:
        self.actions.add(action)

    self.actions.prune(self.actions_percentile)
    self.num_actions = self.actions.size()
    print self.num_actions, "gold actions"

    allowed = self.num_actions - sum(self.actions.disallowed)
    print "num allowed actions:", allowed
    print len(self.actions.roles), "unique roles in action table"


  # Returns suffix(es) of 'word'.
  def get_suffixes(self, word, unicode_chars=None):
    if unicode_chars is None:
      unicode_chars = list(word.decode("utf-8"))
    output = []
    end = min(self.suffixes_max_length, len(unicode_chars))
    for start in xrange(end, 0, -1):
      output.append("".join(unicode_chars[-start:]))
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
    for i in xrange(self.suffix.size()):
      v = self.suffix.value(i)

      if type(v) is unicode:
        v_str = v.encode("utf-8")
      else:
        assert type(v) is str, type(v)
        v_str = v

      writeint(len(v_str), buf)       # number of bytes
      for x in v_str: buf.append(x)   # the bytes themselves
      writeint(len(v), buf)           # number of characters
      if len(v) > 0:
        shorter = v[1:]
        shorter_idx = self.suffix.index(shorter)
        assert shorter_idx is not None, shorter
        writeint(shorter_idx, buf)    # id of the shorter suffix

    return buf


  # Adds LSTM feature to the specification.
  def add_lstm_fixed(self, name, dim, vocab, num=1):
    self.lstm_features.append(
        FeatureSpec(name, dim=dim, vocab=vocab, num=num))

  # Adds fixed feature to the specification.
  def add_ff_fixed(self, name, dim, vocab, num):
    self.ff_fixed_features.append(
        FeatureSpec(name, dim=dim, vocab=vocab, num=num))


  # Adds link feature to the specification.
  def add_ff_link(self, name, dim, activation, num):
    self.ff_link_features.append(
        FeatureSpec(name, dim=dim, activation=activation, num=num))


  # Specifies all fixed and link features.
  def _specify_features(self):
    # LSTM features.
    self.add_lstm_fixed("word", self.words_dim, self.words.size())
    if self.oov_features:
      self.add_lstm_fixed(
          "suffix", self.suffixes_dim, self.suffix.size(), \
          self.suffixes_max_length + 1)  # +1 to account for the empty affix
      self.add_lstm_fixed(
          "capitalization", self.fallback_dim, Spec.CAPITALIZATION_CARDINALITY)
      self.add_lstm_fixed("hyphen", self.fallback_dim, Spec.HYPHEN_CARDINALITY)
      self.add_lstm_fixed(
          "punctuation", self.fallback_dim, Spec.PUNCTUATION_CARDINALITY)
      self.add_lstm_fixed("quote", self.fallback_dim, Spec.QUOTE_CARDINALITY)
      self.add_lstm_fixed("digit", self.fallback_dim, Spec.DIGIT_CARDINALITY)

    self.lstm_input_dim = sum([f.dim for f in self.lstm_features])
    print "LSTM input dim", self.lstm_input_dim
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

    self.add_ff_link("frame-creation-steps", self.link_dim_non_lstm, self.ff_hidden_dim, fl)
    self.add_ff_link("frame-focus-steps", self.link_dim_non_lstm, self.ff_hidden_dim, fl)
    self.add_ff_link("frame-end-lr", self.link_dim_lstm, self.lstm_hidden_dim, fl)
    self.add_ff_link("frame-end-rl", self.link_dim_lstm, self.lstm_hidden_dim, fl)
    self.add_ff_link("history", self.link_dim_non_lstm, self.ff_hidden_dim, self.history_limit)
    self.add_ff_link("lr", self.link_dim_lstm, self.lstm_hidden_dim, 1)
    self.add_ff_link("rl", self.link_dim_lstm, self.lstm_hidden_dim, 1)

    self.ff_input_dim = sum([f.dim for f in self.ff_fixed_features])
    self.ff_input_dim += sum(
        [f.dim * f.num for f in self.ff_link_features])
    print "FF_input_dim", self.ff_input_dim
    assert self.ff_input_dim > 0


  # Builds the spec using 'corpora'.
  def build(self, commons, corpora):
    if type(commons) is str:
      self.commons_path = commons
      commons = sling.Store()
      commons.load(self.commons_path)
      commons.freeze()

    # Prepare lexical dictionaries.
    # For compatibility with DRAGNN, suffixes don't have an OOV item.
    self.commons = commons
    self.words = Lexicon(self.words_normalize_digits)
    self.suffix = Lexicon(self.words_normalize_digits, oov_item=None)

    corpora.rewind()
    corpora.set_gold(False)   # No need to compute gold transitions yet
    for document in corpora:
      for token in document.tokens:
        word = token.text
        self.words.add(word)
        for s in self.get_suffixes(word):
          self.suffix.add(s)
    print "Words:", self.words.size(), "items in lexicon, including OOV"
    print "Suffix:", self.suffix.size(), "items in lexicon"

    # Prepare action table.
    corpora.set_gold(True)
    self._build_action_table(corpora)

    # Add feature specs.
    self._specify_features()

    # Build cascade.
    self.cascade = cascade.ShiftPropbankEvokeCascade(self.actions)
    print self.cascade


  # Loads embeddings for words in the lexicon.
  def load_word_embeddings(self, embeddings_file):
    self.word_embeddings = [None] * self.words.size()
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
    for _ in xrange(size):
      word = ""
      while True:
        ch = f.read(1)
        if ch == " ": break
        word += ch

      vector = list(struct.unpack(fmt, f.read(vector_size)))
      ch = f.read(1)
      assert ch == "\n", "%r" % ch     # end of line expected

      index = self.words.index(word)
      if index != oov and self.word_embeddings[index] is None:
        self.word_embeddings[index] = vector
        count += 1

    f.close()

    self.word_embedding_indices =\
        [i for i, v in enumerate(self.word_embeddings) if v is not None]
    self.word_embeddings = [v for v in self.word_embeddings if v is not None]

    print "Loaded", count, "pre-trained embeddings from file with", size, \
        "vectors. Vectors for remaining", (self.words.size() - count), \
        "words will be randomly initialized."


  # Returns raw indices of LSTM features for all tokens in 'document'.
  def raw_lstm_features(self, document):
    output = []
    chars = []
    categories = []
    for token in document.tokens:
      decoding = list(token.text.decode("utf-8"))
      chars.append(decoding)
      categories.append([unicodedata.category(ch) for ch in decoding])

    for f in self.lstm_features:
      features = Feature()
      output.append(features)
      if f.name == "word":
        for token in document.tokens:
          features.add(self.words.index(token.text))
      elif f.name == "suffix":
        for index, token in enumerate(document.tokens):
          suffixes = self.get_suffixes(token.text, chars[index])
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
        for index in xrange(len(document.tokens)):
          all_punct = all(c[0] == 'P' for c in categories[index])
          some_punct = any(c[0] == 'P' for c in categories[index])

          if all_punct:
            features.add(Spec.ALL_PUNCTUATION)
          elif some_punct:
            features.add(Spec.SOME_PUNCTUATION)
          else:
            features.add(Spec.NO_PUNCTUATION)

      elif f.name == "digit":
        for index in xrange(len(document.tokens)):
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
        for index in xrange(len(document.tokens)):
          value = Spec.NO_QUOTE
          for cat, ch in zip(categories[index], chars[index]):
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
            if token.text == "``":
              value = Spec.OPEN_QUOTE
            elif token.text == "''":
              value = Spec.CLOSE_QUOTE
            if value == Spec.UNKNOWN_QUOTE:
              value = Spec.CLOSE_QUOTE if in_quote else Spec.OPEN_QUOTE
              in_quote = not in_quote
          features.add(value)
      else:
        raise ValueError("LSTM feature '", f.name, "' not implemented")
    return output


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
    else:
      raise ValueError("FF feature '", feature_spec.name, "' not implemented")

    return raw_features


  # Returns link features for the FF unit.
  def translated_ff_link_features(self, feature_spec, state):
    name = feature_spec.name
    num = feature_spec.num

    output = []
    if name == "history":
      for i in xrange(num):
        output.append(None if i >= state.steps else state.steps - i - 1)
    elif name == "lr":
      index = None
      if state.current < state.end:
        index = state.current - state.begin
      output.append(index)
    elif name == "rl":
      index = None
      if state.current < state.end:
        index = state.current - state.begin
      output.append(index)
    elif name == "frame-end-lr":
      for i in xrange(num):
        index = None
        end = state.frame_end_inclusive(i)
        if end != -1:
          index = end - state.begin
        output.append(index)
    elif name == "frame-end-rl":
      for i in xrange(num):
        index = None
        end = state.frame_end_inclusive(i)
        if end != -1:
          index = end - state.begin
        output.append(index)
    elif name == "frame-creation-steps":
      for i in xrange(num):
        step = state.creation_step(i)
        output.append(None if step == -1 else step)
    elif name == "frame-focus-steps":
      for i in xrange(num):
        step = state.focus_step(i)
        output.append(None if step == -1 else step)
    else:
      raise ValueError("Link feature not implemented:" + name)

    return output


  # Debugging methods.
  # Prints raw lstm features with a special prefix so that they can be grepped
  # and compared with another set of feature strings.
  def print_lstm_features(self, document, features):
    assert len(features) == len(self.lstm_features)
    for fidx, f in enumerate(self.lstm_features):
      assert len(document.tokens) == len(features[fidx].indices)
      for i, vals in enumerate(features[fidx].indices):
        text = "(" + document.tokens[i].text + ")"
        if type(vals) is int:
          last = str(vals)
          # For suffixes, also print the feature string.
          if f.name == "suffix":
            last = last + " Value=(" + self.suffix.value(vals) + ")"
          print "LEXDEBUG", f.name, "token", i, text, "=", last
        else:
          for v in vals:
            last = str(v)
            if f.name == "suffix":
              last = last + " Value=(" + self.suffix.value(v) + ")"
            print "LEXDEBUG", f.name, "token", i, text, "=", last


  # Returns feature strings for FF feature indices provided in 'indices'.
  # All indices are assumed to a single feature whose spec is in 'feature_spec'.
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
      print "Taking gold action", gold
      print "On state:", state

      gold_index = self.actions.indices.get(gold, None)
      assert gold_index is not None, "Unknown gold action: %r" % gold
      assert state.is_allowed(gold_index), "Disallowed gold action: %r" % gold
      state.advance(gold)

    print "Final state after", len(document.gold), "actions:", state

