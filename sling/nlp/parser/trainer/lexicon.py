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


# Class for computing and serving a lexicon.
# Usage:
#   lexicon = Lexicon(normalize_digits=...)
#   lexicon.add("foo")
#   lexicon.add("bar")
#   ...
#   index = lexicon.index("foo")
#
# Methods with names starting with an underscore are meant to be internal.
class Lexicon:
  def __init__(self, normalize_digits=True, oov_item="<UNKNOWN>"):
    self.normalize_digits = normalize_digits
    self.item_to_index = {}
    self.index_to_item = {}

    if oov_item is not None:
      self.oov_item = oov_item
      self.oov_index = 0   # Don't change this; OOV is always at position 0
      self.index_to_item[self.oov_index] = self.oov_item
      self.item_to_index[self.oov_item] = self.oov_index
    else:
      self.oov_item = None
      self.oov_index = None


  # Returns whether the lexicon has an OOV item.
  def has_oov(self):
    return self.oov_index is not None


  # Returns the key internally used by the lexicon.
  def _key(self, item):
    if self.normalize_digits:
      return "".join([c if not c.isdigit() else '9' for c in list(item)])
    else:
      return item


  # Loads the lexicon from a text file with one key per line.
  def load(self, vocabfile):
    with open(vocabfile, "r") as f:
      index = 0
      for line in f:
        line = line.strip()
        if line == self.oov_item:
          assert index == self.oov_index, index
        self.item_to_index[line] = index
        self.index_to_item[index] = line
        index += 1


  # Reads the lexicon from a delimited string.
  def read(self, vocabstring, delimiter):
    index = 0
    lines = vocabstring.split(delimiter)
    if lines[-1] == '': lines.pop()
    for line_index, line in enumerate(lines):
      if line == self.oov_item:
        assert index == self.oov_index, index
      self.item_to_index[line] = index
      self.index_to_item[index] = line
      index += 1


  # Adds 'item' to the lexicon.
  def add(self, item):
    item = self._key(item)
    if item not in self.item_to_index:
      i = len(self.item_to_index)
      self.item_to_index[item] = i
      self.index_to_item[i] = item


  # Returns the size of the lexicon, including the OOV item, if any.
  def size(self):
    return len(self.item_to_index)


  # Returns the integer index of 'item' in the lexicon, if present.
  def index(self, item):
    item = self._key(item)
    if item not in self.item_to_index:
      return self.oov_index  # this is None if !has_oov()
    else:
      return self.item_to_index[item]


  # Returns a string representation of the key whose id is 'index'.
  def value(self, index):
    assert index >= 0 and index < len(self.index_to_item), "%r" % index
    return self.index_to_item[index]


  # Returns a string representation of the lexicon.
  def __str__(self):
    s = [self.index_to_item[i] for i in range(self.size())]
    return "\n".join(s)


  # Returns the string representation of the first 'n' keys in the lexicon.
  def first_few(self, prefix="", n=100):
    s = []
    for i in range(min(n, self.size())):
      s.append(prefix + str(i) + " = " + self.index_to_item[i])
    return "\n".join(s)
