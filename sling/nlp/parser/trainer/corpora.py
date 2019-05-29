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

from transition_generator import TransitionGenerator


# Stores a document and its gold transitions.
class AnnotatedDocument(sling.Document):
  def __init__(self, commons, schema, encoded):
    self._store = sling.Store(commons)
    self.object = self._store.parse(encoded, binary=True)
    super(AnnotatedDocument, self).__init__(frame=self.object, schema=schema)
    self.mentions.sort(key=lambda mention: (mention.begin, -mention.length))
    self.gold = []  # sequence of gold transitions


  def size(self):
    return len(self.tokens)


# An iterator over a recordio of documents. It doesn't load all documents into
# memory at once, and so can't shuffle the corpus. It can optionally loop over
# the corpus, and compute transition sequences for the existing frames in the
# document.
class Corpora:
  def __init__(self, recordio, commons, schema=None, gold=False, loop=False):
    self.filename = recordio
    self.commons_owned = False
    if isinstance(commons, str):
      self.commons = sling.Store()
      self.commons.load(commons)
      self.commons_owned = True
    else:
      assert isinstance(commons, sling.Store)
      self.commons = commons

    if schema is None or self.commons_owned:
      schema = sling.DocumentSchema(self.commons)
      if self.commons_owned:
        self.commons.freeze()
    assert schema is not None
    self.schema = schema

    self.reader = sling.RecordReader(recordio)
    self.generator = None
    self.loop = loop
    self.generator = None
    self.set_gold(gold)


  # Iterator interface.
  def __del__(self):
    self.reader.close()


  def __iter__(self):
    return self


  # Sets if the corpora should automatically loop at the end of the corpus.
  def set_loop(self, val):
    self.loop = val


  # Sets if the gold transitions should be added to the document at hand.
  def set_gold(self, gold):
    if gold and self.generator is None:
      self.generator = TransitionGenerator(self.commons)
    elif not gold:
      self.generator = None


  # Returns the next document.
  def __next__(self):
    if self.reader.done():
      if self.loop:
        self.reader.rewind()
      else:
        raise StopIteration

    (_, value) = next(self.reader)
    document = AnnotatedDocument(self.commons, self.schema, value)
    if self.generator is not None:
      document.gold = self.generator.generate(document)
    return document


  # Rewinds to the beginning of the corpus.
  def rewind(self):
    self.reader.rewind()

