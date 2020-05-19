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

"""Wrapper classes for document, token, and mention frames"""

import sling

# Token break types.
NO_BREAK = 0
SPACE_BREAK = 1
LINE_BREAK = 2
SENTENCE_BREAK = 3
PARAGRAPH_BREAK = 4
SECTION_BREAK = 5
CHAPTER_BREAK = 6

class DocumentSchema:
  def __init__(self, store):
    self.isa = store['isa']
    self.document = store['document']
    self.document_text = store['text']
    self.document_tokens = store['tokens']
    self.document_mention = store['mention']
    self.document_theme = store['theme']

    self.token = store['token']
    self.token_index = store['index']
    self.token_word = store['word']
    self.token_start = store['start']
    self.token_size = store['size']
    self.token_break = store['break']
    self.token_pos = store['postag']

    self.phrase = store['phrase']
    self.phrase_begin = store['begin']
    self.phrase_length = store['length']
    self.phrase_evokes = store['evokes']

    self.thing = store['thing']


class Token(object):
  def __init__(self, document, frame, index):
    self.document = document
    self.schema = document.schema
    self.frame = frame
    self.index = index

  @property
  def word(self):
    text = self.frame[self.schema.token_word]
    if text == None:
      start = self.frame[self.schema.token_start]
      if start != None:
        size = self.frame[self.schema.token_size]
        if size == None: size = 1
        text = self.document._text[start : start + size].decode()
    return text

  @word.setter
  def word(self, value):
    self.frame[self.schema.token_word] = value

  @property
  def start(self):
    return self.frame[self.schema.token_start]

  @start.setter
  def start(self, value):
    self.frame[self.schema.token_start] = value

  @property
  def size(self):
    s = self.frame[self.schema.token_size]
    if s == None: s = 1
    return s

  @size.setter
  def size(self, value):
    self.frame[self.schema.token_size] = value

  @property
  def end(self):
    return self.start + self.size

  @property
  def brk(self):
    b = self.frame[self.schema.token_break]
    if b == None: b = SPACE_BREAK if self.index > 0 else NO_BREAK
    return b

  @brk.setter
  def brk(self, value):
    self.frame[self.schema.token_break] = value


class Mention(object):
  def __init__(self, schema, frame):
    self.schema = schema
    self.frame = frame

  @property
  def begin(self):
    return self.frame[self.schema.phrase_begin]

  @begin.setter
  def begin(self, value):
    self.frame[self.schema.phrase_begin] = value

  @property
  def length(self):
    l = self.frame[self.schema.phrase_length]
    if l == None: l = 1
    return l

  @length.setter
  def length(self, value):
    self.frame[self.schema.phrase_length] = value

  @property
  def end(self):
    return self.begin + self.length

  def evokes(self):
    return self.frame(self.schema.phrase_evokes)

  def evoke(self, evoked):
    return self.frame.append(self.schema.phrase_evokes, evoked)

  def evoke_type(self, frame_type):
    f = self.frame.store().frame({self.schema.isa: frame_type})
    self.evoke(f)
    return f


class Document(object):
  def __init__(self, frame=None, store=None, schema=None):
    # Create store, frame, and schema if missing.
    if frame != None:
      store = frame.store()
    if store == None:
      store = sling.Store()
    if schema == None:
      schema = DocumentSchema(store)
    if frame == None:
      frame = store.frame([(schema.isa, schema.document)])

    # Initialize document from frame.
    self.frame = frame
    self.schema = schema
    self._text = frame.get(schema.document_text, binary=True)
    self.tokens = []
    self.mentions = []
    self.themes = []
    self.tokens_dirty = False
    self.mentions_dirty = False
    self.themes_dirty = False

    # Get tokens.
    tokens = frame[schema.document_tokens]
    if tokens != None:
      for t in tokens:
        token = Token(self, t, len(self.tokens))
        self.tokens.append(token)

    # Get mentions.
    for m in frame(schema.document_mention):
      mention = Mention(schema, m)
      self.mentions.append(mention)

    # Get themes.
    for theme in frame(schema.document_theme):
      self.themes.append(theme)

  @property
  def store(self):
    return self.frame.store()

  def add_token(self, word=None, start=None, length=None, brk=SPACE_BREAK):
    slots = []
    if word != None: slots.append((self.schema.token_word, word))
    if start != None: slots.append((self.schema.token_start, start))
    if length != None: slots.append((self.schema.token_length, length))
    if brk != SPACE_BREAK: slots.append((self.schema.token_break, brk))
    token = Token(self, self.store.frame(slots), len(self.tokens))
    self.tokens.append(token)
    self.tokens_dirty = True
    return token

  def add_mention(self, begin, end):
    length = end - begin
    slots = [(self.schema.phrase_begin, begin)]
    if length != 1: slots.append((self.schema.phrase_length, length))
    mention = Mention(self.schema, self.store.frame(slots))
    self.mentions.append(mention)
    self.mentions_dirty = True
    return mention

  def evoke(self, begin, end, frame):
    mention = self.add_mention(begin, end)
    mention.evoke(frame)
    return frame

  def evoke_type(self, begin, end, frame_type):
    mention = self.add_mention(begin, end)
    return mention.evoke_type(frame_type)

  def add_theme(self, theme):
    self.themes.append(theme)
    self.themes_dirty = True

  def update(self):
    # Update tokens in document frame.
    if self.tokens_dirty:
      array = []
      for token in self.tokens: array.append(token.frame)
      self.frame[self.schema.document_tokens] = array
      self.tokens_dirty = False

    # Update mentions in document frame.
    if self.mentions_dirty:
      slots = []
      for mention in self.mentions:
        slots.append((self.schema.document_mention, mention.frame))
      del self.frame[self.schema.document_mention]
      self.frame.extend(slots)
      self.mentions_dirty = False

    # Update themes in document frame.
    if self.themes_dirty:
      slots = []
      for theme in self.themes:
        slots.append((self.schema.document_theme, theme))
      del self.frame[self.schema.document_theme]
      self.frame.extend(slots)
      self.themes_dirty = False

  @property
  def text(self):
    return self._text

  @text.setter
  def text(self, value):
    if isinstance(value, str): value = value.encode()
    self._text = value
    self.frame[self.schema.document_text] = value

  def phrase(self, begin, end):
    parts = []
    for token in self.tokens[begin:end]:
      if len(parts) > 0 and token.brk != NO_BREAK: parts.append(' ')
      parts.append(token.word)
    return ''.join(parts)

  def tolex(self):
    self.update()
    return sling.api.tolex(self.frame)

  def decorate(self):
    self.update()
    index = 0
    for t in self.tokens:
      t.frame.index = index
      if t.frame.word == None: t.word = self.text[t.start:t.end]
      index += 1
    for m in self.mentions:
      if m.frame.name is None:
        m.frame.name = self.phrase(m.begin, m.end)

  def remove_annotations(self):
    if len(self.mentions) > 0:
      self.mentions = []
      self.mentions_dirty = True
    if len(self.themes) > 0:
      self.themes = []
      self.themes_dirty = True
    self.update()

  def refresh_annotations(self):
    self.mentions = []
    for m in self.frame(self.schema.document_mention):
      mention = Mention(self.schema, m)
      self.mentions.append(mention)
    self.mentions_dirty = False

    self.themes = []
    for theme in self.frame(self.schema.document_theme):
      self.themes.append(theme)
    self.themes_dirty = False


class Corpus:
  def __init__(self, filename, commons=None):
    self.input = sling.RecordDatabase(filename)
    self.iter = iter(self.input)
    if commons == None:
      self.commons = sling.Store()
      self.docschema = sling.DocumentSchema(self.commons)
      self.commons.freeze()
    else:
      self.commons = commons
      if "document" in commons:
        self.docschema = sling.DocumentSchema(commons)
      else:
        self.docschema = None

  def __getitem__(self, key):
    data = self.input.lookup(key)
    f = sling.Store(self.commons).parse(data)
    return sling.Document(f, schema=self.docschema)

  def __iter__(self):
    return self

  def __next__(self):
    _, data = self.input.__next__()
    f = sling.Store(self.commons).parse(data)
    return sling.Document(f, schema=self.docschema)

