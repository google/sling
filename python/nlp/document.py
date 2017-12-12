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

class DocumentNames:
  def __init__(self, store):
    self.isa = store['isa']
    self.document = store['/s/document']
    self.document_text = store['/s/document/text']
    self.document_tokens = store['/s/document/tokens']
    self.document_mention = store['/s/document/mention']
    self.document_theme = store['/s/document/theme']

    self.token = store['/s/token']
    self.token_index = store['/s/token/index']
    self.token_text = store['/s/token/text']
    self.token_start = store['/s/token/start']
    self.token_length = store['/s/token/length']
    self.token_break = store['/s/token/break']

    self.phrase = store['/s/phrase']
    self.phrase_begin = store['/s/phrase/begin']
    self.phrase_length = store['/s/phrase/length']
    self.phrase_evokes = store['/s/phrase/evokes']


class Token(object):
  def __init__(self, names, frame):
    self.names = names
    self.frame = frame

  @property
  def index(self):
    return self.frame[self.names.token_index]

  @index.setter
  def index(self, value):
    self.frame[self.names.token_index] = value

  @property
  def text(self):
    return self.frame[self.names.token_text]

  @text.setter
  def text(self, value):
    self.frame[self.names.token_text] = value

  @property
  def start(self):
    return self.frame[self.names.token_start]

  @start.setter
  def start(self, value):
    self.frame[self.names.token_start] = value

  @property
  def length(self):
    l = self.frame[self.names.token_length]
    if l == None: l = 1
    return l

  @length.setter
  def length(self, value):
    self.frame[self.names.token_length] = value

  @property
  def end(self):
    return self.start + self.length

  @property
  def brk(self):
    b = self.frame[self.names.token_break]
    if b == None: b = SPACE_BREAK
    return b

  @brk.setter
  def brk(self, value):
    self.frame[self.names.token_break] = value


class Mention(object):
  def __init__(self, names, frame):
    self.names = names
    self.frame = frame

  @property
  def begin(self):
    return self.frame[self.names.phrase_begin]

  @begin.setter
  def begin(self, value):
    self.frame[self.names.phrase_begin] = value

  @property
  def length(self):
    l = self.frame[self.names.phrase_length]
    if l == None: l = 1
    return l

  @length.setter
  def length(self, value):
    self.frame[self.names.phrase_length] = value

  @property
  def end(self):
    return self.begin + self.length

  def evokes(self):
    return self.frame(self.names.phrase_evokes)

  def evoke(self, evoked):
    return self.frame.append(self.names.phrase_evokes, evoked)


class Document(object):
  def __init__(self, frame=None, store=None, names=None):
    # Create store, frame, and names if missing.
    if frame != None:
      store = frame.store()
    if store == None:
      store = sling.Store()
    if names == None:
      names = DocumentNames(store)
    if frame == None:
      frame = store.frame([(names.isa, names.document)])

    # Initialize document from frame.
    self.frame = frame
    self.names = names
    self.tokens = []
    self.mentions = []
    self.themes = []
    self.tokens_dirty = False
    self.mentions_dirty = False
    self.themes_dirty = False

    # Get tokens.
    tokens = frame[names.document_tokens]
    if tokens != None:
      for t in tokens:
        token = Token(names, t)
        self.tokens.append(token)

    # Get mentions.
    for m in frame(names.document_mention):
      mention = Mention(names, m)
      self.mentions.append(mention)

    # Get themes.
    for theme in frame(names.document_theme):
      self.themes.append(theme)

  def add_token(self, text=None, start=None, length=None, brk=SPACE_BREAK):
    slots = [
      (self.names.isa, self.names.token),
      (self.names.token_index, len(self.tokens)),
    ]
    if text != None: slots.append((self.names.token_text, text))
    if start != None: slots.append((self.names.token_start, start))
    if length != None: slots.append((self.names.token_length, length))
    if brk != SPACE_BREAK: slots.append((self.names.token_break, brk))
    token = Token(self.names, self.frame.store().frame(slots))
    self.tokens.append(token)
    self.tokens_dirty = True
    return token;

  def add_mention(self, begin, end):
    length = end - begin
    slots = [
      (self.names.isa, self.names.phrase),
      (self.names.phrase_begin, begin),
    ]
    if length != 1: slots.append((self.names.phrase_length, length))
    mention = Mention(self.names, self.frame.store().frame(slots))
    self.mentions.append(mention)
    self.mentions_dirty = True
    return mention;

  def add_theme(self, theme):
    self.themes.append(theme)
    self.themes_dirty = True

  def update(self):
    # Update tokens in document frame.
    if self.tokens_dirty:
      array = []
      for token in self.tokens: array.append(token.frame)
      self.frame[self.names.document_tokens] = array
      self.tokens_dirty = False

    # Update mentions in document frame.
    if self.mentions_dirty:
      slots = []
      for mention in self.mentions:
        slots.append((self.names.document_mention, mention.frame))
      del self.frame[self.names.document_mention]
      self.frame.extend(slots)
      self.mentions_dirty = False

    # Update themes in document frame.
    if self.themes_dirty:
      slots = []
      for theme in self.themes:
        slots.append((self.names.document_theme, theme))
      del self.frame[self.names.document_theme]
      self.frame.extend(slots)
      self.themes_dirty = False

  @property
  def text(self):
    return self.frame[self.names.document_text]

  @text.setter
  def text(self, value):
    self.frame[self.names.document_text] = value

