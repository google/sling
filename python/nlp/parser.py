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

"""Wrapper classes for tokenizer, parser, and analyzer"""

import sling

evaluate_frames = sling.api.evaluate_frames

# Global tokenizer.
tokenizer = None

def tokenize(text, store=None, schema=None):
  # Initialize tokenizer if needed.
  global tokenizer
  if tokenizer == None: tokenizer = sling.api.Tokenizer()

  # Create store for document if needed.
  if store == None: store = sling.Store()

  # Tokenize text.
  frame = tokenizer.tokenize(store, text)

  # Return document with tokens.
  return sling.Document(frame, store, schema)


def lex(text, store=None, schema=None):
  # Initialize tokenizer if needed.
  global tokenizer
  if tokenizer == None: tokenizer = sling.api.Tokenizer()

  # Create store for document if needed.
  if store == None: store = sling.Store()

  # Parse LEX-encoded text.
  frame = tokenizer.lex(store, text)

  # Return document with annotations.
  return sling.Document(frame, store, schema)


class Parser:
  def __init__(self, filename, store=None):
    # Create new commons store for parser if needed.
    if store == None:
      self.commons = sling.Store()
    else:
      self.commons = store

    # Load parser.
    self.parser = sling.api.Parser(self.commons, filename)

    # Initialize document schema in commons.
    self.schema = sling.DocumentSchema(self.commons)

    # Freeze store if it is a private commons store for the parser.
    if store == None: self.commons.freeze()

  def parse(self, obj):
    if type(obj) is sling.Document:
      # Parser document.
      obj.update()
      self.parser.parse(obj.frame)
      obj.refresh_annotations()
      return obj
    elif type(obj) is sling.Frame:
      # Parse document frame and return parsed document.
      self.parser.parse(obj)
      return sling.Document(obj)
    else:
      # Create local store for new document.
      store = sling.Store(self.commons)

      # Tokenize text.
      doc = tokenize(str(obj), store=store, schema=self.schema)

      # Parse document.
      self.parser.parse(doc.frame)
      doc.refresh_annotations()
      return doc


class Analyzer:
  def __init__(self, commons, spec):
    # Initialize document schema in commons.
    self.commons = commons
    self.schema = sling.DocumentSchema(self.commons)

    # Load analyzer.
    self.analyzer = sling.api.Analyzer(commons, spec)

  def annotate(self, obj):
    if type(obj) is sling.Document:
      # Analyze document.
      obj.update()
      self.analyzer.annotate(obj.frame)
      obj.refresh_annotations()
      return obj
    elif type(obj) is sling.Frame:
      # Analyze document frame and return annotated document.
      self.analyzer.annotate(obj)
      return sling.Document(obj, schema=self.schema)
    else:
      # Create local store for new document.
      store = sling.Store(self.commons)

      # Tokenize text.
      doc = tokenize(str(obj), store=store, schema=self.schema)

      # Analyze document.
      self.analyzer.annotate(doc.frame)
      doc.refresh_annotations()
      return doc

