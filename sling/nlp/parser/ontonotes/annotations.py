# Copyright 2018 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License")

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

"""Classes for reading annotations in the CONLL format.
"""

import re
import sling

# Represents a span/token/constituent.
class Span:
  def __init__(self, begin, label):
    # Creates an open span with the given begin token and label.
    self.begin = begin
    self.label = label
    self.end = None  # token-exclusive; to be determined later

    # Only for tokens.
    self.text = None
    self.brk = sling.SPACE_BREAK

    # Only for constituency spans.
    self.parent = None   # parent span
    self.children = []   # children spans in left->right order

    # Only true for SRL predicate spans.
    self.predicate = False

    # Head token index.
    self.head = None

  # Returns whether this is a leaf span.
  def leaf(self):
    return len(self.children) == 0

  # Returns whether this only has token constituents as children.
  def leaf_constituent(self):
    for c in self.children:
      if len(c.children) > 0 or c.end > c.begin + 1:
        return False
    return True

  # Returns if the span is complete, i.e. has an end token.
  def ended(self):
    return self.end is not None

  # Marks the span as ended at token 'end'.
  def finish(self, end):
    self.end = end

  # Returns the length of the span.
  def length(self):
    assert self.end is not None, str(self)
    return self.end - self.begin

  # Whether the constituent span represents a conjunction.
  def is_conjunction(self):
    for c in self.children:
      if c.label == 'CC':
        return True
    return False

  # Returns the string representation of the span.
  def __repr__(self):
    s = "[" + str(self.begin) + ":" + str(self.label) + ":"
    if self.ended():
      s += str(self.end)
    else:
      s += '?'
    s += ")"
    return s


# A stack is a list of (possibly nested) spans.
class Stack:
  def __init__(self, nested=False):
    # List of spans in the stack.
    self.spans = []

    # Whether we can assume nestedness of spans and compute parent spans.
    # We only assume it for constituency spans.
    self.nested = nested

    # Most nested open span (assuming nestedness).
    self.open = None

  # Starts a new open span and pushes it on to the stack.
  def start(self, begin, label):
    span = Span(begin, label)
    self.spans.append(span)
    if self.nested:
      if self.open is not None:
        self.open.children.append(span)
      span.parent = self.open
      self.open = span

  # Marks the topmost span with label 'label' as closed.
  def finish(self, end, label=None):
    assert len(self.spans) > 0
    if self.nested:
      assert self.open is not None
      if label is not None:
        assert self.open.label == label
      self.open.finish(end)
      self.open = self.open.parent
    else:
      # Find an open span that matches the label, and close it.
      found = False
      for span in reversed(self.spans):
        if not span.ended() and (label is None or span.label == label):
          span.finish(end)
          found = True
          break
      assert found, (str(self.spans), end, label)

  # Adds a [begin, begin + 1) span.
  def singleton(self, begin, label):
    self.start(begin, label)
    self.finish(begin + 1, label)

  # Returns whether all spans have ended.
  def all_ended(self):
    for s in self.spans:
      if not s.ended():
        return False
    return True

  # Copies a range of spans to 'other'.
  def copy_to(self, other, begin, end=None):
    assert self.open is None
    for s in self.spans:
      if s.begin >= begin and (end is None or s.begin < end):
        assert s.end >= begin
        if end is not None: assert s.end <= end
        other.spans.append(s)

  # Returns the string representation of the stack.
  def __repr__(self):
    return str(self.spans)


# Cache of mentions over a document.
class Mentions:
  def __init__(self, document):
    self.document = document
    self.mentions = {}

  # Gets an existing mention or adds a new one.
  # x,y can either denote an [x, y) interval or
  # x can be a Span.
  def get_or_add(self, x, y=None):
    if isinstance(x, Span):
      y = x.end
      x = x.begin
    assert y is not None
    mention = self.mentions.get((x, y))
    if mention is None:
      mention = self.document.add_mention(x, y)
      self.mentions[(x, y)] = mention
    return mention

  # Returns whether the given span has a mention.
  def has(self, span):
    assert isinstance(span, Span)
    return (span.begin, span.end) in self.mentions


# Stores the end token of the spans.
class SpanEnds:
  def __init__(self, size):
    self.end = [False] * size   # i -> whether a span ends at i

  # Add an end token.
  def add(self, end):
    self.end[end] = True

  # Returns the last end token strictly before 'index'.
  def last_end_before(self, index):
    for i in range(index - 1, -1, -1):
      if self.end[i]:
        return i
    return -1


# Span types.
NER = 1
PRED = 2
ARG = 4
COREF = 8
ALL = NER | PRED | ARG | COREF


# Collection of CONLL annotations for a document.
class Annotations:
  CONSTITUENCY_BEGIN = True
  CONSTITUENCY_END = False

  def __init__(self, converter):
    self.converter = converter
    self.options = converter.options
    self.summary = converter.summary

    # Current sentence index.
    self.sentence = None


  # Generator that returns spans of specified type(s).
  def spans(self, types_mask, label=False):
    if types_mask & NER > 0:
      for s in self.ner.spans:
        yield (s, "NER") if label else s

    if types_mask & (PRED | ARG) > 0:
      for srl in self.srl:
        for s in srl.spans:
          if s.predicate and (types_mask & PRED > 0):
            yield (s, "PRED") if label else s
          elif not s.predicate and (types_mask & ARG > 0):
            yield (s, "ARG") if label else s

    if types_mask & COREF > 0:
      for s in self.coref.spans:
        yield (s, "COREF") if label else s


  # Returns part-of-speech for the specified token.
  def pos(self, index):
    return self.tokens.spans[index].label


  # Processes 'line' which should be in the CONLL format.
  # - either '#begin document', signifying the start of a document part.
  # - or '#end document', signifying the end of a document part.
  # - or a line with the following space-separated fields:
  #   0: Document ID
  #   1: Part number
  #   2: Intra-sentence word number
  #   3: Word
  #   4: Part of speech
  #   5: Parse bit (i.e. constituency info)
  #   6: Predicate lemma
  #   7: Predicate id
  #   8: Word sense
  #   9: Speaker
  #  10: Named Entities
  #  11 to N-1: SRL annotations annotations (one column per predicate)
  #   N: Coreference
  #
  # Calls 'callback' if a document has been parsed completely.
  # Here a 'document' refers to a sentence or part (as per 'self.options').
  def read(self, line, callback):
    if len(line) == 0: return

    input_stats = self.summary.input
    if line.startswith("#begin document"):
      # Start a new document.
      self._start_document()
      input_stats.parts.increment()
      return

    if line.startswith("#end document"):
      # Document is done.
      self._end_document(callback)
      return

    fields = line.split()
    if len(fields) == 0:
      # Separator line between sentences.
      return

    # Field 2: intra-sentence token index.
    token_break = sling.SPACE_BREAK
    if fields[2] == '0':  # new sentence
      if self.options.doc_per_sentence:
        self._end_document(callback)
        self._start_document()

      # Set docid and part.
      self.docid = fields[0]
      self.part = fields[1]

      input_stats.sentences.increment()
      token_break = sling.SENTENCE_BREAK
      self._new_sentence(num_srl_predicates=len(fields) - 12)

    # Reset token break to NO_BREAK for the first token.
    if len(self.tokens.spans) == 0:
      token_break = sling.NO_BREAK

    # Use absolute token index instead of the intra-sentence token index.
    token_index = len(self.tokens.spans)

    # Add token.
    self.tokens.singleton(token_index, label=None)
    token = self.tokens.spans[-1]
    token.brk = token_break

    # fields[3] = token text, fields[4] = POS.
    token.text = fields[3]
    token.label = fields[4]

    # fields[5] = constituency bit.
    for (boundary_type, label) in self._parse_constituents(fields[5]):
      if boundary_type == Annotations.CONSTITUENCY_BEGIN:
        self.constituents.start(token_index, label)
      else:
        assert boundary_type == Annotations.CONSTITUENCY_END
        self.constituents.finish(token_index + 1)

    # Following two fields are ignored.
    # fields[8] = Token's word sense.
    # fields[9] = speaker id.

    # fields[10] = Named entity annotations.
    ner = fields[10]
    if ner != '*':
      beginning = ner[0] == '('
      ending = ner[-1] == ')'
      if beginning and ending:  # e.g. (PERSON)
        label = ner[1:-1]
        self.ner.singleton(token_index, label)
      elif beginning:           # e.g. (TIME*
        assert ner[-1] == '*'
        label = ner[1:-1]
        self.ner.start(token_index, label)
      else:                     # e.g. *)
        assert ending
        assert ner == '*)'
        self.ner.finish(token_index + 1)

    # fields[-1] = Coreference.
    if not self.options.skip_coref:
      coref = fields[-1]
      if coref != "-":
        coref_fields = coref.split('|')
        for coref_field in coref_fields:
          beginning = coref_field[0] == '('
          ending = coref_field[-1] == ')'
          if beginning and ending:
            cluster_id = int(coref_field[1:-1])  # e.g. (8)
            self.coref.singleton(token_index, cluster_id)
          elif beginning:
            cluster_id = int(coref_field[1:])    # e.g. (8
            self.coref.start(token_index, cluster_id)
          else:                                  # e.g. 8)
            assert ending
            cluster_id = int(coref_field[:-1])
            self.coref.finish(token_index + 1, cluster_id)

    # fields[11:-1] = SRL annotations.
    assert len(self.current_srl) == len(fields[11:-1])
    for index, srl in enumerate(fields[11:-1]):
      srl_annotation = self.current_srl[index]
      if srl != '*':
        beginning = srl[0] == '('
        ending = srl[-1] == ')'
        if beginning and ending:  # e.g. (V*) or (ARGM-TMP*)
          assert srl[-2] == '*'
          label = srl[1:-2]
          predicate = False
          if label == 'V':
            predicate = True
            assert fields[6] != '-'
            label = fields[6]                # predicate prefix, e.g. 'live'
            if fields[7] != "-":
              label += "-" + fields[7]       # predicate suffix, e.g. '01'
          srl_annotation.singleton(token_index, '/pb/' + label)
          if predicate:
            srl_annotation.spans[-1].predicate = True
        elif beginning:           # e.g. (ARG2*
          assert srl[-1] == '*'
          label = srl[1:-1]
          assert label != 'V'     # predicates can't be multi-token
          srl_annotation.start(token_index, '/pb/' + label)
        else:                     # e.g. *)
          assert ending
          assert srl == '*)'
          srl_annotation.finish(token_index + 1)


  # Resets stacks for a new document.
  def _start_document(self):
    self.docid = None
    self.part = None

    # Spans for NER, Coref, SRL, Constituency nodes.
    self.ner = Stack()
    self.coref = Stack()
    self.srl = []        # one stack per predicate for all processed sentences
    self.constituents = Stack(nested=True)
    self.tokens = Stack()

    # Under-construction SRL annotations. One stack per predicate in
    # the current sentence. These are reset after each sentence.
    self.current_srl = []


  # Called at the start of each sentence. Allocates space to hold SRL
  # annotations for the new sentence after saves the SRL annotations of the
  # previous sentence (if any).
  def _new_sentence(self, num_srl_predicates):
    # Done with SRL annotations for the previous sentence.
    self._save_srl_annotations()

    # Add one stack per SRL predicate in the new sentence.
    self.current_srl = []
    for _ in range(num_srl_predicates):
      self.current_srl.append(Stack())

    # Set sentence index.
    if self.sentence is None:
      self.sentence = 0
    else:
      self.sentence += 1


  # Called at the end of the document. Writes the annotations to a SLING
  # document and invokes 'callback' with it.
  def _end_document(self, callback=None):
    if len(self.tokens.spans) == 0: return

    # Save SRL annotations for the last sentence.
    self._save_srl_annotations()

    # Add tokens as leaf constituents.
    self._add_token_constituents()

    # Find heads of all constituents.
    for node in self.constituents.spans:
      self.converter.head_finder.find(node)

    # Generate input statistics.
    self._summarize_input()

    # Sanity check: all annotations should be complete.
    assert len(self.current_srl) == 0, self.current_srl
    assert self.ner.all_ended()
    assert self.coref.all_ended()
    assert self.constituents.all_ended()
    for s in self.srl:
      assert s.all_ended()

    # Write the SLING document and invoke the callback.
    if callback is not None:
      store = sling.Store(self.converter.commons)
      document = sling.Document(None, store, self.converter.schema)
      self.write(document)
      callback(document)


  # Saves the current sentence's SRL annotations.
  def _save_srl_annotations(self):
    # SRL spans should be complete.
    for stack in self.current_srl:
      assert stack.all_ended()
    self.srl.extend(self.current_srl)
    self.current_srl = []


  # Complete the children constituents array by inserting tokens:
  # - When there are holes in the token range of the parent.
  # - At leaf constituents.
  def _add_token_constituents(self):
    for constituent in self.constituents.spans:
      children = []
      begin = constituent.begin
      for child in constituent.children:
        for i in range(begin, child.begin):
          s = self.tokens.spans[i]
          children.append(s)
          s.parent = constituent
        children.append(child)
        begin = child.end
      for i in range(begin, constituent.end):
        s = self.tokens.spans[i]
        children.append(s)
        s.parent = constituent
      constituent.children = children

  # Returns phrase(s) from 'span' that are comprised entirely of
  # allowed tokens as per 'disallowed'. Also, if 'pos_tags' is set to a list
  # of POS tags,  then each returned phrase's tokens need to have a POS in
  # that list.
  def _split_noun_phrases(self, span, disallowed, \
    pos_tags=['NNS', 'NN', 'NNP', 'NNPS', 'HYPH']):
    allowed = lambda i: (disallowed is None or not disallowed[i]) and \
      (pos_tags is None or self.pos(i) in pos_tags)

    output = []
    begin = span.begin
    end = begin
    while end < span.end:
      while end < span.end and allowed(end):
        end += 1
      if end > begin:
        # Remove trailing and leading hyphens.
        right = end - 1
        while right >= begin and self.pos(right) == 'HYPH': right -= 1
        left = begin
        while left <= right and self.pos(left) == 'HYPH': left += 1
        if right >= left:
          output.append((left, right + 1))
      while end < span.end and not allowed(end):
        end += 1
      begin = end
    return output

  # Generates noun phrases using constituency information and adds them
  # as extra named entities.
  # - Marks [NML HYPH PP] spans as noun phrases, e.g. [Commander - in - chief],
  #   or [right - of - way]. The inner NML is not processed as a nested span.
  #
  # - Each base NML span (i.e. only has token children) is split into noun
  #   phrases, such that each phrase:
  #   - doesn't overlap with an existing NER/PRED/noun-phrase span, AND
  #   - doesn't contain any conjunction (CC) token, AND
  #   - doesn't begin or end in a HYPH token
  #
  # - Each base NP (i.e. with no NP descendants) is split into noun phrase(s)
  #   using the same heuristic as above for NML spans, with an additional
  #   restriction that each phrase token have a noun/hyphen part-of-speech tag.
  #
  # - Each recursive NP of the form [NP ending in POS, <token constituents>]
  #   yields noun phrase(s) from the token constituents portion. For example,
  #   the NP "Japan's economy and development" would yield 'economy' and
  #   'development' phrases. We need this where the token constituents are
  #   not covered by base NP(s).
  #
  # - Marks each pronoun as a noun-phrase (of type PERSON for some pronouns).
  def _add_noun_phrases(self):
    # Setup token ranges from which we won't generate new noun phrases.
    # Disallow existing NER spans.
    disallowed = [False] * len(self.tokens.spans)
    for span in self.ner.spans:
      for i in range(span.begin, span.end):
        disallowed[i] = True

    # Also disallow predicates, since we also have nominal predicates.
    for srl in self.srl:
      for span in srl.spans:
        if span.predicate:
          for i in range(span.begin, span.end):
            disallowed[i] = True

    # Break noun phrase expansion at conjunctions and commas.
    for token in self.tokens.spans:
      if token.label == 'CC' or token.label == ',':
        disallowed[token.begin] = True

    # Sort in ascending order of constituent lengths.
    spans = [span for span in self.constituents.spans]
    spans.sort(key=lambda s:s.length())
    norm_summary = self.summary.normalization

    # Get NML spans that decompose as [NML HYPH PP].
    for span in spans:
      ch = span.children
      if span.label == 'NML' and len(ch) == 3 and \
        ch[0].label == 'NML' and ch[1].label == 'HYPH' and ch[2].label == 'PP':
          self.ner.start(span.begin, self.options.backoff_type)
          self.ner.finish(span.end)
          added = self.ner.spans[-1]
          example = (self.docid, self._phrase(added), self._pos_sequence(added))
          norm_summary.nml_titles.increment(\
            self._child_sequence(span), example=example)
          for i in range(added.begin, added.end):
            disallowed[i] = True

    # Get noun phrase(s) from each base NML span.
    base = {}  # NML span boundaries -> base NML or not
    for span in spans:
      if span.label != 'NML':
        continue

      key = (span.begin, span.end)
      if key not in base:
        # Leaf NML span. Additionally check if all children are tokens.
        base[key] = span.leaf_constituent()

        # Split this span into allowed portions. Each portion becomes a
        # named entity.
        if base[key]:
          phrases = self._split_noun_phrases(span, disallowed, pos_tags=None)
          for (begin, end) in phrases:
            self.ner.start(begin, self.options.backoff_type)
            self.ner.finish(end)
            added = self.ner.spans[-1]
            example = (self.docid, self._phrase(added))
            norm_summary.base_nml.increment(end - begin, example=example)

        # Mark all NML ancestors as recursive so they won't be processed.
        p = span.parent
        while p is not None:
          if p.label == 'NML':
            base[(p.begin, p.end)] = False
          p = p.parent

      # An NML span should be off-limits upon during subsequent NP-span
      # processing, so mark each NML span (base or recursive) as disallowed.
      for i in range(span.begin, span.end):
        disallowed[i] = True

    # Output noun phrase(s) from each base NP.
    processed_np = set()
    for span in spans:
      if span.label != 'NP': continue

      key = (span.begin, span.end)
      if key not in processed_np:
        processed_np.add(key)

        # Mark ancestor NPs as processed.
        p = span.parent
        while p is not None:
          if p.label == 'NP':
            processed_np.add((p.begin, p.end))
          p = p.parent

        phrases = self._split_noun_phrases(span, disallowed)
        for (begin, end) in phrases:
          self.ner.start(begin, self.options.backoff_type)
          self.ner.finish(end)
          added = self.ner.spans[-1]
          example = (self.docid, self._phrase(added))
          norm_summary.base_np.increment(end - begin, example=example)

          # Disallow the range of the added NP.
          for i in range(added.begin, added.end):
            disallowed[i] = True

    # Handle NPs of the form [Base NP ending in POS, token constituents].
    for span in spans:
      if span.label != 'NP' or len(span.children) < 2 or \
        span.children[0].label != 'NP' or \
        self.pos(span.children[0].end - 1) != 'POS' or \
        not span.children[0].leaf_constituent():
        continue

      other_children_tokens = True
      for i in range(0, len(span.children)):
        if i > 0 and len(span.children[i].children) > 0:
          other_children_tokens = False
          break
      if not other_children_tokens:
        continue

      boundary = Span(span.children[0].end, '')
      boundary.end = span.end
      phrases = self._split_noun_phrases(span, disallowed, \
        pos_tags=['NN', 'NNS', 'HYPH'])
      child_seq = self._child_sequence(span)
      for (begin, end) in phrases:
        self.ner.start(begin, self.options.backoff_type)
        self.ner.finish(end)
        added = self.ner.spans[-1]
        pos_seq = ' '.join([self.pos(i) for i in range(begin, end)])
        example = (self.docid, \
          self._phrase(span) + " -> " + self._phrase(added), child_seq)
        norm_summary.recursive_np.increment(pos_seq, example=example)
        for i in range(begin, end):
          disallowed[i] = True

    # Mark pronouns.
    person_pronouns = ['he', 'she', 'him', 'her', 'himself', 'herself']
    for span in self.tokens.spans:
      if span.label in ['PRP', 'PRP$']:
        label = self.options.backoff_type
        if span.text.lower() in person_pronouns:
          label = "PERSON"
        self.ner.singleton(span.begin, label)


  # Returns a list of beginning and ending constituent spans from 'parse_bit'.
  # Examples:
  # (S(VP*  : S and VP spans begin here.
  # (NP*))) : NP span begins and ends here, 2 other spans end here.
  # *))))   : 4 spans end here.
  #
  # Beginning and ending spans are assumed to be separated by '*'.
  def _parse_constituents(self, parse_bit):
    if parse_bit == '' or parse_bit == '*' or parse_bit == '-':
      return []

    index = parse_bit.find('*')
    assert index != -1, parse_bit
    output = []
    for begin_tag in parse_bit[0:index].split('('):
      if begin_tag != '':
        output.append((Annotations.CONSTITUENCY_BEGIN, begin_tag))

    for _ in range(len(parse_bit[index + 1:])):
      output.append((Annotations.CONSTITUENCY_END, ''))

    return output


  # Returns a map [begin, end) -> lowest constituent that covers it exactly.
  def _constituency_map(self):
    output = {}

    # Process constituents in left->right order. This will automatically
    # give us the lowest constituency span for a given interval.
    for span in self.constituents.spans:
      output[(span.begin, span.end)] = span

    # Also include the tokens.
    for span in self.tokens.spans:
      output[(span.begin, span.end)] = span
    return output


  # Normalizes 'span' using syntax-based heuristics.
  def _normalize_span(self, span, constituents, key=None, example=False):
    begin = span.begin
    end = span.end
    changed = True
    tokens = self.tokens.spans
    norm_summary = self.summary.normalization

    while end > begin + 1:  # can't normalize single word spans
      histogram = None
      original = (begin, end)
      if self.options.trim_trailing_possessives and \
        span.head == end - 1 and tokens[span.head].label == 'POS' and \
        (begin, end - 1) in constituents:
        histogram = norm_summary.possessives
        end -= 1
      elif self.options.drop_leading_articles and \
        tokens[begin].label == 'DT' and begin != span.head:
        histogram = norm_summary.articles
        begin += 1
      elif self.options.descend_prepositions:
        lowest = constituents.get((begin, end), None)
        if lowest is not None and \
          lowest.head == begin and tokens[begin].label in ['IN', 'TO'] and \
          (begin + 1, end) in constituents:
            histogram = norm_summary.prep
            begin += 1

      if (begin, end) == original:
        break
      else:
        # Record the change in the corresponding histogram.
        if histogram is not None:
          if key is None: key = span.label
          e = None
          if example: e = self._context(original[0], original[1], window=0)
          histogram.increment(key, example=e)

        # Recompute head.
        lowest = constituents.get((begin, end), None)
        if lowest is not None and lowest.head is not None:
          span.head = lowest.head
        elif span.head is not None and (span.head < begin or span.head >= end):
          span.head = None

    # Set new span boundaries.
    changed = (span.begin, span.end) != (begin, end)
    span.begin = begin
    span.end = end
    return changed


  # Computes heads of all spans using constituency information.
  def _compute_span_heads(self, constituents):
    for span in self.spans(ALL):
      span.head = None
      lowest = constituents.get((span.begin, span.end), None)
      if lowest is not None and lowest.head is not None:
        span.head = lowest.head
      elif self.options.last_token_as_fallback_head:
        span.head = span.end - 1


  # Returns a head token -> span mapping for NER and SRL Predicate spans.
  def _head_to_span(self):
    heads = {}
    for (span, label) in self.spans(NER | PRED, label=True):
      heads[span.head] = (span, label)
    return heads


  # Returns the POS sequence string for the span's token range.
  def _pos_sequence(self, span):
    seq = []
    for i in range(span.begin, span.end):
      pos = self.pos(i)
      if i == span.head: pos = '[' + pos + ']'
      seq.append(pos)
    return ' '.join(seq)


  # Returns the label sequence string for the span's children.
  def _child_sequence(self, span):
    return ' '.join([c.label for c in span.children])


  # Normalizes all spans.
  def _normalize(self):
    norm_summary = self.summary.normalization
    constituents = self._constituency_map()
    tokens = self.tokens.spans

    # Populate span heads for NER and predicate spans.
    self._compute_span_heads(constituents)

    # Token -> Span that ends there (or None).
    span_ends = SpanEnds(len(tokens))

    # Normalize NER spans.
    for span in self.ner.spans:
      if self._normalize_span(span, constituents, example=True):
        norm_summary.ner.increment()
      span_ends.add(span.end - 1)

    # Normalize SRL spans.
    for span in self.spans(PRED | ARG):
      if self._normalize_span(span, constituents, example=not span.predicate):
        if span.predicate:
          norm_summary.predicates.increment()
          span_ends.add(span.end - 1)
        else:
          norm_summary.arguments.increment()

    # Normalize Coref spans.
    for span in self.spans(COREF):
      if self._normalize_span(span, constituents, key="COREF", example=False):
        norm_summary.coref.increment()

    # Shrink the spans further.
    # Collect head -> span mapping for NER and SRL predicate spans.
    heads = self._head_to_span()

    # Collect spans to be normalized and sort them by length.
    spans = list(self.spans(ARG | COREF, label=True))
    spans.sort(key=lambda s: s[0].length())

    # Normalize spans.
    for span, source in spans:
      orig_start = span.begin
      orig_end = span.end
      lowest = constituents.get((span.begin, span.end), None)

      # Skip if we shouldn't normalize conjunctions.
      if lowest is not None:
        assert lowest.head == span.head, (lowest.head, span.head)
        if lowest.is_conjunction() and not self.options.normalize_conjunctions:
          continue

      # Last ditch effort to set the span head.
      if span.head is None and self.options.last_token_as_fallback_head:
        span.head = span.end - 1

      shrunk_bin = None
      reduced_bin = None

      # Try to shrink the span so that it matches an existing span
      # with the same head.
      if self.options.shrink_using_heads:
        match = heads.get(span.head, None)
        if match is not None:
          span.begin = match[0].begin
          span.end = match[0].end
          shrunk_bin = source + " -> " + heads[span.head][1]

      # If shrinking fails, then reduce the span to its head token.
      if shrunk_bin is None and span.head is not None \
        and self.options.reduce_to_head and \
        (span.begin != span.head or span.end != span.head + 1):
        span.begin = span.head
        span.end = span.head + 1
        reduced_bin = source + " (" + self.pos(span.head) + ")"

      example = (self.docid, self._phrase(orig_start, orig_end), \
          self._phrase(span))

      if reduced_bin:
        norm_summary.reduced.increment(reduced_bin, example=example)
      elif shrunk_bin:
        norm_summary.head.increment(shrunk_bin, example=example)
      else:
        # Couldn't normalize span with any heuristic.
        example = (example[0], example[1])
        norm_summary.none.increment(orig_end - orig_start, example=example)

    # Coref spans in the same cluster can be nested, e.g. [this [itself]].
    # After normalization, they can create duplicates, e.g. [[itself]].
    # Remove those now.
    unique = {(span.begin, span.end): span for span in self.coref.spans}
    self.coref.spans = unique.values()

    # Include particles in predicates.
    if self.options.particles_in_verbs:
      self._include_particles()


  # Expands SRL predicates to include particles, unless the particle is already
  # a part of an SRL argument span.
  def _include_particles(self):
    tokens = self.tokens.spans
    end = len(tokens) - 1
    ends = set()
    for srl in self.srl:
      covered = set()
      for span in srl.spans:
        if not span.predicate:
          for t in range(span.begin, span.end):
            covered.add(t)
      for span in srl.spans:
        if span.predicate and tokens[span.begin].label.startswith('VB'):
          i = span.end   # token index of the particle, if present
          if i < end and i not in covered and tokens[i].label == 'RP':
            ends.add(span.end)
            span.end += 1
            key = tokens[i - 1].text + '_' + tokens[i].text
            key = key.lower()
            self.summary.normalization.particles.increment(key)

    for s in self.spans(PRED | ARG | COREF):
      if s.end in ends and s.begin == s.end - 1:
        s.end += 1


  # Summarizes CONLL annotation statistics for the current document.
  def _summarize_input(self):
    input_stats = self.summary.input
    input_stats.tokens.increment(len(self.tokens.spans))

    # Constituency histogram.
    for c in self.constituents.spans:
      if len(c.children) > 0:  # ignore leaves (=token constituents)
        input_stats.constituents.increment(c.label)

    # POS tag histogram.
    for s in self.tokens.spans:
      input_stats.postag.increment(s.label)

    # Spans not matching any constituents.
    constituents = self._constituency_map()
    for (span, label) in self.spans(ALL, label=True):
      match = constituents.get((span.begin, span.end))
      if match is None:
        input_stats.no_matching_constituents.increment(label,\
          example=(self.docid, self._phrase(span)))

    # Span -> Type(s).
    span_labels = {}
    _add_label = lambda span, label: \
        span_labels.setdefault((span.begin, span.end), []).append(label)

    # Records the length of 'span' in a histogram, if it is unique.
    spans = {}
    def _record_length(span):
      key = (span.begin, span.end)
      if key in spans:
        return
      spans[key] = True
      input_stats.all_length.increment(span.length())

    # Summarize NER spans.
    for span, source in self.spans(NER, label=True):
      input_stats.ner.increment(span.label)
      input_stats.ner_length.increment(span.length())
      _record_length(span)
      _add_label(span, source)

    # Summarize SRL spans.
    for span, source in self.spans(PRED | ARG, label=True):
      _record_length(span)
      _add_label(span, source)
      if span.predicate:
        input_stats.predicates.increment()
        input_stats.srl_predicate_length.increment(span.length())
      else:
        input_stats.arguments.increment(span.label)
        input_stats.srl_argument_length.increment(span.length())

    # Summarize coref spans.
    input_stats.coref.increment(len(self.coref.spans))
    clusters = {}
    for span, source in self.spans(COREF, label=True):
      _record_length(span)
      _add_label(span, source)
      input_stats.coref_length.increment(span.length())
      clusters.setdefault(span.label, []).append(span)
    input_stats.clusters.increment(len(clusters))
    for _, cluster in clusters.items():
      input_stats.coref_size.increment(len(cluster))

    # Compute exact span overlap statistics.
    for span, t in span_labels.items():
      if len(t) == 1:
        input_stats.exact_overlaps.increment(t[0] + ' alone')
        continue

      t.sort()
      example = None
      mark = False
      if len(t) == 2 and t[0] == 'NER' and t[1] == 'PRED':
        example = self._context(span[0], span[1])
        mark = True
      key = ', '.join(t)
      key = re.sub(r"ARG,.* ARG", ">1 ARGS", key)
      input_stats.exact_overlaps.increment(key, example=example)
      if mark:
        input_stats.exact_overlaps.mark(key)


  # Writes constituent spans to the supplied SLING document.
  def _write_constituents(self, document):
    store = document.frame.store()
    schema = document.schema
    constituency_schema = self.converter.constituency_schema

    spans = self.constituents.spans
    constituents = store.array(len(spans))
    document.frame[constituency_schema.constituents] = constituents

    # Add one frame per constituent.
    for i in range(len(spans)):
      span = spans[i]

      frame = store.frame({schema.isa: constituency_schema.constituent})
      frame[document.schema.phrase_begin] = span.begin
      if span.end > span.begin + 1:
        frame[document.schema.phrase_length] = span.end - span.begin
      frame[constituency_schema.tag] = store[span.label]
      constituents[i] = frame
      span.frame = frame

    # Add parent/child/head information to the constituent frames.
    for span in spans:
      slots = {}
      if span.parent is not None:
        slots[constituency_schema.parent] = span.parent.frame
      if span.head is not None:
        slots[constituency_schema.head] = span.head

      # Omit writing token constituents as they are already covered in
      # the tokens array.
      children = [c for c in span.children if hasattr(c, "frame")]
      if len(children) > 0:
        children_array = store.array(len(children))
        slots[constituency_schema.children] = children_array
        for j, child in enumerate(children):
          children_array[j] = child.frame
      span.frame.extend(slots)

  # Returns a windowed context of a span, e.g.
  # left context [span text] right context.
  def _context(self, begin, end=None, window=3):
    if isinstance(begin, Span):
      end = begin.end
      begin = begin.begin
    elif isinstance(begin, int):
      if end is None:
        end = begin + 1
    assert end is not None
    b = max(0, begin - window)
    e = min(len(self.tokens.spans), end + window)
    output = ''

    # Left context.
    if b < begin:
      for i in range(b, begin):
        t = self.tokens.spans[i]
        if t.brk > sling.NO_BREAK and output != '':
          output += ' '
        output += t.text
      output += ' '

    # Span itself.
    output += '['
    for i in range(begin, end):
      t = self.tokens.spans[i]
      if t.brk > sling.NO_BREAK and i > begin:
        output += ' '
      output += t.text
    output += ']'

    # Right context.
    if end < e:
      for i in range(end, e):
        t = self.tokens.spans[i]
        if t.brk > sling.NO_BREAK and output != '':
          output += ' '
        output += t.text

    return output


  # Returns a span without any context.
  def _phrase(self, begin, end=None):
    return self._context(begin, end, window=0)


  # Returns a span's text with the head marked with brackets.
  def _phrase_with_head(self, s):
    result = []
    for i in range(s.begin, s.end):
      word = self.tokens.spans[i].text
      if i == s.head: word = '[' + word + ']'
      result.append(word)
    return ' '.join(result)


  # Stores a particular frame type to be evoked for a span.
  class FrameType:
    def __init__(self, span, type, refer=None):
      self.span = (span.begin, span.end)   # span itself
      self.type = type                     # frame type
      self.refer = refer                   # referring frame, only for coref
      self.frame = None                    # if already evoked, the frame


  # Writes annotations to SLING document(s).
  # Statistics pertaining to conversion to SLING are written to self.summary.
  #
  # Adds frames using the following heuristic.
  #
  # Notation:
  # - 'Normalizing' a span means:
  #   - Drop trailing possessive if it is the head, if enabled.
  #   - Drop leading articles, if enabled.
  #   - Follow prepositions to objects, if enabled.
  #   - For every SRL Arg and Coref span s:
  #     - Skip s if s is a conjunction and conjunctions are to be skipped.
  #     - [shrink_using_heads]: If s has the same head as a previously
  #       normalized span s' and s' is shorter than s, then s = s'
  #     - [reduce_to_head]: Otherwise s = head(s)
  #   - Expand predicates to include particles, if enabled.
  #
  # - Let 'frame_types' be a map: span -> set of frame type(s) for that span.
  #
  # 0. Add noun phrases as NER spans using NML and NP constituents, if enabled.
  # 1. Normalize all spans.
  # 2. For each NER span s:
  #      frame_types[s].add(s.label)
  #
  # 3.  For each predicate span s:
  #       frame_types[s].add(s.label)
  #
  # 4. For each coref cluster c:
  #      For each span s in c:
  #        If s in map:
  #          cluster_type = first NER or SRL predicate type from frame_types[s]
  #          cluster_head = s
  #      If no head is found for c:
  #        cluster_head = earliest span in c
  #        cluster_type = generic_type (e.g. 'thing')
  #      For each span s in c:
  #        frame_types[s].add(cluster_type)
  #
  # 5. For each SRL arg span s:
  #      If s is not in map:
  #        Add a generic type (e.g. 'thing') for s to frame_types[s].
  #
  # 6. Now that the frame type(s) are determined, start evoking them:
  #    - Evoke all frame type(s) from each cluster head.
  #    - From each non-head coref span:
  #        - Refer the frame of its head corresponding to the cluster type.
  #        - Evoke frame(s) of any other type(s) for the span.
  #    - Evoke frame(s) for each remaining NER span.
  #    - Evoke frame(s) for all SRL spans, and link predicates to arguments.
  def write(self, document):
    store = document.store
    schema = document.schema
    isa = schema.isa

    # Set docid.
    docid = self.docid + ':' + str(self.part)
    if self.options.doc_per_sentence:
      docid += '_s' + str(self.sentence)
    document.frame.extend({'/ontonotes/docid': docid})

    # Write tokens.
    for t in self.tokens.spans:
      token = document.add_token(word=t.text, brk=t.brk)
      token.frame[schema.token_pos] = store['/postag/' + t.label]

    # Write the constituents to the document.
    if not self.options.omit_constituents:
      self._write_constituents(document)

    # Add more noun phrases as extra named entities.
    if self.options.extra_noun_phrases:
      self._add_noun_phrases()

    # Normalize all spans.
    self._normalize()

    mentions = Mentions(document)

    # Span -> list of FrameType objects.
    frame_types = {}

    # Adds 'type' as to the list of frame types for 'span'.
    def _add_type(span, type, refer=None):
      key = (span.begin, span.end)
      if key not in frame_types:
        frame_types[key] = []
      for t in frame_types[key]:
        # See if we are already meant to evoke 'type'.
        if t.type == type:
          if t.refer is None: t.refer = refer
          return t
      f = Annotations.FrameType(span, type, refer)
      frame_types[key].append(f)
      return f

    # Record NER span types.
    for span in self.ner.spans:
      _add_type(span, span.label)

    # Record SRL predicate types.
    for span in self.spans(PRED):
      label = span.label
      if self.options.one_predicate_type:
        label = self.options.generic_predicate_type
      _add_type(span, label)

    # Record coref span types. A coref span takes an existing type
    # from 'frame_types', failing which it takes a generic type.
    coref = {}
    cluster_heads = {}
    for span in self.coref.spans:
      if span.label not in coref:
        coref[span.label] = []   # recall that span.label is the cluster id
      coref[span.label].append(span)

    # Sort each cluster.
    for cluster_id, cluster in coref.items():
      cluster.sort(key=lambda span: span.begin)
      head_type = None
      head = None

      # Try to set the cluster head to a span that evokes a non-generic frame.
      for span in cluster:
        key = (span.begin, span.end)
        if key in frame_types:
          head = span
          head_type = frame_types[key][0]
          break

      # Fallback to the first span, and use a generic type.
      if head is None:
        head = cluster[0]
        head_type = _add_type(head, self.options.backoff_type)
      cluster_heads[cluster_id] = head

      # Ask all non-head spans to refer the head type.
      for span in cluster:
        if span != head:
          _add_type(span, head_type.type, refer=head_type)

    # Record type for each SRL argument span. Again, this will try to
    # use existing non-generic frame types wherever possible.
    for srl in self.srl:
      for span in srl.spans:
        if not span.predicate:
          key = (span.begin, span.end)
          if key not in frame_types:
            _add_type(span, self.options.backoff_type)

    # Evokes/refers frame(s) for 'span'.
    def _evoke(span):
      key = (span.begin, span.end)
      assert key in frame_types
      for t in frame_types[key]:
        if t.frame is not None: continue   # frame already evoked
        mention = mentions.get_or_add(span)
        if t.refer is not None:            # refer existing frame
          assert t.refer.frame is not None, (t, t.refer)
          t.frame = t.refer.frame
          mention.evoke(t.refer.frame)
        else:                              # evoke new frame
          t.frame = mention.evoke_type(store[t.type])

    # Start making spans and evoking frames in the document.
    # Evoke cluster head frames, followed by non-heads.
    for cluster_id, cluster in coref.items():
      head = cluster_heads[cluster_id]
      _evoke(head)
      for span in cluster:
        if span is not head:
          _evoke(span)

    # Evoke frames for all NER spans.
    for ner in self.ner.spans:
      _evoke(ner)

    # Evoke frames for all SRL spans.
    for srl in self.srl:
      predicate_frame = None
      for span in srl.spans:
        if span.predicate:
          _evoke(span)

          # Get the predicate frame.
          types = frame_types[(span.begin, span.end)]
          for f in types:
            if f.refer is None:
              predicate_frame = f.frame
              break
          if predicate_frame is None:
            predicate_frame = types[0].frame
      assert predicate_frame is not None, str(srl.spans)

      # Link predicate frame to argument frames.
      for span in srl.spans:
        if not span.predicate:
          _evoke(span)
          frame = frame_types[(span.begin, span.end)][0].frame
          predicate_frame.extend({span.label: frame})

    document.update()
