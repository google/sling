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
    self.head = None     # head token index

  # Returns whether this is a leaf span.
  def leaf(self):
    return len(self.children) == 0

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

  # Whether the span is an SRL predicate.
  def is_predicate(self):
    return self.label.startswith("/pb/pred/")

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
          if label == 'V':
            assert fields[6] != '-'
            label = "/pb/pred/" + fields[6]  # predicate prefix, e.g. 'live'
            if fields[7] != "-":
              label += "-" + fields[7]       # predicate suffix, e.g. '01'
          srl_annotation.singleton(token_index, label)
        elif beginning:           # e.g. (ARG2*
          assert srl[-1] == '*'
          label = srl[1:-1]
          assert label != 'V'     # predicates can't be multi-token
          srl_annotation.start(token_index, label)
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
    for _ in xrange(num_srl_predicates):
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
        for i in xrange(begin, child.begin):
          s = self.tokens.spans[i]
          children.append(s)
          s.parent = constituent
        children.append(child)
        begin = constituent.end
      for i in xrange(begin, constituent.end):
        s = self.tokens.spans[i]
        children.append(s)
        s.parent = constituent
      constituent.children = children


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

    for _ in xrange(len(parse_bit[index + 1:])):
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

    while changed and begin < end - 1:
      changed = False
      histogram = None
      if self.options.drop_leading_articles and \
        tokens[begin].label == 'DT' and \
        not tokens[begin].text[0].isupper():
          histogram = norm_summary.articles
          changed = True
      elif self.options.descend_prepositions:
        lowest = constituents.get((begin, end), None)
        if lowest is not None and lowest.label == 'PP' and \
          lowest.head == begin and tokens[begin].label in ['IN', 'TO'] and \
          (begin + 1, end) in constituents:
            histogram = norm_summary.prep
            changed = True
      if changed:
        if key is None: key = span.label
        e = None
        if example: e = self._context(begin, end, window=0)
        histogram.increment(key, example=e)
        begin += 1
 
    # Set new span boundaries.
    span.begin = begin
    span.end = end
    if changed:
      norm_summary.changed.increment()
    else:
      norm_summary.unchanged.increment()
    return changed


  # Normalizes all spans.
  def _normalize(self):
    norm_summary = self.summary.normalization
    constituents = self._constituency_map()
    tokens = self.tokens.spans

    # Normalize NER spans.
    for span in self.ner.spans:
      if self._normalize_span(span, constituents, example=True):
        norm_summary.ner.increment()

    # Normalize SRL spans.
    for srl in self.srl:
      for span in srl.spans:
        example = not span.is_predicate()
        if self._normalize_span(span, constituents, example=example):
          if span.is_predicate():
            norm_summary.predicates.increment()
          else:
            norm_summary.arguments.increment()
    
    # Normalize Coref spans.
    for span in self.coref.spans:
      if self._normalize_span(span, constituents, key="Coref", example=False):
          norm_summary.coref.increment()

    # Shrink-using-head heuristic.
    if self.options.shrink_using_heads:
      # Collect head -> span mapping for NER and SRL predicate spans.
      heads = {}
      for span in self.ner.spans:
        lowest = constituents.get((span.begin, span.end), None)
        if lowest is None or lowest.head is None:
          continue
        heads[lowest.head] = (span, "NER")
      for srl in self.srl:
        for span in srl.spans:
          if not span.is_predicate(): continue
          lowest = constituents.get((span.begin, span.end), None)
          if lowest is None or lowest.head is None:
            continue
          heads[lowest.head] = (span, "PRED")

      # Try to align SRL argument spans to other spans with the same head.
      for srl in self.srl:
        for span in srl.spans:
          if span.is_predicate():
            continue
          lowest = constituents.get((span.begin, span.end), None)

          # Only normalize if this span is not a conjunction and the matching
          # span is shorter.
          if lowest is None or lowest.head is None or lowest.is_conjunction():
            continue
          if lowest.head in heads:
            match = heads[lowest.head][0]
            if match.length() < span.length():
              span.begin = match.begin
              span.end = match.end
              norm_summary.head.increment("ARG -> " + heads[lowest.head][1])
          else:
            heads[lowest.head] = (span, "ARG")

      # Try to align coref spans to other spans with the same head.
      for span in self.coref.spans:
        lowest = constituents.get((span.begin, span.end), None)
        if lowest is None or lowest.head is None or lowest.is_conjunction():
          continue
        if lowest.head in heads:
          match = heads[lowest.head][0]
          if match.length() < span.length():
            span.begin = match.begin
            span.end = match.end
            norm_summary.head.increment("COREF -> " + heads[lowest.head][1])
           
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
    for srl in self.srl:
      covered = set()
      for span in srl.spans:
        if not span.is_predicate():
          for t in xrange(span.begin, span.end):
            covered.add(t)
      for span in srl.spans:
        if span.is_predicate():
          i = span.end   # token index of the particle, if present
          if i < end and i not in covered and tokens[i].label == 'RP':
            span.end += 1
            key = tokens[i - 1].text + '_' + tokens[i].text
            key = key.lower()
            self.summary.normalization.particles.increment(key)

 
  # Summarizes CONLL annotation statistics for the current document.
  def _summarize_input(self):
    input_stats = self.summary.input
    input_stats.tokens.increment(len(self.tokens.spans))
    for c in self.constituents.spans:
      if len(c.children) > 0:  # ignore leaves (=token constituents)
        input_stats.constituents.increment(c.label)

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
    for span in self.ner.spans:
      input_stats.ner.increment(span.label)
      input_stats.ner_length.increment(span.length())
      _record_length(span)
      _add_label(span, 'NER')

    # Summarize SRL spans.
    for srl in self.srl:
      for span in srl.spans:
        if span.is_predicate():
          input_stats.predicates.increment()
          input_stats.srl_predicate_length.increment(span.length())
          _record_length(span)
          _add_label(span, 'Pred')
        else:
          input_stats.arguments.increment(span.label)
          input_stats.srl_argument_length.increment(span.length())
          _record_length(span)
          _add_label(span, 'Arg')

    # Summarize coref spans.
    input_stats.coref.increment(len(self.coref.spans))
    clusters = {}
    for span in self.coref.spans:
      _record_length(span)
      _add_label(span, 'Coref')
      input_stats.coref_length.increment(span.length())
      clusters.setdefault(span.label, []).append(span)
    input_stats.clusters.increment(len(clusters))
    for _, cluster in clusters.iteritems():
      input_stats.coref_size.increment(len(cluster))

    # Compute exact span overlap statistics.
    for span, t in span_labels.iteritems():
      if len(t) == 1:
        input_stats.exact_overlaps.increment(t[0] + ' alone')
        continue

      t.sort()
      example = None
      mark = False
      if len(t) == 2 and t[0] == 'NER' and t[1] == 'Pred':
        example = self._context(span[0], span[1])
        mark = True
      key = ', '.join(t)
      key = re.sub(r"Arg,.* Arg", ">1 Args", key)
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
    for i in xrange(len(spans)):
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
      for i in xrange(b, begin):
        t = self.tokens.spans[i]
        if t.brk > sling.NO_BREAK and output != '':
          output += ' '
        output += t.text
      output += ' '

    # Span itself.
    output += '['
    for i in xrange(begin, end):
      t = self.tokens.spans[i]
      if t.brk > sling.NO_BREAK and i > begin:
        output += ' '
      output += t.text
    output += ']'

    # Right context.
    if end < e:
      for i in xrange(end, e):
        t = self.tokens.spans[i]
        if t.brk > sling.NO_BREAK and output != '':
          output += ' '
        output += t.text

    return output


  # Returns a span without any context.
  def _phrase(self, begin, end=None):
    return self._context(begin, end, window=0)

  
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
  # - Let 'frame_types' be a map: span -> set of frame type(s) for that span.
  # - 'Normalizing' means:
  #   - Dropping leading articles, if enabled.
  #   - Following prepositions to objects, if enabled.
  #   - Expanding predicates to include particles, if enabled.
  #   - Aligning SRL Arg and Coref spans to other spans with the same head,
  #     if enabled (with the exception of conjunctions).
  #
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
    document.frame.extend({'id': docid})

    # Write tokens.
    for t in self.tokens.spans:
      token = document.add_token(word=t.text, brk=t.brk)
      token.frame[schema.token_pos] = store['/postag/' + t.label]

    # Write the constituents to the document.
    if not self.options.omit_constituents:
      self._write_constituents(document)

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
    for srl in self.srl:
      for span in srl.spans:
        if span.is_predicate():
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
    for cluster_id, cluster in coref.iteritems():
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
        if not span.is_predicate():
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
    for cluster_id, cluster in coref.iteritems():
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
        if span.is_predicate():
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
        if not span.is_predicate():
          _evoke(span)
          frame = frame_types[(span.begin, span.end)][0].frame
          predicate_frame.extend({span.label: frame})

    document.update()
