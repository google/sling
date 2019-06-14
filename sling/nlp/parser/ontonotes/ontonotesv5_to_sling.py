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

"""Main script for converting .gold_conll files from Ontonotes v5 to SLING.

Sample usage:
ONTO=path/to/ontonotes/data/train/data/english/annotations/
python3 path/to/this/script.py --output=/tmp/output.rec --input=$ONTO

# Also import coref annotations (not enabled by default).
python3 path/to/this/script.py --no_skip_coref --input=$ONTO

# Only converted a handful of files.
python3 path/to/this/script.py --max=100 --input=$ONTO

# Also output conversion summary.
python3 path/to/this/script.py --summary=/tmp/summary.txt --input=$ONTO

# Only process files whose names are whitelisted.
python3 path/to/this/script.py --allowed_ids_file=/path/to/ids

By default, the script performs a series of span normalization steps.
These can be disabled by passing:
  --no_trim_trailing_possessives \
  --no_drop_leading_articles \
  --no_descend_prepositions \
  --no_particles_in_verbs \
  --no_shrink_using_heads \
  --no_reduce_to_head \
  --no extra_noun_phrases

See the Options class below for a complete list of options.
"""

import sling
import sling.flags as flags
from annotations import Annotations
from head_finder import HeadFinder
from statistics import Statistics

# Conversion options.
class Options:
  def __init__(self):
    # Whether to omit writing constituents to the SLING document.
    self.omit_constituents = True

    # Name of the file containing CONLL filenames to process.
    # Official filenames are provided here:
    # http://conll.cemantix.org/2012/download/ids/english/coref/
    #
    # These are supposed to have complete coreference and higher coverage
    # SRL annotations than the other files.
    self.allowed_ids_file = 'local/data/corpora/sempar/train.id'

    # Skip importing coreference annotations or not.
    self.skip_coref = True

    # Whether to output one document per sentence or not.
    # Can only be true if skip_coref is True.
    # If false, then one document per part will be output.
    self.doc_per_sentence = True

    # Span construction and normalization options.
    # Remove trailing possessives, e.g. "China 's" is normalized to "China".
    self.trim_trailing_possessives = True

    # Whether or not to drop leading articles, e.g. the incident -> incident.
    self.drop_leading_articles = True

    # Whether or not to descend from prepositions to their objects.
    # E.g. to travel -> travel.
    self.descend_prepositions = True

    # Whether or not to shrink a span to another one with the same head.
    self.shrink_using_heads = True

    # Whether or not to reduce an unshrinkable span to its head.
    self.reduce_to_head = True

    # Whether or not to generate extra noun phrases using constituency info.
    self.extra_noun_phrases = True

    # When head information can't be computed for a span's normalization,
    # whether or not to use its last token as a proxy for the head.
    self.last_token_as_fallback_head = True

    # Apply shrink/reduce/expansion based heuristics to conjunctions too.
    # E.g. "John and Mary" would normalize to the head "Mary".
    self.normalize_conjunctions = True

    # Whether or not to include particles in SRL predicates.
    self.particles_in_verbs = True

    # Whether to use one generic frame type for all SRL predicates (True)
    # or derive the frame type from the predicate label (False).
    self.one_predicate_type = True

    # Generic predicate type to use if 'one_predicate_type' is True.
    self.generic_predicate_type = "/pb/predicate"

    # Fallback frame type for spans without any non-generic type.
    self.backoff_type = "thing"


# Conversion summary.
# This is composed of three major components -- statistics of the input,
# normalization, and output.
class Summary:
  # Statistics of the CONLL input.
  class Input:
    def __init__(self, statistics):
      # Basic counts.
      self.statistics = statistics
      section = statistics.section("Input Statistics")
      self.files = section.counter("Files Converted")
      self.files_skipped = section.counter("Files Skipped")
      self.parts = section.counter("Parts")
      self.sentences = section.counter("Sentences")
      self.tokens = section.counter("Tokens")
      self.predicates = section.counter("SRL Predicate Spans")
      self.clusters = section.counter("Coreference Clusters")
      self.coref = section.counter("Coreference Member Spans")

      # Constituent tag histogram.
      self.constituents = section.histogram("Constituent Tags")
      self.constituents.set_output_options(extremas=True)

      # Pos tag histogram.
      self.postag = section.histogram("Token POS Tags")

      # NER Span histogram (by type).
      self.ner = section.histogram("NER Spans")

      # SRL Argument Span histogram (by type).
      self.arguments = section.histogram("SRL Argument Spans")

      # Coref cluster size.
      self.coref_size = section.histogram('Coref Cluster Size')
      self.coref_size.set_output_options(sort_by='key', max_output_bins=20)

      # Spans not matching any constituents.
      self.no_matching_constituents = section.histogram(\
        "Spans not matching constituents", max_examples_per_bin=3)

      # Span length histograms.
      section = statistics.section("Input Span Lengths")
      self.ner_length = section.histogram("NER Lengths")
      self.srl_predicate_length = section.histogram("SRL Predicate Lengths")
      self.srl_argument_length = section.histogram("SRL Argument Lengths")
      self.coref_length = section.histogram("Coref Span Lengths")
      self.all_length = section.histogram("All Span Lengths (Unique Spans)")
      for h in [self.ner_length, self.srl_predicate_length, \
                self.srl_argument_length, self.coref_length, self.all_length]:
        h.set_output_options(sort_by='key', max_output_bins=30, extremas=True)

      # Inter-annotation overlap statistics.
      section = statistics.section("Input Inter-annotation Statistics")
      self.exact_overlaps = section.histogram(
        "Exact Input Span Overlaps", max_examples_per_bin=5)
      self.exact_overlaps.set_output_options(sort_by='key')


  # Normalization statistics.
  class Normalization:
    def __init__(self, statistics):
      # Basic counts.
      self.statistics = statistics
      section = statistics.section("Normalization")
      self.ner = section.counter("NER Spans Normalized")
      self.predicates = section.counter("SRL Predicate Spans Normalized")
      self.arguments = section.counter("SRL Argument Spans Normalized")
      self.coref = section.counter("Coreference Spans Normalized")

      # Drill-down into individual normalization steps.
      self.possessives = section.histogram("Drop Trailing Possessives",\
        max_examples_per_bin=3)
      self.articles = section.histogram("Drop Articles", max_examples_per_bin=3)
      self.prep = section.histogram("Prep. Object", max_examples_per_bin=3)
      self.head = section.histogram("Shrunk via Head", max_examples_per_bin=20)
      self.reduced = section.histogram(\
        "Reduced to Head", max_examples_per_bin=3)
      self.none = section.histogram("No normalization", max_examples_per_bin=3)
      for h in [self.head, self.reduced, self.none]:
        h.set_output_options(max_output_bins=50)

      # Noun phrases added.
      self.base_np = section.histogram("Noun Phrases Added via Base NPs",\
        max_examples_per_bin=10)
      self.recursive_np = section.histogram(\
        "Noun Phrases Added via Recursive NPs", max_examples_per_bin=30)
      self.base_nml = section.histogram("Noun Phrases Added via Base NMLs",\
        max_examples_per_bin=10)
      self.nml_titles = section.histogram("Noun Phrases Added via NML Titles",\
        max_examples_per_bin=30)

      self.particles = section.histogram("Verb Particle Inclusion")
      self.particles.set_output_options(max_output_bins=20)

  # Statistics computed on output SLING documents.
  class Output:
    def __init__(self, statistics):
      self.statistics = statistics
      section = statistics.section("Output Counts")
      self.docs = section.counter("Documents")
      self.tokens = section.counter("Tokens")
      self.sentences = section.counter("Sentences")
      self.mentions = section.counter("Mentions")
      self.frames = section.counter("Frames")
      self.constituents = section.counter("Constituents")
      self.cluster_size = section.histogram("Coref Cluster Size")
      self.nesting = section.histogram("Span Nesting Depth", \
        max_examples_per_bin=30)

      section = statistics.section("Output Span Lengths")
      self.thing_length = section.histogram("Spans evoking 'thing'",\
        max_examples_per_bin=3)
      self.non_thing_length = section.histogram("Spans not evoking 'thing'")
      self.all_length = section.histogram("Spans evoking any frame")
      for h in [self.all_length, self.thing_length, self.non_thing_length]:
        h.set_output_options(max_output_bins=20, extremas=True)

      section = statistics.section("Output Frames")
      self.num_evokes = \
        section.histogram("No. of frames evoked from a mention",\
                          max_examples_per_bin=3)
      self.types_evoked = \
        section.histogram("Types of frames evoked from a mention")
      self.types_evoked.set_output_options(sort_by='count', max_output_bins=30)


  def __init__(self):
    statistics = Statistics()
    self.statistics = statistics
    self.input = Summary.Input(statistics)
    self.normalization = Summary.Normalization(statistics)
    self.output = Summary.Output(statistics)

  # Returns a string representation of the statistics.
  def __repr__(self):
    return str(self.statistics)


# Cache for constituency related symbols.
class ConstituencySchema:
  def __init__(self, store):
    self.constituents = store['/constituency/constituents']
    self.constituent = store['/constituency/constituent']
    self.tag = store['/constituency/tag']
    self.parent = store['/constituency/parent']
    self.children = store['/constituency/children']
    self.head = store['/constituency/head']


# Main converter class.
class Converter:
  def __init__(self, commons, summary, options, schema=None):
    self.commons = commons
    if schema is None:
      schema = sling.DocumentSchema(commons)
    self.schema = schema
    self.constituency_schema = ConstituencySchema(commons)
    self.summary = summary
    self.options = options
    self.head_finder = HeadFinder(summary.statistics)

  # Returns SLING document(s) made by converting .conll file 'filename'.
  def tosling(self, filename):
    documents = []
    annotations = Annotations(self)
    input_stats = self.summary.input

    # Callback that will be invoked for each SLING document that is built.
    # This could be for each sentence or each document part, as specified.
    def callback(document):
      documents.append(document)

    with open(filename, "r") as f:
      input_stats.files.increment()
      lines = f.readlines()
      for line in lines:
        annotations.read(line, callback)

    for document in documents:
      self._add_output_statistics(document)

    return documents

  # Returns the POS sequence for the tokens in 'mention'.
  # Multiple consecutive tokens with the same tag T are represented as T+.
  def _pos_sequence(self, document, mention):
    seq = []
    for i in range(mention.begin, mention.end):
      pos = document.tokens[i].frame[document.schema.token_pos].id
      if pos.startswith('/postag/'): pos = pos[8:]
      if len(seq) == 0 or (seq[-1] != pos and seq[-1] != pos + '+'):
        seq.append(pos)
      elif not seq[-1].endswith('+'):
        seq[-1] = seq[-1] + '+'
    return ' '.join(seq)

  # Computes output statistics from 'document'.
  def _add_output_statistics(self, document):
    docid = document.frame["/ontonotes/docid"]
    output = self.summary.output

    # Basic counters.
    output.docs.increment()
    output.tokens.increment(len(document.tokens))
    output.mentions.increment(len(document.mentions))
    for m in document.mentions:
      for frame in m.evokes():
        output.frames.increment()
    for token in document.tokens:
      if token.brk == sling.SENTENCE_BREAK:
        output.sentences.increment()
    if len(document.tokens) > 0:
      output.sentences.increment()  # for the last sentence
    c = self.constituency_schema.constituents
    if c is not None and c in document.frame:
      output.constituents.increment(len(document.frame[c]))

    frame_counts = {}
    for m in document.mentions:
      # Histogram of evoked types.
      num = len([_ for _ in m.evokes()])
      types = [f[self.schema.isa].id for f in m.evokes()]
      types.sort()
      types = ', '.join(types)
      output.types_evoked.increment(types)

      # Histogram of number of evoked frames per mention.
      example=None
      if num > 1:
        example = (docid, document.phrase(m.begin, m.end), types)
      output.num_evokes.increment(num, example=example)

      # Histogram of span lengths.
      length = m.end - m.begin
      output.all_length.increment(length)

      if types == self.options.backoff_type:
        # Histogram of lengths of spans that only evoke the backoff type.
        example=None
        bucket = str(length)
        if length > 1:
          example = (docid, document.phrase(m.begin, m.end))
          bucket += ": " + self._pos_sequence(document, m)
        output.thing_length.increment(bucket, example=example)
      else:
        # Histogram of lengths of all other spans.
        output.non_thing_length.increment(length)

      for f in m.evokes():
        if f not in frame_counts:
          frame_counts[f] = 0
        frame_counts[f] += 1

    # Histogram of coref cluster sizes.
    for _, count in frame_counts.items():
      output.cluster_size.increment(count)

    # Histogram of span nesting depth.
    # Crossing spans are denoted via a special bucket.
    mentions = [m for m in document.mentions]
    mentions.sort(key=lambda m: -m.length)   # longer spans first
    covered = [None] * len(document.tokens)  # token -> deepest span over it
    depths = {}                              # span -> depth
    for mention in mentions:
      depth = 0
      key = (mention.begin, mention.end)
      parent = covered[mention.begin]

      # Since we are iterating over spans in decreasing order of length,
      # all tokens in [begin, end) should be covered by the same span (if any).
      # Otherwise [begin, end) represents a crossing span.
      for token in range(mention.begin, mention.end):
        if covered[token] != parent:
          other = parent if parent is not None else covered[token]
          example = (docid, document.phrase(mention.begin, mention.end), \
            document.phrase(other.begin, other.end))
          output.nesting.increment("CROSSING SPAN", example=example)
          break
      if parent is not None:
        depth = 1 + depths[(parent.begin, parent.end)]
      depths[key] = depth
      for i in range(mention.begin, mention.end):
        covered[i] = mention

    for _, depth in depths.items():
      output.nesting.increment(depth)

# Returns true if 'filename' appears in the list of ids in 'allowed_ids'.
def file_allowed(allowed_ids, filename):
  if len(allowed_ids) == 0:
    return True
  _, sep, suffix = filename.partition('data/english/annotations')
  filename = sep + suffix
  return filename in allowed_ids


if __name__ == "__main__":
  import os
  import sys

  flags.define('--input',
               help='CONLL folder name ending in "annotations"',
               default='',
               type=str)
  flags.define('--output',
               help='Output recordio file',
               default='/tmp/output.rec',
               type=str)
  flags.define('--max',
               help='Maximum number of files to process (-1 for all)',
               default=-1,
               type=int)
  flags.define('--summary',
               help='Output file where the summary will be written.',
               default='',
               type=str)
  flags.define('--constituency_schema',
               help='Constituency schema file (used if writing constituents)',
               default='data/nlp/schemas/constituency.sling',
               type=str)
  options = Options()

  # Also add a flag per conversion option.
  for option, value in options.__dict__.items():
    if type(value) is bool:
      assert value == True, option
      flags.define('--' + option,
                   help=option,
                   dest=option,
                   default=True,
                   action='store_true')
      flags.define('--no_' + option,
                   help=option,
                   dest=option,
                   default=True,
                   action='store_false')
    else:
      flags.define('--' + option,
                   help=option,
                   default=value,
                   type=type(value))
  flags.parse()
  for option, value in flags.arg.__dict__.items():
    if option in options.__dict__:
      options.__dict__[option] = value
      print("Setting option", option, "to", value)

  if options.doc_per_sentence:
    assert options.skip_coref, \
      "Per-sentence documents can only be output without coreference."

  commons = sling.Store()
  schema = sling.DocumentSchema(commons)

  if not flags.arg.omit_constituents:
    assert os.path.exists(flags.arg.constituency_schema)
    commons.load(flags.arg.constituency_schema)

  commons.freeze()
  writer = sling.RecordWriter(flags.arg.output)

  # Read allowed ids, if provided.
  allowed_ids = set()
  if len(options.allowed_ids_file) > 0:
    with open(options.allowed_ids_file, 'r') as f:
      for line in f:
        line = line.strip()
        if not line.endswith('.gold_conll'):
          line += '.gold_conll'
        allowed_ids.add(line)
    print(len(allowed_ids), "allowed filenames read")

  # Convert each file in the specified folder.
  summary = Summary()
  converter = Converter(commons, summary, options, schema)
  for root, dirs, files in os.walk(flags.arg.input):
    for filename in files:
      if flags.arg.max >= 0 and summary.input.files.value > flags.arg.max:
        break

      if filename.endswith(".gold_conll"):
        fullname = os.path.join(root, filename)
        if not file_allowed(allowed_ids, fullname):
          summary.input.files_skipped.increment()
          continue

        documents = converter.tosling(fullname)
        for i, document in enumerate(documents):
          docid = document.frame["/ontonotes/docid"]
          writer.write(docid, document.frame.data(binary=True))
        if summary.input.files.value % 200 == 0:
          print("Processed", summary.input.files.value, "files")

  writer.close()
  print("Wrote", summary.output.docs.value, "docs to", flags.arg.output)

  # Write conversion summary.
  if flags.arg.summary != '':
    with open(flags.arg.summary, 'w') as f:
      summary_str = str(summary)
      f.write("CONVERSION SUMMARY\n\n")
      f.write(summary_str)
    print("Summary written to", flags.arg.summary)

