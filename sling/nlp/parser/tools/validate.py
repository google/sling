# Copyright 2018 Google Inc.
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

# Validates a corpora of documents for common errors.
# This script can be run via
#   python /path/to/script.py --input=<recordio>  --commons=<commons_file>
# or imported and used via the validate() method.

import sling
import sys

sys.path.insert(0, "sling/nlp/parser/trainer")
import corpora

# Error checking options.
class Options:
  def __init__(self):
    self.stop_on_first_bad_document = False
    self.allow_mentions_without_frames = False
    self.allow_nil_roles = False
    self.allow_nil_values = False
    self.max_error_examples = 3

# Represents a single error and the context it occurs in.
class Error:
  # Error types.
  BAD_SPAN_BEGIN = 0           # span has a bad begin index
  BAD_SPAN_END = 1             # span has a bad end index
  CROSSING_SPAN = 2            # span crosses another span
  MENTION_WITHOUT_FRAME = 3    # mention without evoked frame
  UNTYPED_EVOKED_FRAME = 4     # evoked frame without a type
  FRAME_TYPE_NOT_GLOBAL = 5    # frame type is a local symbol
  ROLE_IS_NONE = 6             # frame with a nil role
  VALUE_IS_NONE = 7            # frame with a nil value
  ROLE_IS_LOCAL = 8            # frame with a role that is a local symbol
  VALUE_NOT_A_FRAME = 9        # frame with a slot value that is not a frame
  FRAME_NOT_LOCAL = 10         # expected a local frame but didn't get one

  def __init__(self, code, document, args):
    self.code = code

    # Document where the error occurs.
    self.document = document
    assert isinstance(self.document, sling.Document), type(self.document)

    # Error-specific context (e.g. mention, frame etc).
    self.args = args

  # Returns [a limited-size prefix of] the document text.
  def _document_text(self, max_tokens=20):
    s = []
    for index, token in enumerate(self.document.tokens):
      if max_tokens >= 0 and index >= max_tokens: break
      if token.brk == 0 and len(s) > 0:
        s[-1] = s[-1] + token.word
      else:
        s.append(token.word)
    if max_tokens >= 0 and len(self.document.tokens) > max_tokens:
      s.append(" ...")
    return ' '.join(s)

  # Returns a string reprsentation of the error.
  def tostr(self, indent=0):
    self.document.decorate()
    output = []
    output.extend(["Document: " + self._document_text()])
    docid = self.document.frame.id
    if docid is not None:
      output.extend(["DocumentId: " + str(docid)])
    output.extend(["DocumentLength: " + str(len(self.document.tokens))])
    if type(self.args[0]) is sling.Mention:
      output.extend(["Mention: " + self.args[0].frame.data(binary=False)])

    if self.code == Error.BAD_SPAN_BEGIN:
      output.extend(["Begin: " + str(self.args[0].begin)])
    elif self.code == Error.BAD_SPAN_END:
      output.extend(["End: " + str(self.args[0].end)])
    elif self.code == Error.CROSSING_SPAN:
      m2 = self.args[1]
      output.extend(["Mention2: " + m2.frame.data(binary=False)])
    elif self.code == Error.UNTYPED_EVOKED_FRAME:
      f = self.args[1]
      output.extend(["UntypedEvokedFrame: " + f.data(binary=False)])
    elif self.code == Error.FRAME_TYPE_NOT_GLOBAL:
      t = self.args[2]
      output.extend(["NonGlobalType: " + t.data(binary=False)])
    elif self.code == Error.ROLE_IS_NONE:
      f = self.args[0]
      output.extend(["FrameWithNilRole: " + f.data(binary=False)])
    elif self.code == Error.VALUE_IS_NONE:
      f = self.args[0]
      output.extend(["FrameWithNilValue: " + f.data(binary=False)])
    elif self.code == Error.ROLE_IS_LOCAL:
      f = self.args[0]
      role = self.args[1]
      output.extend(["Frame: " + f.data(binary=False)])
      output.extend(["LocalRole: " + role.data(binary=False)])
    elif self.code == Error.VALUE_NOT_A_FRAME:
      f = self.args[1]
      output.extend(["NonFrameValue: " + str(f)])
    elif self.code == Error.FRAME_NOT_LOCAL:
      f = self.args[1]
      output.extend(["NonLocalFrame: " + f.data(binary=False)])

    if indent > 0:
      prefix = ' ' * indent
      for i in xrange(len(output)):
        if i > 0: output[i] = prefix + output[i]
    return '\n'.join(output)


# Returns a string representation of the specified error code.
def _codestr(code):
  assert type(code) is int, code
  for c, value in Error.__dict__.iteritems():
    if type(value) is int and value == code and c[0].isupper():
      return c
  return "<UNKNOWN_ERROR:" + str(code) + ">"

# Represents results of checking a corpora for errors.
class Results:
  def __init__(self, options):
    self.error_counts = {}      # error code -> count
    self.error_examples = {}    # error code -> limited no. of examples
    self.options = options      # error checking options

  # Returns whether there were no errors.
  def ok(self):
    return len(self.error_counts) == 0

  # Creates and adds an error with the specified code and context.
  def error(self, code, args):
    document = args[0]
    args = args[1:]
    if code not in self.error_counts:
      self.error_counts[code] = 0
    self.error_counts[code] += 1
    if self.options.max_error_examples >= 0 and \
      self.error_counts[code] <= self.options.max_error_examples:
      error = Error(code, document, args)
      if code not in self.error_examples:
        self.error_examples[code] = []
      self.error_examples[code].append(error)

  # Aggregates the result set in 'other' to this result set.
  def add(self, other):
    for code, count in other.error_counts.iteritems():
      if code not in self.error_counts:
        self.error_counts[code] = 0
        self.error_examples[code] = []
      self.error_counts[code] += count

      num = len(other.error_examples[code])
      current = len(self.error_examples[code])
      if self.options.max_error_examples >= 0 and \
        num + current > self.options.max_error_examples:
        num = self.options.max_error_examples - current
      if num > 0:
        self.error_examples[code].extend(other.error_examples[code][0:num])

  # Returns the string representation of error checking results.
  def __repr__(self):
    if self.ok():
      return "No errors"

    total = 0
    for code, count in self.error_counts.iteritems():
      total += count

    output = []
    output.append("Total " + str(total) + " errors")
    for code, count in self.error_counts.iteritems():
      output.append("  " +  _codestr(code) + " : " + str(count))

    output.extend(["", "EXAMPLES", "-" * 70, ""])
    for code, examples in self.error_examples.iteritems():
      output.append(_codestr(code) + ":")
      for index, example in enumerate(examples):
        indent = len(str(index) + ") ")
        s = str(index) + ") " + example.tostr(indent)
        output.extend([s, ""])
      output.append("")

    return '\n'.join(output)

# Validates 'frame', which is expected to be a local frame, for errors.
# If 'mention' is not None, then 'frame' is one of the evoked frames from it.
# Validation results are added to 'results'
def _validate_frame(document, mention, frame, options, results):
  if type(frame) is not sling.Frame:
    results.error(Error.VALUE_NOT_A_FRAME, [document, mention, frame])
    return

  if not frame.islocal():
    results.error(Error.FRAME_NOT_LOCAL, [document, mention, frame])
    return

  commons = document.store.globals()

  # Check that the frame type is valid.
  t = frame[document.schema.isa]
  if t is None:
    results.error(Error.UNTYPED_EVOKED_FRAME, [document, mention, frame])
  elif t.islocal():
    results.error(Error.FRAME_TYPE_NOT_GLOBAL, [document, mention, frame, t])

  # Check that frame slots are valid.
  for role, value in frame:
    if not options.allow_nil_roles and role is None:
      results.error(Error.ROLE_IS_NONE, [document, frame])
    if not options.allow_nil_values and value is None:
      results.error(Error.VALUE_IS_NONE, [document, frame])
    if role is not None and type(role) is sling.Frame and role.islocal():
      results.error(Error.ROLE_IS_LOCAL, [document, frame, role])
    # TODO: Add support to see if certain slots (e.g. /pb/ARG0) should always
    # have local values, while others (e.g. measure) should always have global
    # values. This can be read from the schema or specified in 'options'.


# Validates 'document' against common errors.
def _validate(document, options):
  results = Results(options)
  length = len(document.tokens)
  for mention in document.mentions:
    begin = mention.begin
    end = mention.end

    # Check span offsets.
    if begin < 0 or begin >= length:
      results.error(Error.BAD_SPAN_BEGIN, [document, mention])
    if end < 0 or end > length:
      results.error(Error.BAD_SPAN_END, [document, mention])

    # Check for crossing spans.
    for m2 in document.mentions:
      if m2.begin < begin: continue  # don't double count crossing spans
      if m2.begin >= end: break   # mentions are sorted
      if m2.begin < begin and m2.end > begin and m2.end < end:
        results.error(Error.CROSSING_SPAN, [document, mention, m2])
      if m2.begin > begin and m2.end > end:
        results.error(Error.CROSSING_SPAN, [document, mention, m2])

    # Check valid evoked frames.
    num_evoked = 0
    for frame in mention.evokes():
      num_evoked += 1
      _validate_frame(document, mention, frame, options, results)

    if not options.allow_mentions_without_frames and num_evoked == 0:
      results.error(Error.MENTION_WITHOUT_FRAME, [document, mention])

  for frame in document.themes:
    _validate_frame(document, None, frame, options, results)

  return results


# Main entry point.
# Checks the corpora in 'recordio_filename' for errors.
def validate(commons, recordio_filename, output_recordio='', options=Options()):
  schema = None
  if type(commons) is not sling.Store:
    assert type(commons) is str
    filename = commons
    commons = sling.Store()
    commons.load(filename)
    schema = sling.DocumentSchema(commons)
    commons.freeze()
  else:
    schema = sling.DocumentSchema(commons)

  corpus = corpora.Corpora(recordio_filename, commons, schema)
  aggregate = Results(options)
  count = 0
  writer = None
  written = 0
  if output_recordio != '':
    writer = sling.RecordWriter(output_recordio)
  for document in corpus:
    count += 1
    results = _validate(document, options)
    aggregate.add(results)
    if not results.ok() and options.stop_on_first_bad_document:
      print "Stopping after first bad document as requested"
      break
    if writer and results.ok():
      writer.write('', document.frame.data(binary=True))
      written += 1

  if writer:
    writer.close()

  return aggregate, count, written


if __name__ == "__main__":
  import sling.flags as flags
  flags.define('--input',
               help='Input recordio',
               default="",
               type=str,
               metavar='CORPUS_RECORDIO')
  flags.define('--commons',
               help='Commons file name',
               default="",
               type=str,
               metavar='COMMONS')
  flags.define('--max_examples',
               help='Max number of examples per error type',
               default=3,
               type=int,
               metavar='MAX_ERROR_EXAMPLES')
  flags.define('--output',
               help='Output recordio file name for valid documents',
               default="",
               type=str,
               metavar='OUTPUT_RECORDIO')
  flags.parse()

  options = Options()
  options.max_error_examples = flags.arg.max_examples
  results, total, written = validate(
      flags.arg.commons, flags.arg.input, flags.arg.output, options)
  print "Went over", total, "documents"
  if flags.arg.output:
    print "Wrote", written, "valid documents to", flags.arg.output
  print results
