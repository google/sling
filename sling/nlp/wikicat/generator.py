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

# Task for generating exhaustive parses from a Wikipedia category title.
# A parse is a list of annotated spans over the tokenized category title.
# A span annotation comprises of a chain of pids ending in a QID.

import collections
import copy
import sling
import sling.log as log
import string

from collections import defaultdict
from sling.task.workflow import register_task
from util import load_kb

# A span is a [begin, end) range of tokens in the category title string.
# It is resolved to a list of PIDs and a QID along with the QID's prior
# from the phrase table. This resolution also comes with a count of how
# many members of the category have that (PID-chain, QID) as a fact.
Span = collections.namedtuple('Span', 'begin end qid prior pids count')

# Generates an exhaustive list of parses for a category string.
# Only processes categories that pass some basic checks.
class CategoryParseGenerator:
  # Minimum count below which a (pid, qid) span annotation will be ignored.
  MIN_PID_QID_COUNT = 2

  def lookup(self, name):
    handle = self.kb[name]
    assert handle is not None, "%s not in KB" % name
    return handle

  def init(self, task):
    self.kb = load_kb(task)
    self.names = sling.PhraseTable(self.kb, task.input("phrase-table").name)

    self.min_members = int(task.param("min_members"))
    self.num_parses_bins = [1, 2, 3, 5, 10, 20, 50, 100, 200]

    # Lookup some handles in advance.
    self.h_language =  self.lookup("/lang/" + task.param("language"))
    self.h_lang = self.lookup("lang")
    self.main_topic = self.lookup("P301")   # present in topical categories
    self.h_member = self.lookup("/w/item/member")
    self.h_instanceof = self.lookup('P31')
    self.h_subclassof = self.lookup('P279')
    self.h_category = self.lookup('Q4167836')
    self.h_category_contains = self.lookup('P4224')
    self.english = task.param("language") == "en"

    # The following kinds of categories won't be processed.
    self.uninteresting_categories = set([
        self.lookup('Q20769287'),  # disambiguation category
        self.lookup('Q15407973'),  # list category
        #self.lookup('Q56428020'),  # template category
        self.lookup('Q23894233'),  # stub category
        self.lookup('Q24046192'),  # admin category
        self.lookup('Q15647814'),  # user category
        self.lookup('Q20010800'),  # user language category
        self.lookup('Q30432511'),  # meta category
        self.lookup('Q13331174')   # navbox category
    ])

    # These pids will not be considered as resolution for spans.
    self.pids_to_ignore = set([
        self.h_instanceof,         # P31 = instance of
        self.lookup('P279'),       # P279 = subclass of
        self.lookup('P971'),       # P971 = category combines topics
        self.lookup('P4224'),      # P4224 = category contains
    ])

    # These QIDs will not be considered as resolutions for spans.
    self.base_qids = set([
        self.lookup('Q5'),         # human
        self.lookup('Q215627'),    # person
        self.lookup('Q17334923'),  # location
        self.lookup('Q811430'),    # construction
        self.lookup('Q43229'),     # organization
        self.lookup('Q2385804'),   # educational institution
        self.lookup('Q294163'),    # public institution
        self.lookup('Q15401930'),  # product
        self.lookup('Q12737077'),  # occupation
        self.lookup('Q192581'),    # job
        self.lookup('Q4164871'),   # position
        self.lookup('Q216353')     # title
    ])
    self.extractor = sling.api.FactExtractor(self.kb)


  # Returns whether the item is a category.
  def is_category(self, frame):
    return self.h_category in frame(self.h_instanceof)


  # Returns the category title in the specified language.
  def get_title(self, frame):
    for alias in frame("alias"):
      if alias[self.h_lang] == self.h_language:
        return alias.name
    return None


  # Returns true if the category with QID 'category_qid' and members
  # 'category_members' should be rejected from processing or not.
  # If true, also returns a corresponding reason.
  def reject(self, category_qid, category_frame, category_members):
    if self.main_topic in category_frame:
      return (True, "topical")
    if self.min_members >= 0 and len(category_members) < self.min_members:
      return (True, "very_few_members")

    for category_type in category_frame(self.h_instanceof):
      if category_type in self.uninteresting_categories:
        return (True, "uninteresting_category_type")

    title = self.get_title(category_frame)
    if title is None:
      return (True, "no_title_in_language")
    if self.english and title.find("stub") != -1:
      return (True, "stub")

    return (False, "")


  # Returns counts of (QID, PID-chain) for all (PID-chain, QID) facts across
  # all members. (PID-chain, QID) facts that occur multiple times in a single
  # member are counted only once.
  def qid_pid_counts(self, store, members):
    qp_counts = defaultdict(lambda: defaultdict(int))
    seen = set()         # (PID, QID) seen in one member
    for member in members:
      facts = self.extractor.facts(store, member)
      seen.clear()
      for fact in facts:
        if fact in seen:
          continue
        seen.add(fact)
        qid = fact[-1]     # fact = sequence of PIDs followed by a QID
        pids = tuple(fact[:-1])
        qp_counts[qid][pids] += 1

    return qp_counts


  # Computes and returns all spans in the tokenized category title document,
  # that can be resolved to (QID, PID-chain) entries in 'qp_counts'.
  #
  # Spans are reported as a map: token i -> all spans that start at token i.
  def compute_spans(self, document, qp_counts):
    tokens = document.tokens
    size = len(tokens)
    begin_to_spans = [[] for _ in range(size)]
    for begin in range(size):
      # Ignore spans starting in punctuation.
      begin_word = tokens[begin].word
      if begin_word in string.punctuation and begin_word != "(":
        continue

      for end in range(begin + 1, size + 1):
        # Ignore spans ending in punctuation.
        end_word = tokens[begin].word
        if end_word in string.punctuation and end_word != ")":
          continue

        # Special case: allow balanced parentheses.
        if begin_word == "(" and \
          all([t.word != ")" for t in tokens[begin + 1: end]]):
          continue
        if end_word == ")" and \
          all([t.word != "(" for t in tokens[begin: end - 1]]):
          continue

        phrase = document.phrase(begin, end)
        matches = self.names.query(phrase)

        # Also lemmatize plurals.
        if self.english:
          original = phrase
          if phrase.endswith("ies"): phrase = phrase[0:-3] + "y"
          elif phrase.endswith("es"): phrase = phrase[0:-2]
          elif phrase.endswith("s"): phrase = phrase[0:-1]

          if original != phrase:
            # Add more matches.
            existing = set([m.item() for m in matches])
            more_matches = self.names.query(phrase)
            for m in more_matches:
              if m.item() not in existing:
                matches.append(m)

        if len(matches) == 0:
          continue

        total_denom = 1.0 / sum([m.count() for m in matches])
        for match in matches:
          qid = match.item()
          prior = match.count() * total_denom
          if qid in qp_counts:
            for pids, count in qp_counts[qid].items():
              # Ignore low frequency (pid, qid) pairs.
              if count < CategoryParseGenerator.MIN_PID_QID_COUNT:
                continue

              span = Span(begin, end, match.item(), prior, pids, count)
              begin_to_spans[begin].append(span)
    return begin_to_spans


  # Takes all spans and constructs maximal parses from them. Each parse only
  # contains non-overlapping spans.
  #
  # This is computed recursively from right to left.
  # Base case: No parses can start at the end token.
  # Case 1: If no parses start at 'begin' then return all parses
  #         starting at begin + 1.
  # Case 2: If spans s_1, ..., s_k start at 'begin', then return
  #         \Union_i ({s_i} \union ParsesStartingAt(s_i.end))
  def construct_parses(self, begin_to_spans):
    end = len(begin_to_spans)
    parses = {}          # i -> parses starting at token i
    parses[end] = [[]]   # no parses can start after the last token

    for begin in range(end - 1, -1, -1):
      if len(begin_to_spans[begin]) == 0:
        # No spans start at 'begin', so report parses starting at begin + 1.
        parses[begin] = copy.copy(parses[begin + 1])
      else:
        parses[begin] = []
        for span in begin_to_spans[begin]:
          for parse in parses[span.end]:
            parse = copy.copy(parse)
            parse.append(span)
            parses[begin].append(parse)

    # Reverse the spans in each full parse. This will order them by tokens.
    parses[0] = [list(reversed(p)) for p in parses[0]]
    return parses[0]


  # Returns a string representation of a parse.
  def parse_to_str(self, parse):
    output = []
    for span in parse:
      output.append(str(span.pids) + ':' + str(span.qid))
    return ' '.join(output)


  # Returns true if the given span should be dropped.
  def skip_span(self, span):
    if span.qid in self.base_qids:
      return True
    for pid in span.pids:
      if pid in self.pids_to_ignore:
        return True
    return False


  # Post-processes parses by dropping some spans.
  def post_process(self, parses):
    output = []
    seen = set()
    for parse in parses:
      new_parse = []
      for span in parse:
        if not self.skip_span(span):
          new_parse.append(span)
      if len(new_parse) == 0:
        continue
      if len(new_parse) != len(parse):
        # Dropping spans might lead to duplicate parses, so dedup them.
        s = self.parse_to_str(new_parse)
        if s not in seen:
          output.append(new_parse)
          seen.add(s)
      else:
        output.append(parse)
    return output


  # Returns category members that (a) are not categories themselves,
  # (b) satisfy any category_contains property of the category.
  def get_members(self, frame):
    members = [m for m in frame(self.h_member) if not self.is_category(m)]

    if self.h_category_contains not in frame:
      # No other constraint to check.
      return members

    store = frame.store()
    allowed = set([store.resolve(c) for c in frame(self.h_category_contains)])
    output = []
    for member in members:
      valid = False
      for value in member(self.h_instanceof):
        # Fast-check for whether 'value' satisfies category_contains.
        if value in allowed:
          valid = True
          break

        # Slow-check for whether 'value' satisfies category_contains.
        for a in allowed:
          if self.extractor.in_closure(self.h_subclassof, a, value):
            valid = True
            break
        if valid:
          break
      if valid:
        output.append(member)
    return output


  # Runs the parse generation task.
  def run(self, task):
    self.init(task)

    writer = sling.RecordWriter(task.output("output").name)
    rejected = sling.RecordWriter(task.output("rejected").name)
    inputs = [t.name for t in task.inputs("items")]

    for filename in inputs:
      reader = sling.RecordReader(filename)
      for index, (key, value) in enumerate(reader):
        store = sling.Store(self.kb)
        frame = store.parse(value)

        # Only process category items.
        if not self.is_category(frame):
          rejected.write(key, "not_category")
          continue

        # See if the category should be skipped.
        members = self.get_members(frame)
        reject, reason = self.reject(key, frame, members)
        if reject:
          task.increment("skipped_categories/" + reason)
          rejected.write(key, reason)
          continue

        # First, collect the targets of all facts of all category members.
        qp_counts = self.qid_pid_counts(store, members)

        # Next, tokenize the category title.
        title = self.get_title(frame)
        colon = title.find(':')
        title = title[colon + 1:]
        document = sling.tokenize(title, store)

        # Next, find matches for all spans. These are reported as a list,
        # where ith item = spans that begin at token i (possibly an empty list).
        begin_to_spans = self.compute_spans(document, qp_counts)

        # Construct maximal parses with non-overlapping spans.
        parses = self.construct_parses(begin_to_spans)

        # Post-process parses.
        parses = self.post_process(parses)
        if len(parses) == 0 or len(parses) == 1 and len(parses[0]) == 0:
          task.increment("skipped_categories/no_parses")
          rejected.write(key, "no_parses")
          continue

        # Write parses as frames.
        frame = store.frame({"name": title, "members": members})
        frame["document"] = document.frame
        for parse in parses:
          span_array = store.array(len(parse))
          for i, span in enumerate(parse):
            span_array[i] = store.frame({
                "begin": span.begin, "end": span.end, "qid": span.qid,
                "prior": span.prior, "pids": list(span.pids),
                "count": span.count
            })
          parse_frame = store.frame({"spans": span_array})
          frame.append("parse", parse_frame)
        writer.write(key, frame.data(binary=True))
        task.increment("categories_accepted")

        # Compute histogram over number of parses.
        for b in self.num_parses_bins:
          if len(parses) <= b:
            task.increment("#parses <= %d" % b)
        if self.num_parses_bins[-1] < len(parses):
          task.increment("#parses > %d" % self.num_parses_bins[-1])

      reader.close()
    writer.close()
    rejected.close()


register_task("category-parse-generator", CategoryParseGenerator)
