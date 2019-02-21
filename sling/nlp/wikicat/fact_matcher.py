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
import sling.log as log

from collections import defaultdict
from enum import Enum
from sling.task.workflow import register_task
from util import load_kb


"""
Match-type of a proposed fact vs any existing fact.
"""
class FactMatchType(Enum):
  NEW = 0                   # no fact currently exists for the property
  EXACT = 1                 # proposed fact exactly matches an existing one
  SUBSUMES_EXISTING = 2     # proposed fact is coarser than an existing one
  SUBSUMED_BY_EXISTING = 3  # an existing fact is coarser than the proposed one
  CONFLICT = 4              # for unique-valued pids, conflicts with existing
  ADDITIONAL = 5            # proposed fact would be an extra one


"""
FactMatcher matches a proposed (source_qid, pid, target_qid) fact against
any existing (source_qid, pid, _) facts.

For example, consider the category "Danish cyclists", with a candidate parse:
  country_of_citizenship=QID(Denmark) and sport=QID(cycling).

For this parse, we would separately compute and report fact matching
statistics for each of the two assertions over all members of the category.
"""
class FactMatcher:
  """
  Output of match statistics computation across multiple source_qids.
  This is comprised of a histogram over various FactMatchType buckets,
  and corresponding evidences (if enabled).
  """
  class Output:
    def __init__(self, max_evidences=0):
      self.max_evidences = max_evidences
      self.counts = defaultdict(int)      # match type -> count
      self.evidences = defaultdict(list)  # match type -> evidences

    # Adds a match type with corresponding evidence to the output.
    def add(self, match_type, evidence):
      self.counts[match_type] += 1
      if self.max_evidences < 0 or \
        self.max_evidences > len(self.evidences[match_type]):
          self.evidences[match_type].append(evidence)

    # Saves and returns the histogram as a frame. For the ADDITIONAL bucket,
    # the existing fanout is also saved in the frame. This is because when we
    # propose an ADDITIONAL fact, we may want to compute its likelihood based on
    # how many facts already exist.
    def as_frame(self, store):
      buckets = []
      for match_type, count in self.counts.iteritems():
        buckets.append(store.frame([
          ("match_type", match_type.name),
          ("count", count)]))
      buckets = store.array(buckets)
      frame = store.frame({"buckets": buckets})

      if FactMatchType.ADDITIONAL in self.evidences:
        fanouts = defaultdict(int)
        for _, fanout in self.evidences[FactMatchType.ADDITIONAL]:
          fanouts[fanout] += 1

        fanouts = [\
          store.frame({"fanout": f, "count": fanouts[f]}) \
          for f in sorted(fanouts.keys())]
        frame["fanout_for_additional"] = store.frame(\
          {"bucket": bucket for bucket in fanouts})

      return frame

    # String representation of the histogram.
    def __repr__(self):
      keys = list(sorted(self.counts.keys()))
      kv = [(FactMatchType(k), self.counts[k]) for k in keys]
      return '{%s}' % ', '.join(["%s:%d" % (x[0].name, x[1]) for x in kv])


  def __init__(self, kb, extractor):
    self.kb = kb
    self.extractor = extractor
    self.unique_properties = set()
    self.date_properties = set()

    # Collect unique-valued and date-valued properties.
    # The former will be used to compute CONFLICT counts, and the latter need to
    # be processed in a special manner while matching existing facts.
    constraint_role = kb["P2302"]
    unique = kb["Q19474404"]         # single-value constraint
    w_time = kb["/w/time"]
    for prop in kb["/w/entity"]("role"):
      if prop.target == w_time:
        self.date_properties.add(prop)
      for constraint_type in prop(constraint_role):
        if constraint_type == unique or constraint_type["is"] == unique:
          self.unique_properties.add(prop)
    log.info("%d unique-valued properties" % len(self.unique_properties))
    log.info("%d date-valued properties" % len(self.date_properties))


  # Returns whether 'prop' is a date-valued property.
  def _date_valued(self, prop):
    return prop in self.date_properties


  # Returns existing targets for the given property for the given item.
  # The property could be a pid path.
  def _existing_facts(self, store, item, prop, closure):
    assert type(prop) is list
    pid = prop[0]
    facts = self.extractor.facts_for(store, item, [pid], closure)
    output = []
    for fact in facts:
      if list(fact[:-1]) == prop:
        output.append(fact[-1])
    return output


  # Returns whether 'first' is a finer-precision date than 'second'.
  # 'first' and 'second' should be sling.Date objects.
  def _finer_date(self, first, second):
    if first.precision <= second.precision:
      return False
    if second.precision == sling.MILLENNIUM:
      return first.year >= second.year and first.year < second.year + 1000
    if second.precision == sling.CENTURY:
      return first.year >= second.year and first.year < second.year + 100
    if second.precision == sling.DECADE:
      return first.year >= second.year and first.year < second.year + 10
    if second.precision == sling.YEAR:
      return first.year == second.year
    if second.precision == sling.MONTH:
      return first.year == second.year and first.month == second.month

    # Should not reach here.
    return False


  # Reports match type for the proposed fact (item, prop, value) against
  # any existing fact for the same item and property.
  # 'value' should be a sling.Frame object.
  # 'prop' could be a property (sling.Frame) or a pid-path represented either
  # as a sling.Array or a list.
  #
  # Returns the type of the match along with the corresponding evidence.
  def for_item(self, item, prop, value, store=None):
    assert isinstance(value, sling.Frame)
    if isinstance(prop, sling.Frame):
      prop = [prop]
    else:
      prop = list(prop)

    if store is None:
      store = sling.Store(self.kb)

    # Compute existing facts without any backoff.
    exact_facts = self._existing_facts(store, item, prop, False)
    if len(exact_facts) == 0:
      return (FactMatchType.NEW, item)

    if value in exact_facts:
      return (FactMatchType.EXACT, item)

    # For date-valued properties, existing dates could be int or string
    # (which won't match 'value', which is a sling.Frame). For them, we do a
    # more elaborate matching procedure.
    if self._date_valued(prop[-1]):
      proposed_date = sling.Date(value)
      existing_dates = [sling.Date(e) for e in exact_facts]
      for e in existing_dates:
        if e.value() == proposed_date.value():
          return (FactMatchType.EXACT, item)

    # Check whether the proposed fact subsumes an existing fact.
    closure_facts = self._existing_facts(store, item, prop, True)
    if value in closure_facts:
      return (FactMatchType.SUBSUMES_EXISTING, item)

    # Check whether the proposed fact is subsumed by an existing fact.
    for existing in exact_facts:
      if isinstance(existing, sling.Frame):
        if self.extractor.subsumes(store, prop[-1], existing, value):
          return (FactMatchType.SUBSUMED_BY_EXISTING, (item, existing))

    # Again, dates require special treatment.
    if self._date_valued(prop[-1]):
      for e in existing_dates:
        if self._finer_date(proposed_date, e):
          return (FactMatchType.SUBSUMED_BY_EXISTING, (item, e))

    # Check for conflicts in case of unique-valued properties.
    if len(prop) == 1 and prop[0] in self.unique_properties:
      return (FactMatchType.CONFLICT, (item, exact_facts[0]))

    # Proposed fact is an additional one. Report the existing fanout.
    return (FactMatchType.ADDITIONAL, (item, len(exact_facts)))


  # Same as above, but returns a histogram of match types over multiple items.
  def for_items(self, items, prop, value, store=None, max_evidences=0):
    if store is None:
      store = sling.Store(self.kb)
    output = FactMatcher.Output(max_evidences)
    for item in items:
      (match, evidence) = self.for_item(item, prop, value, store)
      output.add(match, evidence)
    return output


  # Same as above, but returns a match-type histogram separately for each
  # proposed (pid, qid) span in 'parse'.
  def for_parse(self, items, parse, store=None, max_evidences=0):
    if store is None:
      store = sling.Store(self.kb)
    retval = []   # ith entry = match statistics for ith span
    for span in parse.spans:
      output = self.for_items(items, span.pids, span.qid, store, max_evidences)
      retval.append(output)
    return retval


  # Same as above, but returns one list of match-type histograms per parse
  # in 'category'. The list of source items is taken to be the members of
  # 'category'.
  #
  # 'category' is a frame that is produced by the initial stages of the category
  # parsing pipeline (cf. parse_generator.py and prelim_ranker.py).
  #
  # Most parses share a lot of common spans, so this method caches and reuses
  # match statistics for such spans.
  def for_parses(self, category, store=None, max_evidences=0):
    if store is None:
      store = sling.Store(self.kb)

    items = category.members
    output = []    # ith entry = match stats for ith parse
    cache = {}     # (pid, qid) -> match stats
    for parse in category("parse"):
      parse_stats = []
      for span in parse.spans:
        key = (span.pids, span.qid)
        stats = None
        if key in cache:
          stats = cache[key]
        else:
          stats = self.for_items(\
            items, span.pids, span.qid, store, max_evidences)
          cache[key] = stats
        parse_stats.append(stats)
      output.append(parse_stats)
    return output


# Task that adds fact matching statistics to each span in each category parse.
class FactMatcherTask:
  def init(self, task):
    self.kb = load_kb(task)
    self.extractor = sling.FactExtractor(self.kb)
    self.matcher = FactMatcher(self.kb, self.extractor)


  # Runs the task over a recordio of category parses.
  def run(self, task):
    self.init(task)
    reader = sling.RecordReader(task.input("parses").name)
    writer = sling.RecordWriter(task.output("output").name)
    for key, value in reader:
      store = sling.Store(self.kb)
      category = store.parse(value)
      matches = self.matcher.for_parses(category, store, max_evidences=-1)
      frame_cache = {}   # (pid, qid) -> frame containing their match statistics
      for parse, parse_match in zip(category("parse"), matches):
        for span, span_match in zip(parse.spans, parse_match):
          span_key = (span.pids, span.qid)
          if span_key not in frame_cache:
            match_frame = span_match.as_frame(store)
            frame_cache[span_key] = match_frame
          span["fact_matches"] = frame_cache[span_key]
      writer.write(key, category.data(binary=True))
      task.increment("fact-matcher/categories-processed")
    reader.close()
    writer.close()

register_task("category-parse-fact-matcher", FactMatcherTask)


# Loads a KB and brings up a shell to compute and debug match statistics.
def shell():
  kb = load_kb("local/data/e/wiki/kb.sling")
  extractor = sling.api.FactExtractor(kb)
  matcher = FactMatcher(kb, extractor)

  parses = "local/data/e/wikicat/filtered-parses.rec"
  db = sling.RecordDatabase(parses)

  while True:
    item = raw_input("Enter item or category QID:")

    # See if a category QID was entered, if so, compute and output match
    # statistics for all its parses.
    value = db.lookup(item)
    if value is not None:
      store = sling.Store(kb)
      category = store.parse(value)
      output = matcher.for_parses(category, store, max_evidences=4)
      print "%s = %s (%d members)" % \
        (item, category.name, len(category.members))
      for idx, (parse, match) in enumerate(zip(category("parse"), output)):
        print "%d. %s" % (idx, ' '.join(parse.signature))
        for span, span_match in zip(parse.spans, match):
          print "  %s = (%s=%s) : %s" % \
            (span.signature, str(list(span.pids)), span.qid, \
             str(span_match))
        print ""
      print ""
      continue

    item = kb[item]

    pids = raw_input("Enter [comma-separated] pid(s):")
    pids = filter(None, pids.replace(' ', '').split(','))
    for pid in pids:
      assert pid in kb, pid
    pids = [kb[p] for p in pids]

    qid = raw_input("Enter qid:")
    assert qid in kb, qid
    qid = kb[qid]

    output = matcher.for_item(item, pids, qid)
    print item, "(" + item.name + ") :", \
      output[0].name, "evidence: ", output[1]
    print ""

