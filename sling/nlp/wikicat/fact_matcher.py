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
import sys

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
  # For CONFLICT, SUBSUMES_EXISTING, and EXACT, we store only a handful
  # of source items. For the rest, we store all source items.
  MAX_SOURCE_ITEMS = 10

  TYPES_WITH_EXEMPLARS_ONLY = set([
      FactMatchType.EXACT,
      FactMatchType.CONFLICT,
      FactMatchType.SUBSUMES_EXISTING
  ])


  """
  Output of match statistics computation across multiple source_qids.
  This is comprised of a histogram over various FactMatchType buckets,
  and corresponding source qids.
  """
  class Output:
    def __init__(self):
      self.counts = defaultdict(int)         # match type -> count
      self.source_items = defaultdict(list)  # match type -> [some] source items

    # Adds a match type with corresponding source item to the output.
    def add(self, match_type, source_item):
      self.counts[match_type] += 1
      if match_type not in FactMatcher.TYPES_WITH_EXEMPLARS_ONLY or \
          len(self.source_items[match_type]) < FactMatcher.MAX_SOURCE_ITEMS:
          self.source_items[match_type].append(source_item)

    # Saves and returns the histogram as a frame.
    def as_frame(self, store):
      buckets = []
      for match_type, count in self.counts.items():
        bucket_frame = store.frame([
          ("match_type", match_type.name),
          ("count", count),
          ("source_items", self.source_items[match_type])])
        if match_type in self.source_items:
          items = []
        buckets.append(bucket_frame)
      buckets = store.array(buckets)
      frame = store.frame({"buckets": buckets})

      return frame


    # String representation of the histogram.
    def __repr__(self):
      keys = list(sorted(self.counts.keys()))
      kv = [(k, self.counts[k]) for k in keys]
      return '{%s}' % ', '.join(["%s:%d" % (x[0].name, x[1]) for x in kv])


  def __init__(self, kb, extractor):
    self.kb = kb
    self.extractor = extractor
    self.unique_properties = set()
    self.date_properties = set()
    self.location_properties = set()

    # Collect unique-valued, date-valued, and location-valued properties.
    # The former will be used to compute CONFLICT counts, and the latter need to
    # be processed in a special manner while matching existing facts.
    constraint_role = kb["P2302"]
    unique = kb["Q19474404"]         # single-value constraint
    w_time = kb["/w/time"]
    w_item = kb["/w/item"]
    p_subproperty_of = kb["P1647"]
    p_location = kb["P276"]
    for prop in kb["/w/entity"]("role"):
      if prop.target == w_time:
        self.date_properties.add(prop)
      if prop.target == w_item:
        for role, value in prop:
          if role == p_subproperty_of:
            if kb.resolve(value) == p_location:
              self.location_properties.add(prop)
      for constraint_type in prop(constraint_role):
        if constraint_type == unique or constraint_type["is"] == unique:
          self.unique_properties.add(prop)

    log.info("%d unique-valued properties" % len(self.unique_properties))
    log.info("%d date-valued properties" % len(self.date_properties))
    log.info("%d location-valued properties" % len(self.location_properties))

    # Set closure properties.
    self.closure_properties = {}
    self.p_subclass = kb["P279"]
    self.p_parent_org = kb["P749"]
    p_located_in = kb["P131"]
    for p in self.location_properties:
      self.closure_properties[p] = p_located_in

    # 'Educated at' -> 'Part of'.
    self.closure_properties[kb["P69"]] = kb["P361"]


  # Returns whether 'prop' is a date-valued property.
  def _date_valued(self, prop):
    return prop in self.date_properties


  # Returns existing targets for the given property for the given item.
  # The property could be a pid path.
  def _facts_without_backoff(self, store, item, prop):
    assert type(prop) is list
    pid = prop[0]
    facts = self.extractor.facts_for(store, item, [pid], False)
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


  # Returns if 'coarse' subsumes 'fine' by following 'closure_property' edges.
  def subsumes(self, store, closure_property, coarse, fine):
    if coarse == fine:
      return True

    if closure_property is not None:
      return self.extractor.in_closure(closure_property, coarse, fine)
    else:
      return self.extractor.in_closure(self.p_subclass, coarse, fine) \
          or self.extractor.in_closure(self.p_parent_org, coarse, fine)


  # Returns a FactMatchType value depending on how 'proposed' compares against
  # 'existing' as a target of the property 'prop'.
  # 'existing' should be a list of existing target(s) (possibly empty).
  def match_type(self, store, prop, existing, proposed):
    existing = [self.kb.resolve(e) for e in existing]
    proposed = self.kb.resolve(proposed)

    if len(existing) == 0:
      return FactMatchType.NEW

    exact = False
    subsumes = False
    subsumed = False

    # For date-valued properties, existing dates could be int or string
    # (which won't match 'proposed', which could be a sling.Frame).
    # For them, we do a more elaborate matching procedure.
    if self._date_valued(prop):
      existing_dates = [sling.Date(e) for e in existing]
      proposed_date = sling.Date(proposed)
      for e in existing_dates:
        exact |= e.value() == proposed_date.value()
        subsumes |= self._finer_date(e, proposed_date)
        subsumed |= self._finer_date(proposed_date, e)
    else:
      closure_property = self.closure_properties.get(prop, None)
      for e in existing:
        exact |= e == proposed
        if isinstance(e, sling.Frame):
          subsumes |= self.subsumes(store, closure_property, proposed, e)
          subsumed |= self.subsumes(store, closure_property, e, proposed)

    if exact: return FactMatchType.EXACT
    if subsumes: return FactMatchType.SUBSUMES_EXISTING
    if subsumed: return FactMatchType.SUBSUMED_BY_EXISTING
    if prop in self.unique_properties: return FactMatchType.CONFLICT
    return FactMatchType.ADDITIONAL


  # Reports match type for the proposed fact (item, prop, value) against
  # any existing fact for the same item and property.
  # 'value' should be a sling.Frame object.
  # 'prop' could be a property (sling.Frame) or a pid-path represented either
  # as a sling.Array or a list.
  #
  # Returns the type of the match.
  def for_item(self, item, prop, value, store=None):
    assert isinstance(value, sling.Frame)
    if isinstance(prop, sling.Frame):
      prop = [prop]
    else:
      prop = list(prop)

    if store is None:
      store = sling.Store(self.kb)

    # Compute existing facts without any backoff.
    existing = self._facts_without_backoff(store, item, prop)

    return self.match_type(store, prop[-1], existing, value)


  # Same as above, but returns a histogram of match types over multiple items.
  def for_items(self, items, prop, value, store=None):
    if store is None:
      store = sling.Store(self.kb)
    output = FactMatcher.Output()
    for item in items:
      match = self.for_item(item, prop, value, store)
      output.add(match, item)
    return output


  # Same as above, but returns one list of match-type histograms per parse
  # in 'category'. The list of source items is taken to be the members of
  # 'category'.
  #
  # 'category' is a frame that is produced by the initial stages of the category
  # parsing pipeline (cf. parse_generator.py and prelim_ranker.py).
  #
  # Most parses share a lot of common spans, so this method caches and reuses
  # match statistics for such spans.
  def for_parses(self, category, store=None):
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
          stats = self.for_items(items, span.pids, span.qid, store)
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
      matches = self.matcher.for_parses(category, store)
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
      output = matcher.for_parses(category, store)
      print("%s = %s (%d members)" % \
            (item, category.name, len(category.members)))
      for idx, (parse, match) in enumerate(zip(category("parse"), output)):
        print("%d. %s" % (idx, ' '.join(parse.signature)))
        for span, span_match in zip(parse.spans, match):
          print("  %s = (%s=%s) : %s" % \
                (span.signature, str(list(span.pids)), span.qid, \
                 str(span_match)))
        print()
      print()
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
    print(item, "(" + item.name + ") :", output.name)
    print()


# Runs a few hard-coded fact-matching tests.
def test_fact_matcher():
  RED = "\033[1;31m"
  GREEN = "\033[0;32m"
  RESET = "\033[0;0m"

  def error(entry, message):
    sys.stdout.write(RED)
    print("[FAILED] ", end='')
    sys.stdout.write(RESET)
    print(entry, ":", message)

  def success(entry):
    sys.stdout.write(GREEN)
    print("[SUCCESS] ", end='')
    sys.stdout.write(RESET)
    print(entry)

  kb = load_kb("local/data/e/wiki/kb.sling")
  extractor = sling.api.FactExtractor(kb)
  matcher = FactMatcher(kb, extractor)

  # Test cases.
  tuples = []

  # Adds the given test case and its reverse test case too (if possible).
  def add(pid, existing, proposed, match_type):
    tuples.append((pid, existing, proposed, match_type))

    # Add the reverse case.
    if match_type != FactMatchType.NEW and existing != proposed:
      rev_type = match_type
      if match_type == FactMatchType.SUBSUMED_BY_EXISTING:
        rev_type = FactMatchType.SUBSUMES_EXISTING
      if match_type == FactMatchType.SUBSUMES_EXISTING:
        rev_type = FactMatchType.SUBSUMED_BY_EXISTING
      tuples.append((pid, proposed, existing, rev_type))

  # Place of birth, Kapiolani Medical Center, Honolulu.
  add("P19", "Q6366688", "Q18094", FactMatchType.SUBSUMES_EXISTING)

  # Place of birth, Kapiolani Medical Center, US.
  add("P19", "Q6366688", "Q30", FactMatchType.SUBSUMES_EXISTING)

  # Place of birth, <no existing value>, US.
  add("P19", "", "Q30", FactMatchType.NEW)

  # Place of birth, US, US.
  add("P19", "Q30", "Q30", FactMatchType.EXACT)

  # Place of birth, Honolulu, Chicago.
  add("P19", "Q18094", "Q1297", FactMatchType.CONFLICT)

  # Children, Malia Obama, Sasha Obama.
  add("P40", "Q15070044", "Q15070048", FactMatchType.ADDITIONAL)

  # Date-valued properties: int values.
  # Note: P585 = point in time (unique valued), P580 = start time (non unique)
  add("P585", 1961, 19610804, FactMatchType.SUBSUMED_BY_EXISTING)
  add("P585", 1961, 196108, FactMatchType.SUBSUMED_BY_EXISTING)
  add("P585", 1961, 1961, FactMatchType.EXACT)
  add("P585", 1961, 196, FactMatchType.SUBSUMES_EXISTING) # 196 = 196X (decade)
  add("P585", 1961, 19, FactMatchType.SUBSUMES_EXISTING)  # 19 = 19XX (century)
  add("P585", 1961, 1, FactMatchType.SUBSUMES_EXISTING)   # 1 = 1XXX (millenium)
  add("P585", 1962, 19610804, FactMatchType.CONFLICT)
  add("P585", 1962, 196108, FactMatchType.CONFLICT)
  add("P585", 1962, 1961, FactMatchType.CONFLICT)
  add("P580", 1961, 1962, FactMatchType.ADDITIONAL)

  # Date-valued properties: string values.
  add("P585", "1961", "1961-08-04", FactMatchType.SUBSUMED_BY_EXISTING)
  add("P585", "1961", "1961-08", FactMatchType.SUBSUMED_BY_EXISTING)
  add("P585", "1961", "1961", FactMatchType.EXACT)
  add("P585", "1961", "196*", FactMatchType.SUBSUMES_EXISTING)  # decade
  add("P585", "1961", "19**", FactMatchType.SUBSUMES_EXISTING)  # century
  add("P585", "1961", "1***", FactMatchType.SUBSUMES_EXISTING)  # millenium
  add("P585", "1962", "1961-08-04", FactMatchType.CONFLICT)
  add("P585", "1962", "1961-08", FactMatchType.CONFLICT)
  add("P585", "1962", "1961", FactMatchType.CONFLICT)
  add("P580", "1961", "1962-08", FactMatchType.ADDITIONAL)

  # Date-valued properties: QID values. These are only available for years,
  # decades, and millenia.
  q1961 = "Q3696"
  q1962 = "Q2764"
  q196x = "Q35724"
  q197x = "Q35014"
  q19xx = "Q6927"
  q1xxx = "Q25860"
  add("P585", q196x, q1961, FactMatchType.SUBSUMED_BY_EXISTING)
  add("P585", q1xxx, q1961, FactMatchType.SUBSUMED_BY_EXISTING)
  add("P585", q1961, q1961, FactMatchType.EXACT)
  add("P585", q1961, q1962, FactMatchType.CONFLICT)
  add("P585", q196x, q197x, FactMatchType.CONFLICT)
  add("P585", q19xx, q197x, FactMatchType.SUBSUMED_BY_EXISTING)
  add("P580", q1961, q197x, FactMatchType.ADDITIONAL)

  # Date-valued properties: proposed and existing values have different types.
  add("P585", q1961, 1961, FactMatchType.EXACT)
  add("P585", q196x, 196, FactMatchType.EXACT)
  add("P585", q19xx, 19, FactMatchType.EXACT)
  add("P585", q1xxx, 1, FactMatchType.EXACT)
  add("P585", q196x, 1961, FactMatchType.SUBSUMED_BY_EXISTING)
  add("P585", q1961, 19610804, FactMatchType.SUBSUMED_BY_EXISTING)
  add("P585", q1961, 19, FactMatchType.SUBSUMES_EXISTING)
  add("P585", q1961, "1961", FactMatchType.EXACT)
  add("P585", q196x, "196*", FactMatchType.EXACT)
  add("P585", q19xx, "19**", FactMatchType.EXACT)
  add("P585", q196x, "1961", FactMatchType.SUBSUMED_BY_EXISTING)
  add("P585", q196x, "1961-08-04", FactMatchType.SUBSUMED_BY_EXISTING)
  add("P585", q1961, "196*", FactMatchType.SUBSUMES_EXISTING)
  add("P585", "", "196*", FactMatchType.NEW)
  add("P585", q1961, "1962", FactMatchType.CONFLICT)
  add("P585", 1963, "1962", FactMatchType.CONFLICT)
  add("P580", q1961, "1962", FactMatchType.ADDITIONAL)
  add("P580", 1963, "1962", FactMatchType.ADDITIONAL)

  # Genre, melodrama, drama.
  add("P136", "Q191489", "Q21010853", FactMatchType.SUBSUMES_EXISTING)

  # Genre, trip-hop, electronic music.
  add("P136", "Q205560", "Q9778", FactMatchType.SUBSUMES_EXISTING)

  # Genre, rock and roll, electronic music.
  add("P136", "Q7749", "Q9778", FactMatchType.ADDITIONAL)

  # Educated at, Harvard Law School, Harvard University.
  add("P69", "Q49122", "Q13371", FactMatchType.SUBSUMES_EXISTING)

  # Educated at, Harvard Law School, Yale University.
  add("P69", "Q49122", "Q49112", FactMatchType.ADDITIONAL)

  # Employer, Airbus, Airbus SE.
  add("P108", "Q67", "Q2311", FactMatchType.SUBSUMES_EXISTING)

  # Employer, Airbus, Boeing.
  add("P108", "Q67", "Q66", FactMatchType.ADDITIONAL)

  # Occupation, sports cyclist, cyclist.
  add("P106", "Q2309784", "Q2125610", FactMatchType.SUBSUMES_EXISTING)

  # Occupation, sports cyclist, cricketer.
  add("P106", "Q2309784", "Q12299841", FactMatchType.ADDITIONAL)

  store = sling.Store(kb)
  total_successes = 0
  for entry in tuples:
    pid, existing, proposed, expected = entry
    if pid not in kb:
      error(entry, "%s not in KB" % pid)
      continue

    pid = kb[pid]
    if isinstance(existing, str) and existing != "" and existing in kb:
      existing = kb[existing]
    if isinstance(proposed, str) and proposed in kb:
      proposed = kb[proposed]

    if existing == "":
      existing = []
    else:
      existing = [existing]
    actual = matcher.match_type(store, pid, existing, proposed)
    if actual == expected:
      success(entry)
      total_successes += 1
    else:
      error(entry, "Got %s, but expected %s" % (actual.name, expected.name))
  print("Total successful tests: %d out of %d" % (total_successes, len(tuples)))


if __name__ == '__main__':
  test_fact_matcher()
