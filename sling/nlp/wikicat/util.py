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


import collections
import sling
import sling.log as log

from collections import defaultdict

# Pool of loaded KBs.
_kb_cache = {}

# Loads a KB into the global pool if it is not already loaded.
def load_kb(task):
  if type(task) is str:
    filename = task  # assume filename
  else:
    filename = task.input("kb").name

  if filename in _kb_cache:
    log.info("Retrieving cached KB")
    return _kb_cache[filename]
  else:
    kb = sling.Store()
    kb.load(filename)
    log.info("Knowledge base read")
    kb.lockgc()
    kb.freeze()
    kb.unlockgc()
    log.info("Knowledge base frozen")
    _kb_cache[filename] = kb
    return kb


# Returns the full signature string for 'parse'.
def full_parse_signature(parse):
  return ' '.join(list(parse.signature))


# Returns the coarse signature string for 'parse'.
def coarse_parse_signature(parse):
  return ' '.join(list(parse.coarse_signature))


# Returns the specified type of signature for 'span'.
def span_signature(span, signature_type):
  if signature_type == "full":
    return span.signature
  elif signature_type == "coarse":
    return span.coarse_signature
  else:
    raise ValueError(signature_type)


# Returns the specified type of signature for 'parse'.
def parse_signature(parse, signature_type):
  if signature_type == "full":
    return full_parse_signature(parse)
  elif signature_type == "coarse":
    return coarse_parse_signature(parse)
  else:
    raise ValueError(signature_type)


# Returns list of span signatures from 'parse_signature'.
def parse_to_span_signatures(parse_signature):
  ls = parse_signature.split(' ')
  return [x for x in ls if x[0] == '$' and x[1:].find('=$') != -1]


# Stores fact match counts + limited examples from a given parse.
class MatchCounts:
  def __init__(self, max_examples=5):
    self.counts = defaultdict(int)
    self.examples = defaultdict(set)
    self.max_examples = max_examples


  # Clears all counts.
  def clear(self):
    self.counts.clear()
    self.examples.clear()


  # Adds a bucket from a parse span to the counts and exemplars.
  def add(self, bucket):
    match_type = bucket.match_type
    self.counts[match_type] += bucket.count
    existing = len(self.examples[match_type])
    if self.max_examples < 0:
      self.examples[match_type].update(bucket.source_items)
    elif existing < self.max_examples:
      diff = self.max_examples - existing
      diff = min(diff, len(bucket.source_items))
      self.examples[match_type].update(bucket.source_items[0:diff])


  # Merges counts and examples from 'other' into self.
  def merge(self, other):
    for k, v in other.counts.items():
      self.counts[k] += v
      existing = len(self.examples[k])
      if existing < self.max_examples:
        diff = self.max_examples - existing
        diff = min(diff, len(other.examples[k]))
        for x in other.examples[k]:
          self.examples[k].add(x)
          if len(self.examples[k]) >= self.max_examples:
            break


  # Returns a dictionary representation of the counts.
  def to_dict(self):
    result = {}
    for k, v in self.counts.items():
      d = {}
      d["count"] = v
      d["examples"] = [e.id for e in self.examples.get(k, [])]
      result[k] = d
    return result


# Returns an instance of MatchCounts for 'span'.
def fact_matches_for_span(span, counts=None, max_examples=5):
  if counts is None:
    counts = MatchCounts(max_examples)
  if "fact_matches" in span:
    for bucket in span.fact_matches.buckets:
      counts.add(bucket)
  return counts


# Returns an instance of MatchCounts across all spans of 'parse'.
def fact_matches_for_parse(parse, counts=None, max_examples=5):
  if counts is None:
    counts = MatchCounts(max_examples)
  for span in parse.spans:
    fact_matches_for_span(span, counts)
  return counts

