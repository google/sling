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


# Returns a dictionary of FactMatchType -> count for 'span'.
def fact_matches_for_span(span, counts=None):
  if counts is None:
    counts = defaultdict(int)
  if "fact_matches" in span:
    for bucket in span.fact_matches.buckets:
      counts[bucket.match_type] += bucket.count
  return counts


# Returns a dictionary of FactMatchType -> count across all spans of 'parse'.
def fact_matches_for_parse(parse, counts=None):
  if counts is None:
    counts = defaultdict(int)
  for span in parse.spans:
    fact_matches_for_span(span, counts)
  return counts

