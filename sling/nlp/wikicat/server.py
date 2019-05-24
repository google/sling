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


"""
Backend for browsing category parses.
"""

import collections
import cgi
import json
import os
import sling
import sling.flags as flags
import sling.log as log
import tempfile
import util

from http.server import BaseHTTPRequestHandler, HTTPServer
from collections import defaultdict
from fact_matcher import FactMatchType
from urllib.parse import urlparse, parse_qs


# Returns the list of fact match types.
def fact_match_type_names():
  return [t.name for t in FactMatchType]


# Scores a given parse using specified match-type weights.
class Metric:
  def __init__(self, metric_str):
    self.weights = {}  # string match-type -> weight
    for t in fact_match_type_names():
      self.weights[t] = 0.0

    d = dict([x.split(':') for x in metric_str.split(',')])
    for (k, v) in d.items():
      self.weights[k] = float(v)


  # Returns the score of 'match_counts' as per the weights.
  def score(self, match_counts):
    s = 0.0
    for match_type, count in match_counts.counts.items():
      s += self.weights[match_type] * count
    return s


# Stores a usable subset of spans in a signature.
class SpanSubset:
  def __init__(self, span_subset_str=None):
    self.allowed = None
    if span_subset_str is not None and span_subset_str != "-1":
      self.allowed = [int(x) for x in span_subset_str.split(',')]


  # Returns whether span numbered 'index' is allowed.
  def is_allowed(self, index):
    return self.allowed is None or index in self.allowed


# Wrapper around a parse.
class Parse:
  def __init__(self, unique_id, category, frame, parse):
    self.id = unique_id          # unique int id for the parse
    self.parse = parse           # the parse itself
    self.category = category     # QID for the category
    self.frame = frame           # frame of the category
    self.span_counts = []        # i -> match counts for ith span in parse

    # Score and counts as per a set of spans. Changes as the user tinkers
    # with various settings.
    self.score = 0.0
    self.counts = None


  # Computes and caches the match counts for each span from the parse frame.
  def compute_span_counts(self):
    if len(self.span_counts) == 0:
      for span in self.parse.spans:
        c = util.fact_matches_for_span(span)
        self.span_counts.append(c)


  # Computes aggregate match counts across all selected spans.
  def counts_for(self, span_subset=None, counts=None):
    if counts is None:
      counts = util.MatchCounts()
    for index, count in enumerate(self.span_counts):
      if span_subset is None or span_subset.is_allowed(index):
        counts.merge(count)
    return counts


  # Returns a list of (span signature, span match counts) for all selected spans
  # as per 'span_subset'.
  def selected_span_counts(self, span_subset=None):
    output = []
    for index, counts in enumerate(self.span_counts):
      if span_subset is None or span_subset.is_allowed(index):
        output.append((self.parse.spans[index].signature, counts.to_dict()))
    return output


  # Computes a list of name-parts for 'parse'.
  def name_parts(self, span_subset=None, use_span_signature=False):
    def _word(text, token):
      size = token.size if token.size is not None else 1
      return text[token.start:token.start + size]

    tokens = self.frame.document.tokens
    text = self.frame.document.text
    begin = 0
    parts = []
    for index, span in enumerate(self.parse.spans):
      if begin < span.begin:
        words = ' '.join([_word(text, t) for t in tokens[begin:span.begin]])
        parts.append((words, "", ""))

      words = ' '.join([_word(text, t) for t in tokens[span.begin:span.end]])
      if span_subset is None or span_subset.is_allowed(index):
        if use_span_signature:
          words = span.signature
        parts.append((words, '.'.join([x.id for x in span.pids]), span.qid.id))
      else:
        parts.append((words, "", ""))
      begin = span.end
    if begin < len(tokens):
      words = ' '.join([_word(text, t) for t in tokens[begin:]])
      parts.append((words, "", ""))

    return [{
        "text": p[0],
        "pqid": p[1] + "=" + p[2] if p[1] != "" else "",
      } for p in parts]


# Decides whether a parse should be selected for further processing.
class ParseSelector:
  def __init__(self, formula_str):
    typenames = fact_match_type_names()
    for symbol in ["NAME", "SCORE", "INTERESTING", "QID"]:
      assert symbol not in typenames, symbol

    if formula_str is None or formula_str == "":
      formula_str = "True"  # every parse is selected
    self.formula = formula_str


  # Returns whether 'parse' should be selected. Assumes that 'parse' has its
  # counts (for the relevent subset of spans) filled in.
  def select(self, parse):
    formula = self.formula
    if formula == "True":
      return True

    # Replace symbols by their values. This is just a regex replace.
    c = parse.counts.counts
    for t in fact_match_type_names():
      formula = formula.replace("$" + t, str(c.get(t, 0)))

    # Replace other special fields.
    formula = formula.replace("$SCORE", str(parse.score))
    formula = formula.replace("$INTERESTING", \
      str(c.get("NEW", 0) + c.get("ADDITIONAL", 0) + \
      c.get("SUBSUMED_BY_EXISTING", 0)))
    formula = formula.replace("$NAME", "parse.frame.name")
    formula = formula.replace("$QID", "parse.category")
    return eval(formula) is True


# Wraps a request from the front-end to the server.
#
# A request is comprised of a text 'query' (e.g. a signature or category),
# a set of weights (captured by 'metric'), a set of span indices
# (captured by 'span_subset'; relevant only for signature queries), and
# a parse-selection formula captured by 'parse_selector'.
class Request:
  def __init__(self, query, metric, span_subset=None, \
               parse_selector=None, topk=500):
    self.query = query
    self.metric = Metric(metric)
    self.span_subset = SpanSubset(span_subset)
    self.parse_selector = ParseSelector(parse_selector)
    self.topk = topk


# Response to a signature query.
class SignatureResponse:
  def __init__(self, request):
    self.request = request
    self.error_message = None
    self.span_signatures = util.parse_to_span_signatures(request.query)

    # All counts are after the parse selector has come into play.
    self.counts_across_spans = util.MatchCounts()
    self.counts_across_selected_spans = util.MatchCounts()
    self.counts_by_span = [util.MatchCounts() for _ in self.span_signatures]

    # Top selected parses shown.
    self.top_parses = []

    # Statistics of selected parses that aren't not shown.
    self.top_parses_not_shown = {"num": 0, "counts": util.MatchCounts()}

    # Statistics of unselected parses.
    self.unselected_parses = {"num": 0, "counts": util.MatchCounts()}


  # Adds 'parse' to the response.
  def add(self, parse):
    for index, span in enumerate(parse.parse.spans):
      counts = parse.span_counts[index]
      self.counts_across_spans.merge(counts)
      self.counts_by_span[index].merge(counts)
      if self.request.span_subset.is_allowed(index):
        self.counts_across_selected_spans.merge(counts)
    self.top_parses.append(parse)


  # Records 'parse' as been rejected from the response.
  def reject(self, parse):
    self.unselected_parses["num"] += 1
    self.unselected_parses["counts"].merge(parse.counts)


  # Trims the response to top-k as per user's metric.
  def trim(self):
    k = self.request.topk
    self.top_parses.sort(key=lambda parse: -parse.score);
    self.top_parses_not_shown["num"] = max(0, len(self.top_parses) - k)
    c = self.top_parses_not_shown["counts"]
    for parse in self.top_parses[k:]:
      parse.counts_for(self.request.span_subset, c)
    self.top_parses = self.top_parses[:k]


  # Sets error message.
  def error(self, message):
    self.error_message = message


  # Converts the response to JSON for communicating back to the front-end.
  def to_json(self):
    data = {}
    data["response_type"] = "signature"
    if self.error_message is not None:
      data["error"] = self.error_message
      return json.dumps(data)

    data["num_total_spans"] = len(self.counts_by_span)
    data["num_selected_spans"] = sum([
        self.request.span_subset.is_allowed(i) \
            for i in range(len(self.counts_by_span))])

    data["per_span"] = [
      {
        "selected" : self.request.span_subset.is_allowed(index),
        "signature" : self.span_signatures[index],
        "counts": self.counts_by_span[index].to_dict()
      } for index, count in enumerate(self.counts_by_span)
    ]
    data["total_spans"] = self.counts_across_spans.to_dict()
    data["selected_spans"] = self.counts_across_selected_spans.to_dict()

    data["parses"] = [{
      "parse_id": p.id,
      "category_qid": p.category,
      "name": p.frame.name,
      "name_parts": p.name_parts(self.request.span_subset),
      "score": round(p.score, 2),
      "counts": p.counts.to_dict(),
      "selected_span_counts": p.selected_span_counts(self.request.span_subset)
    } for p in self.top_parses]

    data["parses_not_shown"] = {
      "num": self.top_parses_not_shown["num"],
      "counts": self.top_parses_not_shown["counts"].to_dict()
    }
    data["parses_not_selected"] = {
      "num": self.unselected_parses["num"],
      "counts": self.unselected_parses["counts"].to_dict()
    }

    return json.dumps(data)


# Response to a category request.
class CategoryResponse:
  def __init__(self, request):
    self.request = request
    self.num_extra = 0
    self.rejected = 0
    self.parses = []


  # Adds 'parse' as a candidate parse for the category.
  def add(self, parse):
    self.parses.append(parse)


  # Rejects 'parse' as a candidate parse for the category.
  def reject(self, parse):
    self.rejected += 1


  # Trims the response to keep only top-k parses.
  def trim(self, topk):
    self.parses.sort(key=lambda p: -p.score)
    self.num_extra = max(0, len(self.parses) - topk)
    self.parses = self.parses[0:topk]


  # Converts the response to JSON for communicating back to the front-end.
  def to_json(self):
    d = {}
    d["response_type"] = "category"
    d["rejected"] = self.rejected
    d["extra"] = self.num_extra
    parses = []
    for parse in self.parses:
      entry = {
        "signature" : " ".join(parse.parse.signature),
        "name_parts" : parse.name_parts(span_subset=None, \
            use_span_signature=True),
        "parse_id" : parse.id,
        "score" : round(parse.score, 2),
        "counts" : parse.counts.to_dict()
      }
      parses.append(entry)
    d["parses"] = parses
    return json.dumps(d)


# Response for a recordio generation request.
class RecordioResponse:
  def __init__(self, request, store):
    self.request = request
    self.error_message = None
    self.num_facts = defaultdict(int)
    self.store = store

    # Recordio will be generated only for these match types.
    self.match_types = [
      FactMatchType.NEW.name,
      FactMatchType.ADDITIONAL.name,
      FactMatchType.SUBSUMED_BY_EXISTING.name
    ]
    self.frames = []
    self.output_file = ""


  # Records an error message during recordio generation.
  def error(self, message):
    self.error_message = message


  # Adds facts implied by 'parse' to the set of facts that will be written.
  def add(self, parse):
    for index, span in enumerate(parse.parse.spans):
      if not self.request.span_subset.is_allowed(index):
        continue

      pid = span.pids
      qid = span.qid
      if len(pid) > 1:  # skip multi-hop facts
        continue
      else:
        pid = pid[0]

      for bucket in span.fact_matches.buckets:
        if bucket.match_type in self.match_types:
          for item in bucket.source_items:
            frame = self.store.frame([("item", item)])
            frame.facts = self.store.frame([(pid, qid)])
            frame.provenance = self.store.frame([
                ("category", parse.category),
                ("method", "Member of Category:%s" % parse.frame.name)
            ])
            frame.comment = "%s : %s = %s" % (item, span.signature, qid)
            frame.match_type = bucket.match_type
            self.num_facts[(pid.id, bucket.match_type)] += 1
            self.frames.append(frame)


  # Writes the facts to 'filename'.
  def write(self, filename):
    self.output_file = filename
    writer = sling.RecordWriter(filename)
    for frame in self.frames:
      writer.write(frame.item.id, frame.data(binary=True))


  # Returns a JSON-formatted summary of recordio generation.
  def to_json(self):
    ls = []
    for (pid, match_type), count in self.num_facts.items():
      ls.append((pid, match_type, count))
    d = {}
    d["counts"] = ls
    d["output_file" ] = self.output_file
    return json.dumps(d)


# Aggregate statistics for a signature across all its parses allowed by
# the parse selection formula.
class SignatureStats:
  def __init__(self, signature):
    self.signature = signature
    self.score = 0.0                   # aggregate score as per user-weights
    self.counts = util.MatchCounts()   # aggregate match counts
    self.example_category = None       # exemplar category for the parse
    self.rejected = 0                  # no. of parses rejected by the formula
    self.selected = 0                  # no. of parses allowed by the formula


  # Records 'parse' as a rejected parse for the signature.
  def reject_parse(self, parse):
    self.rejected += 1


  # Records 'parse' as a selected parse for the signature.
  def add_parse(self, parse, request):
    self.selected += 1
    self.counts.merge(parse.counts)
    self.score += parse.score
    if self.example_category is None:
      self.example_category = parse.frame.name


  # Returns a dictionary form of the statistics.
  def to_dict(self):
    data = {}
    data["signature"] = self.signature
    data["example_category"] = self.example_category
    data["score"] = round(self.score, 2)
    data["rejected"] = self.rejected
    data["selected"] = self.selected
    data["counts"] = self.counts.to_dict()
    return data


# Container for parses, categories, and mappings that are read only once
# and persisted across various HTTP requests.
class ServerGlobals:
  # Initializes the globals from 'parses_filename'.
  def init(self, parses_filename, output_dir):
    self.output_dir = output_dir
    reader = sling.RecordReader(parses_filename)
    self.category_name_to_qid = {}                 # category name -> qid
    self.category_frame = {}                       # category qid -> frame
    self.category_parses = {}                      # category qid -> parses
    self.signature_to_parse = defaultdict(list)    # signature -> parse
    self.store = sling.Store()
    self.num_parses = 0
    for index, (qid, value) in enumerate(reader):
      if (index + 1) % 20000 == 0:
        log.info("%d categories read" % index)
      qid = qid.decode('utf-8')
      frame = self.store.parse(value)
      self.category_name_to_qid[frame.name] = qid
      self.category_frame[qid] = frame
      self.category_parses[qid] = []
      for parse in frame("parse"):
        element = Parse(self.num_parses, qid, frame, parse)
        signature = util.full_parse_signature(parse)
        self.signature_to_parse[signature].append(element)
        self.category_parses[qid].append(element)
        self.num_parses += 1
    self.store.lockgc()
    self.store.freeze()
    self.store.unlockgc()


  # Returns whether 'key' is a category QID or name.
  def is_category(self, key):
    return key in self.category_frame or key in self.category_name_to_qid

  # Returns a list of (parse, selected?) tuples for each parse matching
  # 'signature'. Selection and scoring is done as per 'request'.
  # parse.counts and parse.score are filled in for each parse.
  #
  # If multiple parses matching 'signature' belong to the same category, then
  # only the highest scoring one (as per 'request') is returned.
  def parses_for_signature(self, signature, request):
    parses = []
    for parse in self.signature_to_parse[signature]:
      parse.compute_span_counts()
      parse.counts = parse.counts_for(request.span_subset)
      parse.score = request.metric.score(parse.counts)
      parses.append((parse, request.parse_selector.select(parse)))
    parses.sort(key=lambda t: -t[0].score)

    output = []
    seen = set()
    for parse, selected in parses:
      if parse.category not in seen:
        output.append((parse, selected))
        seen.add(parse.category)
    return output


  # Returns the JSON response for a top signatures request.
  def handle_top(self, request):
    all_stats = []   # i -> SignatureStats for ith signature
    for signature in self.signature_to_parse:
      signature_stats = SignatureStats(signature)
      for parse, selected in self.parses_for_signature(signature, request):
        if selected:
          signature_stats.add_parse(parse, request)
        else:
          signature_stats.reject_parse(parse)
      if signature_stats.selected > 0:
        all_stats.append(signature_stats)
    all_stats.sort(key=lambda s: -s.score)

    if request.topk >= 0:
      all_stats = all_stats[:request.topk]
    all_stats = [s.to_dict() for s in all_stats]
    return json.dumps({"response_type":"top", "stats":all_stats})


  # Returns the JSON response for a signature query.
  def handle_signature(self, request):
    response = SignatureResponse(request)
    signature = request.query
    if signature in self.signature_to_parse:
      for parse, selected in self.parses_for_signature(signature, request):
        if selected:
          response.add(parse)
        else:
          response.reject(parse)
      response.trim()
    else:
      response.error("Unknown input:" + signature)
    return response


  # Returns the JSON response for a category query.
  def handle_category(self, request):
    response = CategoryResponse(request)
    qid = request.query
    if qid in self.category_name_to_qid:
      qid = self.category_name_to_qid[qid]

    parses = self.category_parses[qid]
    for parse in parses:
      parse.compute_span_counts()
      parse.counts = parse.counts_for(span_subset=None)
      parse.score = request.metric.score(parse.counts)
      if request.parse_selector.select(parse):
        response.add(parse)
      else:
        response.reject(parse)

    response.trim(request.topk)
    return response


  # Handles a recordio generation request and returns a summary.
  def handle_recordio(self, request):
    response = RecordioResponse(request, sling.Store(self.store))
    signature = request.query
    if signature in self.signature_to_parse:
      filename = ""
      first = True
      for parse, selected in self.parses_for_signature(signature, request):
        if first:
          # Make an appropriate filename from the PID(s) involved.
          first = False
          for index, span in enumerate(parse.parse.spans):
            if request.span_subset.is_allowed(index):
              filename = filename + ".".join([p.id for p in span.pids]) + "_"
        if selected:
          response.add(parse)

      # Generate the full filename for the recordio.
      (fd, filename) = tempfile.mkstemp(\
        suffix=".rec", prefix=filename, dir=self.output_dir)
      response.write(filename)
      os.close(fd)
    else:
      response.error("No such signature:" + signature)
    return response


# This will be visible inside the browser, which is a BaseHTTPRequestHandler.
server_globals = ServerGlobals()

# Main class for browsing category parses.
class BrowserService(BaseHTTPRequestHandler):
  MIME_TYPES = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".ico": "image/x-icon",
    ".css": "text/css",
    ".js": "text/javascript",
    ".html": "text/html; charset=utf-8",
    ".txt": "text/plain; charset=utf-8",
  }


    # Writes 'obj', which could be bytes or a string, to the response.
  def write(self, obj):
    if type(obj) is bytes:
      self.wfile.write(obj)
    else:
      self.wfile.write(bytes(obj, 'utf-8'))


  # Sends default HTTP response headers.
  def _set_headers(self, mimetype="text/html"):
    self.send_response(200)
    self.send_header('Content-type', mimetype)
    self.end_headers()


  # Computes the returns the fact-matching score for the given parse. Weights
  # for each bucket-type are given in 'weights' (FactMatchType.name -> float).
  def parse_fact_score(self, parse, weights):
    score = 0.0
    match_counts = util.fact_matches_for_parse(parse, max_examples=0)
    for match_type, count in match_counts.counts.items():
      score += count * weights[match_type]
    return score


  # Returns the (mime type, whether content is static?) for 'path'.
  def get_mime_type(self, path):
    default_output = ("text/json; charset=utf-8", False)
    if path.startswith("/wikicat/query?") or \
      path.startswith("/wikicat/recordio?") or \
      path.startswith("/wikicat/basic") or \
      path.startswith("/wikicat/weights"):
      return default_output

    ext = path.rfind(".")
    if ext == -1:
      return default_output
    ext = path[ext:]
    if ext not in BrowserService.MIME_TYPES:
      return ("", False)
    return (BrowserService.MIME_TYPES[ext], True)


  # Returns default weights for all match types.
  def default_weights(self):
    wts = {
      "NEW": 0.1,
      "EXACT": 1.0,
      "SUBSUMES_EXISTING": 0.05,
      "SUBSUMED_BY_EXISTING": 0.5,
      "CONFLICT": -20,
      "ADDITIONAL": 0.05
    }
    result = []
    for t in fact_match_type_names():
      wt = wts.get(t, 0.0)
      result.append((t, wt))
    return json.dumps(result)


  # Returns very basic statistics about the parses.
  def basic_stats(self):
    data = {}
    data["num_categories"] = len(server_globals.category_parses)
    data["num_parses"] = server_globals.num_parses
    return json.dumps(data)


  # Converts 'params' into a Request object.
  def params_to_request(self, params):
    return Request(\
        params["query"][0], params["metric"][0], \
        span_subset=params["spans"][0],
        parse_selector=params["selector"][0])


  # Overridden method for generating the head of the response.
  def do_HEAD(self):
    self._set_headers()


  # Overridden GET request handler.
  def do_GET(self):
    print(self.client_address)
    # Accept only local requests.
    if self.client_address[0] not in ['127.0.0.1', 'localhost']:
      return

    params = parse_qs(urlparse(self.path).query)
    print(params, self.path)
    if self.path.startswith("/wikicat/"):
      (mime_type, static) = self.get_mime_type(self.path)
      if static and mime_type != "":
        # Static content is served by reading the specified file.
        # All static content is based off the sling/nlp/wikicat/app folder.
        self._set_headers(mime_type)
        path = os.getcwd() + "/sling/nlp/wikicat/app/" + self.path[9:]
        with open(path, "rb") as f:
          self.write(f.read())
      elif not static:
        self._set_headers(mime_type)
        if self.path.startswith("/wikicat/basic"):
          response = self.basic_stats()
        elif self.path.startswith("/wikicat/weights"):
          response = self.default_weights()
        elif self.path.startswith("/wikicat/query?"):
          request = self.params_to_request(params)
          if request.query.lower() == "top":
            response = server_globals.handle_top(request)
          elif server_globals.is_category(request.query):
            response = server_globals.handle_category(request).to_json()
          else:
            response = server_globals.handle_signature(request).to_json()
        elif self.path.startswith("/wikicat/recordio?"):
          request = self.params_to_request(params)
          response = server_globals.handle_recordio(request).to_json()
        self.write(response)


  # Overridden method for responding to POST requests.
  def do_POST(self):
    if self.client_address[0] not in ['127.0.0.1', 'localhost']:
      return
    raise ValueError("No POST handler! Use GET instead.")


if __name__ == "__main__":
  flags.define("--port",
               help="port number for the HTTP server",
               default=8001,
               type=int,
               metavar="PORT")
  flags.define("--parses",
               help="Recordio of category parses",
               default="local/data/e/wikicat/parses-with-match-statistics.rec",
               type=str,
               metavar="FILE")
  flags.define("--output",
               help="Output dir where Wikibot recordios will be generated.",
               default="local/data/e/wikicat/facts",
               type=str,
               metavar="DIR")
  flags.parse()
  log.info('Reading parses from %s' % flags.arg.parses)
  server_globals.init(flags.arg.parses, flags.arg.output)

  if not os.path.exists(flags.arg.output):
    os.makedirs(flags.arg.output)
  log.info("Output recordios will be dumped to %s" % flags.arg.output)

  server_address = ('', flags.arg.port)
  httpd = HTTPServer(server_address, BrowserService)
  log.info('Running: http://localhost:%d/wikicat/index.html' % flags.arg.port)
  httpd.serve_forever()
