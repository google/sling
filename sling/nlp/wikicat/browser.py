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
Browser for viewing category parses.
"""

import cgi
import sling
import sling.flags as flags
import sling.log as log
import util
import SocketServer

from BaseHTTPServer import BaseHTTPRequestHandler, HTTPServer
from collections import defaultdict
from fact_matcher import FactMatchType

"""
Container for global variables.

Note that the browser is implemented as a subclass of BaseHTTPRequestHandler.
It is constructed for each incoming HTTP request and deleted once the request is
processed. Therefore any global variables (e.g. signature -> parse mapping)
live in a BrowserGlobals object outside the handler.
"""
class BrowserGlobals:
  # Initializes the globals from 'parses_filename'.
  def init(self, parses_filename, output_dir):
    self.output_dir = output_dir
    reader = sling.RecordReader(parses_filename)
    self.category_name_to_qid = {}                      # category name -> qid
    self.category_frame = {}                            # category qid -> frame
    self.full_signature_to_parse = defaultdict(list)    # signature -> parse
    self.coarse_signature_to_parse = defaultdict(list)  # signature -> parse
    self.store = sling.Store()
    for index, (qid, value) in enumerate(reader):
      if index > 0 and index % 20000 == 0:
        log.info("%d categories read" % index)
      frame = self.store.parse(value)
      self.category_name_to_qid[frame.name] = qid
      self.category_frame[qid] = frame
      for parse in frame("parse"):
        element = (qid, frame, parse)
        full_signature = util.full_parse_signature(parse)
        self.full_signature_to_parse[full_signature].append(element)
        coarse_signature = util.coarse_parse_signature(parse)
        self.coarse_signature_to_parse[coarse_signature].append(element)


# This will be visible inside the browser, which is a BaseHTTPRequestHandler.
browser_globals = BrowserGlobals()

# Holds statistics for a single signature.
class SignatureStats:
  def __init__(self):
    # Exemplar category QID, frame, and parse with this signature.
    self.example_qid = None
    self.example_category = None
    self.example_parse = None

    # Aggregate stats across all (category, parse) pairs with this signature.
    # If a category has >=2 parses with this signature, then only the highest
    # scoring parse is considered.

    # Total score across all parses.
    self.score = 0.0

    # Total fact-matching statistics.
    self.fact_matches = util.MatchCounts()

    # Non-deduped total number of member items.
    self.members = 0

    # Total number of categories.
    self.num = 0


  # Set the exemplar if it is not already set.
  def example(self, qid, category, parse):
    if self.example_qid is not None:
      return
    self.example_qid = qid
    self.example_category = category
    self.example_parse = parse


# Main class for browsing category parses.
class Browser(BaseHTTPRequestHandler):
  # Sends default HTTP response headers.
  def _set_headers(self):
    self.send_response(200)
    self.send_header('Content-type', 'text/html')
    self.end_headers()


  # Various utility methods to write the HTML response.
  #
  # Writes a beginning HTML tag. Tag attributes are taken from 'kwargs'.
  def _begin(self, tag, **kwargs):
    s = "<" + tag
    for k, v in kwargs.iteritems():
      if v is None or k == 'colspan' and v == 1: continue
      s += " " + k + "='" + str(v) + "'"
    s += ">"
    self.wfile.write(s)


  # Writes a beginning and ending HTML tag, e.g.
  # <input type='text' name='foo' />
  def _begin_end(self, tag, **kwargs):
    s = "<" + tag
    for k, v in kwargs.iteritems():
      if v is None or k == 'colspan' and v == 1: continue
      s += " " + k + "='" + str(v) + "'"
    s += "/>"
    self.wfile.write(s)


  # Writes ending HTML tag. If 'tag' is a list of HTML tag names,
  # then ending tags for all members of the list are generated in order.
  def _end(self, tag):
    if type(tag) is list:
      for n in tag:
        self.wfile.write("</" + n + ">")
    else:
      self.wfile.write("</" + tag + ">")


  # Writes the string version of 'text'.
  def _text(self, text):
    self.wfile.write(str(text))


  # Writes HTML to render 'text' in 'color', with a tooltip of 'hover'.
  def _color_text(self, text, color=None, hover=None):
    style = None
    if color is not None:
      style = 'color:' + color
    self._tag("span", text, style=style, title=hover)


  # Writes a beginning tag, some text, and end of tag.
  # Tag attributes are taken from 'kwargs'.
  def _tag(self, tag, text, **kwargs):
    self._begin(tag, **kwargs)
    if text is not None:
      self.wfile.write(str(text))
    self._end(tag)


  # Writes an HTML linebreak.
  def _br(self):
    self._begin_end("br")


  # Writes CSS styles to the response.
  def write_styles(self):
    styles = '''
      <style type="text/css">
        table.main_table {
          border-collapse: collapse;
          border-top: 2px solid #eeeeee;
          border-bottom: 2px solid #eeeeee;
        }

        .main_table thead {
          background: #dddddd;
        }

        .main_table th {
          padding: 10px;
          border-left: 2px solid #dddddd;
          border-right: 2px solid #dddddd;
        }

        .main_table tbody tr {
          background: #efefef;
          border-right: 2px solid #dddddd;
          border-bottom: 2px solid #dddddd;
        }

        .main_table td {
          padding: 6px;
          border-left: 2px solid #dddddd;
          border-right: 2px solid #dddddd;
        }

        table.span_fact_match {
          font-size:11px;
          color:#333333;
          border-top: 2px solid #dddddd;
          background-color: #efefef;
          text-align: center;
          border-collapse: collapse;
        }

        td.numeric {
          text-align: right;
        }

        td.sep {
          background: #dddddd;
        }

        .fact_match_count {
          color: blue;
          position: relative;
          display: inline-block;
        }

        .fact_match_examples {
          visibility: hidden;
          background-color: white;
          width: 200px;
          color: black;
          position: absolute;
          text-align: left;
          padding: 2px 2px;
          z-index: 1;
        }

        .fact_match_count:hover .fact_match_examples {
          visibility: visible;
        }
      </style>
      '''
    self.wfile.write(styles)


  # Writes JS event handlers to the response.
  def write_script(self):
    script = '''
    <script language="javascript">
    function onclick_handler(target, submit_main_form) {
      if (target != null) {
        document.getElementById("main_form_input").value = target;
      }
      if (submit_main_form) {
        document.getElementById("main_form").submit();
      }
    }

    function set_weight(short_name, value) {
      value = value.toString();
      document.getElementById('main_form_wt_' + short_name).value = value;
    }

    function empty_wts() {
      set_weight('new', 0);
      set_weight('exact', 0);
      set_weight('additional', 0);
      set_weight('conflict', 0);
      set_weight('subsumes_existing', 0);
      set_weight('subsumed_by_existing', 0);
    }

    function set_headroom() {
      empty_wts();
      set_weight('new', 1);
      set_weight('additional', 0.5);
    }

    function set_penalized_headroom() {
      set_headroom();
      set_weight('conflict', -1);
    }

    function set_conflicts() {
      empty_wts();
      set_weight('conflict', -50);
    }

    function set_existing() {
      empty_wts();
      set_weight('exact', 1);
      set_weight('subsumes_existing', 1);
      set_weight('subsumed_by_existing', 1);
    }
    </script>
    '''
    self.wfile.write(script)


  # Writes a table header column to the response.
  def _add_column(self, label):
    self._tag("th", str(label))


  # Writes a separator table column to the response.
  def _separator(self, header=True):
    tag = "th" if header else "td"
    self._begin(tag + " class='sep'")
    self._end(tag)


  # Writes a table cell with the given contents.
  def _cell(self, contents, numeric=False):
    if type(contents) is float:
      contents = "%.4f" % contents
    if numeric:
      self._begin("td class='numeric'")
    else:
      self._begin("td")
    self._text(str(contents))
    self._end("td")


  # Writes an empty table cell.
  def _empty_cell(self):
    self._tag("td", "&nbsp;")


  # Writes an anchor whose clicking triggers the main form's submission with the
  # given argument.
  def _form_anchor(self, display, value):
    handler = 'onclick_handler("%s", true);return false;' % value
    self._tag("a", display, onclick=handler, href='')


  # Computes the returns the fact-matching score for the given parse. Weights
  # for each bucket-type are given in 'weights' (FactMatchType.name -> float).
  def parse_fact_score(self, parse, weights):
    score = 0.0
    match_counts = util.fact_matches_for_parse(parse, max_examples=0)
    for match_type, count in match_counts.counts.iteritems():
      score += count * weights[match_type]
    return score


  # Reads fact-matching weights from the corresponding fields in 'form'.
  def fact_match_weights(self, form):
    weights = {}
    for t in FactMatchType:
      name = t.name
      field = "main_form_wt_" + name.lower()
      if field not in form:
        weights[name] = 0.0
      else:
        weights[name] = float(form.getvalue(field))
    return weights


  # Computes and returns the specified kind of score for the given parse.
  def parse_score(self, category, parse, score_type, fact_match_weights):
    if score_type == "num_members":
      return len(category.members)
    if score_type == "prelim_parse_score":
      return parse.score
    if score_type == "fact_matching_score":
      return self.parse_fact_score(parse, fact_match_weights)
    else:
      raise ValueError(score_type)


  # Writes all the settings options to the response. The options are defaulted
  # to their existing values in 'main_form'.
  def write_settings(self, main_form):
    # Returns the option value from 'main_form'.
    def old_value(name, fallback):
      if main_form is None:
        return fallback
      return main_form.getvalue(name)

    # Makes and returns a <select> list with the given id, options, and value.
    def make_select_list(list_id, option_name_values, default_value):
      self._begin("select", id=list_id, name=list_id)
      selected = old_value(list_id, default_value)
      for (name, value) in option_name_values:
        if selected == value:
          self._tag("option", name, value=value, selected="true")
        else:
          self._tag("option", name, value=value)
      self._end("select")

    # Default weights for computing fact-match scores.
    default_weights = {
      FactMatchType.NEW: 0.1,
      FactMatchType.EXACT: 1,
      FactMatchType.SUBSUMES_EXISTING: 0.5,
      FactMatchType.SUBSUMED_BY_EXISTING: 0.9,
      FactMatchType.CONFLICT: -50,
      FactMatchType.ADDITIONAL: 0.05
    }

    self._begin("div", style="background-color:#cccccc")
    self._tag("h3", "Settings")

    # Options for setting fact-match bucket weights.
    self._tag("b", "Weights for computing fact-matching scores:")
    self._br()
    for t in FactMatchType:
      self._text("&nbsp;" + t.name + ": ")
      field_name = "main_form_wt_" + t.name.lower()
      value = old_value(field_name, default_weights[t])
      self._begin_end("input", type="text", size=5, \
          value=value, id=field_name, name=field_name)
    self._begin("font", size="-1px")
    self._br()

    # Short-cuts for using pre-set weights.
    self._text("(Short-cuts: ")
    self._tag("button", "Headroom", onclick="set_headroom();return false;")
    self._tag("button", "Headroom minus conflicts",\
      onclick="set_penalized_headroom();return false;")
    self._tag("button", "Conflicts", onclick="set_conflicts();return false;")
    self._tag("button", "Existing", onclick="set_existing();return false;")
    self._end("font")
    self._br()

    # Option to choose the metric for sorting parses.
    self._tag("b", "Sort categories/parses by: ")
    make_select_list("main_form_sort_metric", [
      ("Fact-matching score", "fact_matching_score"),
      ("Number of members", "num_members"),
      ("Prelim parse score", "prelim_parse_score")], "fact_matching_score")
    self._br()

    # Option to select the parse signature to use.
    self._tag("b", "Signature type: ")
    make_select_list("main_form_signature_type", [
      ("Coarse", "coarse"), ("Full", "full")], "full")
    self._br()

    # Generate checkboxes for all other miscellaneous options.
    self._tag("b", "Other Options: ")
    self._br()
    for (name, box) in [
        ("span QID", "span_qid"),
        ("preliminary parse scores", "prelim_parse_scores"),
        ("span scores", "span_scores"),
        ("fact-matching statistics", "fact_matching_statistics"),
        ("span-level fact-matching Statistics", "span_fact_match_stats"),
        ("categories with the same signature", "similar_categories")]:
      name = "Show " + name
      box = "main_form_show_" + box
      self._text(" " + name + ": ")
      checked = old_value(box, None)
      if main_form is None and box == "main_form_show_fact_matching_statistics":
        checked = "on"
      box_args = {"id": box, "name": box, "type": "checkbox"}
      if checked == "on":
        box_args["checked"] = checked
      self._begin_end("input", **box_args)
      self._br()
    self._end("div")


  # Writes the main form, which on surface only has the input text field.
  # On submission, this form copies all the settings from the settings form,
  # and then submits.
  def write_main_form(self, form):
    self._text(" Loaded %d categories with %d full and %d coarse signatures" % \
      (len(browser_globals.category_name_to_qid),
       len(browser_globals.full_signature_to_parse),
       len(browser_globals.coarse_signature_to_parse)))
    self._br()

    # Main input box.
    self._text(" Enter category QID/name or full/coarse signature or 'top': ")
    value = form.getvalue("main_form_input") if form is not None else None
    self._begin_end("input", id="main_form_input", name="main_form_input", \
      type="text", size=100, value=value)

    # Set the functionality mode of the main form to 'browse'.
    self._begin_end("input", type="hidden", id="main_form_functionality", \
                    name="main_form_functionality", value="browse")

    # Submit button.
    self._text("&nbsp;")
    self._begin_end("input", type="submit")

    # Short-cut for showing top signatures.
    self._br()
    self._form_anchor("Top Signatures", "top")


  # Writes the part of the page before the results. This consists of the
  # settings and the main form.
  def write_entry_page(self, form=None):
    self._begin("html")
    self._begin("head")
    self._begin_end("link", rel="icon", href="data:,")
    self.write_styles()
    self._text('\n')
    self.write_script()
    self._text('\n')
    self._end("head")
    self._begin("body")
    self._begin("form", id="main_form", method="POST", action="", \
        onsubmit="onclick_handler(null, false);")
    self.write_settings(form)
    self.write_main_form(form)
    self._end("form")
    self._br()


  # Writes a recordio for facts for a given signature.
  def write_recordio(self, form):
    filename = form.getvalue("recordio_filename")
    signature = form.getvalue("recordio_signature")
    chosen_categories = form.getvalue("recordio_categories")
    all_chosen = chosen_categories == 'ALL'
    chosen_categories = set(chosen_categories.split(","))

    # See which spans in the signature should be focused on.
    num_spans = int(form.getvalue("recordio_num_spans"))
    chosen_spans = []
    for index in xrange(num_spans):
      if form.getvalue("recordio_span%d" % index) == "on":
        chosen_spans.append(index)

    parses = None
    if signature in browser_globals.coarse_signature_to_parse:
      parses = browser_globals.coarse_signature_to_parse[signature]
    else:
      parses = browser_globals.full_signature_to_parse[signature]

    allowed_match_types = set(form.getvalue("recordio_match_types").split(","))
    store = browser_globals.store
    writer = sling.RecordWriter(filename)

    counts = defaultdict(int)
    categories_seen = set()
    for (category_qid, category_frame, parse) in parses:
      # Skip category if we shouldn't extract facts for its members.
      if not all_chosen and category_qid not in chosen_categories:
        continue

      if category_qid in categories_seen:
        continue

      categories_seen.add(category_qid)
      for index in chosen_spans:
        span = parse.spans[index]
        pid = span.pids
        qid = span.qid

        # Can't upload multi-hop facts yet.
        if len(pid) > 1:
          continue

        pid = pid[0]
        matches = util.fact_matches_for_span(span, max_examples=-1)
        for match_type, examples in matches.examples.iteritems():
          if match_type not in allowed_match_types:
            continue

          for member in examples:
            frame = store.frame([("item", member)])
            frame.facts = store.frame([(pid, qid)])
            frame.provenance = store.frame([
                ("category", category_qid),
                ("method", "Member of Category:%s" % category_frame.name)
            ])
            frame.comment = "%s : %s = %s" % \
                (member.name if member.name is not None else member, \
                 span.signature,
                 qid.name if qid.name is not None else qid)
            counts[(pid, match_type)] += 1
            writer.write(member.id, frame.data(binary=True))
    writer.close()
    self._text("Wrote recordio to: " + str(filename))
    self._br()
    self._text("Fact counts by property: %s" % dict(counts))


  # Overridden method for generating the head of the response.
  def do_HEAD(self):
    self._set_headers()


  # Overridden method for responding to GET requests. This is called at the very
  # start when the browser is loaded, and also to get the page's thumbnail icon.
  def do_GET(self):
    self._set_headers()
    if self.path == "/":
      # For the first landing on the page, just generate the empty form.
      self.write_entry_page()
      self._end(["body", "html"])


  # Overridden method for responding to POST requests. This is the main method,
  # which is called whenever the form is submitted.
  def do_POST(self):
    self._set_headers()

    # Parse the form fields.
    form = cgi.FieldStorage(
        fp=self.rfile,
        headers=self.headers,
        environ={'REQUEST_METHOD': 'POST'}
    )

    recordio = form.getvalue("main_form_functionality") == "recordio"
    if recordio:
      self.write_recordio(form)
      return

    # Mirror the filled out form in the response.
    self.write_entry_page(form)

    # See if the input is a category name or qid.
    main_input = form.getvalue("main_form_input")
    if main_input == "top":
      self.handle_top_signatures(form)
    else:
      if main_input in browser_globals.category_name_to_qid:
        main_input = browser_globals.category_name_to_qid[main_input]
        self.handle_category(main_input, form)
      elif main_input[0] == 'Q' and main_input[1:].isdigit():
        if main_input in browser_globals.category_frame:
          self.handle_category(main_input, form)
        else:
          self.bad_input("Can't find category: %s" % main_input)
      elif main_input in browser_globals.coarse_signature_to_parse:
        self.handle_signature(main_input, \
            browser_globals.coarse_signature_to_parse[main_input], form)
      elif main_input in browser_globals.full_signature_to_parse:
        self.handle_signature(main_input, \
          browser_globals.full_signature_to_parse[main_input], form)
      else:
        self.bad_input("Unknown input: %s" % main_input)
    self._end(["body", "html"])


  # Generates an error message for bad inputs.
  def bad_input(self, message):
    self._color_text(message, color="red")


  # Writes fact-match counts as table cells.
  # Hovering on the cell exposes a list of examples.
  def write_fact_match_counts(self, match_counts):
    for t in FactMatchType:
      count = "-"
      examples = []
      if t.name in match_counts.counts:
        count = match_counts.counts[t.name]
        examples = match_counts.examples[t.name]
      self._begin("td class='numeric'")
      self._begin("div class='fact_match_count'")
      self._text(count)
      if len(examples) > 0:
        self._begin("div class='fact_match_examples'")
        self._tag("b", "Exemplar source items")
        self._br()
        self._begin("ul")
        for example in examples:
          link = "https://www.wikidata.org/wiki/%s" % example.id
          self._begin("li")
          self._tag("a", example.id, target="_blank", href=link)
          self._end("li")
        self._end("ul")
        self._end("div")
      self._end(["div", "td"])


  # Writes table header for the main table.
  def write_main_table_header(self, *column_groups):
    self._begin("table class='main_table'")
    self._begin("thead")
    for index, group in enumerate(column_groups):
      if group is None:
        continue
      if index != 0:
        self._separator()
      if type(group) is str:
        self._add_column(group)
      else:
        for column in group:
          self._add_column(column)
    self._end("thead")


  # Handler for 'top signatures' query.
  def handle_top_signatures(self, form):
    max_rows = 200
    fact_weights = self.fact_match_weights(form)
    score_type = form.getvalue("main_form_sort_metric")
    signature_type = form.getvalue("main_form_signature_type")

    mapping = browser_globals.full_signature_to_parse
    if signature_type == "coarse":
      mapping = browser_globals.coarse_signature_to_parse

    # Sort all (signature, category, parse) tuples with the specified metric.
    scored = []
    for signature, parse_list in mapping.iteritems():
      for qid, category, parse in parse_list:
        score = self.parse_score(category, parse, score_type, fact_weights)
        scored.append((signature, qid, category, parse, score))
    scored.sort(key=lambda x: -x[4])

    # Group by signature, and aggregate relevant information.
    all_stats = {}
    seen = set()
    for signature, qid, category, parse, score in scored:
      if signature not in all_stats:
        all_stats[signature] = SignatureStats()
      stats = all_stats[signature]
      stats.example(qid, category, parse)
      key = (signature, qid)
      if key not in seen:
        seen.add(key)
        stats.members += len(category.members)
        stats.score += score
        stats.num += 1
        util.fact_matches_for_parse(parse, stats.fact_matches)

    # Take only the 'max_rows' top signatures as per the aggregated scores.
    all_stats = list(all_stats.iteritems())
    all_stats.sort(key=lambda x: -x[1].score)
    all_stats = all_stats[:max_rows]

    # Display them in a tabular form.
    self.write_main_table_header(
      ["Signature", "Example Category", "Score", "#Members / #Categories"],
      [t.name for t in FactMatchType])

    for signature, stats in all_stats:
      self._begin("tr")
      self._begin("td")
      self._form_anchor(signature, signature)
      self._end("td")
      self._begin("td")
      self._form_anchor(stats.example_category.name, stats.example_qid)
      self._end("td")
      self._cell(stats.score, numeric=True)
      self._cell("%d / %d" % (stats.members, stats.num), numeric=True)
      self._separator(header=False)
      self.write_fact_match_counts(stats.fact_matches)
      self._end("tr")
    self._end("table")


  # Handler for signature inputs.
  def handle_signature(self, signature, categories, form):
    score_type = form.getvalue("main_form_sort_metric")
    fact_weights = self.fact_match_weights(form)
    signature_type = form.getvalue("main_form_signature_type")

    # Sort all parses with this signature.
    output = []
    for (qid, category, parse) in categories:
      score = self.parse_score(category, parse, score_type, fact_weights)
      output.append((qid, category, parse, score))
    output.sort(key=lambda x: -x[3])

    # Get fact-matching statistics. Consider only the top parse for a category
    # if it has >1 parses with the same signature.
    category_count = defaultdict(int)
    match_counts = util.MatchCounts()
    span_match_counts = defaultdict(util.MatchCounts)
    num_members = 0
    for qid, category, parse, score in output:
      category_count[qid] += 1
      if category_count[qid] > 1:
        continue
      num_members += len(category.members)
      util.fact_matches_for_parse(parse, match_counts)
      for span in parse.spans:
        span_signature = util.span_signature(span, signature_type)
        util.fact_matches_for_span(span, span_match_counts[span_signature])

    # Write a summary.
    self._tag("div",
      "<b>%s</b>: in %d (category, parse) pairs across %d categories" % \
      (signature, len(categories), len(category_count)))
    self._tag("b", "#Items across categories: ")
    self._text("%d (incl. possible duplicates)" % num_members)

    self._br()
    self._tag("b", "Span-level fact-matching statistics")
    self.write_main_table_header(
      ["Span Signature"],
      [t.name for t in FactMatchType])
    for span_signature, span_matches in span_match_counts.iteritems():
      self._begin("tr")
      self._cell(span_signature)
      self._separator(header=False)
      self.write_fact_match_counts(span_matches)
      self._end("tr")
    self._begin("tr")
    self._cell("All")
    self._separator(header=False)
    self.write_fact_match_counts(match_counts)
    self._end("tr")
    self._end("table")

    # Give an option to generate a recordio file.
    if signature_type == "full":
      self._br()
      self._begin("table class='main_table'")
      self._begin("tr")
      self._begin("td")
      self._begin("form", id="recordio_form", method="POST", action="", \
                  target="_blank")
      self._begin_end("input", type="hidden", name="main_form_functionality", \
                      value="recordio")
      self._begin_end("input", type="hidden", name="recordio_signature", \
                      id="recordio_signature", value=signature)
      self._tag("b", "Generate recordio for this signature")
      self._br()
      self._br()
      self._text("Filename: ")
      filename = "local/data/e/wikicat/" + signature.replace("$", "_") + ".rec"
      self._begin_end("input", type="text", size=100, \
                      name="recordio_filename", value=filename)
      self._br()
      self._text("Category QIDs ('ALL' for all): ")
      self._begin_end("input", type="text", size=100, value="ALL", \
                      name="recordio_categories")
      self._br()
      self._text("Generate facts for these types:")
      self._begin_end("input", type="text", size=100, \
                      value="NEW,ADDITIONAL,SUBSUMED_BY_EXISTING", \
                      name="recordio_match_types")
      self._br()
      self._text("Generate the following facts:")
      self._br()
      count = 0
      for token in signature.split():
        if token[0] == '$' and token[1:].find("$") >= 0:
          name = "recordio_span%d" % count
          self._text("&nbsp;&nbsp;" + token + " ")
          self._begin_end("input", type="checkbox", name=name, id=name, \
                          checked="on")
          self._br()
          count += 1
      self._begin_end("input", type="hidden", name="recordio_num_spans", \
                      id="recordio_total_spans", value=count)
      self._begin_end("input", type="submit")
      self._end(["form", "td", "tr", "table"])

    # Write the individual parses in a tabular form.
    self._br()
    self._tag("b", "Categories with parses matching '" + signature + "'")
    seen = set()
    max_rows = 200
    self.write_main_table_header(
      ["Category", "Prelim parse score", "#Members", "Fact-matching score"],
      [t.name for t in FactMatchType])
    row_count = 0
    for qid, category, parse, score in output:
      if row_count >= max_rows:
        break
      if qid in seen:
        continue
      row_count += 1
      seen.add(qid)
      self._begin("tr")
      self._begin("td")
      self._form_anchor(qid + ": " + category.name, qid)
      if category_count[qid] > 1:
        more = category_count[qid] - 1
        self._text(" (%d more parse%s)" % (more, "" if more == 1 else "s"))
      self._end("td")
      self._cell(parse.score, numeric=True)
      self._cell(len(category.members), numeric=True)
      self._cell("%.4f" % self.parse_fact_score(parse, fact_weights), True)

      self._separator(header=False)
      counts = util.fact_matches_for_parse(parse)
      self.write_fact_match_counts(counts)
      self._end("tr")
    self._end("table")


  # Handler for category inputs.
  # Writes all the parses for a given category qid.
  def handle_category(self, qid, form):
    def is_on(name):
      return form.getvalue("main_form_" + name) == "on"

    # Various options.
    show_span_qid = is_on("show_span_qid")
    show_prelim_parse_scores = is_on("show_prelim_parse_scores")
    show_span_scores = is_on("show_span_scores")
    show_fact_matches = is_on("show_fact_matching_statistics")
    show_span_fact_matches = is_on("show_span_fact_match_stats")
    show_similar_categories = is_on("show_similar_categories")
    signature_type = form.getvalue("main_form_signature_type")
    metric = form.getvalue("main_form_sort_metric")
    fact_weights = self.fact_match_weights(form)

    frame = browser_globals.category_frame[qid]
    document = sling.Document(frame=frame.document)

    num = len([p for p in frame("parse")])
    self._tag("div", "<b>%s = %s</b>: %d members, %d parses" % \
              (qid, frame.name, len(frame.members), num))
    self._br()

    # Write the parses in a tabular format.
    show_prelim_parse_scores &= metric != "prelim_parse_score"
    self.write_main_table_header(
      "Signature",
      [t.word for t in document.tokens],
      "Metric",
      "Prelim Scores" if show_prelim_parse_scores else None,
      [t.name for t in FactMatchType] if show_fact_matches else None,
      "Matching Categories" if show_similar_categories else None)

    # Each parse is written as one row.
    parses = [(parse, self.parse_score(frame, parse, metric, fact_weights)) \
      for parse in frame("parse")]
    parses.sort(key=lambda x: -x[1])
    for parse, metric_value in parses:
      signature = util.parse_signature(parse, signature_type)

      self._begin("tr")
      self._begin("td")
      self._form_anchor(signature, signature)
      self._end("td")
      self._separator(header=False)
      prev_span_end = -1
      for span in parse.spans:
        for index in xrange(prev_span_end + 1, span.begin):
          self._empty_cell()

        self._begin("td", colspan=span.end-span.begin, align='middle')
        text = util.span_signature(span, signature_type)
        if show_span_qid:
          text += " (" + str(span.qid) + ")"
        title = '.'.join([str(p) for p in span.pids]) + ' = ' + str(span.qid)
        if "name" in span.qid:
          title += " (" + span.qid[name] + ")"
        self._tag("span", text, title=title)

        if show_span_scores and "prior" in span:
          self._br()
          self._text("%s = %0.4f" % ("prior", span.prior))

        if show_span_fact_matches:
          local_counts = util.fact_matches_for_span(span)
          self._br()
          self._begin("table class='span_fact_match'")
          self._begin("thead")
          for t in FactMatchType:
            self._tag("th", t.name)
          self._end("thead")
          self._begin("tr")
          self.write_fact_match_counts(local_counts)
          self._end(["tr", "table"])

        self._end("td")
        prev_span_end = span.end - 1

      for index in xrange(prev_span_end + 1, len(document.tokens)):
        self._empty_cell()

      self._separator(header=False)
      if type(metric_value) is int:
        self._cell(metric_value)
      else:
        self._cell("%.4f" % metric_value)

      if show_prelim_parse_scores:
        self._separator(header=False)
        self._begin("td class='numeric'")
        for score_type in ["prior", "member_score", "cover"]:
          if score_type in parse:
            self._text("%s = %0.4f" % (score_type, parse[score_type]))
            self._br()
        if "score" in parse:
          self._color_text("Overall = %0.4f" % parse.score, "blue")
        self._end("td")

      if show_fact_matches:
        self._separator(header=False)
        total_fact_counts = util.fact_matches_for_parse(parse)
        self.write_fact_match_counts(total_fact_counts)

      if show_similar_categories:
        self._separator(header=False)
        self._begin("td")
        limit = 5
        signature_mapping = browser_globals.full_signature_to_parse
        if signature_type == "coarse":
          signature_mapping = browser_globals.coarse_signature_to_parse
        seen = set()
        for (other_qid, other_category, other_parse) in \
          signature_mapping[signature]:
          if len(seen) >= limit:
            break
          if other_qid != qid and other_qid not in seen:
            seen.add(other_qid)
            self._text(other_category.name)
            self._form_anchor(" (= %s)" % other_qid, other_qid)
            self._text(" (%0.4f)" % other_parse.score)
            self._br()
        self._end("td")
      self._end("tr")
    self._end("table")


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
               default="local/data/e/wikicat/",
               type=str,
               metavar="DIR")
  flags.parse()
  log.info('Reading parses from %s' % flags.arg.parses)
  browser_globals.init(flags.arg.parses, flags.arg.output)
  server_address = ('', flags.arg.port)
  httpd = HTTPServer(server_address, Browser)
  log.info('Starting HTTP Server on port %d' % flags.arg.port)
  httpd.serve_forever()
