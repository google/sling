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

"""Class for updating wikidata with extracted facts from a record file."""
import pywikibot
import sling
import json
import sys
import datetime
import sling.flags as flags

flags.define("--first",
             help="first record to update",
             default=0,
             type=int)

flags.define("--last",
             help="last record to update",
             default=sys.maxsize,
             type=int)

flags.define("--test",
             help="use test record file",
             default=False,
             action='store_true')

flags.define("--verbose",
             help="verbose mode",
             default=False,
             action='store_true')

flags.define("--batch",
             help="number of records to update",
             default=3,
             type=int)

flags.define("--input",
             help="File in 'local/data/e/wikibot/' to process",
             default="birth-dates",
             type=str)


precision_map = {
  sling.MILLENNIUM: pywikibot.WbTime.PRECISION['millenia'],
  sling.CENTURY: pywikibot.WbTime.PRECISION['century'],
  sling.DECADE: pywikibot.WbTime.PRECISION['decade'],
  sling.YEAR: pywikibot.WbTime.PRECISION['year'],
  sling.MONTH: pywikibot.WbTime.PRECISION['month'],
  sling.DAY: pywikibot.WbTime.PRECISION['day']
}

class StoreFactsBot:
  def __init__(self):
    self.site = pywikibot.Site("wikidata", "wikidata")
    self.repo = self.site.data_repository()

    time_str = datetime.datetime.now().isoformat("-")[:19].replace(":","-")
    if flags.arg.test:
      record_file_name = "local/data/e/wikibot/test-" + flags.arg.input + ".rec"
      time_str = "test-" + time_str
    else:
      record_file_name = "local/data/e/wikibot/" + flags.arg.input + ".rec"
    status_file_name = "local/logs/wikibotlog-" + time_str + ".rec"
    self.record_file = sling.RecordReader(record_file_name)
    self.status_file = sling.RecordWriter(status_file_name)

    self.store = sling.Store()
    self.store.lockgc()
    print("loading kb")
    self.store.load("local/data/e/wiki/kb.sling")
    print("kb loaded")

    self.page_cat = self.store["/wp/page/category"]

    self.date_of_birth = self.store['P569']
    self.date_of_death = self.store['P570']

    self.n_item = self.store["item"]
    self.n_facts = self.store["facts"]
    self.n_provenance = self.store["provenance"]
    self.n_category = self.store["category"]
    self.n_url = self.store["url"]
    self.n_method = self.store["method"]
    self.n_status = self.store["status"]
    self.n_revision = self.store["revision"]
    self.n_skipped = self.store["skipped"]
    self.store.freeze()
    self.rs = sling.Store(self.store)

    self.wiki = {'fr': 'Q8447',    'en': 'Q328',    'da': 'Q181163', \
                 'pt': 'Q11921',   'fi': 'Q175482', 'es': 'Q8449', \
                 'pl': 'Q1551807', 'de': 'Q48183',  'nl': 'Q10000', \
                 'sv': 'Q169514',  'it': 'Q11920',  'no': 'Q191769'}
    self.languages = self.wiki.keys()
    self.wiki_sources = {}
    for lang, wp in self.wiki.items():
      # P143 means 'imported from Wikimedia project'
      source_claim = pywikibot.Claim(self.repo, "P143")
      target = pywikibot.ItemPage(self.repo, wp)
      source_claim.setTarget(target)
      self.wiki_sources[lang] = source_claim
    self.record_db = {}
    fname = "local/data/e/wiki/{}/documents@10.rec"
    for lang in self.languages:
      self.record_db[lang] = sling.RecordDatabase(fname.format(lang))

    # inferred from
    self.source_claim = pywikibot.Claim(self.repo, "P3452")
    # Wikimedia import URL
    self.url_source_claim = pywikibot.Claim(self.repo, "P4656")
    # imported from Wikimedia project
    self.wp_source_claim = pywikibot.Claim(self.repo, "P143")
    self.en_wp = pywikibot.ItemPage(self.repo, "Q328")
    self.wp_source_claim.setTarget(self.en_wp)

    # referenced (on)
    self.time_claim = pywikibot.Claim(self.repo, "P813")
    today = datetime.date.today()
    time_target = pywikibot.WbTime(year=today.year,
                                   month=today.month,
                                   day=today.day)
    self.time_claim.setTarget(time_target)

    self.uniq_prop = {self.date_of_birth, self.date_of_death}
    kb = self.store
    # Collect unique-valued properties.
    # They will be used to update claims in Wikidata accordingly.
    constraint_role = kb["P2302"]
    unique = kb["Q19474404"]         # single-value constraint
    for prop in kb["/w/entity"]("role"):
      for constraint_type in prop(constraint_role):
        if kb.resolve(constraint_type) == unique:
          self.uniq_prop.add(prop)

  def __del__(self):
    self.status_file.close()
    self.record_file.close()

  def get_sources(self, h_item, category):
    sources = [self.time_claim]
    cat_item = pywikibot.ItemPage(self.repo, category)
    if cat_item.exists() and not cat_item.isRedirectPage():
      source_claim = pywikibot.Claim(self.repo, "P3452") # inferred from
      source_claim.setTarget(cat_item)
      sources.append(source_claim)
    h_cat = self.store[category]
    for lang in self.languages:
      item_doc = self.record_db[lang].lookup(h_item.id)
      if item_doc is None: continue
      item_frame = sling.Store(self.store).parse(item_doc)
      if h_cat in item_frame(self.page_cat):
        sources.append(self.wiki_sources[lang])
    return sources

  def get_url_sources(self, url):
    if '"' in url: url = "https://en.wikipedia.org"
    self.url_source_claim.setTarget(url)
    return [self.url_source_claim, self.time_claim]

  def get_wp_sources(self):
    return [self.wp_source_claim]

  def all_WP(self, sources):
    if not sources: return True
    for source in sources:
      if source and "P143" not in source:
        if "P3452" not in source: return False
        for claim in source["P3452"]:
          target = claim.getTarget()
          target.get()
          if not target.labels["en"].startswith("Category:"):
            return False
    return True

  def ever_had_prop(self, wd_item, prop):
    # Up to 150 revisions covers the full history of 99% of e.g. human items
    revisions = wd_item.revisions(total=150, content=True)
    for revision in revisions:
      try:
        revision_text = json.loads(revision.text)
        claims = revision_text['claims']
      except:
        pass # unable to extract claims - move to next revision
      else:
        if prop in claims: return True
    return False

  def ever_had(self, wd_item, prop, target):
    # Up to 150 revisions covers the full history of 99% human items
    revisions = wd_item.revisions(total=150, content=True)
    for revision in revisions:
      try:
        revision_text = json.loads(revision.text)
        claims = revision_text['claims']
      except:
        continue # unable to extract claims - move to next revision
      if prop not in claims: continue
      for claim in claims[prop]:
        try:
          old_claim = pywikibot.Claim.fromJSON(self.repo, claim)
        except:
          continue # unable to extract prop claim - move to next prop claim
        if old_claim.target_equals(target): return True
    return False

  def log_status_skip(self, item, facts, error):
    if flags.arg.verbose: print("Skipping", item.id, " -- ", facts, error)
    status_record = self.rs.frame({
      self.n_item: item,
      self.n_facts: facts,
      self.n_status: self.rs.frame({self.n_skipped: error})
    })
    self.status_file.write(str(item.id), status_record.data(binary=True))

  def log_status_stored(self, item, facts, rev_id):
    if flags.arg.verbose: print("Storing", item.id, " -- ", facts)
    url = "https://www.wikidata.org/w/index.php?title="
    url += str(item.id)
    url += "&type=revision&diff="
    url += rev_id
    status_record = self.rs.frame({
      self.n_item: item,
      self.n_facts: facts,
      self.n_status: self.rs.frame({
        self.n_revision: rev_id,
        self.n_url: url
      })
    })
    self.status_file.write(str(item.id), status_record.data(binary=True))

  def get_wbtime(self, date):
    precision = precision_map[date.precision] # sling to wikidata
    if date.precision <= sling.YEAR:
      return pywikibot.WbTime(year=date.year, precision=precision)
    if date.precision == sling.MONTH:
      return pywikibot.WbTime(year=date.year,
                              month=date.month,
                              precision=precision)
    if date.precision == sling.DAY:
      return pywikibot.WbTime(year=date.year,
                              month=date.month,
                              day=date.day,
                              precision=precision)
    return None

  def store_records(self, records, batch_size=3):
    updated = 0
    recno = 0
    url_prefix = "https://www.wikidata.org/wiki/"
    oldval = ""
    for item_bytes, record in records:
      item_str = item_bytes.decode()
      recno += 1
      if recno < flags.arg.first:
        print("Skipping record number", recno)
        continue
      if recno > flags.arg.last: break
      if updated >= batch_size:
        print("Hit batch size of", batch_size)
        break
      print(updated, "records updated. Processing", url_prefix + item_str)
      fact_record = self.rs.parse(record)
      item = fact_record[self.n_item]
      facts = fact_record[self.n_facts]
      provenance = fact_record[self.n_provenance]
      if self.rs[item_str] != item:
        self.log_status_skip(item, facts, "inconsistent input")
        continue # read next record in the file
      wd_item = pywikibot.ItemPage(self.repo, item_str)
      if not wd_item.exists():
        self.log_status_skip(item, facts, "page does not exist")
        continue
      if wd_item.isRedirectPage():
        self.log_status_skip(item, facts, "redirect page")
        continue
      try:
        wd_item.get()
        wd_claims = wd_item.claims
      except:
        self.log_status_skip(item, facts, "exception getting claims")
        continue
      # Process facts / claims
      for prop, val in facts:
        prop_str = str(prop.id)
        fact = self.rs.frame({prop: val})
        claim = pywikibot.Claim(self.repo, prop_str)
        if provenance[self.n_category] and val != oldval:
          oldval = val
          print(provenance[self.n_category],
                self.store[provenance[self.n_category]].name)
        if prop in self.uniq_prop:
          if prop_str not in wd_claims:
            if self.ever_had_prop(wd_item, prop_str):
              self.log_status_skip(item, fact, "already had property")
              continue
          if claim.type == "time":
            date = sling.Date(val) # parse date from val
            target = self.get_wbtime(date)
            if target is None:
              self.log_status_skip(item, facts, "date precision exception")
              continue
            if prop_str in wd_claims:
              if len(wd_claims[prop_str]) > 1: # more than one property already
                self.log_status_skip(item, fact, "has property more than once")
                continue
              old = wd_claims[prop_str][0].getTarget()
              if old is None:
                print("+++++++++++++++++++++++++++++++++++++++++++", \
                  item_str, "had unknown date", \
                  "+++++++++++++++++++++++++++++++++++++++++++")
                continue
              if old.precision >= target.precision:
                err_str = "precise date already exists"
                self.log_status_skip(item, fact, err_str)
                continue
              if old.year != date.year:
                self.log_status_skip(item, fact, "conflicting year in date")
                continue
              if old.precision >= pywikibot.WbTime.PRECISION['month'] and \
                 old.month != date.month:
                self.log_status_skip(item, fact, "conflicting month in date")
                continue
              # Item already has property with a same year less precise date.
              # Ensure sources are all WP or empty
              if not self.all_WP(wd_claims[prop_str][0].getSources()):
                self.log_status_skip(item, fact, "date with non-WP source(s)")
                continue
              print("++++++++++++++ Removing old date for", item_str)
              wd_item.removeClaims(wd_claims[prop_str])
          elif claim.type == 'wikibase-item':
            if prop_str in wd_claims:
              self.log_status_skip(item, fact, "already has property")
              continue
            target = pywikibot.ItemPage(self.repo, val.id)
          else:
            # TODO add location and possibly other types
            print("Error: Unknown claim type", claim.type)
            continue
        else: # property not unique
          if claim.type == 'wikibase-item':
            target = pywikibot.ItemPage(self.repo, val.id)
          elif claim.type == "time":
            target = self.get_wbtime(val)
            if target is None:
              self.log_status_skip(item, facts, "date precision exception")
              continue
          else:
            # TODO add location and possibly other types
            print("Error: Unknown claim type", claim.type)
            continue
          if prop_str in wd_claims:
            old_fact = False
            for clm in wd_claims[prop_str]:
              if clm.target_equals(target):
                self.log_status_skip(item, fact, "already has fact")
                old_fact = True
            if old_fact: continue
          if self.ever_had(wd_item, prop_str, target):
            self.log_status_skip(item, fact, "already had fact")
            continue
        if provenance[self.n_category]:
          s = str(provenance[self.n_category])
          sources = self.get_sources(item, s)
        elif provenance[self.n_url]:
          s = str(provenance[self.n_url])
          sources = self.get_wp_sources()
        else:
          continue
        summary = provenance[self.n_method] + " " + s
        claim.setTarget(target)
        if len(sources) > 0:
          claim.addSources(sources)
        wd_item.addClaim(claim, summary=summary)
        rev_id = str(wd_item.latest_revision_id)
        self.log_status_stored(item, fact, rev_id)
        updated += 1
      print("Record: ", recno, end=' ')
      if flags.arg.verbose: print("Item:", item.id, "Record: ", recno)
    print("Last record:", recno, "Total:", updated, "records updated.")


  def run(self):
    self.store_records(self.record_file, flags.arg.batch)

if __name__ == '__main__':
  flags.parse()
  sfb = StoreFactsBot()
  sfb.run()

