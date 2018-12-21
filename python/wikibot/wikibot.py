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
             default=sys.maxint,
             type=int)

flags.define("--test",
             help="use test record file",
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
    self.n_item = self.store["item"]
    self.n_facts = self.store["facts"]
    self.n_provenance = self.store["provenance"]
    self.n_category = self.store["category"]
    self.n_url = self.store["url"]
    self.n_method = self.store["method"]
    self.n_status = self.store["status"]
    self.n_revision = self.store["revision"]
    self.n_url = self.store["url"]
    self.n_skipped = self.store["skipped"]
    self.store.freeze()
    self.rs = sling.Store(self.store)

    self.source_claim = pywikibot.Claim(self.repo, "P3452") # inferred from
    self.url_source_claim = pywikibot.Claim(self.repo, "P4656") # Wm import URL
    self.time_claim = pywikibot.Claim(self.repo, "P813") # referenced (on)
    today = datetime.date.today()
    time_target = pywikibot.WbTime(year=today.year,
                                   month=today.month,
                                   day=today.day)
    self.time_claim.setTarget(time_target)

  def __del__(self):
    self.status_file.close()
    self.record_file.close()

  def get_sources(self, category):
    source_target = pywikibot.ItemPage(self.repo, category)
    self.source_claim.setTarget(source_target)
    return [self.source_claim, self.time_claim]

  def get_url_sources(self, url):
    if '"' in url: url = "https://en.wikipedia.org"
    self.url_source_claim.setTarget(url)
    return [self.url_source_claim, self.time_claim]

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

  def log_status_skip(self, item, facts, error):
    print "Skipping", str(item), " -- ", error
    status_record = self.rs.frame({
      self.n_item: item,
      self.n_facts: facts,
      self.n_status: self.rs.frame({self.n_skipped: error})
    })
    self.status_file.write(str(item), status_record.data(binary=True))

  def log_status_stored(self, item, facts, rev_id):
    url = "https://www.wikidata.org/w/index.php?title="
    url += str(item)
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
    self.status_file.write(str(item), status_record.data(binary=True))

  def store_records(self, records, batch_size=3):
    updated = 0
    recno = 0
    for item_str, record in records:
      recno += 1
      if recno < flags.arg.first:
        print "Skipping record number", recno
        continue
      if recno > flags.arg.last: break
      if updated >= batch_size:
        print "Hit batch size of", batch_size
        break
      print "Processing https://www.wikidata.org/wiki/" + item_str
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
        prop_str = str(prop)
        fact = self.rs.frame({prop: val})
        if prop_str not in wd_claims and self.ever_had_prop(wd_item, prop_str):
          self.log_status_skip(item, fact, "already had property")
          continue
        claim = pywikibot.Claim(self.repo, prop_str)
        if claim.type == "time":
          date = sling.Date(val) # parse date from record
          precision = precision_map[date.precision] # sling to wikidata
          if date.precision <= sling.YEAR:
            target = pywikibot.WbTime(year=date.year, precision=precision)
          elif date.precision == sling.MONTH:
            target = pywikibot.WbTime(year=date.year,
                                      month=date.month,
                                      precision=precision)
          elif date.precision == sling.DAY:
            target = pywikibot.WbTime(year=date.year,
                                      month=date.month,
                                      day=date.day,
                                      precision=precision)
          else:
            self.log_status_skip(item, facts, "date precision exception")
            continue
          if prop_str in wd_claims:
            if len(wd_claims[prop_str]) > 1: # more than one property already
              self.log_status_skip(item, fact, "already has property")
              continue
            old = wd_claims[prop_str][0].getTarget()
            if old is None:
              wd_item.removeClaims(wd_claims[prop_str])
            elif not(old.precision < precision and old.year == date.year):
              self.log_status_skip(item, fact, "already has property")
              continue
            else:
              # item already has property with a same year less precise date
              claim = wd_claims[prop_str][0]
        elif claim.type == 'wikibase-item':
          if prop_str in wd_claims:
            self.log_status_skip(item, fact, "already has property")
            continue
          target = pywikibot.ItemPage(self.repo, val)
        else:
          # TODO add location and possibly other types
          print "Error: Unknown claim type", claim.type
          continue
        if provenance[self.n_category]:
          s = str(provenance[self.n_category])
          sources = self.get_sources(s)
        elif provenance[self.n_url]:
          s = str(provenance[self.n_url])
          sources = self.get_url_sources(s)
        else:
          continue
        summary = provenance[self.n_method] + " " + s
        if prop_str in wd_claims and claim in wd_claims[prop_str]:
          claim.changeTarget(target)
        else:
          claim.setTarget(target)
          wd_item.addClaim(claim, summary=summary)
        rev_id = str(wd_item.latest_revision_id)
        claim.addSources(sources)
        self.log_status_stored(item, fact, rev_id)
        updated += 1
      print item, recno
    print "Last record:", recno, "Total:", updated, "records updated."


  def run(self):
    self.store_records(self.record_file, flags.arg.batch)

if __name__ == '__main__':
  flags.parse()
  sfb = StoreFactsBot()
  sfb.run()

