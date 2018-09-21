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
      record_file_name = "local/data/e/wikibot/test-birth-dates.rec"
      time_str = "test-" + time_str
    else:
      record_file_name = "local/data/e/wikibot/birth-dates.rec"
    status_file_name = "local/data/e/wikibot/wikibotlog-"+time_str+".rec"

    self.record_file = sling.RecordReader(record_file_name)
    self.status_file = sling.RecordWriter(status_file_name)

    self.store = sling.Store()
    self.n_item = self.store["item"]
    self.n_facts = self.store["facts"]
    self.n_provenance = self.store["provenance"]
    self.n_category = self.store["category"]
    self.n_method = self.store["method"]
    self.n_status = self.store["status"]
    self.n_revision = self.store["revision"]
    self.n_url = self.store["url"]
    self.n_skipped = self.store["skipped"]
    self.store.freeze()
    self.rs = sling.Store(self.store)

    self.source_claim = pywikibot.Claim(self.repo, "P3452") # inferred from
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
      if recno < flags.arg.first: continue
      if recno > flags.arg.last: break
      recno += 1
      if updated >= batch_size:
        print "Hit batch size of", batch_size
        break
      print "Processing", item_str
      fact_record = self.rs.parse(record)
      item = fact_record[self.n_item]
      facts = fact_record[self.n_facts]
      provenance = fact_record[self.n_provenance]
      if self.rs[item_str] != item:
        self.log_status_skip(item, facts, "inconsistent input")
        continue # read next record in the file
      wd_item = pywikibot.ItemPage(self.repo, item_str)
      if wd_item.isRedirectPage():
        self.log_status_skip(item, facts, "redirect page")
        continue
      wd_claims = wd_item.get().get('claims')
      # Process facts / claims
      for prop, val in facts:
        prop_str = str(prop)
        fact = self.rs.frame({prop: val})
        if prop_str in wd_claims:
          self.log_status_skip(item, fact, "already has property")
          continue
        if self.ever_had_prop(wd_item, prop_str):
          self.log_status_skip(item, fact, "already had property")
          continue
        claim = pywikibot.Claim(self.repo, prop_str)
        if claim.type == "time":
          date = sling.Date(val) # parse date from record
          precision = precision_map[date.precision] # sling to wikidata
          target = pywikibot.WbTime(year=date.year, precision=precision)
        elif claim.type == 'wikibase-item':
          target = pywikibot.ItemPage(self.repo, val)
        else:
          # TODO add location and possibly other types
          print "Error: Unknown claim type", claim.type
          continue
        claim.setTarget(target)
        cat_str = str(provenance[self.n_category])
        summary = provenance[self.n_method] + " " + cat_str
        wd_item.addClaim(claim, summary=summary)
        rev_id = str(wd_item.latest_revision_id)
        claim.addSources(self.get_sources(cat_str))
        self.log_status_stored(item, fact, rev_id)
        updated += 1
      print item
    print "Last record.", updated, "records updated."

  def run(self):
    self.store_records(self.record_file, batch_size=2)

if __name__ == '__main__':
  flags.parse()
  sfb = StoreFactsBot()
  sfb.run()

