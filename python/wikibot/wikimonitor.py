# Copyright 2018 Google Inc.
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

"""Class defining a dashboard of the status of Sling updates to WikiData."""

import pywikibot
import sling
import sling.flags as flags
import glob
import datetime

flags.define("--test",
             help="use test record file",
             default=False,
             action='store_true')

flags.define("--countonly",
             help="only count changes - no status",
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

class WikiMonitor:
  def __init__(self):
    self.site = pywikibot.Site("wikidata", "wikidata")
    self.repo = self.site.data_repository()

    self.path = "local/logs/"
    test = ""
    if flags.arg.test: test = "test-"
    self.pattern = self.path + "wikibotlog-" + test + "20*.rec"

    self.store = sling.Store()
    self.n_item = self.store["item"]
    self.n_facts = self.store["facts"]
    self.n_status = self.store["status"]
    self.n_revision = self.store["revision"]
    self.n_skipped = self.store["skipped"]
    self.store.freeze()

  def process_log_data(self, files):
    no_of_files = len(files)
    file_no = 0
    rs = sling.Store(self.store)
    skipped = 0
    updated = 0
    errors = 0
    deleted = 0
    changed = 0
    redirected = 0
    updates = {}
    for r_file in files:
      file_no += 1
      print("Processing {:4d} of {} ({})".format(file_no, no_of_files, r_file))
      reader = sling.RecordReader(r_file)
      last_updated = updated
      for item_bytes, record in reader:
        item_str = item_bytes.decode()
        rec = rs.parse(record)
        status = rec[self.n_status]
        if self.n_skipped in status:
          skipped += 1
          continue
        elif self.n_revision not in status:
          print("ERROR - unknown status")
          errors += 1
          continue
        updated += 1
        if flags.arg.countonly: continue
        wd_item = pywikibot.ItemPage(self.repo, item_str)
        if wd_item.isRedirectPage():
          redirected += 1
          continue
        wd_claims = wd_item.get().get('claims')
        facts = rec[self.n_facts]
        for prop, val in facts:
          p_claims =  wd_claims.get(str(prop), [])
          if not p_claims:
            deleted += 1
            continue
          for wd_claim in p_claims:
            if wd_claim.type == "time":
              date = sling.Date(val) # parse date from record
              precision = precision_map[date.precision] # sling to wikidata
              target = pywikibot.WbTime(year=date.year, precision=precision)
              if target.toTimestr() != wd_claim.target.toTimestr():
                print("https://www.wikidata.org/wiki/" + item_str,  str(prop))
                print("Old:", target.toTimestr())
                print("New:", wd_claim.target.toTimestr())
            elif wd_claim.type == 'wikibase-item':
              target = pywikibot.ItemPage(self.repo, val)
            else:
              # TODO add location and possibly other types
              print("Error: Unknown claim type", claim.type)
              continue
            if not wd_claim.target_equals(target):
              changed += 1
      reader.close()
      print(updated - last_updated)
      f = r_file.split("-")
      date = int(f[1] + f[2] + f[3])
      if date not in updates: updates[date] = 0
      updates[date] += (updated - last_updated)
    print(skipped, "skipped,", updated, "updated,", deleted, "deleted,", \
          changed, "changed,", errors, "error records in file")
    print("Done processing last file")
    # Print number of accumulated updates over time
    first = min(updates)
    acc_upd = 0
    d = datetime.date(first // 10000, (first % 10000) // 100, first % 100)
    while d <= datetime.date.today():
      num = d.year * 10000 + d.month * 100 + d.day
      if num in updates: acc_upd += updates[num]
      print(d.strftime("%Y-%m-%d") + "," + str(acc_upd))
      d += datetime.timedelta(days = 1)

  def run(self):
    self.process_log_data(glob.glob(self.pattern))

if __name__ == '__main__':
  flags.parse()
  sfb = WikiMonitor()
  sfb.run()

