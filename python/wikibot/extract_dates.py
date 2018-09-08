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

"""Class for extracting birth date facts and storing them in a record file"""

import sling
import re
import sys
import json

class ExtractDates:
  def __init__(self):
    self.kb = sling.Store()
    self.kb.load("local/data/e/wiki/kb.sling")
    self.instanceof = self.kb['P31']
    self.has_part = self.kb['P527']
    self.part_of = self.kb['P361']
    self.item_category = self.kb['/w/item/category']
    self.date_of_birth = self.kb['P569']
    self.wikimedia_category = self.kb['Q4167836']
    self.date_types = [
      self.kb['Q29964144'], # year BC
      self.kb['Q577'],      # year
      self.kb['Q39911'],    # decade
      self.kb['Q578'],      # century
      self.kb['Q36507'],    # millennium
    ]
    self.human = self.kb['Q5']
    self.item = self.kb["item"]
    self.facts = self.kb["facts"]
    self.provenance = self.kb["provenance"]
    self.category = self.kb["category"]
    self.method = self.kb["method"]

    self.calendar = sling.Calendar(self.kb)
    self.names = sling.PhraseTable(self.kb,
                                   "local/data/e/wiki/en/phrase-table.repo")
    self.kb.freeze()
    self.date_type = {}
    self.conflicts = 0
    self.out_file = "local/data/e/wikibot/birth-dates.rec"

  def find_date(self, phrase):
    for item in self.names.lookup(phrase):
      for cls in item(self.instanceof):
        clsr = cls.resolve()
        if clsr in self.date_types:
          self.date_type[item] = clsr
          return item
    return None

  def dated_categories(self, pattern):
    cats = {}
    pat = re.compile(pattern)

    for item in self.kb:
      is_category = False
      for cls in item(self.instanceof):
        if cls == self.wikimedia_category: is_category = True
      if not is_category: continue
      name = str(item.name)
      m = pat.match(name)  # , re.IGNORECASE
      if m is not None:
        date = self.find_date(m.group(1))
        if date is not None:
          cats[item] = date
    print len(cats), " dated categories found for pattern", pattern
    return cats

  def most_specific_date(self, dates):
    if dates is None: return None
    if len(dates) == 1: return dates[0]
    dt = {}
    for cat, date in dates:
      dti = self.date_type[date]
      if not dti: return None # no date type found for the date
      if dti in dt:
        self.conflicts += 1
        return None # conflict from two dates of the same date type
      dt[dti] = (cat, date)
    # TODO: check that there is consistency across date types, e.g., can't have
    # both born in the year 1911 and in the decade the 1920s
    for i in range(len(self.date_types)):
      if self.date_types[i] in dt:
        return dt[self.date_types[i]]
    return None

  def find_births(self):
    record_file = sling.RecordWriter(self.out_file)
    records = 0

    for item in self.kb:
      if item[self.instanceof] != self.human: continue
      if item[self.date_of_birth] is not None: continue
      cat_dates = []
      # Collect all the item's birth categories in cat_dates
      for cat in item(self.item_category):
        cat_birth_date = self.birth_cats.get(cat)
        if cat_birth_date is None: continue
        cat_dates.append((cat, cat_birth_date))
      if not cat_dates: continue # no birth categories found for item
      msd = self.most_specific_date(cat_dates)
      if msd is None: continue
      (birth_cat, birth_date) = msd
      records += 1
      store = sling.Store(self.kb)
      facts = store.frame({
        self.date_of_birth: self.calendar.value(sling.Date(birth_date))
      })
      provenance = store.frame({
        self.category: birth_cat,
        self.method: "Member of a birth category, '" + birth_cat.name + "'"
      })
      fact = store.frame({
        self.item: item,
        self.facts: facts,
        self.provenance: provenance
      })
      record_file.write(item.id, fact.data(binary=True))

    record_file.close()
    print records, "birth date records written to file:", self.out_file
    print self.conflicts, "conflicts encountered"

  def run(self):
    self.birth_cats = self.dated_categories("Category:(.+) births")
    self.find_births()

if __name__ == '__main__':
  extract_dates = ExtractDates()
  extract_dates.run()

