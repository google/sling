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

"""Class for extracting birth date facts and storing them in a record file"""

import sling
import sling.flags as flags
import re
import sys
import json

flags.define("--prop",
             help="The property to extract dates for",
             default="birth",
             type=str)

class ExtractDates:
  def __init__(self):
    self.kb = sling.Store()
    self.kb.lockgc()
    self.kb.load("local/data/e/wiki/kb.sling", snapshot=True)
    self.instanceof = self.kb['P31']
    self.has_part = self.kb['P527']
    self.part_of = self.kb['P361']
    self.subclass = self.kb['P279']
    self.item_category = self.kb['/w/item/category']
    self.date_of_birth = self.kb['P569']
    self.date_of_death = self.kb['P570']
    self.inception = self.kb['P571']
    self.dad = self.kb['P576'] # dissolved, abolished or demolished
    self.wikimedia_category = self.kb['Q4167836']
    self.date_types = [
      self.kb['Q29964144'], # year BC
      self.kb['Q577'],      # year
      self.kb['Q39911'],    # decade
      self.kb['Q578'],      # century
      self.kb['Q36507'],    # millennium
    ]
    self.human = self.kb['Q5']
    self.business = self.kb['Q4830453']
    self.organization = self.kb['Q43229']
    self.item = self.kb["item"]
    self.facts = self.kb["facts"]
    self.provenance = self.kb["provenance"]
    self.category = self.kb["category"]
    self.method = self.kb["method"]

    self.names = sling.PhraseTable(self.kb,
                                   "local/data/e/wiki/en/phrase-table.repo")
    self.kb.freeze()
    self.date_type = {}
    self.conflicts = 0

    self.months = {
      "January": 1,
      "February": 2,
      "March": 3,
      "April": 4,
      "May": 5,
      "June": 6,
      "July": 7,
      "August": 8,
      "September": 9,
      "October": 10,
      "November": 11,
      "December": 12
    }

  def subclasses(self, cls):
    subcls = {}
    subcls[int(cls.id[1:len(cls.id)])] = True
    changed = True
    while changed:
      changed = False
      for item in self.kb:
        try:
          num = int(item.id[1:len(item.id)])
        except:
          continue
        if num in subcls: continue
        for supercls in item(self.subclass):
          try:
            supernum = int(supercls.id[1:len(supercls.id)])
          except:
            continue
          if supernum in subcls:
            changed = True
            subcls[num] = True
            break
    return subcls

  def find_date(self, phrase):
    for item in self.names.lookup(phrase):
      for cls in item(self.instanceof):
        clsr = cls.resolve()
        if clsr in self.date_types:
          self.date_type[item] = clsr
          return item
    return None

  def dated_categories(self, pattern, group=1):
    cats = {}
    rec = re.compile(pattern)
    for item in self.kb:
      if self.wikimedia_category not in item(self.instanceof): continue
      m = rec.match(str(item.name))  # , re.IGNORECASE
      if m is not None:
        date = self.find_date(m.group(group))
        if date is not None:
          cats[item] = date
    print len(cats), "dated categories found for pattern", pattern
    return cats

  def most_specific_date(self, dates):
    if dates is None: return None
    if len(dates) == 1: return dates[0]
    dt = {}
    for cat, date in dates:
      dti = self.date_type[date]
      if not dti: return None # no date type found for the date
      if dti in dt:
        (old_cat, old_date) = dt[dti]
        if old_date != date:
          self.conflicts += 1
          return None # conflict from two dates of the same date type
      dt[dti] = (cat, date)
    # TODO: check that there is consistency across date types, e.g., can't have
    # both born in the year 1911 and in the decade the 1920s
    for i in range(len(self.date_types)):
      if self.date_types[i] in dt:
        return dt[self.date_types[i]]
    return None

  def is_org(self, item):
    for cls in item(self.instanceof):
      try:
        num = int(cls.id[1:len(cls.id)])
      except:
        continue
      if num in self.org_cls: return True
    return False

  def find_inceptions(self, inc_cats):
    self.out_file = "local/data/e/wikibot/inc-dates.rec"
    record_file = sling.RecordWriter(self.out_file)
    records = 0
    store = sling.Store(self.kb)
    types = {}

    for item in self.kb:
      if self.wikimedia_category in item(self.instanceof): continue
      if self.human in item(self.instanceof): continue
      if not self.is_org(item): continue
      name = item.name
      if name is not None and name.startswith("Category:"): continue
      if item[self.inception] is not None: continue
      cat_dates = []
      # Collect all the item's inception categories in cat_dates
      for cat in item(self.item_category):
        cat_inc_date = inc_cats.get(cat)
        if cat_inc_date is None: continue
        cat_dates.append((cat, cat_inc_date))
      if not cat_dates: continue # no inception categories found for item
      msd = self.most_specific_date(cat_dates)
      if msd is None: continue
      (inc_cat, inc_date) = msd
      records += 1

      facts = store.frame({
        self.inception: sling.Date(inc_date).value()
      })
      provenance = store.frame({
        self.category: inc_cat,
        self.method: "Member of an inception category, '" + inc_cat.name + "'"
      })
      fact = store.frame({
        self.item: item,
        self.facts: facts,
        self.provenance: provenance
      })
      record_file.write(item.id, fact.data(binary=True))

    record_file.close()
    print records, "inception date records written to file:", self.out_file
    print self.conflicts, "conflicts encountered"

  def find_deaths(self, death_cats):
    self.out_file = "local/data/e/wikibot/death-dates.rec"
    record_file = sling.RecordWriter(self.out_file)
    records = 0

    for item in self.kb:
      if self.human not in item(self.instanceof): continue
      if item[self.date_of_death] is not None: continue
      cat_dates = []
      # Collect all the item's death categories in cat_dates
      for cat in item(self.item_category):
        cat_death_date = death_cats.get(cat)
        if cat_death_date is None: continue
        cat_dates.append((cat, cat_death_date))
      if not cat_dates: continue # no death categories found for item
      msd = self.most_specific_date(cat_dates)
      if msd is None: continue
      (death_cat, death_date) = msd
      records += 1
      store = sling.Store(self.kb)
      facts = store.frame({
        self.date_of_death: sling.Date(death_date).value()
      })
      provenance = store.frame({
        self.category: death_cat,
        self.method: "Member of a death category, '" + death_cat.name + "'"
      })
      fact = store.frame({
        self.item: item,
        self.facts: facts,
        self.provenance: provenance
      })
      record_file.write(item.id, fact.data(binary=True))

    record_file.close()
    print records, "death date records written to file:", self.out_file
    print self.conflicts, "conflicts encountered"

  def find_births(self, birth_cats):
    self.out_file = "local/data/e/wikibot/birth-dates.rec"
    record_file = sling.RecordWriter(self.out_file)
    records = 0

    for item in self.kb:
      if self.human not in item(self.instanceof): continue
      if item[self.date_of_birth] is not None: continue
      cat_dates = []
      # Collect all the item's birth categories in cat_dates
      for cat in item(self.item_category):
        cat_birth_date = birth_cats.get(cat)
        if cat_birth_date is None: continue
        cat_dates.append((cat, cat_birth_date))
      if not cat_dates: continue # no birth categories found for item
      msd = self.most_specific_date(cat_dates)
      if msd is None: continue
      (birth_cat, birth_date) = msd
      records += 1
      store = sling.Store(self.kb)
      facts = store.frame({
        self.date_of_birth: sling.Date(birth_date).value()
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
    if flags.arg.prop == "birth":
      birth_cats = self.dated_categories("Category:(.+) births")
      self.find_births(birth_cats)
    elif flags.arg.prop == "death":
      death_cats = self.dated_categories("Category:(.+) deaths")
      self.find_deaths(death_cats)
    elif flags.arg.prop == "inception":
      self.org_cls = self.subclasses(self.organization)
      inc_cats = self.dated_categories("Category:(.+) established in (.+)", 2)
      self.find_inceptions(inc_cats)
    else:
      print "'--prop' flags must be either 'birth', 'death', or 'inception'"

if __name__ == '__main__':
  flags.parse()
  extract_dates = ExtractDates()
  extract_dates.run()

