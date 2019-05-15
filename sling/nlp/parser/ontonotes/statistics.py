# Copyright 2018 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License")

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

"""Utility classes for generating histograms and counters.
"""

# Returns a list of pretty-formatted strings for an input list of tuples.
#
# For each tuple in 'tuples':
#   The first item is left justified and the second item is right justified.
#   The third item (if present) is treated as a float (e.g. probability).
#   The fourth item (if present) if true, adds an asterix at the end.
def pretty_print(tuples):
  output = []
  if len(tuples) == 0:
    return output

  key_width = 2 + max(map(lambda x: len(str(x[0])), tuples))
  value_width = max(map(lambda x: len(str(x[1])), tuples))
  for t in tuples:
    k = str(t[0]).ljust(key_width)
    v = str(t[1]).rjust(value_width)
    if len(t) > 2 and t[2] is not None:
      if type(t[2]) is float:
        v += '  ' + ("%.2f" % t[2]).rjust(8) + '%'
      else:
        v += str(t[2])
    if len(t) > 3 and t[3] == True:
        v += '  *'
    output.append(k + v)
  return output


# Basic counter.
class Counter:
  def __init__(self, name):
    self.name = name
    self.value = 0

  # Increments the counter.
  def increment(self, count=1):
    self.value += count

# Histogram that holds counts and optionally examples for each bin.
class Histogram:
  class Bin:
    def __init__(self):
      self.count = 0
      self.examples = []
      self.duplicates = False   # whether we have duplicate examples
      self._mark = False        # whether this bin is to be denoted specially

    # Increments the bin.
    def add(self, count=1):
      self.count += count

    # Adds an example to the bin, up to a max of 'limit'.
    def add_example(self, example, limit):
      if len(self.examples) >= limit:
        return False
      for e in self.examples:
        if e == example:
          self.duplicates = True
          return False
      self.examples.append(example)
      return True

    # Marks the bin as special.
    def mark(self):
      self._mark = True


  # Initializes the histogram.
  def __init__(self, name, max_examples=-1, max_examples_per_bin=0):
    self.name = name

    self.bins = {}    # bin name -> bin
    self.total = 0    # total count across bins

    self.max_examples_per_bin = max_examples_per_bin  # max examples per bin
    self.max_examples_total = max_examples            # max overall examples
    self.num_examples = 0                             # total no. of examples

    # Output options. These are denoted via a leading underscore.
    self._example_printer = None   # custom method for printing examples
    self._sort_by = 'count'        # one of ['key'/'bin', 'count'/'value']
    self._max_output_bins = -1     # how many bins to output
    self._extremas = False         # whether to output min/max bins
    self._cdf = True               # whether to output cumulative probabilities

  # Sets output options from 'kwargs'. Only keys that correspond to the output
  # options are allowed, without the leading underscore (see above).
  def set_output_options(self, **kwargs):
    for key, value in kwargs.items():
      assert '_' + key in self.__dict__.keys(), 'Unknown option:' + key
      if value is not None:
        if key == 'sort_by':
          assert value in ['key', 'bin', 'value', 'count'], value
        self.__dict__['_' + key] = value

  # Marks the bin 'key' as special.
  def mark(self, key):
    if key not in self.bins:
      self.bins[key] = Histogram.Bin()
    self.bins[key].mark()

  # Increments the count of bin 'key' and also stores an example for it.
  def increment(self, key, count=1, example=None):
    self.total += count
    if key not in self.bins:
      self.bins[key] = Histogram.Bin()

    b = self.bins[key]
    b.add(count)
    if example is not None and count == 1:
      if self.num_examples < self.max_examples_total or \
        self.max_examples_total < 0:
        if b.add_example(example, self.max_examples_per_bin):
          self.num_examples += 1

  # Returns a string representation of the histogram.
  def __repr__(self):
    output = []
    output.append('Histogram: ' + self.name)
    output.append('-' * len(output[-1]))
    if self.total == 0:
      output.append('<empty>')
      return '\n'.join(output)

    if self._extremas:
      extremas = []
      keys = sorted(self.bins.keys())
      first = keys[0]
      last = keys[-1]
      extremas.append(('First Bin    = ' + str(first), self.bins[first].count))
      extremas.append(('Last Bin     = ' + str(last), self.bins[last].count))
      biggest = max(self.bins.items(), key=lambda x:x[1].count)
      extremas.append(('Biggest Bin  = ' + str(biggest[0]), biggest[1].count))
      smallest = max(self.bins.items(), key=lambda x:-x[1].count)
      extremas.append(('Smallest Bin = ' + str(smallest[0]), smallest[1].count))
      output.extend(pretty_print(extremas))
      output.append('')

    # Output bin counts in requested order.
    items = list(self.bins.items())

    if self._sort_by in ['count', 'value']:
      # Sort in descending order of count.
      items.sort(key=lambda x:-x[1].count)
    else:
      # Sort in increasing order of key.
      items.sort()

    length = len(items)
    running_total = 0
    denom = self.total / 100.0
    for i in range(length):
      key, val = items[i]
      skip = self._max_output_bins >= 0 and i != length - 1 \
          and i >= self._max_output_bins
      if skip and i == self._max_output_bins:
        skip_total = 0
        skipped = items[i:length-1]  # we won't skip the last bin
        mark = False
        for _, b in skipped:
          skip_total += b.count
          if b._mark: mark = True
        running_total += skip_total
        key = '<' + str(len(skipped)) + ' bins>'
        items.append((key, skip_total, running_total / denom, mark))
      elif skip:
        continue
      else:
        running_total += val.count
        items.append((key, val.count, running_total / denom, val._mark))

    items = items[length:]
    items.append(('Total', self.total))
    output.extend(pretty_print(items))

    # Print examples for each bin.
    if self.num_examples > 0:
      output.append('')
      for item in items:
        key = item[0]
        if key not in self.bins: continue
        b = self.bins[key]
        if len(b.examples) > 0:
          output.append('Examples for ' + str(key))
          if b.duplicates:
            output.append('(Ignoring duplicates)')
          for i, example in enumerate(b.examples):
            lines = None
            if self._example_printer is not None:
              lines = self._example_printer(example)
            else:
              lines = [str(example)]
            lines[0] = str(i) + '. ' + lines[0]
            output.extend(lines)
          output.append('')

    return '\n'.join(output)


# A section is a group of histograms and counters.
class Section:
  def __init__(self, name):
    self.name = name
    self.counters = []
    self.histograms = []

  # Adds and returns a counter with the specified name.
  def counter(self, name):
    for c in self.counters:
      if c.name == name: return c
    self.counters.append(Counter(name))
    return self.counters[-1]

  # Adds and returns a histogram with the specified name and args.
  def histogram(self, name, **kwargs):
    for h in self.histograms:
      if h.name == name: return h
    self.histograms.append(Histogram(name, **kwargs))
    return self.histograms[-1]

  # Returns a string representation of the section.
  def __repr__(self):
    output = ('=' * 80) + '\n' + self.name + '\n' + ('=' * 80) + '\n'

    items = [(c.name, c.value) for c in self.counters]
    output += '\n'.join(pretty_print(items))
    if len(self.counters) > 0:
      output += '\n\n'

    output += '\n\n'.join([str(h) for h in self.histograms])
    if len(self.histograms) > 0:
      output += '\n\n'

    return output


# A list of sections.
class Statistics:
  def __init__(self):
    self.sections = []

  # Adds a new section and returns it.
  def section(self, name):
    for s in self.sections:
      if s.name == name:
        return s
    s = Section(name)
    self.sections.append(s)
    return s

  # Adds a new counter and returns it.
  # If the section name is not provided, it is taken from
  # 'name', which is assumed to be of the form '<section_name>/<counter_name>'.
  def counter(self, section_name, name=None):
    if name is None:
      section_name, _, name = section_name.partition('/')
      assert section_name != '', section_name
      assert name != '', section_name
    return self.section(section_name).counter(name)

  # Adds a new histogram and returns it.
  # If the section name is not provided, it is taken from 'name',
  # which is assumed to be of the form '<section_name>/<histogram_name>'.
  def histogram(self, section_name, name=None, **kwargs):
    if name is None:
      section_name, _, name = section_name.partition('/')
      assert section_name != '', section_name
      assert name != '', section_name
    return self.section(section_name).histogram(name, **kwargs)

  # Returns a string representation of the statistics.
  def __repr__(self):
    return '\n'.join([str(s) for s in self.sections]) + '\n'


