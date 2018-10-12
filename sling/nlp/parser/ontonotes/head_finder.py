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

"""Finds heads for constituency spans using a head rules table. """

# Types of head rules.
LEFT = 1       # for a given tag, find the leftmost child with that tag
RIGHT = 2      # for a given tag, find the rightmost child with that tag
LEFTDIS = 3    # find the leftmost child with tag amongst given tags
RIGHTDIS = 4   # find the rightmost child with tag amongst given tags


# Head rule.
class Rule:
  def __init__(self, type, tags):
    self.type = type   # e.g. LEFT
    self.tags = tags   # POS tags list


# Head rules table.
# It is a map: constituent tag -> list of rules ordered by descending priority.
class RulesTable:
  # Backoff POS tag.
  UNKNOWN_POS = '<UNK>'

  # Creates an empty rules table.
  def __init__(self):
    self.table = {}
    self.default_rules = []

  # Adds a rule to the table.
  # 'parent' is the constituency tag, rule_type is the type of rule to
  # be added and 'tags' is the comma-separated POS tags string for that rule.
  def add(self, parent, rule_type, tags):
    rules = self.table.get(parent, None)
    if rules is None:
      rules = []
      self.table[parent] = rules
    tags = tags.split(',')

    # Add backoff POS tag if no tags are specified.
    if len(tags) == 0:
      tags.append(RulesTable.UNKNOWN_POS)
    rules.append(Rule(rule_type, tags))

  # Adds a default rule to the table.
  def default(self, type):
    self.default_rules = [Rule(type, tags=[RulesTable.UNKNOWN_POS])]

  # Returns the rules list for the constituency tag 'parent', falling back
  # to the default rules.
  def get(self, parent):
    return self.table.get(parent, self.default_rules)


# Returns Michael Collins' head rules as per his 1999 thesis.
def collins_head_table():
  table = [
    ('ADJP', LEFT, \
     'NNS,QP,NN,$,ADVP,JJ,VBN,VBG,ADJP,JJR,NP,JJS,DT,FW,RBR,RBS,SBAR,RB'),
    ('ADVP', RIGHT, 'RB,RBR,RBS,FW,ADVP,TO,CD,JJR,JJ,IN,NP,JJS,NN'),
    ('CONJP', RIGHT, 'CC,RB,IN'),
    ('FRAG', RIGHT, ''),
    ('INTJ', LEFT, ''),
    ('LST', RIGHT, 'LS,:'),
    ('NAC', LEFT, \
     'NN,NNS,NNP,NNPS,NP,NAC,EX,$,CD,QP,PRP,VBG,JJ,JJS,JJR,ADJP,FW'),
    ('NX', LEFT, ''),
    ('PP', RIGHT, 'IN,TO,VBG,VBN,RP,FW'),
    ('PRN', LEFT, ''),
    ('PRT', RIGHT, 'RP'),
    ('QP', LEFT, '$,IN,NNS,NN,JJ,RB,DT,CD,NCD,QP,JJR,JJS'),
    ('RRC', RIGHT, 'VP,NP,ADVP,ADJP,PP'),
    ('S', LEFT, 'TO,IN,VP,S,SBAR,ADJP,UCP,NP'),
    ('SBAR', LEFT, 'WHNP,WHPP,WHADVP,WHADJP,IN,DT,S,SQ,SINV,SBAR,FRAG'),
    ('SBARQ', LEFT, 'SQ,S,SINV,SBARQ,FRAG'),
    ('SINV', LEFT, 'VBZ,VBD,VBP,VB,MD,VP,S,SINV,ADJP,NP'),
    ('SQ', LEFT, 'VBZ,VBD,VBP,VB,MD,VP,SQ'),
    ('UCP', RIGHT, ''),
    ('VP', LEFT, 'TO,VBD,VBN,MD,VBZ,VB,VBG,VBP,AUX,AUXG,VP,ADJP,NN,NNS,NP'),
    ('WHADJP', LEFT, 'CC,WRB,JJ,ADJP'),
    ('WHADVP', RIGHT, 'CC,WRB'),
    ('WHNP', LEFT, 'WDT,WP,WP$,WHADJP,WHPP,WHNP'),
    ('WHPP', RIGHT, 'IN,TO,FW'),
    ('X', RIGHT, ''),
    ('NP', RIGHTDIS, 'NN,NNP,NNPS,NNS,NX,POS,JJR,NML'),
    ('NP', LEFT, 'NP'),
    ('NP', RIGHTDIS, '$,ADJP,PRN'),
    ('NP', RIGHT, 'CD'),
    ('NP', RIGHTDIS, 'JJ,JJS,RB,QP'),
    ('NML', RIGHTDIS, 'NN,NNP,NNPS,NNS,NX,NML,POS,JJR'),
    ('NML', LEFT, 'NP,PRP'),
    ('NML', RIGHTDIS, '$,ADJP,JJP,PRN'),
    ('NML', RIGHT, 'CD'),
    ('NML', RIGHTDIS, 'JJ,JJS,RB,QP,DT,WDT,RBR,ADVP')
  ]
  rules = RulesTable()
  rules.default(LEFT)
  for t in table:
    rules.add(t[0], t[1], t[2])
  return rules


# Returns Yamada and Matsumoto head rules (Yamada and Matsumoto, IWPT 2003).
def yamada_matsumoto_head_table():
  rules = RulesTable()

  # By default, report the first non-punctuation child from the left as head.
  rules.default(LEFT)

  table = [
    ('NP', RIGHTDIS, 'POS,NN,NNP,NNPS,NNS'),
    ('NP', RIGHT, 'NX,JJR,CD,JJ,JJS,RB,QP,NP'),
    ('ADJP', RIGHT, \
     'NNS,QP,NN,$,ADVP,JJ,VBN,VBG,ADJP,JJR,NP,JJS,DT,FW,RBR,RBS,SBAR,RB'),
    ('ADVP', LEFT, 'RB,RBR,RBS,FW,ADVP,TO,CD,JJR,JJ,IN,NP,JJS,NN'),
    ('CONJP', LEFT, 'CC,RB,IN'),
    ('FRAG', LEFT, ''),
    ('INTJ', RIGHT, ''),
    ('LST', LEFT, 'LS,:'),
    ('NAC', RIGHTDIS, 'NN,NNS,NNP,NNPS'),
    ('NAC', RIGHT, 'NP,NAC,EX,$,CD,QP,PRP,VBG,JJ,JJS,JJR,ADJP,FW'),
    ('PP', LEFT, 'IN,TO,VBG,VBN,RP,FW'),
    ('PRN', RIGHT, ''),
    ('PRT', LEFT, 'RP'),
    ('QP', RIGHT, '$,IN,NNS,NN,JJ,RB,DT,CD,NCD,QP,JJR,JJS'),
    ('RRC', LEFT, 'VP,NP,ADVP,ADJP,PP'),
    ('S', RIGHT, 'TO,IN,VP,S,SBAR,ADJP,UCP,NP'),
    ('SBAR', RIGHT, 'WHNP,WHPP,WHADVP,WHADJP,IN,DT,S,SQ,SINV,SBAR,FRAG'),
    ('SBARQ', RIGHT, 'SQ,S,SINV,SBARQ,FRAG'),
    ('SINV', RIGHT, 'VBZ,VBD,VBP,VB,MD,VP,S,SINV,ADJP,NP'),
    ('SQ', RIGHT, 'VBZ,VBD,VBP,VB,MD,VP,SQ'),
    ('UCP', LEFT, ''),
    ('VP', LEFT, 'VBD,VBN,MD,VBZ,VB,VBG,VBP,VP,ADJP,NN,NNS,NP'),
    ('WHADJP', RIGHT, 'CC,WRB,JJ,ADJP'),
    ('WHADVP', LEFT, 'CC,WRB'),
    ('WHNP', RIGHT, 'WDT,WP,WP$,WHADJP,WHPP,WHNP'),
    ('WHPP', LEFT, 'IN,TO,FW'),
    ('NX', RIGHTDIS, 'POS,NN,NNP,NNPS,NNS'),
    ('NX', RIGHT, 'NX,JJR,CD,JJ,JJS,RB,QP,NP'),
    ('X', RIGHT, '')
  ]
  for t in table:
    rules.add(t[0], t[1], t[2])
  return rules


# Head finder takes a head rules table and a constituency node, and outputs
# the token index of the head for that node. The node is assumed to be a rooted
# tree that goes all the way down to individual tokens with POS tags.
class HeadFinder:
  def __init__(self, statistics=None, rulestype="collins"):
    self.punctuation = [".", ",", "(", ")", ":", "``", "''"]
    self.rules = None
    if rulestype == "collins":
      self.rules = collins_head_table()
    else:
      self.rules = yamada_matsumoto_head_table()

    # Various counters.
    self.num_default = None               # default rule usage
    self.num_backoff = None               # backoff head computation
    self.num_none_heads = None            # no head could be found
    self.num_total = None                 # total invocations
    self.backoff_usage_histogram = None   # constituency tag -> backoff heads
    self.default_usage_histogram = None   # constituency tag -> default rules
    if statistics is not None:
      self.num_default = statistics.counter("HeadFinder/DefaultRuleUsage")
      self.num_backoff = statistics.counter("HeadFinder/BackoffHeads")
      self.num_none_heads = statistics.counter("HeadFinder/NoHead")
      self.num_total = statistics.counter("HeadFinder/Total")
      self.default_usage_histogram = \
          statistics.histogram("HeadFinder/DefaultRuleUsageByTag")
      self.backoff_usage_histogram = \
          statistics.histogram("HeadFinder/BackoffUsageByTag")

  # Returns whether POS tag 'tag' is not a punctuation.
  def not_punctuation(self, tag):
    return tag not in self.punctuation

  # Computes head token for node 'root' as per 'rule'.
  # Nodes are assumed to have the following fields: children, head, label.
  # Children's heads should already have been computed.
  #
  # Returns the tuple (head, whether backoff was used).
  def head_from_rule(self, root, rule, force):
    backoff = None
    children = root.children
    if rule.type in [RIGHT, RIGHTDIS]:
      children = reversed(root.children)   # right->left children traversal

    if rule.type in [LEFT, RIGHT]:
      for tag in rule.tags:
        for child in children:
          if child.label == tag:
            return child.head, False
          if backoff is None and force and self.not_punctuation(child.label):
            backoff = child.head
    else:
      assert rule.type in [LEFTDIS, RIGHTDIS], rule.type
      for child in children:
        for tag in rule.tags:
          if child.label == tag:
            return child.head, False
        if backoff is None and force and self.not_punctuation(child.label):
          backoff = child.head

    if backoff is not None and self.num_backoff is not None:
      self.num_backoff.increment()

    return backoff, backoff is not None

  # Recursively finds the head for all nodes starting at 'root'.
  def find(self, root):
    if root.head is not None:
      return root.head

    if self.num_total is not None:
      self.num_total.increment()

    if root.leaf():
      assert root.begin == root.end - 1   # should be token
      root.head = root.begin              # token is its own head
    elif len(root.children) == 1:
      root.head = self.find(root.children[0])
    else:
      # Find heads of all children.
      for child in root.children:
        self.find(child)

      # Apply rules to select a head.
      rules = self.rules.get(root.label)
      if rules is self.rules.default_rules and self.num_default is not None:
        self.num_default.increment()
        self.default_usage_histogram.increment(root.label)
      for rule in rules:
        head, via_backoff = self.head_from_rule(root, rule, rule == rules[-1])
        if head is not None:
          root.head = head
          if via_backoff and self.backoff_usage_histogram is not None:
            self.backoff_usage_histogram.increment(root.label)
          break

      if root.head is None and self.num_none_heads is not None:
        self.num_none_heads.increment()

    return root.head
