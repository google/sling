# Copyright 2017 Google Inc.
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

"""Convert PropBank rolesets to SLING schemas.
"""

import xml.etree.ElementTree as ET
import os
import sys

# The PropBank repository is expected to be checked out in a directory parallel
# to the SLING directory. The PropBank frame can be checked out from here:
# https://github.com/propbank/propbank-frames
propbank_path = '../propbank/frames/'

# Output file for SLING schemas.
sling_output_fn = '/tmp/propbank.sling'

# Projections to VerbNet can be added by changing this.
output_vn_projections = False

def StringLiteral(str):
  """Convert string to string literal."""
  literal = '"' + str.replace('\\', '\\\\').replace('"', '\\"') + '"'
  return literal.encode('utf8')

def PropBankId(name):
  """Convert PropBank name to SLING id."""
  underscore = name.find('_')
  period = name.find('.')
  if underscore != -1 and period > underscore:
    name = name[:underscore] + name[period:]
  return '/pb/' + name.replace(".", "-")

def VerbNetId(name):
  """Convert VerbNet id to SLING id."""
  return  '/vn/' + name.strip().lower().replace('.', '-')

def VerbNetRole(name):
  """Convert VerbNet role to SLING id."""
  return  '/vn/class/' + name.lower()

# POS tag mapping.
pos_mapping = {
  'v': 'verb',
  'n': 'noun',
  'j': 'adjective',
}

# Parse all the PropBank frame files.
schemas = []
out = open(sling_output_fn, 'w')
for filename in os.listdir(propbank_path):
  # Read and parse XML file with PropBank roleset definition.
  if not filename.endswith('.xml'): continue
  fn = propbank_path + filename
  tree = ET.parse(fn)
  root = tree.getroot()

  for predicate in root.iterfind('predicate'):
    # Get lemma and POS.
    lemma = predicate.get('lemma').replace("_", " ")

    for roleset in predicate.iterfind('roleset'):
      id = roleset.get('id')
      name = roleset.get('name')
      frameid = PropBankId(id)
      schemas.append(frameid)

      # Output PropBank roleset.
      print >> out, '; PropBank roleset ' + id + ' for \'' + lemma + '\''
      print >> out, '{=' + frameid + ' :/pb/roleset +/pb/frame'

      print >> out, '  name: ' + StringLiteral(id)
      print >> out, '  description: ' + StringLiteral(name)
      print >> out, '  family: /schema/propbank'

      # Output predicates
      aliases = roleset.find('aliases')
      for alias in aliases.iterfind('alias'):
        lemma = alias.text.replace('_', ' ')
        pos = alias.get('pos')
        if pos in pos_mapping:
          print >> out, '  trigger: {lemma: ' + StringLiteral(lemma) + \
                        ' pos: ' + pos_mapping[pos] + '}'

      # Output PropBank roles.
      print >> out
      roles = roleset.find('roles')
      vnmap = {}
      for role in roles.iterfind('role'):
        descr = role.get('descr')
        n = role.get('n')
        f = role.get('f')
        if n == 'm':
          if f is not None:
            n = 'M-' + f.upper()
          else:
            n = 'M'

        arg = 'arg' + n.lower()
        roleid = frameid + '/' + arg

        for vnrole in role.iterfind('vnrole'):
          cls = VerbNetId(vnrole.get('vncls'))
          theta = VerbNetRole(vnrole.get('vntheta'))
          if not cls in vnmap: vnmap[cls] = {}
          rolemap = vnmap[cls]
          rolemap[roleid] = theta

        print >> out, '  role: {=' + roleid + \
                      ' :slot +/pb/' + arg + \
                      ' name: "ARG' + n + \
                      '" description: ' + StringLiteral(descr) + \
                      ' source: ' + frameid +'}'

      # Output PropBank to VerbNet projections.
      if output_vn_projections:
        for vn in vnmap:
          print >> out
          print >> out, '  ; Projection to VerbNet class ' + vn
          print >> out, '  projection: {:mapping +construction'
          print >> out, '    input_schema: ' + frameid
          print >> out, '    output_schema: ' + vn
          print >> out, '    binding: [ input hastype ' + frameid + ' ]'
          print >> out, '    binding: [ output hastype ' + vn + ' ]'
          rolemap = vnmap[vn]
          for arg in sorted(rolemap.keys()):
            print >> out, '    binding: [ input ' + arg + ' equals output ' + \
                          rolemap[arg] + ' ]'

          print >> out, '  }'

      print >> out, '}'
      print >> out

# Write schema catalog.
print >> out, '{=/schema/propbank :schema_family name: "PropBank schemas"'
print >> out, '  precompute_templates: 1'
print >> out, '  precompute_projections: 1'
for schema in schemas: print >> out, '  member_schema: ' + schema
print >> out, '}'

out.close()

