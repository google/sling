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

# Tool to make a commons store from recordio(s) of documents.
#
# Sample usage:
# python path/to/this/script.py \
#    --input=train.rec,dev.rec,test.rec --output=/tmp/commons.sling

import sling

def build(recordio_filenames, output_filename):
  commons = sling.Store()
  schema = sling.DocumentSchema(commons)
  commons.freeze()

  symbol_names = {}
  symbol_names["/s/thing"] = 1

  # Adds handle's id to 'symbol_names' if it is already not in 'commons'.
  def add(handle):
    if type(handle) is not sling.Frame or handle.id is None: return

    id_str = str(handle.id)
    if commons[id_str] is not None: return

    if id_str not in symbol_names: symbol_names[id_str] = 0
    symbol_names[id_str] += 1

  for filename in recordio_filenames:
    reader = sling.RecordReader(filename)
    for key, value in reader:
      store = sling.Store(commons)
      document = sling.Document(store.parse(value), schema=schema)

      for mention in document.mentions:
        for frame in mention.evokes():
          for slot_role, slot_value in frame:
            add(slot_role)
            add(slot_value)

      for theme in document.themes:
        for slot_role, slot_value in theme:
          add(slot_role)
          add(slot_value)

  output = sling.Store()
  schema = sling.DocumentSchema(output)

  for name, count in symbol_names.iteritems():
    output.frame({"id": name})
  output.freeze()
  output.save(output_filename, binary=True)
  return output, symbol_names


if __name__ == "__main__":
  import sling.flags as flags

  flags.define('--input',
               help='Comma separated list of recordio files',
               default="",
               type=str,
               metavar='COMMA_SEPARATED_RECORDIOS')

  flags.define('--output',
               help='Output commons file name',
               default="/tmp/commons.sling",
               type=str,
               metavar='OUTPUT_COMMONS_FILENAME')

  flags.parse()
  _, symbols = build(flags.arg.input.split(","), flags.arg.output)
  print "Commons written to", flags.arg.output, "with", len(symbols), "symbols"


