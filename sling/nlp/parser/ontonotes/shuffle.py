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

# Shuffle training corpus.

import random
import sling
import sling.flags as flags

flags.define('--input',
             help='input file with documents')
flags.define('--output',
             help='output for shuffled documents')
flags.define('--seed',
             help='seed for shuffling the corpus',
             default="314159",
             type=int,
             metavar='NUM')

if __name__ == '__main__':
  flags.parse()

  # Read input corpus.
  reader = sling.RecordReader(flags.arg.input)
  records = [(key, value) for key, value in reader]
  reader.close()

  # Shufle documents.
  r = random.Random(flags.arg.seed)
  r.shuffle(records)

  # Write shuffled documents to output.
  writer = sling.RecordWriter(flags.arg.output)
  for key, value in records:
    writer.write(key, value)
  writer.close()

