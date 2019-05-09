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

# Convert SLING documents from version 1 to version 2.

import sling
import sys

# Check arguments.
if len(sys.argv) != 3:
  print("usage:", sys.argv[0], "<v1 doc rec input>", "<v2 doc rec output>")
  sys.exit(1)

# Intialize commons store.
commons = sling.Store()
commons.parse("""
{=document =/s/document}
{=url =/s/document/url}
{=title =/s/document/title}
{=text =/s/document/text}
{=tokens =/s/document/tokens}
{=mention =/s/document/mention}
{=theme =/s/document/theme}
{=token =/s/token}
{=index =/s/token/index}
{=start =/s/token/start}
{=size =/s/token/length}
{=break =/s/token/break}
{=word =/s/token/text}
{=phrase =/s/phrase}
{=begin =/s/phrase/begin}
{=length =/s/phrase/length}
{=evokes =/s/phrase/evokes}
""")
commons.freeze()

# Convert documents.
num_docs = 0
fin = sling.RecordReader(sys.argv[1])
fout = sling.RecordWriter(sys.argv[2])
for key, value in fin:
  store = sling.Store(commons)
  f = store.parse(value)
  fout.write(key, f.data(binary=True))
  num_docs += 1

fin.close()
fout.close()
print(num_docs, "documents converted")

