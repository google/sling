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


import sling
import sling.log as log

# Pool of loaded KBs.
_kb_cache = {}

def load_kb(task):
  if type(task) is str:
    filename = task  # assume filename
  else:
    filename = task.input("kb").name

  if filename in _kb_cache:
    log.info("Retrieving cached KB")
    return _kb_cache[filename]
  else:
    kb = sling.Store()
    kb.load(filename)
    log.info("Knowledge base read")
    kb.lockgc()
    kb.freeze()
    kb.unlockgc()
    log.info("Knowledge base frozen")
    _kb_cache[filename] = kb
    return kb


