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

"""Workflow builder for downloading wiki dumps"""

import os
from urllib.request import urlopen
import _strptime
import time

import sling
import sling.task.corpora as corpora
import sling.flags as flags
import sling.log as log
from sling.task.workflow import *

# Number of concurrent downloads.
download_concurrency = 0

# Task for downloading wiki dumps.
class UrlDownload:
  def run(self, task):
    # Get task parameters.
    name = task.param("shortname")
    url = task.param("url")
    ratelimit = task.param("ratelimit", 0)
    chunksize = task.param("chunksize", 64 * 1024)
    priority = task.param("priority", 0)
    output = task.output("output")
    log.info("Download " + name + " from " + url)

    # Make sure directory exists.
    directory = os.path.dirname(output.name)
    if not os.path.exists(directory): os.makedirs(directory)

    # Do not overwrite existing file unless flag is set.
    if not flags.arg.overwrite and os.path.exists(output.name):
      raise Exception("file already exists: " + output.name + \
                      " (use --overwrite to overwrite existing files)")

    # Hold-off on low-prio tasks
    if priority > 0: time.sleep(priority)

    # Wait until we are below the rate limit.
    global download_concurrency
    if ratelimit > 0:
      while download_concurrency >= ratelimit: time.sleep(10)
      download_concurrency += 1

    # Download from url to file.
    if ratelimit > 0: log.info("Start download of " + url)
    conn = urlopen(url)
    last_modified = time.mktime(time.strptime(conn.headers['last-modified'],
                                              "%a, %d %b %Y %H:%M:%S GMT"))
    total_bytes = "bytes_downloaded"
    bytes = name + "_bytes_downloaded"
    with open(output.name, 'wb') as f:
      while True:
        chunk = conn.read(chunksize)
        if not chunk: break
        f.write(chunk)
        task.increment(total_bytes, len(chunk))
        task.increment(bytes, len(chunk))
    os.utime(output.name, (last_modified, last_modified))
    if ratelimit > 0: download_concurrency -= 1
    log.info(name + " downloaded")

register_task("url-download", UrlDownload)

class DownloadWorkflow:
  def __init__(self, name=None, wf=None):
    if wf == None: wf = Workflow(name)
    self.wf = wf

  #---------------------------------------------------------------------------
  # Wikipedia dumps
  #---------------------------------------------------------------------------

  def wikipedia_dump(self, language=None):
    """Resource for wikipedia dump. This can be downloaded from wikimedia.org
    and contains a full dump of Wikipedia in a particular language. This is
    in XML format with the articles in Wiki markup format."""
    if language == None: language = flags.arg.language
    return self.wf.resource(corpora.wikipedia_dump(language),
                            format="xml/wikipage")

  def download_wikipedia(self, url=None, dump=None, language=None):
    if language == None: language = flags.arg.language
    if url == None: url = corpora.wikipedia_url(language)
    if dump == None: dump = self.wikipedia_dump(language)
    priority = 1
    if language == "en": priority = 0

    with self.wf.namespace(language + "-wikipedia-download"):
      download = self.wf.task("url-download")
      download.add_params({
        "language": language,
        "url": url,
        "shortname": language + "wiki",
        "ratelimit": 2,
        "priority": priority,
      })
      download.attach_output("output", dump)
      return dump

  #---------------------------------------------------------------------------
  # Wikidata dumps
  #---------------------------------------------------------------------------

  def wikidata_dump(self):
    """Resource for wikidata dump. This can be downloaded from wikimedia.org
    and contains a full dump of Wikidata in JSON format."""
    return self.wf.resource(corpora.wikidata_dump(), format="text/json")

  def download_wikidata(self, url=None, dump=None):
    if url == None: url = corpora.wikidata_url()
    if dump == None: dump = self.wikidata_dump()

    with self.wf.namespace("wikidata-download"):
      download = self.wf.task("url-download")
      download.add_params({
        "url": url,
        "shortname": "wikidata",
      })
      download.attach_output("output", dump)
      return dump

