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

"""Workflow for silver-labeling of Wikipedia articles"""

import os
import sling.flags as flags
import sling.task.corpora as corpora
from sling.task import *
from sling.task.wiki import WikiWorkflow

class SilverWorkflow:
  def __init__(self, name=None, wf=None):
    if wf == None: wf = Workflow(name)
    self.wf = wf
    self.wiki = WikiWorkflow(wf=wf)

  def workdir(self, language=None):
    if language == None:
      return flags.arg.workdir + "/silver"
    else:
      return flags.arg.workdir + "/silver/" + language

  #---------------------------------------------------------------------------
  # IDF table
  #---------------------------------------------------------------------------

  def idftable(self, language=None):
    """Resource for IDF table."""
    if language == None: language = flags.arg.language
    return self.wf.resource("idf.repo",
                            dir=self.workdir(language),
                            format="repository")

  def build_idf(self, language=None):
    # Build IDF table from Wikipedia.
    if language == None: language = flags.arg.language
    documents = self.wiki.wikipedia_documents(language)

    with self.wf.namespace(language + "-idf"):
      # Collect words.
      wordcounts = self.wf.shuffle(
        self.wf.map(documents, "vocabulary-mapper", format="message/count",
                    params={
                      "min_document_length": 200,
                      "only_lowercase": True
                    })
      )

      # Build IDF table.
      builder = self.wf.task("idf-table-builder", params={"threshold": 30})
      self.wf.connect(wordcounts, builder)
      builder.attach_output("repository", self.idftable(language))

  #---------------------------------------------------------------------------
  # Silver-labeled documents
  #---------------------------------------------------------------------------

  def silver_documents(self, language=None):
    """Resource for silver-labeled documents."""
    if language == None: language = flags.arg.language
    return self.wf.resource("silver@10.rec",
                            dir=self.workdir(language),
                            format="records/document")

  def silver_annotation(self, indocs=None, outdocs=None, language=None):
    if indocs == None: indocs = self.wiki.wikipedia_documents(language)
    if outdocs == None: outdocs = self.silver_documents(language)
    if language == None: language = flags.arg.language
    phrases = corpora.repository("data/wiki/" + language) + "/phrases.txt"

    with self.wf.namespace(language + "-silver"):
      mapper = self.wf.task("document-processor", "labeler")

      mapper.add_annotator("mentions")
      mapper.add_annotator("anaphora")
      mapper.add_annotator("phrase-structure")
      mapper.add_annotator("relations")

      mapper.add_param("resolve", True)
      mapper.add_param("language", language)
      mapper.attach_input("commons", self.wiki.knowledge_base())
      mapper.attach_input("aliases", self.wiki.phrase_table(language))
      mapper.attach_input("dictionary", self.idftable(language))
      if os.path.isfile(phrases):
        mapper.attach_input("phrases", self.wf.resource(phrases, format="lex"))

      self.wf.connect(self.wf.read(indocs), mapper)
      output = self.wf.channel(mapper, format="message/document")
      return self.wf.write(output, outdocs)

