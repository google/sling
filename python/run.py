#!/usr/bin/python2.7
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

"""Run SLING processing"""

import sling
import sling.flags as flags
import sling.log as log
import sling.task.corpora as corpora
import sling.task.download as download
import sling.task.wiki as wiki
import sling.task.embedding as embedding
import sling.task.workflow as workflow

# Command-line flags.
flags.define("--download_wikidata",
             help="download wikidata dump",
             default=False,
             action='store_true')

flags.define("--download_wikipedia",
             help="download wikipedia dump(s)",
             default=False,
             action='store_true')

flags.define("--import_wikidata",
             help="convert wikidata to sling format",
             default=False,
             action='store_true')

flags.define("--import_wikipedia",
             help="convert wikidata dump(s) to sling format",
             default=False,
             action='store_true')

flags.define("--parse_wikipedia",
             help="parse wikipedia(s)",
             default=False,
             action='store_true')

flags.define("--merge_categories",
             help="merge categories for Wikipedias into items",
             default=False,
             action='store_true')

flags.define("--invert_categories",
             help="invert categories from item categories to category members",
             default=False,
             action='store_true')

flags.define("--compute_item_popularity",
             help="compute item popularity from alias counts",
             default=False,
             action='store_true')

flags.define("--fuse_items",
             help="fuse items from wikidata and wikipedia",
             default=False,
             action='store_true')

flags.define("--build_kb",
             help="build knowledge base",
             default=False,
             action='store_true')

flags.define("--extract_names",
             help="extract names for items",
             default=False,
             action='store_true')

flags.define("--build_nametab",
             help="build name table",
             default=False,
             action='store_true')

flags.define("--build_phrasetab",
             help="build name table",
             default=False,
             action='store_true')

flags.define("--extract_vocabulary",
             help="extract vocabulary for word embeddings",
             default=False,
             action='store_true')

flags.define("--build_wiki",
             help="run all workflow for building knowledge base",
             default=False,
             action='store_true')

flags.define("--train_word_embeddings",
             help="train word embeddings",
             default=False,
             action='store_true')

flags.define("--extract_fact_lexicon",
             help="extract fact and category lexicons",
             default=False,
             action='store_true')

flags.define("--extract_facts",
             help="extract facts from knowledge base",
             default=False,
             action='store_true')

flags.define("--train_fact_embeddings",
             help="train fact and category embeddings",
             default=False,
             action='store_true')

flags.define("--dryrun",
             help="build worflows but do not run them",
             default=False,
             action='store_true')

flags.define("--monitor",
             help="port number for task monitor (0 means no monitor)",
             default=6767,
             type=int,
             metavar="PORT")

flags.define("--logdir",
             help="directory where workflow logs are stored",
             default="local/logs",
             metavar="DIR")

def run_workflow(wf):
  # In dryrun mode the workflow is just dumped without running it.
  if flags.arg.dryrun:
    print wf.wf.dump()
    return

  # Start workflow.
  log.info("start workflow")
  wf.wf.start()

  # Wait until workflow completes. Poll every second to make the workflow
  # interruptible.
  done = False
  while not done: done = wf.wf.wait(1000)

def download_corpora():
  if flags.arg.download_wikidata or flags.arg.download_wikipedia:
    wf = download.DownloadWorkflow("wiki-download")

    # Download wikidata dump.
    if flags.arg.download_wikidata:
      wf.download_wikidata()

    # Download wikipedia dumps.
    if flags.arg.download_wikipedia:
      for language in flags.arg.languages:
        wf.download_wikipedia(language=language)

    run_workflow(wf)

def import_wiki():
  if flags.arg.import_wikidata or flags.arg.import_wikipedia:
    wf = wiki.WikiWorkflow("wiki-import")
    # Import wikidata.
    if flags.arg.import_wikidata:
      log.info("Import wikidata")
      wf.wikidata()

    # Import wikipedia(s).
    if flags.arg.import_wikipedia:
      for language in flags.arg.languages:
        log.info("Import " + language + " wikipedia")
        wf.wikipedia(language=language)

    run_workflow(wf)

def parse_wikipedia():
  # Convert wikipedia pages to SLING documents.
  if flags.arg.parse_wikipedia:
    for language in flags.arg.languages:
      log.info("Parse " + language + " wikipedia")
      wf = wiki.WikiWorkflow(language + "-wikipedia-parsing")
      wf.parse_wikipedia(language=language)
      run_workflow(wf)

def fuse_items():
  # Merge categories from wikipedias.
  if flags.arg.merge_categories:
    log.info("Merge wikipedia categories")
    wf = wiki.WikiWorkflow("category-merging")
    wf.merge_wikipedia_categories()
    run_workflow(wf)

  # Invert categories.
  if flags.arg.invert_categories:
    log.info("Invert categories")
    wf = wiki.WikiWorkflow("category-inversion")
    wf.invert_wikipedia_categories()
    run_workflow(wf)

  # Compute item popularity.
  if flags.arg.compute_item_popularity:
    log.info("Compute item popularity")
    wf = wiki.WikiWorkflow("item-popularity")
    wf.compute_item_popularity()
    run_workflow(wf)

  # Fuse items.
  if flags.arg.fuse_items:
    log.info("Fuse items")
    wf = wiki.WikiWorkflow("fuse-items")
    wf.fuse_items()
    run_workflow(wf)


def build_knowledge_base():
  # Build knowledge base repository.
  if flags.arg.build_kb:
    log.info("Build knowledge base repository")
    wf = wiki.WikiWorkflow("knowledge-base")
    wf.build_knowledge_base()
    run_workflow(wf)

  # Extract item names from wikidata and wikipedia.
  if flags.arg.extract_names:
    for language in flags.arg.languages:
      log.info("Extract " + language + " names")
      wf = wiki.WikiWorkflow(language + "-name-extraction")
      wf.extract_names(language=language)
      run_workflow(wf)

  # Build name table.
  if flags.arg.build_nametab:
    for language in flags.arg.languages:
      log.info("Build " + language + " name table")
      wf = wiki.WikiWorkflow(language + "-name-table")
      wf.build_name_table(language=language)
      run_workflow(wf)

  # Build phrase table.
  if flags.arg.build_phrasetab:
    for language in flags.arg.languages:
      log.info("Build " + language + " phrase table")
      wf = wiki.WikiWorkflow(language + "-phrase-table")
      wf.build_phrase_table(language=language)
      run_workflow(wf)

def train_embeddings():
  # Extract vocabulary for word embeddings.
  if flags.arg.extract_vocabulary:
    for language in flags.arg.languages:
      log.info("Extract " + language + " vocabulary")
      wf = embedding.EmbeddingWorkflow(language + "-vocabulary")
      wf.extract_vocabulary(language=language)
      run_workflow(wf)

  # Train word embeddings.
  if flags.arg.train_word_embeddings:
    for language in flags.arg.languages:
      log.info("Train " + language + " word embeddings")
      wf = embedding.EmbeddingWorkflow(language + "-word-embeddings")
      wf.train_word_embeddings(language=language)
      run_workflow(wf)

  # Extract vocabulary for fact and category embeddings.
  if flags.arg.extract_fact_lexicon:
    log.info("Extract fact and category lexicons")
    wf = embedding.EmbeddingWorkflow("fact-lexicon")
    wf.extract_fact_lexicon()
    run_workflow(wf)

  # Extract facts from knowledge base.
  if flags.arg.extract_facts:
    log.info("Extract facts from knowledge base")
    wf = embedding.EmbeddingWorkflow("fact-extraction")
    wf.extract_facts()
    run_workflow(wf)

  # Train fact and category embeddings.
  if flags.arg.train_fact_embeddings:
    log.info("Train fact and category embeddings")
    wf = embedding.EmbeddingWorkflow("fact-embeddings")
    wf.train_fact_embeddings()
    run_workflow(wf)


if __name__ == '__main__':
  # Parse command-line arguments.
  flags.parse()

  if flags.arg.build_wiki:
    flags.arg.import_wikidata = True
    flags.arg.import_wikipedia = True
    flags.arg.parse_wikipedia = True
    flags.arg.merge_categories = True
    flags.arg.invert_categories = True
    flags.arg.compute_item_popularity = True
    flags.arg.fuse_items = True
    flags.arg.build_kb = True
    flags.arg.extract_names = True
    flags.arg.build_nametab = True
    flags.arg.build_phrasetab = True

  # Start task monitor.
  if flags.arg.monitor > 0: workflow.start_monitor(flags.arg.monitor)

  # Run workflows.
  download_corpora()
  import_wiki()
  parse_wikipedia()
  fuse_items()
  build_knowledge_base()
  train_embeddings()

  # Stop task monitor.
  if flags.arg.monitor > 0: workflow.stop_monitor()
  workflow.save_workflow_log(flags.arg.logdir)

  # Done.
  log.info("Done")
