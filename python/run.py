#!/usr/bin/python3
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
import sling.task.silver as silver
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

flags.define("--extract_wikilinks",
             help="extract link graph from wikipedias",
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

flags.define("--build_wiki",
             help="run all workflow for building knowledge base",
             default=False,
             action='store_true')

flags.define("--extract_vocabulary",
             help="extract vocabulary for word embeddings",
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

flags.define("--train_fact_plausibility",
             help="train fact plausibility model",
             default=False,
             action='store_true')

flags.define("--build_idf",
             help="build IDF table from wikipedia",
             default=False,
             action='store_true')

flags.define("--silver_annotation",
             help="annotate wikipedia documents with silver annotations",
             default=False,
             action='store_true')

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

    workflow.run(wf.wf)

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

    workflow.run(wf.wf)

def parse_wikipedia():
  # Convert wikipedia pages to SLING documents.
  if flags.arg.parse_wikipedia:
    for language in flags.arg.languages:
      log.info("Parse " + language + " wikipedia")
      wf = wiki.WikiWorkflow(language + "-wikipedia-parsing")
      wf.parse_wikipedia(language=language)
      workflow.run(wf.wf)

def build_knowledge_base():
  # Merge categories from wikipedias.
  if flags.arg.merge_categories:
    log.info("Merge wikipedia categories")
    wf = wiki.WikiWorkflow("category-merging")
    wf.merge_wikipedia_categories()
    workflow.run(wf.wf)

  # Invert categories.
  if flags.arg.invert_categories:
    log.info("Invert categories")
    wf = wiki.WikiWorkflow("category-inversion")
    wf.invert_wikipedia_categories()
    workflow.run(wf.wf)

  # Extract link graph.
  if flags.arg.extract_wikilinks:
    log.info("Extract link graph")
    wf = wiki.WikiWorkflow("link-graph")
    wf.extract_links()
    workflow.run(wf.wf)

  # Fuse items.
  if flags.arg.fuse_items:
    log.info("Fuse items")
    wf = wiki.WikiWorkflow("fuse-items")
    wf.fuse_items()
    workflow.run(wf.wf)

  # Build knowledge base repository.
  if flags.arg.build_kb:
    log.info("Build knowledge base repository")
    wf = wiki.WikiWorkflow("knowledge-base")
    wf.build_knowledge_base()
    workflow.run(wf.wf)

def build_alias_tables():
  # Extract item names from wikidata and wikipedia.
  if flags.arg.extract_names:
    for language in flags.arg.languages:
      log.info("Extract " + language + " names")
      wf = wiki.WikiWorkflow(language + "-name-extraction")
      wf.extract_names(language=language)
      workflow.run(wf.wf)

  # Build name table.
  if flags.arg.build_nametab:
    for language in flags.arg.languages:
      log.info("Build " + language + " name table")
      wf = wiki.WikiWorkflow(language + "-name-table")
      wf.build_name_table(language=language)
      workflow.run(wf.wf)

  # Build phrase table.
  if flags.arg.build_phrasetab:
    for language in flags.arg.languages:
      log.info("Build " + language + " phrase table")
      wf = wiki.WikiWorkflow(language + "-phrase-table")
      wf.build_phrase_table(language=language)
      workflow.run(wf.wf)

def train_embeddings():
  # Extract vocabulary for word embeddings.
  if flags.arg.extract_vocabulary:
    for language in flags.arg.languages:
      log.info("Extract " + language + " vocabulary")
      wf = embedding.EmbeddingWorkflow(language + "-vocabulary")
      wf.extract_vocabulary(language=language)
      workflow.run(wf.wf)

  # Train word embeddings.
  if flags.arg.train_word_embeddings:
    for language in flags.arg.languages:
      log.info("Train " + language + " word embeddings")
      wf = embedding.EmbeddingWorkflow(language + "-word-embeddings")
      wf.train_word_embeddings(language=language)
      workflow.run(wf.wf)

  # Extract vocabulary for fact and category embeddings.
  if flags.arg.extract_fact_lexicon:
    log.info("Extract fact and category lexicons")
    wf = embedding.EmbeddingWorkflow("fact-lexicon")
    wf.extract_fact_lexicon()
    workflow.run(wf.wf)

  # Extract facts from knowledge base.
  if flags.arg.extract_facts:
    log.info("Extract facts from knowledge base")
    wf = embedding.EmbeddingWorkflow("fact-extraction")
    wf.extract_facts()
    workflow.run(wf.wf)

  # Train fact and category embeddings.
  if flags.arg.train_fact_embeddings:
    log.info("Train fact and category embeddings")
    wf = embedding.EmbeddingWorkflow("fact-embeddings")
    wf.train_fact_embeddings()
    workflow.run(wf.wf)

  # Train fact plausibility model.
  if flags.arg.train_fact_plausibility:
    log.info("Train fact plausibility model")
    wf = embedding.EmbeddingWorkflow("plausibility")
    wf.train_fact_plausibility()
    workflow.run(wf.wf)


def silver_annotation():
  # Extract IDF table.
  if flags.arg.build_idf:
    wf = silver.SilverWorkflow("idf-table")
    for language in flags.arg.languages:
      log.info("Build " + language + " IDF table")
      wf.build_idf(language=language)
    workflow.run(wf.wf)

  # Run silver-labeling of Wikipedia documents.
  if flags.arg.silver_annotation:
    for language in flags.arg.languages:
      log.info("Silver-label " + language + " wikipedia")
      wf = silver.SilverWorkflow(language + "-silver")
      wf.silver_annotation(language=language)
      workflow.run(wf.wf)


if __name__ == '__main__':
  # Parse command-line arguments.
  flags.parse()

  if flags.arg.build_wiki:
    flags.arg.import_wikidata = True
    flags.arg.import_wikipedia = True
    flags.arg.parse_wikipedia = True
    flags.arg.merge_categories = True
    flags.arg.invert_categories = True
    flags.arg.extract_wikilinks = True
    flags.arg.fuse_items = True
    flags.arg.build_kb = True
    flags.arg.extract_names = True
    flags.arg.build_nametab = True
    flags.arg.build_phrasetab = True

  # Run workflows.
  workflow.startup()
  download_corpora()
  import_wiki()
  parse_wikipedia()
  build_knowledge_base()
  build_alias_tables()
  train_embeddings()
  silver_annotation()
  workflow.shutdown()

  # Done.
  log.info("Done")

