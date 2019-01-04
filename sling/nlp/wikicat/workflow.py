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

# Workflow for generating and scoring parses for Wikipedia category titles.

import sling
import sling.flags as flags
import sling.task.corpora as corpora
from sling.task.workflow import Workflow, start_monitor

# Workflow tasks.
import generator
import prelim_ranker


class CategoryParsingWorkflow:
  def __init__(self, name, outdir, wf=None):
    if wf is None: wf = Workflow(name)
    self.wf = wf
    self.outdir = outdir


  # Adds a KB input to 'task'.
  def kb_input(self, task, kb_dir=None):
    if kb_dir is None:
      kb_dir = corpora.wikidir()
    kb = self.wf.resource(file="kb.sling", dir=kb_dir, format="store/frame")
    task.attach_input("kb", kb)


  # Stage 1: Generate all parses.
  def generate_parses(self, language, min_members):
    with self.wf.namespace("generate-parses"):
      generator = self.wf.task("category-parse-generator")
      generator.add_params({
        "language": language,
        "min_members": min_members
      })
      wikidir = corpora.wikidir()
      self.kb_input(generator, kb_dir=wikidir)

      items = self.wf.resource(
          file="items@10.rec", dir=wikidir, format="records/frame")
      generator.attach_input("items", items)

      phrase_table_dir = wikidir + "/" + language
      phrase_table = self.wf.resource(
          "phrase-table.repo", dir=phrase_table_dir, format="text/frame")
      generator.attach_input("phrase-table", phrase_table)

      output = self.wf.resource(
          "generated-parses.rec", dir=self.outdir, format="records/frame")
      generator.attach_output("output", output)
      rejected = self.wf.resource(
          "rejected-categories.rec", dir=self.outdir, format="records/text")
      generator.attach_output("rejected", rejected)
      return output


  # Stage 2: Generate a preliminary ranking of the parses, keeping the top-k.
  def prelim_rank_parses(self, input_parses, topk):
    with self.wf.namespace("rank-parses"):
      ranker = self.wf.task("prelim-category-parse-ranker")
      ranker.add_params({"max_parses": topk})
      self.kb_input(ranker)
      ranker.attach_input("input", input_parses)
      output = self.wf.resource(
          "filtered-parses.rec", dir=self.outdir, format="records/frame")
      ranker.attach_output("output", output)
      return output


if __name__ == '__main__':
  flags.define("--port",
               help="port number for task monitor (0 means no monitor)",
               default=6767,
               type=int,
               metavar="PORT")
  flags.define("--output",
               help="Output directory",
               default="local/data/e/wikicat",
               type=str,
               metavar="DIR")
  flags.define("--lang",
               help="Language to process",
               default="en",
               type=str,
               metavar="LANG")
  flags.define("--min_members",
               help="Reject categories with less than these many members",
               default=5,
               type=int,
               metavar="MIN")
  flags.define("--topk",
               help="No. of top-k preliminary ranked parses to consider",
               default=50,
               type=int,
               metavar="TOPK")

  flags.parse()
  categories = CategoryParsingWorkflow("category-parsing", flags.arg.output)
  generated = categories.generate_parses(flags.arg.lang, flags.arg.min_members)
  filtered = categories.prelim_rank_parses(generated, flags.arg.topk)
  print categories.wf.dump()

  start_monitor(flags.arg.port)
  categories.wf.start()
  done = False
  while not done:
    done = categories.wf.wait(1000)

