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

# Top-level PyTorch training script.

import os
import sling
import sling.flags as flags
import sys
import torch
from functools import partial

sys.path.insert(0, "sling/nlp/parser/trainer")
import commons_from_corpora as commons_builder
from pytorch_modules import Caspar
from train_util import Resources, setup_training_flags
from trainer import Hyperparams, Trainer, dev_accuracy


# Checks that all arguments in 'ls' are set in 'args'.
def check_present(args, ls):
  for x in ls:
    val = getattr(args, x, None)
    assert val is not None, "%r should be present" % x
    if type(val) is str:
      assert val != "", "%r should be set" % x


def train(args):
  check_present(args, ["train_corpus", "output_folder", "dev_corpus"])

  # Setting an explicit seed for the sake of determinism.
  torch.manual_seed(1)

  # Make commons store if needed.
  if args.commons == '' or not os.path.exists(args.commons):
    if args.commons == '':
      import tempfile
      f = tempfile.NamedTemporaryFile(delete=False)
      print "Will create a commons store at", f.name
      args.commons = f.name
      f.close()
    else:
      print "No commons found at", args.commons, ", creating it..."
    _, symbols = commons_builder.build(
      [args.train_corpus, args.dev_corpus], args.commons)
    print "Commons created at", args.commons, "with", len(symbols), \
      "symbols besides the usual ones."

  resources = Resources()
  resources.load(commons_path=args.commons,
                 train_path=args.train_corpus,
                 word_embeddings_path=args.word_embeddings,
                 small_spec=args.small)

  caspar = Caspar(resources.spec)
  caspar.initialize()

  tmp_folder = os.path.join(args.output_folder, "tmp")
  if not os.path.exists(tmp_folder):
    os.makedirs(tmp_folder)

  evaluator = partial(dev_accuracy,
                      resources.commons_path,
                      resources.commons,
                      args.dev_corpus,
                      resources.schema,
                      tmp_folder)

  output_file_prefix = os.path.join(args.output_folder, "caspar")
  hyperparams = Hyperparams(args)
  print "Using hyperparameters:", hyperparams

  trainer = Trainer(caspar, hyperparams, evaluator, output_file_prefix)
  trainer.train(resources.train)


if __name__ == '__main__':
  setup_training_flags(flags)
  flags.define('--small',
               help='Small dimensions (for testing)',
               default=False,
               type=bool)
  flags.parse()
  train(flags.arg)
