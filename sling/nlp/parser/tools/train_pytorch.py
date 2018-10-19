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
from corpora import Corpora
from pytorch_modules import Caspar
from spec import Spec
from trainer import Hyperparams, Trainer, dev_accuracy
from train_util import mem, setup_training_flags


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
      fname = os.path.join(args.output_folder, "commons")
      print "Will create a commons store at", fname
      args.commons = fname
    else:
      print "No commons found at", args.commons, ", creating it..."
    _, symbols = commons_builder.build(
      [args.train_corpus, args.dev_corpus], args.commons)
    print "Commons created at", args.commons, "with", len(symbols), \
      "symbols besides the usual ones."

  # Make the training spec.
  spec = Spec(args.small)
  spec.build(args.commons, args.train_corpus)

  # Load word embeddings.
  if args.word_embeddings != "" and args.word_embeddings is not None:
    spec.load_word_embeddings(args.word_embeddings)
    print "After loading pre-trained word embeddings", mem()

  # Initialize the model with the spec.
  caspar = Caspar(spec)
  caspar.initialize()

  tmp_folder = os.path.join(args.output_folder, "tmp")
  if not os.path.exists(tmp_folder):
    os.makedirs(tmp_folder)

  evaluator = partial(dev_accuracy,
                      args.commons,
                      args.dev_corpus,
                      tmp_folder)

  output_file_prefix = os.path.join(args.output_folder, "caspar")
  hyperparams = Hyperparams(args)
  print "Using hyperparameters:", hyperparams

  trainer = Trainer(caspar, hyperparams, evaluator, output_file_prefix)
  train = Corpora(args.train_corpus, spec.commons, gold=True)
  trainer.train(train)


if __name__ == '__main__':
  setup_training_flags(flags)
  flags.define('--small',
               help='Small dimensions (for testing)',
               default=False,
               action='store_true')
  flags.parse()
  train(flags.arg)
