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
from sling.myelin.flow import Flow
import sys
import torch
from functools import partial

sys.path.insert(0, "sling/nlp/parser/trainer")
import commons_from_corpora as commons_builder
import random
from corpora import Corpora
from pytorch_modules import Caspar
from spec import Spec
from trainer import Hyperparams, Trainer, dev_accuracy
from train_util import *


def train(args):
  check_present(
      args,
      ["train_corpus", "output_folder", "dev_corpus", "train_shuffle_seed"])

  train_corpus_path = args.train_corpus
  if args.train_shuffle_seed > 0:
    reader = sling.RecordReader(args.train_corpus)
    items = [(key, value) for key, value in reader]
    reader.close()
    r = random.Random(args.train_shuffle_seed)
    r.shuffle(items)
    train_corpus_path = os.path.join(args.output_folder, "train_shuffled.rec")
    writer = sling.RecordWriter(train_corpus_path)
    for key, value in items:
      writer.write(key, value)
    writer.close()
    print("Wrote shuffled train corpus to %s using seed %d" % \
          (train_corpus_path, args.train_shuffle_seed))

  # Setting an explicit seed for the sake of determinism.
  torch.manual_seed(1)

  # Make commons store if needed.
  if args.commons == '' or not os.path.exists(args.commons):
    if args.commons == '':
      fname = os.path.join(args.output_folder, "commons")
      print("Will create a commons store at", fname)
      args.commons = fname
    else:
      print("No commons found at", args.commons, ", creating it...")
    _, symbols = commons_builder.build(
      [train_corpus_path, args.dev_corpus], args.commons)
    print("Commons created at", args.commons, "with", len(symbols), \
        "symbols besides the usual ones.")

  # Make the training spec.
  spec = Spec()
  spec.build(args.commons, train_corpus_path)

  # Initialize the model with the spec and any word embeddings.
  caspar = Caspar(spec)
  embeddings_file = args.word_embeddings
  if embeddings_file == '': embeddings_file = None
  caspar.initialize(embeddings_file)

  tmp_folder = os.path.join(args.output_folder, "tmp")
  if not os.path.exists(tmp_folder):
    os.makedirs(tmp_folder)

  evaluator = partial(dev_accuracy, args.dev_corpus, tmp_folder)

  output_file_prefix = os.path.join(args.output_folder, "caspar")
  hyperparams = Hyperparams(args)
  print("Using hyperparameters:", hyperparams)

  trainer = Trainer(caspar, hyperparams, evaluator, output_file_prefix)
  train = Corpora(train_corpus_path, spec.commons, gold=True)
  trainer.train(train)


if __name__ == '__main__':
  setup_training_flags(flags)
  flags.parse()
  train(flags.arg)
