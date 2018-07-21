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

import os
import psutil
import sling
import subprocess

from datetime import datetime

from corpora import Corpora
from spec import Spec

# Evaluates gold vs test documents, which are assumed to be aligned,
# and returns a dict of their results.
def frame_evaluation(gold_corpus_path, test_corpus_path, commons_path):
  try:
    output = subprocess.check_output(
        ['bazel-bin/sling/nlp/parser/tools/evaluate-frames',
         '--gold_documents=' + gold_corpus_path,
         '--test_documents=' + test_corpus_path,
         '--commons=' + commons_path],
        stderr=subprocess.STDOUT)
  except subprocess.CalledProcessError as e:
    print("Evaluation failed: ", e.returncode, e.output)
    return {'eval_metric': 0.0}

  eval_output = {}
  for line in output.splitlines():
    line = line.rstrip()
    print "Evaluation Metric: ", line
    parts = line.split('\t')
    assert len(parts) == 2, "%r" % line
    eval_output[parts[0]] = float(parts[1])
    if line.startswith("SLOT_F1"):
      eval_output['eval_metric'] = float(parts[1])

  assert eval_output.has_key('eval_metric'), "%r" % str(eval_output)
  return eval_output


# Methods for returning date and memory usage respectively.
def now():
  return "[" + str(datetime.now()) + "]"


def mem():
  p = psutil.Process(os.getpid())
  info = p.memory_info()
  return "(rss=%r vms=%r)" % (info.rss, info.vms)


# Sets up commonly used command-line training flags.
def setup_training_flags(flags):
  flags.define('--output_folder',
               help='Output directory where flow will be saved',
               default="",
               type=str,
               metavar='DIR')
  flags.define('--commons',
               help='Path to the commons store file',
               default="local/data/corpora/sempar/commons.sling",
               type=str,
               metavar='COMMONS_FILE')
  flags.define('--train_corpus', '--train',
               help='Path to the train corpus recordio',
               default="local/data/corpora/sempar/train.rec",
               type=str,
               metavar='TRAIN_RECORDIO')
  flags.define('--dev_corpus', '--dev',
               help='Path to the dev corpus recordio',
               default="local/data/corpora/sempar/dev.rec",
               type=str,
               metavar='DEV_RECORDIO')
  flags.define('--word_embeddings',
               help='(Optional) Path to the word embeddings file',
               default="local/data/corpora/sempar/word2vec-32-embeddings.bin",
               type=str,
               metavar='WORD_EMBEDDINGS_FILE')

  # Training hyperparameters.
  # Notable omissions: decay_steps, dropout_rate.
  flags.define('--train_steps', '--steps',
               help='Number of training batches to use',
               default=100000,
               type=int,
               metavar='NUM_BATCHES')
  flags.define('--report_every',
               help='Checkpoint interval in number of batches',
               default=3000,
               type=int,
               metavar='NUM_BATCHES_BEFORE_CHECKPOINTING')
  flags.define('--batch_size', '--batch',
               help='Batch size',
               default=8,
               type=int,
               metavar='BATCH_SIZE')
  flags.define('--learning_method', '--optimizer',
               help='Optimizer to use: adam or sgd',
               default='adam',
               type=str,
               metavar='OPTIMIZER')
  flags.define('--use_moving_average',
               help='Whether or not to use exponential moving averaging',
               default=True,
               action='store_true')
  flags.define('--moving_average_coeff',
               help='Exponential moving coefficient to use',
               default=0.9999,
               type=float,
               metavar='MOVING_AVERAGE_COEFFICIENT')
  flags.define('--gradient_clip_norm', '--gradient-clip',
               help='Gradient clip norm to use (0.0 to disable clipping)',
               default=1.0,
               type=float)
  flags.define('--learning_rate', '--alpha',
               help='Learning rate for the optimizer',
               default=0.0005,
               type=float,
               metavar='ALPHA')
  flags.define('--adam_beta1',
               help='beta1 hyperparameter for the Adam optimizer',
               default=0.01,
               type=float,
               metavar='BETA1')
  flags.define('--adam_beta2',
               help='beta2 hyperparameter for the Adam optimizer',
               default=0.999,
               type=float,
               metavar='BETA2')
  flags.define('--adam_eps',
               help='epsilon hyperparameter for the Adam optimizer',
               default=1e-5,
               type=float,
               metavar='EPSILON')
  flags.define('--l2_coeff', '-l2',
               help='L2 regularization coefficient',
               default=0.0001,
               type=float,
               metavar='L2')


# Wrapper around training resources.
class Resources:
  def __init__(self):
    self.commons_path = None
    self.commons = None
    self.schema = None
    self.train = None
    self.spec = None


  # Loads the common store, an iterator over a recordio training corpus,
  # computes the Spec from the corpus, and loads any optional word embeddings.
  def load(self,
           commons_path,
           train_path,
           word_embeddings_path=None,
           small_spec=False):
    print "Loading training resources"
    print "Initial memory usage", mem()
    self.commons_path = commons_path
    self.commons = sling.Store()
    self.commons.load(commons_path)
    self.commons.freeze()
    self.schema = sling.DocumentSchema(self.commons)

    self.train = Corpora(
        train_path, self.commons, self.schema, gold=True, loop=False)
    print "Pointed to training corpus in", train_path, mem()

    self.spec = Spec(small_spec)
    self.spec.commons_path = commons_path
    self.spec.build(self.commons, self.train)
    print "After building spec", mem()

    if word_embeddings_path != "" and word_embeddings_path is not None:
      self.spec.load_word_embeddings(word_embeddings_path)
      print "After loading pre-trained word embeddings", mem()

