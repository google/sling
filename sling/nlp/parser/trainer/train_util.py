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
import resource
import sling

from datetime import datetime

# Checks that all arguments in 'ls' are set in 'args'.
def check_present(args, ls):
  for x in ls:
    val = getattr(args, x, None)
    assert val is not None, "%r should be present" % x
    if type(val) is str:
      assert val != "", "%r should be set" % x


# Evaluates gold vs test documents, which are assumed to be aligned,
# and returns a dict of their results.
def frame_evaluation(gold_corpus_path, test_corpus_path, commons):
  metrics = sling.evaluate_frames(commons, gold_corpus_path, test_corpus_path)

  eval_output = {}
  for metric in metrics:
    name = metric[0]
    print("Evaluation Metric: ", metric)
    eval_output[name] = metric[1]
    if name == "SLOT_F1":
      eval_output['eval_metric'] = metric

  assert 'eval_metric' in eval_output, "%r" % str(eval_output)
  return eval_output


# Methods for returning date and memory usage respectively.
def now():
  return "[" + str(datetime.now()) + "]"


def mem():
  return "(rss=%r)" % resource.getrusage(resource.RUSAGE_SELF).ru_maxrss


# Sets up commonly used command-line training flags.
def setup_training_flags(flags):
  flags.define('--train_shuffle_seed',
               help='Seed for shuffling the train corpus. Not shuffled if -1.',
               default="314159",
               type=int,
               metavar='NUM')
  flags.define('--output_folder',
               help='Output directory where flow will be saved',
               default="",
               type=str,
               metavar='DIR')
  flags.define('--commons',
               help='Path to the commons store file',
               default="",
               type=str,
               metavar='FILE')
  flags.define('--train_corpus', '--train',
               help='Path to the train corpus recordio',
               default="local/data/corpora/caspar/train.rec",
               type=str,
               metavar='FILE')
  flags.define('--dev_corpus', '--dev',
               help='Path to the dev corpus recordio',
               default="local/data/corpora/caspar/dev.rec",
               type=str,
               metavar='FILE')
  flags.define('--word_embeddings',
               help='(Optional) Path to the word embeddings file',
               default="local/data/corpora/caspar/word2vec-32-embeddings.bin",
               type=str,
               metavar='FILE')

  # Training hyperparameters.
  # Notable omissions: decay_steps, dropout_rate.
  flags.define('--train_steps', '--steps',
               help='Number of training batches to use',
               default=100000,
               type=int,
               metavar='NUM')
  flags.define('--report_every',
               help='Checkpoint interval in number of batches',
               default=3000,
               type=int,
               metavar='NUM')
  flags.define('--batch_size', '--batch',
               help='Batch size',
               default=8,
               type=int,
               metavar='NUM')
  flags.define('--learning_method', '--optimizer',
               help='Optimizer to use: adam or sgd',
               default='adam',
               type=str,
               metavar='STR')
  flags.define('--use_moving_average',
               help='Whether or not to use exponential moving averaging',
               default=True,
               action='store_true')
  flags.define('--moving_average_coeff',
               help='Exponential moving coefficient to use',
               default=0.9999,
               type=float,
               metavar='FLOAT')
  flags.define('--gradient_clip_norm', '--gradient-clip',
               help='Gradient clip norm to use (0.0 to disable clipping)',
               default=1.0,
               type=float,
               metavar='FLOAT')
  flags.define('--learning_rate', '--alpha',
               help='Learning rate for the optimizer',
               default=0.0005,
               type=float,
               metavar='FLOAT')
  flags.define('--adam_beta1',
               help='beta1 hyperparameter for the Adam optimizer',
               default=0.01,
               type=float,
               metavar='FLOAT')
  flags.define('--adam_beta2',
               help='beta2 hyperparameter for the Adam optimizer',
               default=0.999,
               type=float,
               metavar='FLOAT')
  flags.define('--adam_eps',
               help='epsilon hyperparameter for the Adam optimizer',
               default=1e-5,
               type=float,
               metavar='FLOAT')
  flags.define('--l2_coeff', '-l2',
               help='L2 regularization coefficient',
               default=0.0001,
               type=float,
               metavar='FLOAT')


# Sets up commonly used runtime flags.
def setup_runtime_flags(flags):
  flags.define('--parser',
               help='Parser flow file',
               default="",
               type=str,
               metavar='FILE')
  flags.define('--input',
               help='Path to the input recordio file',
               default="",
               type=str,
               metavar='FILE')
  flags.define('--output',
               help='Path to the output recordio file',
               default="",
               type=str,
               metavar='FILE')
  flags.define('--evaluate',
               help='Whether to evaluate against input documents',
               default=False,
               action='store_true')
  flags.define('--trace',
               help='Whether to write tracing information in documents',
               default=False,
               action='store_true')
