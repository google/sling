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

"""Creates a DRAGNN Sempar graph from a MasterSpec and training with it.
Although it can be called by itself, it is best invoked from train.sh
"""

import glob
import os
import subprocess
import sys
import tensorflow as tf
import zipfile

sys.path.insert(0, "third_party/syntaxnet")

from dragnn.protos import spec_pb2
from dragnn.python import dragnn_ops
from dragnn.python import graph_builder
from dragnn.python import trainer_lib
from dragnn.python import check
from google.protobuf import text_format
from convert import convert_model

tf.load_op_library('bazel-bin/nlp/parser/trainer/sempar.so')

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('master_spec', '',
                    'Path to a complete dragnn master spec text proto.')
flags.DEFINE_string('hyperparams', '', 'Training grid spec text proto.')
flags.DEFINE_string('output_folder', '', 'Full path of the output folder.')
flags.DEFINE_string('commons', '', 'Path to commons.')
flags.DEFINE_string('train_corpus', '', 'Training corpus.')
flags.DEFINE_string('dev_corpus', '', 'Dev corpus with gold frames.')
flags.DEFINE_string('dev_corpus_without_gold', '',
                    'Dev corpus without gold frames.')
flags.DEFINE_string('tf_master', '',
                    'TensorFlow execution engine to connect to.')
flags.DEFINE_integer('train_steps', 200000, 'Number of training steps')
flags.DEFINE_integer('report_every', 500, 'Checkpoint interval')
flags.DEFINE_integer('batch_size', 8, 'Training batch size')
flags.DEFINE_string('flow', '', 'Myelin flow file for model output')

def read_corpus(file_pattern):
  docs = []
  if zipfile.is_zipfile(file_pattern):
    with zipfile.ZipFile(file_pattern, 'r') as zipreader:
      docs = [None] * len(zipreader.namelist())
      for index, fname in enumerate(zipreader.namelist()):
        docs[index] = zipreader.read(fname)
  else:
    filenames = glob.glob(file_pattern)
    docs = [None] * len(filenames)
    for index, name in enumerate(filenames):
      with open(name, 'r') as file:
        docs[index] = file.read()
  print len(docs), "files in", file_pattern
  return docs


def evaluator(gold_docs, test_docs):
  check.Eq(len(gold_docs), len(test_docs), "Unequal #docs during evaluation")

  folder = os.path.join(FLAGS.output_folder, "tmp_docs")
  empty_dir(folder)

  gold_zip_name = os.path.join(folder, "dev.gold.zip")
  test_zip_name = os.path.join(folder, "dev.test.zip")

  with zipfile.ZipFile(gold_zip_name,  'w') as gold:
    for i in xrange(len(gold_docs)):
      filename = "gold." + str(i)
      gold.writestr(filename, gold_docs[i])
    gold.close()

  with zipfile.ZipFile(test_zip_name,  'w') as test:
    for i in xrange(len(test_docs)):
      filename = "test." + str(i)
      test.writestr(filename, test_docs[i])
    test.close()

  try:
    output = subprocess.check_output(
        ['bazel-bin/nlp/parser/trainer/evaluate-frames',
         '--gold_documents=' + gold_zip_name,
         '--test_documents=' + test_zip_name,
         '--commons=' + FLAGS.commons],
        stderr=subprocess.STDOUT)
  except subprocess.CalledProcessError as e:
    print("Evaluation failed: ", e.returncode, e.output)
    return {'eval_metric': 0.0}

  eval_output = {}
  for line in output.splitlines():
    line = line.rstrip()
    tf.logging.info("Evaluation Metric: %s", line)
    parts = line.split('\t')
    check.Eq(len(parts), 2, line)
    eval_output[parts[0]] = float(parts[1])
    if line.startswith("SLOT_F1"):
      eval_output['eval_metric'] = float(parts[1])

  check.IsTrue(eval_output.has_key('eval_metric'), str(eval_output))
  return eval_output


def empty_dir(folder):
  if tf.gfile.IsDirectory(folder):
    tf.gfile.DeleteRecursively(folder)
  elif tf.gfile.Exists(folder):
    tf.gfile.Remove(folder)
  tf.gfile.MakeDirs(folder)


def main(unused_argv):
  tf.logging.set_verbosity(tf.logging.INFO)
  # Read hyperparams and master spec.
  hyperparam_config = spec_pb2.GridPoint()
  text_format.Parse(FLAGS.hyperparams, hyperparam_config)
  print hyperparam_config
  master_spec = spec_pb2.MasterSpec()

  with file(FLAGS.master_spec, 'r') as fin:
    text_format.Parse(fin.read(), master_spec)
  fin.close()

  # Make output folder
  if not os.path.isdir(FLAGS.output_folder):
    os.makedirs(FLAGS.output_folder)

  # Construct TF Graph.
  graph = tf.Graph()

  with graph.as_default():
    builder = graph_builder.MasterBuilder(master_spec, hyperparam_config)

    # Construct default per-component targets.
    default_targets = [
        spec_pb2.TrainTarget(
            name=component.name,
            max_index=idx + 1,
            unroll_using_oracle=[False] * idx + [True])
        for idx, component in enumerate(master_spec.component)
        if (component.transition_system.registered_name != 'shift-only')
    ]

    # Add default and manually specified targets.
    trainers = []
    for target in default_targets:
      trainers += [builder.add_training_from_config(target)]
    check.Eq(len(trainers), 1, "Expected only one training target (FF unit)")

    # Construct annotation and saves. Will use moving average if enabled.
    annotator = builder.add_annotation()
    builder.add_saver()

    # Add backwards compatible training summary.
    summaries = []
    for component in builder.components:
      summaries += component.get_summaries()
    merged_summaries = tf.summary.merge_all()

    # Construct target to initialize variables.
    tf.group(tf.global_variables_initializer(), name='inits')

  # Prepare tensorboard dir.
  events_dir = os.path.join(FLAGS.output_folder, "tensorboard")
  empty_dir(events_dir)
  summary_writer = tf.summary.FileWriter(events_dir, graph)
  print "Wrote events (incl. graph) for Tensorboard to folder:", events_dir
  print "The graph can be viewed via"
  print "  tensorboard --logdir=" + events_dir
  print "  then navigating to http://localhost:6006 and clicking on 'GRAPHS'"

  with graph.as_default():
    tf.set_random_seed(hyperparam_config.seed)

  # Read train and dev corpora.
  print "Reading corpora..."
  train_corpus = read_corpus(FLAGS.train_corpus)
  dev_corpus_with_gold = read_corpus(FLAGS.dev_corpus)
  dev_corpus_without_gold = read_corpus(FLAGS.dev_corpus_without_gold)

  # Prepare checkpoint folder.
  checkpoint_path = os.path.join(FLAGS.output_folder, 'checkpoints/best')
  checkpoint_dir = os.path.dirname(checkpoint_path)
  empty_dir(checkpoint_dir)

  with tf.Session(FLAGS.tf_master, graph=graph) as sess:
    # Make sure to re-initialize all underlying state.
    sess.run(tf.global_variables_initializer())

    # Run training.
    trainer_lib.run_training(
        sess, trainers, annotator,
        evaluator, [0], # pretrain_steps
        [FLAGS.train_steps], train_corpus, dev_corpus_without_gold,
        dev_corpus_with_gold, FLAGS.batch_size, summary_writer,
        FLAGS.report_every, builder.saver, checkpoint_path)

    # Convert model to a Myelin flow.
    if len(FLAGS.flow) != 0:
      tf.logging.info('Saving flow to %s', FLAGS.flow)
      flow = convert_model(master_spec, sess)
      flow.save(FLAGS.flow)

  tf.logging.info('Best checkpoint written to %s', checkpoint_path)


if __name__ == '__main__':
  FLAGS.alsologtostderr = True
  tf.app.run()
