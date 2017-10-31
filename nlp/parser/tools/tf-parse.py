# Copyright 2017 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License")

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

"""Runs a TF Sempar model on a corpus.
"""

import io
import subprocess
import sys
import tensorflow as tf
import timeit
import zipfile

sys.path.insert(0, "third_party/syntaxnet")

from dragnn.protos import spec_pb2
from dragnn.python import graph_builder
from google.protobuf import text_format
from tensorflow.python.platform import gfile

tf.load_op_library('bazel-bin/nlp/parser/trainer/sempar.so')

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string("parser_dir", "", "Directory containing Tensorflow model.")
flags.DEFINE_string("commons", "", "Common store.")
flags.DEFINE_string("corpus", "", "Evaluation / Benchmarking corpus.")
flags.DEFINE_string("output", "", "Output zip file name for annotated corpus.")
flags.DEFINE_integer("threads", 8, "Tensorflow's intra/inter-op parallelism.")
flags.DEFINE_integer("batch_size", 1024, "Maximum batch size.")
flags.DEFINE_bool("evaluate", False, "Perform evaluation.")

# Reads a serialized corpus into memory.
def read_corpus(file_pattern):
  docs = []
  if file_pattern.endswith(".zip"):
    with gfile.GFile(file_pattern, 'r') as f:
      buf = io.BytesIO(f.read())
      with zipfile.ZipFile(buf, 'r') as zipreader:
        docs = [None] * len(zipreader.namelist())
        for index, fname in enumerate(zipreader.namelist()):
          docs[index] = zipreader.read(fname)
  else:
    filenames = gfile.Glob(file_pattern)
    docs = [None] * len(filenames)
    for index, name in enumerate(filenames):
      with gfile.GFile(name, 'r') as f:
        docs[index] = f.read()
  print len(docs), "files in", file_pattern
  return docs


def write_corpus(filename, prefix, data):
  buf = io.BytesIO()
  with zipfile.ZipFile(buf, 'w') as z:
    for i in xrange(len(data)):
      entry = prefix + str(i)
      z.writestr(entry, data[i])
  z.close()

  with gfile.GFile(filename, 'w') as f:
    f.write(buf.getvalue())


def main(argv):
  tf.logging.set_verbosity(tf.logging.INFO)
  session_config = tf.ConfigProto(
      log_device_placement=False,
      intra_op_parallelism_threads=FLAGS.threads,
      inter_op_parallelism_threads=FLAGS.threads)

  master_spec = spec_pb2.MasterSpec()
  master_spec_file = FLAGS.parser_dir + "/master_spec"
  with gfile.GFile(master_spec_file, 'r') as fin:
    text_format.Parse(fin.read(), master_spec)

  tf.logging.info('Building the graph')
  g = tf.Graph()
  with g.as_default(), tf.device('/device:CPU:0'):
    hyperparam_config = spec_pb2.GridPoint()
    hyperparam_config.use_moving_average = True
    builder = graph_builder.MasterBuilder(master_spec, hyperparam_config)
    annotator = builder.add_annotation()
    builder.add_saver()

  with tf.Session(graph=g, config=session_config) as sess:
    tf.logging.info('Initializing variables...')
    sess.run(tf.global_variables_initializer())

    tf.logging.info('Loading from checkpoint...')
    checkpoint = FLAGS.parser_dir + "/checkpoints/best"
    sess.run('save/restore_all', {'save/Const:0': checkpoint})

    # Annotate the corpus.
    corpus = read_corpus(FLAGS.corpus)
    annotated = []
    annotation_time = 0
    for start in range(0, len(corpus), FLAGS.batch_size):
      end = min(start + FLAGS.batch_size, len(corpus))
      feed_dict = {annotator['input_batch']: corpus[start:end]}
      start_time = timeit.default_timer()
      output = sess.run(annotator['annotations'], feed_dict=feed_dict)
      annotation_time += (timeit.default_timer() - start_time)
      annotated.extend(output)

    tf.logging.info("Wall clock time for %s annotation: %f seconds",
                    len(annotated), annotation_time)

  output_file = FLAGS.output
  if FLAGS.evaluate and len(FLAGS.output) == 0:
    output_file = "/tmp/annotated.zip"
    tf.logging.info('--output not provided, will write annotated docs to %s',
                    output_file)

  if FLAGS.evaluate or len(FLAGS.output) != 0:
    # Write the annotated corpus to disk as a zip file.
    write_corpus(output_file, "test.", annotated);
    tf.logging.info('Wrote %d annotated docs to %s',
                    len(annotated), output_file)

  if FLAGS.evaluate:
    # Evaluate against gold annotations.
    try:
      eval_output_lines = subprocess.check_output(
          ['bazel-bin/nlp/parser/tools/evaluate-frames',
           '--gold_documents=' + FLAGS.corpus,
           '--test_documents=' + output_file,
           '--commons=' + FLAGS.commons],
          stderr=subprocess.STDOUT)

      eval_output = {}
      eval_metric = -1
      for line in eval_output_lines.splitlines():
        line = line.rstrip()
        tf.logging.info("Evaluation Metric: %s", line)
        parts = line.split('\t')
        assert len(parts) == 2, line
        eval_output[parts[0]] = float(parts[1])
        if line.startswith("SLOT_F1"):
          eval_metric = float(parts[1])
      assert eval_metric != -1, "Missing SLOT F1"
      tf.logging.info('Overall Evaluation Metric: %f', eval_metric)
    except subprocess.CalledProcessError as e:
      print("Evaluation failed: ", e.returncode, e.output)


if __name__ == '__main__':
  FLAGS.alsologtostderr = True
  tf.app.run()
