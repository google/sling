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

"""A program for creating and dumping a DRAGNN graph."""

import os.path
import tensorflow as tf

from dragnn.protos import spec_pb2
from dragnn.python import dragnn_ops
from dragnn.python import graph_builder
from google.protobuf import text_format

import dragnn.python.load_dragnn_cc_impl

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('master_spec', '', 'Path to a dragnn master spec proto.')
flags.DEFINE_string('hyperparams', '', 'Path to a training grid spec proto.')
flags.DEFINE_string('output_folder', '', 'Full path of the output folder.')

def create_graph(master_spec, hyperparam_config):
  with tf.Graph().as_default() as graph:
    builder = graph_builder.MasterBuilder(master_spec, hyperparam_config)

    # Construct default per-component targets.
    default_targets = [
        spec_pb2.TrainTarget(
            name=component.name,
            max_index=idx + 1,
            unroll_using_oracle=[False] * idx + [True])
        for idx, component in enumerate(master_spec.component)
        if (component.transition_system.registered_name != 'shift-only' and
            component.transition_system.registered_name != 'once')
    ]

    # Add default and manually specified targets.
    for target in default_targets:
      builder.add_training_from_config(target)

    # Construct annotation and saves. Will use moving average if enabled.
    builder.add_annotation()
    builder.add_annotation('annotation_with_trace', enable_tracing=True)
    builder.add_saver()

    # Add backwards compatible training summary.
    summaries = []
    for component in builder.components:
      summaries += component.get_summaries()
    summaries.append(
        tf.contrib.deprecated.scalar_summary('Global step', builder.master_vars[
            'step']))
    summaries.append(
        tf.contrib.deprecated.scalar_summary(
            'Learning rate', builder.master_vars['learning_rate']))
    tf.identity(
        tf.contrib.deprecated.merge_summary(summaries),
        name='training/summary/summary')

    # Construct target to initialize variables.
    tf.group(tf.global_variables_initializer(), name='inits')
    return graph


def main(unused_argv):
  hyperparam_config = spec_pb2.GridPoint()
  text_format.Parse(FLAGS.hyperparams, hyperparam_config)
  master_spec = spec_pb2.MasterSpec()

  with file(FLAGS.master_spec, 'r') as fin:
    text_format.Parse(fin.read(), master_spec)
  fin.close()

  graph = create_graph(master_spec, hyperparam_config)

  events_dir = os.path.join(FLAGS.output_folder, "logs")
  writer = tf.summary.FileWriter(events_dir, graph)
  writer.close()
  print "Wrote events (incl. graph) for Tensorboard to folder:", events_dir
  print "The graph can be viewed via"
  print "  tensorboard --logs=" + events_dir
  print "  then navigating to http://localhost:6006 and clicking on 'GRAPHS'"

  graph_file = os.path.join(FLAGS.output_folder, "graph")
  with file(graph_file, 'w') as fout:
    fout.write(graph.as_graph_def().SerializeToString())
  fout.close()
  print "Also wrote serialized graph proto separately to", graph_file


if __name__ == '__main__':
  FLAGS.alsologtostderr = True
  tf.app.run()
