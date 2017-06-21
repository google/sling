#!/bin/bash
#
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

# Script for training a Sempar model from scratch.
#
# It should be run from the top-level folder (i.e. the one which contains the
# 'nlp' and 'frame' subfolders).
# Usage:
#   /path/to/this/script <train filepattern> <path to commons> <output folder>
#
# It takes as input:
# - A file pattern of training documents with gold annotations. Each file should
#   correspond to one serialized Document frame.
# - Path to the commons store.
# - Name of the output folder. All generated resources e.g. feature lexicons,
#   action table, master spec, TF graph, trained model etc. will be dumped here.

# It performs the following steps:
# - Builds the action table.
# - Builds resources needed by the features.
# - Builds a complete MasterSpec proto.
# - Builds a TF graph using the master spec and default hyperparameters.

# Tweaks:
# - The features and component attributes (e.g. hidden layer size) are
#   hard-coded in generate-master-spec.cc and can be changed there.
# - The default training hyperparameters are hard-coded in this script and
#   can be changed in $HYPERPARAMS below.

set -eux

readonly TRAIN_FILEPATTERN=$1
readonly COMMONS=$2
readonly OUTPUT_FOLDER=$3

HYPERPARAMS="learning_rate:0.0005 decay_steps=800000 "
HYPERPARAMS+="seed:1 learning_method:'adam' "
HYPERPARAMS+="use_moving_average:true dropout_rate:0.9 "
HYPERPARAMS+="gradient_clip_norm:1.0 adam_beta1:0.01 "
HYPERPARAMS+="adam_beta2:0.999 adam_eps:0.0001"

bazel build -c opt nlp/parser/trainer:generate-master-spec

bazel-bin/nlp/parser/trainer/generate-master-spec \
  --documents=${TRAIN_FILEPATTERN} \
  --commons=${COMMONS}

python nlp/parser/trainer/graph-builder-main.py \
  --master_spec="${OUTPUT_FOLDER}/master_spec" \
  --hyperparams=${HYPERPARAMS} \
  --output_folder=${OUTPUT_FOLDER}

echo "Done."
